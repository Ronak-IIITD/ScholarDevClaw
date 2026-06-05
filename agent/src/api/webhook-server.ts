import { createServer, IncomingMessage, Server, ServerResponse } from 'node:http';
import { createHmac, timingSafeEqual } from 'node:crypto';
import { EventEmitter } from 'node:events';
import { logger } from '../utils/logger.js';

export const WEBHOOK_SIGNATURE_HEADER = 'x-hub-signature-256';
export const WEBHOOK_EVENT_HEADER = 'x-github-event';
export const WEBHOOK_DELIVERY_HEADER = 'x-github-delivery';

const DEFAULT_MAX_BODY_BYTES = 5 * 1024 * 1024; // 5 MB cap on incoming payloads
const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;

export interface WebhookServerOptions {
  /** HMAC SHA-256 secret used to verify GitHub signatures. */
  secret?: string;
  /** Maximum allowed request body size in bytes (default 5 MB). */
  maxBodyBytes?: number;
  /** Per-socket request timeout in ms (default 30s). */
  requestTimeoutMs?: number;
  /** Health-check path (default '/healthz'). */
  healthPath?: string;
  /** Webhook path (default '/webhooks/github'). */
  webhookPath?: string;
}

export interface WebhookRequest {
  /** Raw event name (e.g. 'pull_request', 'push', 'ping'). */
  event: string;
  /** Unique delivery id from GitHub. */
  deliveryId: string;
  /** Parsed JSON payload. */
  payload: unknown;
  /** Raw body string (available when handlers need it). */
  rawBody: string;
}

type WebhookHandler = (req: WebhookRequest) => void | Promise<void>;

/**
 * Minimal HTTP server that receives GitHub webhooks. The server:
 * - Verifies HMAC SHA-256 signatures when a secret is configured
 * - Parses JSON bodies up to a configurable size cap
 * - Emits typed events (pull_request, check_run, push, ping, ...)
 *   on the internal EventEmitter
 * - Returns 202 on accepted events so GitHub does not retry
 */
export class WebhookServer extends EventEmitter {
  private server: Server | null = null;
  private readonly options: Required<Omit<WebhookServerOptions, 'secret'>> & {
    secret: string | undefined;
  };
  private listeningPort: number | null = null;
  /** Map of event name -> map of original handler -> wrapped listener. */
  private readonly wrappedHandlers = new Map<string, Map<WebhookHandler, WebhookHandler>>();

  constructor(options: WebhookServerOptions = {}) {
    super();
    this.options = {
      secret: options.secret,
      maxBodyBytes: options.maxBodyBytes ?? DEFAULT_MAX_BODY_BYTES,
      requestTimeoutMs: options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS,
      healthPath: options.healthPath ?? '/healthz',
      webhookPath: options.webhookPath ?? '/webhooks/github',
    };
  }

  /**
   * Verify a GitHub-style signature header against a raw body.
   * Returns true when signatures match (or when no secret is configured).
   */
  verifySignature(rawBody: string, signatureHeader: string | undefined): boolean {
    if (!this.options.secret) {
      // No secret configured: accept every request. Callers can opt-in by
      // setting a secret at construction time.
      return true;
    }
    if (!signatureHeader) return false;
    if (!signatureHeader.startsWith('sha256=')) return false;
    const provided = signatureHeader.slice('sha256='.length);
    const expected = createHmac('sha256', this.options.secret)
      .update(rawBody, 'utf8')
      .digest('hex');

    if (provided.length !== expected.length) return false;
    try {
      return timingSafeEqual(Buffer.from(provided, 'hex'), Buffer.from(expected, 'hex'));
    } catch {
      return false;
    }
  }

  /**
   * Register a handler for a specific GitHub event. Typed handlers are
   * dispatched in registration order; handler errors are caught and logged
   * so a single bad handler cannot break the chain.
   */
  on(event: string, handler: WebhookHandler): this {
    const wrapped = async (req: WebhookRequest) => {
      try {
        await handler(req);
      } catch (err) {
        logger.error('Webhook handler error', {
          event: req.event,
          deliveryId: req.deliveryId,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    };
    if (!this.wrappedHandlers.has(event)) {
      this.wrappedHandlers.set(event, new Map());
    }
    this.wrappedHandlers.get(event)!.set(handler, wrapped);
    return super.on(event, wrapped);
  }

  /** Remove a previously registered handler. */
  off(event: string, handler: WebhookHandler): this {
    const wrapped = this.wrappedHandlers.get(event)?.get(handler);
    if (wrapped) {
      this.wrappedHandlers.get(event)!.delete(handler);
      if (this.wrappedHandlers.get(event)!.size === 0) {
        this.wrappedHandlers.delete(event);
      }
      return super.off(event, wrapped);
    }
    // Fallback: best-effort removal by reference.
    return super.off(event, handler as (...args: unknown[]) => void);
  }

  /** Number of registered handlers for an event (test helper). */
  listenerCount(event: string): number {
    return this.wrappedHandlers.get(event)?.size ?? 0;
  }

  /** Port the server is currently bound to, or null if not started. */
  get port(): number | null {
    return this.listeningPort;
  }

  /**
   * Start listening on the given port. Port 0 lets the OS pick a free port,
   * useful for tests.
   */
  async start(port: number = 0, host: string = '127.0.0.1'): Promise<number> {
    if (this.server) {
      throw new Error('WebhookServer is already running');
    }

    this.server = createServer((req, res) => this.handleRequest(req, res));
    this.server.requestTimeout = this.options.requestTimeoutMs;

    await new Promise<void>((resolve, reject) => {
      const onError = (err: Error) => {
        this.server?.removeListener('listening', onListening);
        reject(err);
      };
      const onListening = () => {
        this.server?.removeListener('error', onError);
        resolve();
      };
      this.server!.once('error', onError);
      this.server!.once('listening', onListening);
      this.server!.listen(port, host);
    });

    const address = this.server.address();
    this.listeningPort = typeof address === 'object' && address ? address.port : port;
    logger.info('Webhook server started', {
      port: this.listeningPort,
      host,
      webhookPath: this.options.webhookPath,
      secretConfigured: Boolean(this.options.secret),
    });
    return this.listeningPort;
  }

  /** Stop the server and release the port. */
  async stop(): Promise<void> {
    if (!this.server) return;
    const server = this.server;
    this.server = null;
    this.listeningPort = null;
    await new Promise<void>((resolve) => {
      server.close(() => resolve());
    });
    logger.info('Webhook server stopped');
  }

  private async handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
    if (req.method !== 'POST' && req.method !== 'GET') {
      this.sendJson(res, 405, { error: 'method_not_allowed' });
      return;
    }

    const url = req.url ?? '/';

    if (url === this.options.healthPath) {
      this.sendJson(res, 200, { status: 'ok' });
      return;
    }

    if (url !== this.options.webhookPath) {
      this.sendJson(res, 404, { error: 'not_found' });
      return;
    }

    if (req.method === 'GET') {
      // GitHub sends a delivery probe on creation; accept it.
      this.sendJson(res, 200, { status: 'ready' });
      return;
    }

    const readResult = await this.readBody(req, this.options.maxBodyBytes);
    if (readResult.kind === 'too_large') {
      this.sendJson(res, 413, { error: 'payload_too_large' });
      return;
    }
    if (readResult.kind === 'error') {
      this.sendJson(res, 400, { error: 'read_error' });
      return;
    }
    const rawBody = readResult.body;

    const signature = this.headerValue(req, WEBHOOK_SIGNATURE_HEADER);
    if (!this.verifySignature(rawBody, signature)) {
      logger.warn('Rejected webhook with invalid signature', {
        event: this.headerValue(req, WEBHOOK_EVENT_HEADER),
        delivery: this.headerValue(req, WEBHOOK_DELIVERY_HEADER),
      });
      this.sendJson(res, 401, { error: 'invalid_signature' });
      return;
    }

    let payload: unknown;
    try {
      payload = rawBody.length > 0 ? JSON.parse(rawBody) : {};
    } catch (err) {
      logger.warn('Failed to parse webhook JSON body', {
        error: err instanceof Error ? err.message : String(err),
      });
      this.sendJson(res, 400, { error: 'invalid_json' });
      return;
    }

    const event = this.headerValue(req, WEBHOOK_EVENT_HEADER) ?? 'unknown';
    const deliveryId = this.headerValue(req, WEBHOOK_DELIVERY_HEADER) ?? 'unknown';
    const webhookRequest: WebhookRequest = { event, deliveryId, payload, rawBody };

    // Respond 202 first so GitHub does not retry, then run handlers.
    this.sendJson(res, 202, { status: 'accepted', event, deliveryId });

    this.dispatch(webhookRequest).catch((err) => {
      logger.error('Webhook handler threw', {
        event,
        deliveryId,
        error: err instanceof Error ? err.message : String(err),
      });
    });
  }

  private async dispatch(req: WebhookRequest): Promise<void> {
    // Emit both the typed event and the wildcard. The typed listeners are
    // already wrapped in error handling by `on()`.
    this.emit(req.event, req);
    this.emit('*', req);
  }

  private readBody(
    req: IncomingMessage,
    maxBytes: number,
  ): Promise<
    { kind: 'ok'; body: string } | { kind: 'too_large' } | { kind: 'error' }
  > {
    return new Promise((resolve) => {
      const chunks: Buffer[] = [];
      let total = 0;
      let settled = false;

      const finish = (
        result:
          | { kind: 'ok'; body: string }
          | { kind: 'too_large' }
          | { kind: 'error' },
      ) => {
        if (settled) return;
        settled = true;
        resolve(result);
      };

      req.on('data', (chunk: Buffer) => {
        if (settled) return;
        total += chunk.length;
        if (total > maxBytes) {
          // Stop accumulating, but keep the request readable so the response
          // can be sent back on the same socket.
          req.pause();
          finish({ kind: 'too_large' });
          return;
        }
        chunks.push(chunk);
      });
      req.on('end', () => {
        if (settled) return;
        finish({ kind: 'ok', body: Buffer.concat(chunks).toString('utf8') });
      });
      req.on('error', (err) => {
        if (settled) return;
        logger.warn('Failed to read webhook body', {
          error: err instanceof Error ? err.message : String(err),
        });
        finish({ kind: 'error' });
      });
    });
  }

  private headerValue(req: IncomingMessage, name: string): string | undefined {
    const raw = req.headers[name];
    if (Array.isArray(raw)) return raw[0];
    return raw;
  }

  private sendJson(res: ServerResponse, status: number, body: Record<string, unknown>): void {
    res.statusCode = status;
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify(body));
  }
}
