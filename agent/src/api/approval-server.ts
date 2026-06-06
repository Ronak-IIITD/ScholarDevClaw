/**
 * Approval HTTP Server
 *
 * A minimal HTTP API for managing approval gates from an external UI
 * (or CLI). The server is a thin shell over `ApprovalAudit` — it handles
 * request parsing, bearer-token auth, body size limits, and request
 * timeouts, then delegates all read/write logic to the audit module.
 *
 * Endpoints:
 *
 *   GET  /health
 *     → { status: "ok", uptimeMs, pendingCount }
 *
 *   GET  /approvals/pending
 *     → { pending: PendingApprovalEntry[] }
 *
 *   GET  /approvals/runs/:runId
 *     → { runId, approvals: ApprovalRecord[] }
 *
 *   GET  /approvals/audit?integrationId=...
 *     → IntegrationAudit
 *
 *   POST /approvals/:integrationId/:phase/decide
 *     body: { action: "approved" | "rejected", notes?: string }
 *     → { ok: true, integrationId, phase, action }
 *
 *   POST /approvals/batch
 *     body: { decisions: [{ integrationId, phase, action, notes? }, ...] }
 *     → { results: [{ ok, error? }, ...] }
 *
 * Auth: when `authKey` is configured, requests must carry
 * `Authorization: Bearer <key>`. When no key is configured, the server
 * refuses to start (we never want to ship an open approval API).
 */

import { createServer, IncomingMessage, Server, ServerResponse } from 'node:http';
import { timingSafeEqual } from 'node:crypto';
import { logger } from '../utils/logger.js';
import { ApprovalAudit, type IntegrationAudit, type PendingApprovalEntry } from './approval-audit.js';
import type { ApprovalRecord } from '../utils/run-store.js';

const DEFAULT_MAX_BODY_BYTES = 1 * 1024 * 1024; // 1 MB
const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;

export const APPROVAL_AUTH_HEADER = 'authorization';
export const APPROVAL_BEARER_PREFIX = 'Bearer ';

export interface ApprovalServerOptions {
  /** Bearer-token secret. Server refuses to start without it. */
  authKey: string;
  /** Maximum allowed request body size in bytes (default 1 MB). */
  maxBodyBytes?: number;
  /** Per-socket request timeout in ms (default 30s). */
  requestTimeoutMs?: number;
  /** Health-check path (default '/health'). */
  healthPath?: string;
  /** Base path for approval endpoints (default '/approvals'). */
  basePath?: string;
}

export interface ApprovalServerDeps {
  /** Audit aggregator. */
  audit: ApprovalAudit;
}

/**
 * JSON response helper that always sets the right Content-Type and
 * status code. Errors use a stable { error, code? } shape.
 */
function sendJson(
  res: ServerResponse,
  status: number,
  body: unknown,
): void {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(payload, 'utf8'),
    'Cache-Control': 'no-store',
  });
  res.end(payload);
}

function sendError(
  res: ServerResponse,
  status: number,
  message: string,
  code?: string,
): void {
  sendJson(res, status, code ? { error: message, code } : { error: message });
}

/**
 * Read the request body with a hard size cap. Resolves with the raw
 * string (utf8) on success, throws on overflow or read error.
 */
async function readBody(
  req: IncomingMessage,
  maxBytes: number,
): Promise<string> {
  return await new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;
    let aborted = false;

    req.on('data', (chunk: Buffer) => {
      if (aborted) return;
      total += chunk.length;
      if (total > maxBytes) {
        aborted = true;
        const err = new Error('Request body exceeds maximum size');
        (err as Error & { statusCode?: number }).statusCode = 413;
        req.destroy();
        reject(err);
        return;
      }
      chunks.push(chunk);
    });

    req.on('end', () => {
      if (aborted) return;
      resolve(Buffer.concat(chunks).toString('utf8'));
    });

    req.on('error', (err) => {
      if (aborted) return;
      reject(err);
    });
  });
}

/**
 * Constant-time bearer-token comparison. Returns false for any input
 * that doesn't match the expected prefix or length.
 */
export function verifyBearer(provided: string | undefined, expected: string): boolean {
  if (!provided) return false;
  if (!provided.startsWith(APPROVAL_BEARER_PREFIX)) return false;
  const token = provided.slice(APPROVAL_BEARER_PREFIX.length);
  if (token.length !== expected.length) return false;
  try {
    return timingSafeEqual(Buffer.from(token, 'utf8'), Buffer.from(expected, 'utf8'));
  } catch {
    return false;
  }
}

/**
 * Coerce an arbitrary string path-parameter into a positive integer
 * phase number. Returns null when the input is not a valid phase.
 */
export function parsePhase(raw: string): number | null {
  if (!/^[0-9]+$/.test(raw)) return null;
  const n = Number.parseInt(raw, 10);
  if (!Number.isInteger(n) || n < 0 || n > 99) return null;
  return n;
}

/**
 * HTTP server for the approval gate. Exposes typed request handlers
 * over `ApprovalAudit`. See module docstring for the endpoint catalog.
 */
export class ApprovalServer {
  private server: Server | null = null;
  private listeningPort: number | null = null;
  private readonly startedAt = Date.now();
  private readonly options: Required<Omit<ApprovalServerOptions, 'authKey'>> & { authKey: string };
  private readonly deps: ApprovalServerDeps;

  constructor(options: ApprovalServerOptions, deps: ApprovalServerDeps) {
    if (!options.authKey) {
      throw new Error('ApprovalServer requires a non-empty authKey');
    }
    this.options = {
      authKey: options.authKey,
      maxBodyBytes: options.maxBodyBytes ?? DEFAULT_MAX_BODY_BYTES,
      requestTimeoutMs: options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS,
      healthPath: options.healthPath ?? '/health',
      basePath: options.basePath ?? '/approvals',
    };
    this.deps = deps;
  }

  /**
   * Start listening on the given port. Pass 0 to bind to a random
   * free port (useful for tests). Returns the actual bound port.
   */
  async start(port: number): Promise<number> {
    if (this.server) {
      throw new Error('ApprovalServer already started');
    }

    const server = createServer((req, res) => {
      this.handleRequest(req, res).catch((err) => {
        logger.error('Approval request failed', {
          error: err instanceof Error ? err.message : String(err),
        });
        if (!res.headersSent) {
          sendError(res, 500, 'Internal server error');
        } else {
          res.end();
        }
      });
    });

    server.requestTimeout = this.options.requestTimeoutMs;

    await new Promise<void>((resolve, reject) => {
      const onError = (err: Error) => {
        server.removeListener('listening', onListening);
        reject(err);
      };
      const onListening = () => {
        server.removeListener('error', onError);
        resolve();
      };
      server.once('error', onError);
      server.once('listening', onListening);
      server.listen(port, '127.0.0.1');
    });

    this.server = server;
    const addr = server.address();
    if (addr && typeof addr === 'object') {
      this.listeningPort = addr.port;
    }
    logger.info('Approval server listening', { port: this.listeningPort });
    return this.listeningPort ?? port;
  }

  async stop(): Promise<void> {
    const server = this.server;
    if (!server) return;
    this.server = null;
    this.listeningPort = null;
    await new Promise<void>((resolve) => {
      server.close(() => resolve());
    });
  }

  /** Returns the port the server is bound to, or null if not started. */
  get port(): number | null {
    return this.listeningPort;
  }

  // -------------------------------------------------------------------
  // Request handling
  // -------------------------------------------------------------------

  private async handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const url = req.url ?? '/';
    const method = (req.method ?? 'GET').toUpperCase();
    const path = url.split('?')[0];

    // Health check is unauthenticated by design.
    if (method === 'GET' && path === this.options.healthPath) {
      await this.handleHealth(res);
      return;
    }

    // Everything else requires a valid bearer token.
    if (!verifyBearer(req.headers[APPROVAL_AUTH_HEADER] as string | undefined, this.options.authKey)) {
      sendError(res, 401, 'Unauthorized', 'unauthorized');
      return;
    }

    try {
      if (method === 'GET' && path === `${this.options.basePath}/pending`) {
        await this.handleListPending(res);
        return;
      }
      if (method === 'GET' && path.startsWith(`${this.options.basePath}/runs/`)) {
        const runId = decodeURIComponent(path.slice(`${this.options.basePath}/runs/`.length));
        await this.handleGetRunApprovals(res, runId);
        return;
      }
      if (method === 'GET' && path === `${this.options.basePath}/audit`) {
        const integrationId = this.getQueryParam(url, 'integrationId');
        if (!integrationId) {
          sendError(res, 400, 'Missing required query parameter: integrationId', 'missing_param');
          return;
        }
        await this.handleGetAudit(res, integrationId);
        return;
      }
      if (method === 'POST' && path === `${this.options.basePath}/batch`) {
        await this.handleBatch(req, res);
        return;
      }
      if (method === 'POST' && this.matchesDecidePath(path)) {
        const { integrationId, phase } = this.parseDecidePath(path);
        if (phase === null) {
          sendError(res, 400, 'Invalid phase in URL', 'invalid_phase');
          return;
        }
        await this.handleDecide(req, res, integrationId, phase);
        return;
      }

      sendError(res, 404, 'Not found', 'not_found');
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const statusCode = (err as Error & { statusCode?: number }).statusCode ?? 500;
      logger.error('Approval request handler error', { error: message, statusCode });
      sendError(res, statusCode, message);
    }
  }

  private matchesDecidePath(path: string): boolean {
    const prefix = `${this.options.basePath}/`;
    if (!path.startsWith(prefix)) return false;
    const rest = path.slice(prefix.length);
    // Format: <integrationId>/<phase>/decide
    const parts = rest.split('/');
    return parts.length === 3 && parts[2] === 'decide';
  }

  private parseDecidePath(path: string): { integrationId: string; phase: number | null } {
    const prefix = `${this.options.basePath}/`;
    const rest = path.slice(prefix.length);
    const parts = rest.split('/');
    return {
      integrationId: decodeURIComponent(parts[0]),
      phase: parsePhase(parts[1]),
    };
  }

  private getQueryParam(url: string, name: string): string | null {
    const queryStart = url.indexOf('?');
    if (queryStart < 0) return null;
    const params = new URLSearchParams(url.slice(queryStart + 1));
    const value = params.get(name);
    return value && value.length > 0 ? value : null;
  }

  // -------------------------------------------------------------------
  // Endpoint handlers
  // -------------------------------------------------------------------

  private async handleHealth(res: ServerResponse): Promise<void> {
    let pendingCount = 0;
    try {
      const pending = await this.deps.audit.listPending();
      pendingCount = pending.length;
    } catch (err) {
      logger.warn('Health check: pending count failed', {
        error: err instanceof Error ? err.message : String(err),
      });
    }
    sendJson(res, 200, {
      status: 'ok',
      uptimeMs: Date.now() - this.startedAt,
      pendingCount,
    });
  }

  private async handleListPending(res: ServerResponse): Promise<void> {
    const pending = await this.deps.audit.listPending();
    sendJson(res, 200, { pending });
  }

  private async handleGetRunApprovals(res: ServerResponse, runId: string): Promise<void> {
    if (!runId) {
      sendError(res, 400, 'Missing runId in URL', 'missing_param');
      return;
    }
    const approvals = await this.deps.audit.getRunApprovals(runId);
    sendJson(res, 200, { runId, approvals });
  }

  private async handleGetAudit(res: ServerResponse, integrationId: string): Promise<void> {
    const audit: IntegrationAudit = await this.deps.audit.getIntegrationAudit(integrationId);
    sendJson(res, 200, audit);
  }

  private async handleDecide(
    req: IncomingMessage,
    res: ServerResponse,
    integrationId: string,
    phase: number,
  ): Promise<void> {
    const body = await readBody(req, this.options.maxBodyBytes);
    const parsed = this.parseDecideBody(body);
    if ('error' in parsed) {
      sendError(res, 400, parsed.error, 'invalid_body');
      return;
    }
    const [result] = await this.deps.audit.decide([
      { integrationId, phase, action: parsed.action, notes: parsed.notes },
    ]);
    if (!result.ok) {
      sendError(res, 502, result.error ?? 'Upstream decision failed', 'upstream_error');
      return;
    }
    sendJson(res, 200, { ok: true, integrationId, phase, action: parsed.action });
  }

  private async handleBatch(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const body = await readBody(req, this.options.maxBodyBytes);
    let parsed: unknown;
    try {
      parsed = JSON.parse(body);
    } catch (err) {
      sendError(res, 400, 'Invalid JSON body', 'invalid_json');
      return;
    }
    if (
      !parsed || typeof parsed !== 'object'
      || !Array.isArray((parsed as { decisions?: unknown }).decisions)
    ) {
      sendError(res, 400, 'Body must be { decisions: [...] }', 'invalid_body');
      return;
    }
    const rawDecisions = (parsed as { decisions: unknown[] }).decisions;
    const decisions = [];
    for (let i = 0; i < rawDecisions.length; i += 1) {
      const d = this.validateBatchDecision(rawDecisions[i]);
      if ('error' in d) {
        sendError(res, 400, `decisions[${i}]: ${d.error}`, 'invalid_body');
        return;
      }
      decisions.push(d.value);
    }
    const results = await this.deps.audit.decide(decisions);
    const ok = results.every((r) => r.ok);
    sendJson(res, ok ? 200 : 207, { ok, results });
  }

  private parseDecideBody(
    body: string,
  ):
    | { action: 'approved' | 'rejected'; notes?: string }
    | { error: string } {
    let parsed: unknown;
    try {
      parsed = body.length === 0 ? {} : JSON.parse(body);
    } catch {
      return { error: 'Invalid JSON body' };
    }
    if (!parsed || typeof parsed !== 'object') {
      return { error: 'Body must be a JSON object' };
    }
    const obj = parsed as { action?: unknown; notes?: unknown };
    if (obj.action !== 'approved' && obj.action !== 'rejected') {
      return { error: 'action must be "approved" or "rejected"' };
    }
    const out: { action: 'approved' | 'rejected'; notes?: string } = { action: obj.action };
    if (typeof obj.notes === 'string' && obj.notes.length > 0) {
      out.notes = obj.notes;
    }
    return out;
  }

  private validateBatchDecision(
    raw: unknown,
  ):
    | {
        value: { integrationId: string; phase: number; action: 'approved' | 'rejected'; notes?: string };
      }
    | { error: string } {
    if (!raw || typeof raw !== 'object') {
      return { error: 'must be an object' };
    }
    const r = raw as { integrationId?: unknown; phase?: unknown; action?: unknown; notes?: unknown };
    if (typeof r.integrationId !== 'string' || r.integrationId.length === 0) {
      return { error: 'integrationId must be a non-empty string' };
    }
    if (typeof r.phase !== 'number' || !Number.isInteger(r.phase) || r.phase < 0 || r.phase > 99) {
      return { error: 'phase must be an integer in [0, 99]' };
    }
    if (r.action !== 'approved' && r.action !== 'rejected') {
      return { error: 'action must be "approved" or "rejected"' };
    }
    const value: {
      integrationId: string;
      phase: number;
      action: 'approved' | 'rejected';
      notes?: string;
    } = {
      integrationId: r.integrationId,
      phase: r.phase,
      action: r.action,
    };
    if (typeof r.notes === 'string' && r.notes.length > 0) {
      value.notes = r.notes;
    }
    return { value };
  }
}

// Re-export for convenience so callers don't have to import two modules
// when they want to format a pending row.
export { formatPendingRow } from './approval-audit.js';
export type { PendingApprovalEntry, ApprovalRecord };
