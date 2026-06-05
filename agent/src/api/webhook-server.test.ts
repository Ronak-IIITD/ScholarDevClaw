import { createHmac } from 'node:crypto';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../utils/logger.js', () => ({
  logger: {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  },
}));

import {
  WebhookServer,
  WEBHOOK_DELIVERY_HEADER,
  WEBHOOK_EVENT_HEADER,
  WEBHOOK_SIGNATURE_HEADER,
  type WebhookRequest,
} from './webhook-server.js';

const SECRET = 'super-secret-token';

function sign(body: string, secret: string = SECRET): string {
  return 'sha256=' + createHmac('sha256', secret).update(body, 'utf8').digest('hex');
}

async function postJson(
  port: number,
  body: string,
  headers: Record<string, string> = {},
): Promise<Response> {
  return await fetch(`http://127.0.0.1:${port}/webhooks/github`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...headers },
    body,
  });
}

describe('WebhookServer', () => {
  let server: WebhookServer;
  let port: number;

  describe('verifySignature', () => {
    it('accepts every signature when no secret is configured', () => {
      const s = new WebhookServer();
      expect(s.verifySignature('{}', undefined)).toBe(true);
      expect(s.verifySignature('{}', 'sha256=deadbeef')).toBe(true);
    });

    it('accepts a correctly signed body', () => {
      const s = new WebhookServer({ secret: SECRET });
      const body = '{"hello":"world"}';
      expect(s.verifySignature(body, sign(body))).toBe(true);
    });

    it('rejects a tampered body', () => {
      const s = new WebhookServer({ secret: SECRET });
      const body = '{"hello":"world"}';
      const signature = sign(body);
      expect(s.verifySignature(body + ' ', signature)).toBe(false);
    });

    it('rejects a body signed with a different secret', () => {
      const s = new WebhookServer({ secret: SECRET });
      const body = '{"hello":"world"}';
      expect(s.verifySignature(body, sign(body, 'other-secret'))).toBe(false);
    });

    it('rejects when the signature header is missing', () => {
      const s = new WebhookServer({ secret: SECRET });
      expect(s.verifySignature('{}', undefined)).toBe(false);
    });

    it('rejects malformed signature headers', () => {
      const s = new WebhookServer({ secret: SECRET });
      expect(s.verifySignature('{}', 'md5=abc')).toBe(false);
      expect(s.verifySignature('{}', 'sha256=zz')).toBe(false);
    });
  });

  describe('HTTP behavior', () => {
    beforeEach(async () => {
      server = new WebhookServer({ secret: SECRET });
      port = await server.start(0);
    });

    afterEach(async () => {
      await server.stop();
    });

    it('responds 200 to health check', async () => {
      const res = await fetch(`http://127.0.0.1:${port}/healthz`);
      expect(res.status).toBe(200);
      const body = (await res.json()) as { status: string };
      expect(body.status).toBe('ok');
    });

    it('responds 200 to GET on the webhook path', async () => {
      const res = await fetch(`http://127.0.0.1:${port}/webhooks/github`);
      expect(res.status).toBe(200);
    });

    it('responds 404 for unknown routes', async () => {
      const res = await fetch(`http://127.0.0.1:${port}/nope`);
      expect(res.status).toBe(404);
    });

    it('rejects unsigned POSTs with 401 when secret is configured', async () => {
      const res = await postJson(port, '{"a":1}', {
        [WEBHOOK_EVENT_HEADER]: 'ping',
        [WEBHOOK_DELIVERY_HEADER]: 'd-1',
      });
      expect(res.status).toBe(401);
    });

    it('accepts valid signed POSTs and returns 202', async () => {
      const body = JSON.stringify({ zen: 'hello' });
      const res = await postJson(port, body, {
        [WEBHOOK_SIGNATURE_HEADER]: sign(body),
        [WEBHOOK_EVENT_HEADER]: 'ping',
        [WEBHOOK_DELIVERY_HEADER]: 'd-2',
      });
      expect(res.status).toBe(202);
      const json = (await res.json()) as { status: string; event: string };
      expect(json.status).toBe('accepted');
      expect(json.event).toBe('ping');
    });

    it('rejects oversize bodies with 413', async () => {
      const tiny = new WebhookServer({ secret: SECRET, maxBodyBytes: 16 });
      const p = await tiny.start(0);
      try {
        const body = 'x'.repeat(1024);
        const res = await postJson(p, body, {
          [WEBHOOK_SIGNATURE_HEADER]: sign(body),
          [WEBHOOK_EVENT_HEADER]: 'push',
        });
        expect(res.status).toBe(413);
      } finally {
        await tiny.stop();
      }
    });

    it('rejects invalid JSON with 400', async () => {
      const body = '{ not json';
      const res = await postJson(port, body, {
        [WEBHOOK_SIGNATURE_HEADER]: sign(body),
        [WEBHOOK_EVENT_HEADER]: 'push',
      });
      expect(res.status).toBe(400);
    });
  });

  describe('event dispatch', () => {
    beforeEach(async () => {
      server = new WebhookServer({ secret: SECRET });
      port = await server.start(0);
    });

    afterEach(async () => {
      await server.stop();
    });

    it('invokes registered handler with parsed payload', async () => {
      const received: WebhookRequest[] = [];
      server.on('pull_request', (req) => {
        received.push(req);
        return Promise.resolve();
      });

      const payload = {
        action: 'opened',
        number: 7,
        pull_request: { id: 1, title: 'Test PR' },
        repository: { full_name: 'owner/repo' },
      };
      const body = JSON.stringify(payload);
      const res = await postJson(port, body, {
        [WEBHOOK_SIGNATURE_HEADER]: sign(body),
        [WEBHOOK_EVENT_HEADER]: 'pull_request',
        [WEBHOOK_DELIVERY_HEADER]: 'd-99',
      });
      expect(res.status).toBe(202);

      // Handler runs after response; wait a tick for the dispatch.
      await new Promise((r) => setTimeout(r, 10));

      expect(received).toHaveLength(1);
      expect(received[0].event).toBe('pull_request');
      expect(received[0].deliveryId).toBe('d-99');
      expect((received[0].payload as { number: number }).number).toBe(7);
    });

    it('emits a wildcard event in addition to the typed one', async () => {
      const wildcard: WebhookRequest[] = [];
      const typed: WebhookRequest[] = [];
      server.on('*', (req) => {
        wildcard.push(req);
        return Promise.resolve();
      });
      server.on('check_run', (req) => {
        typed.push(req);
        return Promise.resolve();
      });

      const body = JSON.stringify({ check_run: { status: 'completed' } });
      await postJson(port, body, {
        [WEBHOOK_SIGNATURE_HEADER]: sign(body),
        [WEBHOOK_EVENT_HEADER]: 'check_run',
        [WEBHOOK_DELIVERY_HEADER]: 'd-3',
      });

      await new Promise((r) => setTimeout(r, 10));

      expect(typed).toHaveLength(1);
      expect(wildcard).toHaveLength(1);
    });

    it('catches handler errors without failing the request', async () => {
      const good: WebhookRequest[] = [];
      server.on('push', () => {
        throw new Error('boom');
      });
      server.on('push', (req) => {
        good.push(req);
        return Promise.resolve();
      });

      const body = JSON.stringify({ ref: 'refs/heads/main' });
      const res = await postJson(port, body, {
        [WEBHOOK_SIGNATURE_HEADER]: sign(body),
        [WEBHOOK_EVENT_HEADER]: 'push',
        [WEBHOOK_DELIVERY_HEADER]: 'd-4',
      });
      expect(res.status).toBe(202);

      await new Promise((r) => setTimeout(r, 10));

      // The second handler still ran despite the first throwing.
      expect(good).toHaveLength(1);
    });

    it('off() removes a previously registered handler', () => {
      const handler = () => undefined;
      server.on('push', handler);
      expect(server.listenerCount('push')).toBe(1);
      server.off('push', handler);
      expect(server.listenerCount('push')).toBe(0);
    });
  });

  describe('lifecycle', () => {
    it('start() throws when the server is already running', async () => {
      const s = new WebhookServer();
      const p = await s.start(0);
      try {
        await expect(s.start(0)).rejects.toThrow(/already running/i);
      } finally {
        await s.stop();
      }
      // p kept to silence unused warning on some toolchains
      expect(p).toBeGreaterThan(0);
    });

    it('stop() is a no-op when never started', async () => {
      const s = new WebhookServer();
      await expect(s.stop()).resolves.toBeUndefined();
    });

    it('exposes the bound port after start', async () => {
      const s = new WebhookServer();
      const p = await s.start(0);
      try {
        expect(s.port).toBe(p);
      } finally {
        await s.stop();
      }
      expect(s.port).toBeNull();
    });
  });

  describe('no-secret mode', () => {
    it('accepts unsigned POSTs when no secret is configured', async () => {
      const s = new WebhookServer();
      const seen: string[] = [];
      s.on('ping', (req) => {
        seen.push(req.event);
        return Promise.resolve();
      });
      const p = await s.start(0);
      try {
        const res = await postJson(p, '{"zen":"hi"}', {
          [WEBHOOK_EVENT_HEADER]: 'ping',
          [WEBHOOK_DELIVERY_HEADER]: 'd-5',
        });
        expect(res.status).toBe(202);
        await new Promise((r) => setTimeout(r, 10));
        expect(seen).toEqual(['ping']);
      } finally {
        await s.stop();
      }
    });
  });
});
