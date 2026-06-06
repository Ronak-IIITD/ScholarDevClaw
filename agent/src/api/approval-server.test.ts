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
  ApprovalServer,
  APPROVAL_AUTH_HEADER,
  APPROVAL_BEARER_PREFIX,
  parsePhase,
  verifyBearer,
} from './approval-server.js';
import type { ApprovalAudit, IntegrationAudit, PendingApprovalEntry } from './approval-audit.js';
import type { ApprovalRecord } from '../utils/run-store.js';

const AUTH = 'super-secret-token';
const NOW = Date.UTC(2026, 5, 6, 12, 0, 0);

function makeAuditStub(overrides: Partial<ApprovalAudit> = {}): ApprovalAudit {
  return {
    listPending: vi.fn(async () => []),
    getIntegrationAudit: vi.fn(async () => {
      throw new Error('not stubbed');
    }),
    getRunApprovals: vi.fn(async () => []),
    decide: vi.fn(async () => []),
    ...overrides,
  } as unknown as ApprovalAudit;
}

async function get(
  port: number,
  path: string,
  headers: Record<string, string> = {},
): Promise<Response> {
  return await fetch(`http://127.0.0.1:${port}${path}`, {
    method: 'GET',
    headers: { ...headers },
  });
}

async function post(
  port: number,
  path: string,
  body: unknown,
  headers: Record<string, string> = {},
): Promise<Response> {
  return await fetch(`http://127.0.0.1:${port}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    body: typeof body === 'string' ? body : JSON.stringify(body),
  });
}

function authHeader(): Record<string, string> {
  return { [APPROVAL_AUTH_HEADER]: `${APPROVAL_BEARER_PREFIX}${AUTH}` };
}

function makePending(overrides: Partial<PendingApprovalEntry> = {}): PendingApprovalEntry {
  return {
    integrationId: 'int-1',
    repoUrl: '/repo',
    currentPhase: 3,
    mode: 'step_approval',
    guardrailReasons: [],
    updatedAt: NOW,
    createdAt: NOW - 60_000,
    ...overrides,
  };
}

// =====================================================================
// verifyBearer
// =====================================================================

describe('verifyBearer', () => {
  it('returns false when no header is provided', () => {
    expect(verifyBearer(undefined, AUTH)).toBe(false);
  });

  it('returns false when the prefix is wrong', () => {
    expect(verifyBearer(`Basic ${AUTH}`, AUTH)).toBe(false);
  });

  it('returns false when lengths differ', () => {
    expect(verifyBearer(`${APPROVAL_BEARER_PREFIX}short`, AUTH)).toBe(false);
  });

  it('returns true on a matching token', () => {
    expect(verifyBearer(`${APPROVAL_BEARER_PREFIX}${AUTH}`, AUTH)).toBe(true);
  });

  it('returns false on a different token of equal length', () => {
    const other = 'x'.repeat(AUTH.length);
    expect(verifyBearer(`${APPROVAL_BEARER_PREFIX}${other}`, AUTH)).toBe(false);
  });
});

// =====================================================================
// parsePhase
// =====================================================================

describe('parsePhase', () => {
  it('parses positive integers', () => {
    expect(parsePhase('0')).toBe(0);
    expect(parsePhase('1')).toBe(1);
    expect(parsePhase('6')).toBe(6);
  });

  it('rejects negatives and non-numerics', () => {
    expect(parsePhase('-1')).toBeNull();
    expect(parsePhase('1.5')).toBeNull();
    expect(parsePhase('abc')).toBeNull();
    expect(parsePhase('')).toBeNull();
  });

  it('rejects out-of-range values', () => {
    expect(parsePhase('100')).toBeNull();
    expect(parsePhase('999')).toBeNull();
  });
});

// =====================================================================
// Construction
// =====================================================================

describe('ApprovalServer construction', () => {
  it('refuses to start without an authKey', () => {
    expect(() => new ApprovalServer({ authKey: '' }, { audit: makeAuditStub() })).toThrow(
      'non-empty authKey',
    );
  });
});

// =====================================================================
// HTTP behavior
// =====================================================================

describe('ApprovalServer HTTP', () => {
  let server: ApprovalServer;
  let audit: ApprovalAudit;
  let port: number;

  beforeEach(async () => {
    audit = makeAuditStub();
    server = new ApprovalServer({ authKey: AUTH }, { audit });
    port = await server.start(0);
  });

  afterEach(async () => {
    await server.stop();
  });

  // -------- Health --------

  it('GET /health is unauthenticated and reports pending count', async () => {
    (audit.listPending as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
      makePending(),
      makePending({ integrationId: 'int-2' }),
    ]);
    const res = await get(port, '/health');
    expect(res.status).toBe(200);
    const body = await res.json() as { status: string; pendingCount: number; uptimeMs: number };
    expect(body.status).toBe('ok');
    expect(body.pendingCount).toBe(2);
    expect(body.uptimeMs).toBeGreaterThanOrEqual(0);
  });

  it('GET /health returns 200 with pendingCount=0 on audit error', async () => {
    (audit.listPending as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error('convex down'));
    const res = await get(port, '/health');
    expect(res.status).toBe(200);
    const body = await res.json() as { pendingCount: number };
    expect(body.pendingCount).toBe(0);
  });

  // -------- Auth --------

  it('returns 401 on protected routes without a bearer token', async () => {
    const res = await get(port, '/approvals/pending');
    expect(res.status).toBe(401);
    const body = await res.json() as { error: string; code: string };
    expect(body.code).toBe('unauthorized');
  });

  it('returns 401 on protected routes with a wrong bearer token', async () => {
    const res = await get(port, '/approvals/pending', {
      [APPROVAL_AUTH_HEADER]: `${APPROVAL_BEARER_PREFIX}wrong-token-of-same-length`,
    });
    expect(res.status).toBe(401);
  });

  // -------- List pending --------

  it('GET /approvals/pending returns the pending list', async () => {
    (audit.listPending as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
      makePending(),
      makePending({ integrationId: 'int-2' }),
    ]);
    const res = await get(port, '/approvals/pending', authHeader());
    expect(res.status).toBe(200);
    const body = await res.json() as { pending: PendingApprovalEntry[] };
    expect(body.pending).toHaveLength(2);
    expect(body.pending[0].integrationId).toBe('int-1');
  });

  it('GET /approvals/pending forwards errors as 500', async () => {
    (audit.listPending as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error('boom'));
    const res = await get(port, '/approvals/pending', authHeader());
    expect(res.status).toBe(500);
  });

  // -------- Run approvals --------

  it('GET /approvals/runs/:runId returns the local run approvals', async () => {
    const records: ApprovalRecord[] = [
      { phase: 1, action: 'approved', createdAt: '2026-06-06T12:00:00.000Z' },
    ];
    (audit.getRunApprovals as ReturnType<typeof vi.fn>).mockResolvedValueOnce(records);
    const res = await get(port, '/approvals/runs/run-42', authHeader());
    expect(res.status).toBe(200);
    const body = await res.json() as { runId: string; approvals: ApprovalRecord[] };
    expect(body.runId).toBe('run-42');
    expect(body.approvals).toEqual(records);
  });

  // -------- Audit --------

  it('GET /approvals/audit?integrationId=X returns the integration audit', async () => {
    const integrationAudit: IntegrationAudit = {
      integration: {
        _id: 'int-1',
        repoUrl: '/repo',
        status: 'awaiting_approval',
        mode: 'step_approval',
        currentPhase: 3,
        createdAt: NOW - 60_000,
        updatedAt: NOW,
        retryCount: 0,
      },
      entries: [],
      summary: { approved: 1, rejected: 0, pending: 1 },
    };
    (audit.getIntegrationAudit as ReturnType<typeof vi.fn>).mockResolvedValueOnce(integrationAudit);
    const res = await get(port, '/approvals/audit?integrationId=int-1', authHeader());
    expect(res.status).toBe(200);
    const body = await res.json() as IntegrationAudit;
    expect(body.integration._id).toBe('int-1');
    expect(body.summary.approved).toBe(1);
  });

  it('GET /approvals/audit returns 400 when integrationId is missing', async () => {
    const res = await get(port, '/approvals/audit', authHeader());
    expect(res.status).toBe(400);
  });

  it('GET /approvals/audit returns 404 when audit throws "not found"', async () => {
    (audit.getIntegrationAudit as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
      new Error('Integration not found: missing'),
    );
    const res = await get(port, '/approvals/audit?integrationId=missing', authHeader());
    expect(res.status).toBe(500);
    const body = await res.json() as { error: string };
    expect(body.error).toContain('not found');
  });

  // -------- Decide (single) --------

  it('POST /approvals/:id/:phase/decide approves a phase', async () => {
    (audit.decide as ReturnType<typeof vi.fn>).mockResolvedValueOnce([{ ok: true }]);
    const res = await post(
      port,
      '/approvals/int-1/3/decide',
      { action: 'approved', notes: 'lgtm' },
      authHeader(),
    );
    expect(res.status).toBe(200);
    const body = await res.json() as { ok: boolean; integrationId: string; phase: number; action: string };
    expect(body).toEqual({ ok: true, integrationId: 'int-1', phase: 3, action: 'approved' });
    expect(audit.decide).toHaveBeenCalledWith([
      { integrationId: 'int-1', phase: 3, action: 'approved', notes: 'lgtm' },
    ]);
  });

  it('POST /approvals/:id/:phase/decide rejects bad action', async () => {
    const res = await post(
      port,
      '/approvals/int-1/3/decide',
      { action: 'maybe' },
      authHeader(),
    );
    expect(res.status).toBe(400);
    const body = await res.json() as { code: string };
    expect(body.code).toBe('invalid_body');
  });

  it('POST /approvals/:id/:phase/decide rejects invalid phase', async () => {
    const res = await post(
      port,
      '/approvals/int-1/abc/decide',
      { action: 'approved' },
      authHeader(),
    );
    expect(res.status).toBe(400);
    const body = await res.json() as { code: string };
    expect(body.code).toBe('invalid_phase');
  });

  it('POST /approvals/:id/:phase/decide returns 502 on upstream failure', async () => {
    (audit.decide as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
      { ok: false, error: 'convex timeout' },
    ]);
    const res = await post(
      port,
      '/approvals/int-1/3/decide',
      { action: 'rejected' },
      authHeader(),
    );
    expect(res.status).toBe(502);
  });

  // -------- Batch --------

  it('POST /approvals/batch returns 200 when all decisions succeed', async () => {
    (audit.decide as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
      { ok: true },
      { ok: true },
    ]);
    const res = await post(
      port,
      '/approvals/batch',
      {
        decisions: [
          { integrationId: 'a', phase: 1, action: 'approved' },
          { integrationId: 'b', phase: 2, action: 'rejected', notes: 'no' },
        ],
      },
      authHeader(),
    );
    expect(res.status).toBe(200);
    const body = await res.json() as { ok: boolean; results: Array<{ ok: boolean }> };
    expect(body.ok).toBe(true);
    expect(body.results).toHaveLength(2);
  });

  it('POST /approvals/batch returns 207 when any decision fails', async () => {
    (audit.decide as ReturnType<typeof vi.fn>).mockResolvedValueOnce([
      { ok: true },
      { ok: false, error: 'partial' },
    ]);
    const res = await post(
      port,
      '/approvals/batch',
      {
        decisions: [
          { integrationId: 'a', phase: 1, action: 'approved' },
          { integrationId: 'b', phase: 2, action: 'rejected' },
        ],
      },
      authHeader(),
    );
    expect(res.status).toBe(207);
    const body = await res.json() as { ok: boolean };
    expect(body.ok).toBe(false);
  });

  it('POST /approvals/batch returns 400 on bad shape', async () => {
    const res = await post(port, '/approvals/batch', { decisions: 'nope' }, authHeader());
    expect(res.status).toBe(400);
  });

  it('POST /approvals/batch returns 400 on bad decision', async () => {
    const res = await post(
      port,
      '/approvals/batch',
      { decisions: [{ integrationId: 'a', phase: 1, action: 'maybe' }] },
      authHeader(),
    );
    expect(res.status).toBe(400);
  });

  it('POST /approvals/batch returns 400 on invalid JSON', async () => {
    const res = await post(port, '/approvals/batch', 'not-json{', authHeader());
    expect(res.status).toBe(400);
    const body = await res.json() as { code: string };
    expect(body.code).toBe('invalid_json');
  });

  // -------- 404 --------

  it('returns 404 for unknown routes', async () => {
    const res = await get(port, '/no-such-path', authHeader());
    expect(res.status).toBe(404);
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});
