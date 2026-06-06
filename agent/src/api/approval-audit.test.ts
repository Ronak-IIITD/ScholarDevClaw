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
  ApprovalAudit,
  formatAuditLine,
  formatPendingRow,
  type AuditEntry,
  type PendingApprovalEntry,
} from './approval-audit.js';
import type { ConvexClientWrapper, Integration, Approval } from './convex.js';
import type { RunStore, ApprovalRecord } from '../utils/run-store.js';

const NOW = Date.UTC(2026, 5, 6, 12, 0, 0);

function makeIntegration(overrides: Partial<Integration> = {}): Integration {
  return {
    _id: 'int-1',
    repoUrl: '/repo',
    status: 'awaiting_approval',
    mode: 'step_approval',
    currentPhase: 3,
    createdAt: NOW - 60_000,
    updatedAt: NOW,
    retryCount: 0,
    ...overrides,
  };
}

function makeApproval(overrides: Partial<Approval> = {}): Approval {
  return {
    _id: 'ap-1',
    integrationId: 'int-1',
    phase: 3,
    action: 'approved',
    notes: 'looks good',
    createdAt: NOW - 10_000,
    ...overrides,
  };
}

function makeConvexStub(overrides: Partial<ConvexClientWrapper> = {}): ConvexClientWrapper {
  return {
    listIntegrations: vi.fn(),
    listApprovals: vi.fn(),
    getIntegration: vi.fn(),
    createApproval: vi.fn(),
    ...overrides,
  } as unknown as ConvexClientWrapper;
}

function makeRunStoreStub(approvals: ApprovalRecord[] = []): RunStore {
  return {
    getApprovals: vi.fn(async () => approvals),
  } as unknown as RunStore;
}

describe('formatAuditLine', () => {
  it('formats a basic entry', () => {
    const entry: AuditEntry = {
      timestamp: NOW,
      phase: 3,
      kind: 'approved',
      action: 'approved',
      details: 'notes="lgtm"',
    };
    const line = formatAuditLine(entry);
    expect(line).toContain('2026-06-06T12:00:00.000Z');
    expect(line).toContain('phase=3');
    expect(line).toContain('approved');
    expect(line).toContain('action=approved');
    expect(line).toContain('notes="lgtm"');
  });

  it('omits action when not present', () => {
    const entry: AuditEntry = {
      timestamp: NOW,
      phase: 0,
      kind: 'integration_created',
      details: 'repo=/x',
    };
    const line = formatAuditLine(entry);
    expect(line).not.toContain('action=');
  });

  it('uses phase=- for integration-level events', () => {
    const entry: AuditEntry = {
      timestamp: NOW,
      phase: 0,
      kind: 'integration_failed',
      details: 'timeout',
    };
    const line = formatAuditLine(entry);
    expect(line).toContain('phase=-');
    expect(line).toContain('integration_failed');
  });
});

describe('formatPendingRow', () => {
  it('renders a row with all fields', () => {
    const entry: PendingApprovalEntry = {
      integrationId: 'int-42',
      repoUrl: '/repo',
      currentPhase: 3,
      mode: 'step_approval',
      awaitingReason: 'phase 3 done',
      guardrailReasons: ['big diff'],
      confidence: 0.8,
      updatedAt: NOW,
      createdAt: NOW - 1000,
    };
    const line = formatPendingRow(entry);
    expect(line).toContain('int-42');
    expect(line).toContain('phase=3');
    expect(line).toContain('mode=step_approval');
    expect(line).toContain('reason=phase 3 done');
    expect(line).toContain('1 guardrail');
    expect(line).toContain('repo=/repo');
  });

  it('truncates long reasons', () => {
    const long = 'x'.repeat(100);
    const entry: PendingApprovalEntry = {
      integrationId: 'int-1',
      repoUrl: '/r',
      currentPhase: 1,
      mode: 'step_approval',
      awaitingReason: long,
      guardrailReasons: [],
      updatedAt: NOW,
      createdAt: NOW,
    };
    const line = formatPendingRow(entry);
    expect(line).toContain('...');
    expect(line.length).toBeLessThan(300);
  });

  it('uses - for missing reason', () => {
    const entry: PendingApprovalEntry = {
      integrationId: 'int-1',
      repoUrl: '/r',
      currentPhase: 1,
      mode: 'autonomous',
      guardrailReasons: [],
      updatedAt: NOW,
      createdAt: NOW,
    };
    const line = formatPendingRow(entry);
    expect(line).toContain('reason=-');
  });

  it('omits guardrail suffix when none', () => {
    const entry: PendingApprovalEntry = {
      integrationId: 'int-1',
      repoUrl: '/r',
      currentPhase: 1,
      mode: 'step_approval',
      guardrailReasons: [],
      updatedAt: NOW,
      createdAt: NOW,
    };
    const line = formatPendingRow(entry);
    expect(line).not.toContain('guardrail');
  });
});

describe('ApprovalAudit.listPending', () => {
  it('returns integrations in awaiting_approval status, oldest first', async () => {
    const i1 = makeIntegration({ _id: 'a', updatedAt: NOW - 30_000 });
    const i2 = makeIntegration({ _id: 'b', updatedAt: NOW - 10_000 });
    const i3 = makeIntegration({ _id: 'c', updatedAt: NOW - 20_000 });
    const convex = makeConvexStub({
      listIntegrations: vi.fn(async (status?: string) => {
        expect(status).toBe('awaiting_approval');
        return [i1, i2, i3];
      }),
    });
    const audit = new ApprovalAudit(convex);
    const result = await audit.listPending();
    expect(result.map((r) => r.integrationId)).toEqual(['a', 'c', 'b']);
  });

  it('returns an empty array when nothing is pending', async () => {
    const convex = makeConvexStub({
      listIntegrations: vi.fn(async (_status?: string) => []),
    });
    const audit = new ApprovalAudit(convex);
    const result = await audit.listPending();
    expect(result).toEqual([]);
  });

  it('preserves guardrail reasons and awaiting reason', async () => {
    const integration = makeIntegration({
      awaitingReason: 'phase 3',
      guardrailReasons: ['size', 'protected branch'],
    });
    const convex = makeConvexStub({
      listIntegrations: vi.fn(async () => [integration]),
    });
    const audit = new ApprovalAudit(convex);
    const [entry] = await audit.listPending();
    expect(entry.awaitingReason).toBe('phase 3');
    expect(entry.guardrailReasons).toEqual(['size', 'protected branch']);
  });
});

describe('ApprovalAudit.getIntegrationAudit', () => {
  it('throws when the integration does not exist', async () => {
    const convex = makeConvexStub({
      getIntegration: vi.fn(async () => null),
    });
    const audit = new ApprovalAudit(convex);
    await expect(audit.getIntegrationAudit('missing')).rejects.toThrow(
      'Integration not found: missing',
    );
  });

  it('merges integration lifecycle with approval records into a timeline', async () => {
    const integration = makeIntegration();
    const approvals = [
      makeApproval({ phase: 1, action: 'approved', createdAt: NOW - 50_000 }),
      makeApproval({ phase: 2, action: 'approved', createdAt: NOW - 30_000 }),
    ];
    const convex = makeConvexStub({
      getIntegration: vi.fn(async () => integration),
      listApprovals: vi.fn(async () => approvals),
    });
    const audit = new ApprovalAudit(convex);
    const result = await audit.getIntegrationAudit('int-1');
    // 1 created + 2 approvals = 3 entries.
    expect(result.entries).toHaveLength(3);
    expect(result.entries[0].kind).toBe('integration_created');
    expect(result.entries[1].kind).toBe('approved');
    expect(result.entries[1].phase).toBe(1);
    expect(result.entries[2].kind).toBe('approved');
    expect(result.entries[2].phase).toBe(2);
    // Sorted ascending by timestamp.
    const timestamps = result.entries.map((e) => e.timestamp);
    expect([...timestamps].sort((a, b) => a - b)).toEqual(timestamps);
  });

  it('appends an integration_failed entry when status is failed', async () => {
    const integration = makeIntegration({
      status: 'failed',
      errorMessage: 'patch failed to apply',
      updatedAt: NOW,
    });
    const convex = makeConvexStub({
      getIntegration: vi.fn(async () => integration),
      listApprovals: vi.fn(async () => []),
    });
    const audit = new ApprovalAudit(convex);
    const result = await audit.getIntegrationAudit('int-1');
    const failed = result.entries.find((e) => e.kind === 'integration_failed');
    expect(failed).toBeDefined();
    expect(failed!.details).toContain('patch failed to apply');
  });

  it('summarizes approved/rejected/pending counts', async () => {
    const integration = makeIntegration({ status: 'awaiting_approval' });
    const approvals = [
      makeApproval({ phase: 1, action: 'approved' }),
      makeApproval({ phase: 2, action: 'approved' }),
      makeApproval({ phase: 3, action: 'rejected' }),
    ];
    const convex = makeConvexStub({
      getIntegration: vi.fn(async () => integration),
      listApprovals: vi.fn(async () => approvals),
    });
    const audit = new ApprovalAudit(convex);
    const result = await audit.getIntegrationAudit('int-1');
    expect(result.summary).toEqual({ approved: 2, rejected: 1, pending: 1 });
  });

  it('summary pending is 0 when status is not awaiting_approval', async () => {
    const integration = makeIntegration({ status: 'completed' });
    const convex = makeConvexStub({
      getIntegration: vi.fn(async () => integration),
      listApprovals: vi.fn(async () => [makeApproval({ action: 'approved' })]),
    });
    const audit = new ApprovalAudit(convex);
    const result = await audit.getIntegrationAudit('int-1');
    expect(result.summary).toEqual({ approved: 1, rejected: 0, pending: 0 });
  });
});

describe('ApprovalAudit.getRunApprovals', () => {
  it('returns empty when no RunStore is configured', async () => {
    const audit = new ApprovalAudit(makeConvexStub());
    expect(await audit.getRunApprovals('r1')).toEqual([]);
  });

  it('returns the approval records from the RunStore', async () => {
    const records: ApprovalRecord[] = [
      { phase: 1, action: 'approved', createdAt: '2026-06-06T12:00:00.000Z' },
      { phase: 2, action: 'rejected', notes: 'no', createdAt: '2026-06-06T12:01:00.000Z' },
    ];
    const audit = new ApprovalAudit(makeConvexStub(), {
      runStore: makeRunStoreStub(records),
    });
    const result = await audit.getRunApprovals('r1');
    expect(result).toEqual(records);
  });

  it('returns an empty array and warns when RunStore throws', async () => {
    const audit = new ApprovalAudit(makeConvexStub(), {
      runStore: {
        getApprovals: vi.fn(async () => {
          throw new Error('disk full');
        }),
      } as unknown as RunStore,
    });
    const result = await audit.getRunApprovals('r1');
    expect(result).toEqual([]);
  });
});

describe('ApprovalAudit.decide', () => {
  it('applies each decision and reports per-decision success', async () => {
    const createApproval = vi.fn(async () => undefined);
    const audit = new ApprovalAudit(makeConvexStub({ createApproval }));
    const results = await audit.decide([
      { integrationId: 'a', phase: 1, action: 'approved' },
      { integrationId: 'b', phase: 2, action: 'rejected', notes: 'no' },
    ]);
    expect(results).toEqual([{ ok: true }, { ok: true }]);
    expect(createApproval).toHaveBeenCalledTimes(2);
    expect(createApproval).toHaveBeenNthCalledWith(1, 'a', 1, 'approved', undefined);
    expect(createApproval).toHaveBeenNthCalledWith(2, 'b', 2, 'rejected', 'no');
  });

  it('continues on per-decision failure and returns the error', async () => {
    const createApproval = vi
      .fn()
      .mockResolvedValueOnce(undefined)
      .mockRejectedValueOnce(new Error('convex down'))
      .mockResolvedValueOnce(undefined);
    const audit = new ApprovalAudit(makeConvexStub({ createApproval }));
    const results = await audit.decide([
      { integrationId: 'a', phase: 1, action: 'approved' },
      { integrationId: 'b', phase: 2, action: 'rejected' },
      { integrationId: 'c', phase: 3, action: 'approved' },
    ]);
    expect(results).toEqual([
      { ok: true },
      { ok: false, error: 'convex down' },
      { ok: true },
    ]);
  });

  it('returns an empty array when given no decisions', async () => {
    const audit = new ApprovalAudit(makeConvexStub());
    expect(await audit.decide([])).toEqual([]);
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});
