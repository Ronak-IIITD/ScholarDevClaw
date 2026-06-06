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
  parseApprovalsArgs,
  runApprovalsCommand,
  type ApprovalCliDeps,
} from './approval-cli.js';
import type { ApprovalAudit, PendingApprovalEntry, IntegrationAudit } from './approval-audit.js';
import type { ApprovalServer } from './approval-server.js';

const NOW = Date.UTC(2026, 5, 6, 12, 0, 0);

function makeAudit(overrides: Partial<ApprovalAudit> = {}): ApprovalAudit {
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

function makeDeps(overrides: Partial<ApprovalCliDeps> = {}): {
  deps: ApprovalCliDeps;
  output: ReturnType<typeof vi.fn>;
  audit: ApprovalAudit;
} {
  const output = vi.fn();
  const audit = overrides.audit ?? makeAudit();
  const deps: ApprovalCliDeps = {
    audit,
    output,
    ...overrides,
  };
  return { deps, output, audit };
}

// =====================================================================
// parseApprovalsArgs
// =====================================================================

describe('parseApprovalsArgs', () => {
  it('returns help for empty argv', () => {
    expect(parseApprovalsArgs([])).toEqual({ kind: 'help' });
  });

  it('returns help for help subcommand', () => {
    expect(parseApprovalsArgs(['help'])).toEqual({ kind: 'help' });
    expect(parseApprovalsArgs(['--help'])).toEqual({ kind: 'help' });
    expect(parseApprovalsArgs(['-h'])).toEqual({ kind: 'help' });
  });

  it('returns list', () => {
    expect(parseApprovalsArgs(['list'])).toEqual({ kind: 'list' });
  });

  it('parses approve with id and phase', () => {
    expect(parseApprovalsArgs(['approve', 'int-1', '3'])).toEqual({
      kind: 'decide',
      action: 'approved',
      integrationId: 'int-1',
      phase: 3,
    });
  });

  it('parses approve with notes', () => {
    expect(parseApprovalsArgs(['approve', 'int-1', '3', '--notes', 'lgtm'])).toEqual({
      kind: 'decide',
      action: 'approved',
      integrationId: 'int-1',
      phase: 3,
      notes: 'lgtm',
    });
  });

  it('parses reject with reason', () => {
    expect(parseApprovalsArgs(['reject', 'int-1', '3', '--reason', 'no'])).toEqual({
      kind: 'decide',
      action: 'rejected',
      integrationId: 'int-1',
      phase: 3,
      notes: 'no',
    });
  });

  it('parses audit with id', () => {
    expect(parseApprovalsArgs(['audit', 'int-1'])).toEqual({
      kind: 'audit',
      integrationId: 'int-1',
    });
  });

  it('parses audit without id', () => {
    expect(parseApprovalsArgs(['audit'])).toEqual({ kind: 'audit' });
  });

  it('parses serve with port and auth-key', () => {
    expect(parseApprovalsArgs(['serve', '--port', '9000', '--auth-key', 'secret'])).toEqual({
      kind: 'serve',
      port: 9000,
      authKey: 'secret',
    });
  });

  it('parses serve with default port', () => {
    expect(parseApprovalsArgs(['serve', '--auth-key', 'secret'])).toEqual({
      kind: 'serve',
      port: 8088,
      authKey: 'secret',
    });
  });

  it('rejects unknown subcommand', () => {
    expect(() => parseApprovalsArgs(['frobnicate'])).toThrow(/Unknown subcommand/);
  });

  it('rejects missing integrationId', () => {
    expect(() => parseApprovalsArgs(['approve'])).toThrow(/Missing <integrationId>/);
  });

  it('rejects missing phase', () => {
    expect(() => parseApprovalsArgs(['approve', 'int-1'])).toThrow(/Missing <phase>/);
  });

  it('rejects non-numeric phase', () => {
    expect(() => parseApprovalsArgs(['approve', 'int-1', 'abc'])).toThrow(/non-negative integer/);
  });

  it('rejects out-of-range phase', () => {
    expect(() => parseApprovalsArgs(['approve', 'int-1', '100'])).toThrow(/in 0\.\.99/);
  });

  it('rejects serve without auth key', () => {
    const before = process.env.SCHOLARDEVCLAW_APPROVAL_AUTH_KEY;
    delete process.env.SCHOLARDEVCLAW_APPROVAL_AUTH_KEY;
    try {
      expect(() => parseApprovalsArgs(['serve'])).toThrow(/auth key/);
    } finally {
      if (before !== undefined) process.env.SCHOLARDEVCLAW_APPROVAL_AUTH_KEY = before;
    }
  });

  it('uses SCHOLARDEVCLAW_APPROVAL_AUTH_KEY for serve when no --auth-key', () => {
    const before = process.env.SCHOLARDEVCLAW_APPROVAL_AUTH_KEY;
    process.env.SCHOLARDEVCLAW_APPROVAL_AUTH_KEY = 'env-secret';
    try {
      expect(parseApprovalsArgs(['serve'])).toEqual({
        kind: 'serve',
        port: 8088,
        authKey: 'env-secret',
      });
    } finally {
      if (before !== undefined) {
        process.env.SCHOLARDEVCLAW_APPROVAL_AUTH_KEY = before;
      } else {
        delete process.env.SCHOLARDEVCLAW_APPROVAL_AUTH_KEY;
      }
    }
  });

  it('rejects invalid port', () => {
    expect(() => parseApprovalsArgs(['serve', '--port', 'abc', '--auth-key', 'k'])).toThrow(
      /Invalid --port/,
    );
  });
});

// =====================================================================
// runApprovalsCommand — help
// =====================================================================

describe('runApprovalsCommand — help', () => {
  it('prints usage and exits 0', async () => {
    const { deps, output } = makeDeps();
    const result = await runApprovalsCommand(['help'], deps);
    expect(result.exitCode).toBe(0);
    expect(result.subcommand).toBe('help');
    expect(output).toHaveBeenCalled();
    const text = output.mock.calls[0][0] as string;
    expect(text).toContain('Usage: scholardevclaw approvals');
  });
});

// =====================================================================
// runApprovalsCommand — list
// =====================================================================

describe('runApprovalsCommand — list', () => {
  it('prints an empty message when nothing is pending', async () => {
    const audit = makeAudit({ listPending: vi.fn(async () => []) });
    const { deps, output } = makeDeps({ audit });
    const result = await runApprovalsCommand(['list'], deps);
    expect(result.exitCode).toBe(0);
    expect(result.subcommand).toBe('list');
    expect(output).toHaveBeenCalledWith('No integrations are currently awaiting approval.');
  });

  it('prints formatted rows for each pending entry', async () => {
    const audit = makeAudit({
      listPending: vi.fn(async () => [
        makePending({ integrationId: 'int-a' }),
        makePending({ integrationId: 'int-b', currentPhase: 1 }),
      ]),
    });
    const { deps, output } = makeDeps({ audit });
    const result = await runApprovalsCommand(['list'], deps);
    expect(result.exitCode).toBe(0);
    expect(output.mock.calls[0][0]).toContain('Pending approvals (2)');
    const allOutput = output.mock.calls.map((c) => c[0] as string).join('\n');
    expect(allOutput).toContain('int-a');
    expect(allOutput).toContain('int-b');
  });

  it('exits 2 when audit.listPending throws', async () => {
    const audit = makeAudit({
      listPending: vi.fn(async () => {
        throw new Error('convex timeout');
      }),
    });
    const { deps, output } = makeDeps({ audit });
    const result = await runApprovalsCommand(['list'], deps);
    expect(result.exitCode).toBe(2);
    expect(output).toHaveBeenCalledWith(expect.stringContaining('convex timeout'));
  });
});

// =====================================================================
// runApprovalsCommand — decide
// =====================================================================

describe('runApprovalsCommand — decide', () => {
  it('approves a phase and exits 0 on success', async () => {
    const decide = vi.fn(async () => [{ ok: true }]);
    const audit = makeAudit({ decide });
    const { deps, output } = makeDeps({ audit });
    const result = await runApprovalsCommand(['approve', 'int-1', '3'], deps);
    expect(result.exitCode).toBe(0);
    expect(result.subcommand).toBe('approved');
    expect(decide).toHaveBeenCalledWith([
      { integrationId: 'int-1', phase: 3, action: 'approved' },
    ]);
    expect(output.mock.calls.some((c) => (c[0] as string).includes('Done.'))).toBe(true);
  });

  it('rejects a phase with reason', async () => {
    const decide = vi.fn(async () => [{ ok: true }]);
    const audit = makeAudit({ decide });
    const { deps } = makeDeps({ audit });
    const result = await runApprovalsCommand(
      ['reject', 'int-1', '3', '--reason', 'too risky'],
      deps,
    );
    expect(result.exitCode).toBe(0);
    expect(result.subcommand).toBe('rejected');
    expect(decide).toHaveBeenCalledWith([
      { integrationId: 'int-1', phase: 3, action: 'rejected', notes: 'too risky' },
    ]);
  });

  it('exits 2 when decide reports failure', async () => {
    const decide = vi.fn(async () => [{ ok: false, error: 'forbidden' }]);
    const audit = makeAudit({ decide });
    const { deps, output } = makeDeps({ audit });
    const result = await runApprovalsCommand(['approve', 'int-1', '3'], deps);
    expect(result.exitCode).toBe(2);
    expect(output).toHaveBeenCalledWith(expect.stringContaining('forbidden'));
  });
});

// =====================================================================
// runApprovalsCommand — audit
// =====================================================================

describe('runApprovalsCommand — audit', () => {
  it('prints the audit timeline for an integration', async () => {
    const trail: IntegrationAudit = {
      integration: {
        _id: 'int-1',
        repoUrl: '/repo',
        status: 'completed',
        mode: 'step_approval',
        currentPhase: 6,
        createdAt: NOW - 100_000,
        updatedAt: NOW,
        retryCount: 0,
      },
      entries: [
        {
          timestamp: NOW - 100_000,
          phase: 0,
          kind: 'integration_created',
          details: 'repo=/repo',
        },
        {
          timestamp: NOW - 50_000,
          phase: 3,
          kind: 'approved',
          action: 'approved',
          details: '',
        },
      ],
      summary: { approved: 1, rejected: 0, pending: 0 },
    };
    const audit = makeAudit({ getIntegrationAudit: vi.fn(async () => trail) });
    const { deps, output } = makeDeps({ audit });
    const result = await runApprovalsCommand(['audit', 'int-1'], deps);
    expect(result.exitCode).toBe(0);
    const allOutput = output.mock.calls.map((c) => c[0] as string).join('\n');
    expect(allOutput).toContain('Audit for int-1');
    expect(allOutput).toContain('integration_created');
    expect(allOutput).toContain('approved');
    expect(allOutput).toContain('approved=1');
  });

  it('falls back to listing pending when no integrationId is given', async () => {
    const audit = makeAudit({ listPending: vi.fn(async () => []) });
    const { deps, output } = makeDeps({ audit });
    const result = await runApprovalsCommand(['audit'], deps);
    expect(result.exitCode).toBe(0);
    expect(output).toHaveBeenCalledWith(expect.stringContaining('pending approval queue'));
  });

  it('exits 2 on upstream failure', async () => {
    const audit = makeAudit({
      getIntegrationAudit: vi.fn(async () => {
        throw new Error('not found');
      }),
    });
    const { deps } = makeDeps({ audit });
    const result = await runApprovalsCommand(['audit', 'missing'], deps);
    expect(result.exitCode).toBe(2);
  });
});

// =====================================================================
// runApprovalsCommand — parse error
// =====================================================================

describe('runApprovalsCommand — parse errors', () => {
  it('prints the parse error and exits 1', async () => {
    const { deps, output } = makeDeps();
    const result = await runApprovalsCommand(['frobnicate'], deps);
    expect(result.exitCode).toBe(1);
    expect(result.subcommand).toBe('parse_error');
    expect(output.mock.calls[0][0]).toMatch(/Unknown subcommand/);
  });
});

// =====================================================================
// runApprovalsCommand — serve
// =====================================================================

describe('runApprovalsCommand — serve', () => {
  let originalSdcKey: string | undefined;
  beforeEach(() => {
    originalSdcKey = process.env.SCHOLARDEVCLAW_APPROVAL_AUTH_KEY;
    delete process.env.SCHOLARDEVCLAW_APPROVAL_AUTH_KEY;
  });
  afterEach(() => {
    if (originalSdcKey !== undefined) {
      process.env.SCHOLARDEVCLAW_APPROVAL_AUTH_KEY = originalSdcKey;
    } else {
      delete process.env.SCHOLARDEVCLAW_APPROVAL_AUTH_KEY;
    }
  });

  it('starts the server and reports the bound port', async () => {
    const fakeServer = {
      start: vi.fn(async (port: number) => port),
      stop: vi.fn(async () => undefined),
    } as unknown as ApprovalServer;
    const createServer = vi.fn(() => fakeServer);
    // startServer throws immediately, which is how we short-circuit
    // the infinite block in runServe. The CLI catches the throw and
    // exits with code 2; what we care about is that the server was
    // created and start was attempted.
    const startServer = vi.fn(async () => {
      throw new Error('test: stop blocking');
    });
    const { deps, output } = makeDeps({ createServer, startServer });
    const result = await runApprovalsCommand(
      ['serve', '--port', '9000', '--auth-key', 'k'],
      deps,
    );
    expect(createServer).toHaveBeenCalledWith(expect.anything(), 'k');
    expect(startServer).toHaveBeenCalled();
    // Error surfaced via the output channel.
    expect(output.mock.calls.some((c) => (c[0] as string).includes('test: stop blocking'))).toBe(true);
    expect(result.exitCode).toBe(2);
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});
