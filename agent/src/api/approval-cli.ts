/**
 * Approval CLI
 *
 * Subcommand dispatcher for the `approvals` agent command. Pure logic
 * with no direct I/O of its own — everything that needs to talk to
 * Convex or the filesystem goes through an injected `ApprovalAudit`,
 * and everything that prints goes through an injected `output`
 * function. This makes the module trivially testable.
 *
 * Subcommands:
 *
 *   approvals list
 *     List every integration currently blocked on an approval gate.
 *
 *   approvals approve <integrationId> <phase> [--notes "..."]
 *     Approve a specific phase of an integration.
 *
 *   approvals reject <integrationId> <phase> [--reason "..."] [--notes "..."]
 *     Reject a specific phase of an integration.
 *
 *   approvals audit [integrationId]
 *     Print the audit timeline for one (or all known) integrations.
 *
 *   approvals serve [--port N] [--auth-key <key>]
 *     Start the HTTP server (see approval-server.ts) and block until
 *     the process is killed.
 */

import { logger } from '../utils/logger.js';
import { formatAuditLine, formatPendingRow, ApprovalAudit } from './approval-audit.js';
import { ApprovalServer } from './approval-server.js';

export type OutputFn = (line: string) => void;

export interface ApprovalCliDeps {
  audit: ApprovalAudit;
  output?: OutputFn;
  /** Optional factory for ApprovalServer (used by `serve`); defaults to real class. */
  createServer?: (audit: ApprovalAudit, authKey: string) => ApprovalServer;
  /**
   * Override the port-binding call (used by tests so we don't actually
   * bind to a TCP port). Returns the bound port.
   */
  startServer?: (server: ApprovalServer, port: number) => Promise<number>;
}

export interface ApprovalCliResult {
  /** Process exit code (0 = success, 1 = user error, 2 = upstream error). */
  exitCode: number;
  /** Subcommand that ran (for logging / introspection). */
  subcommand: string;
}

const SUBCOMMANDS = new Set(['list', 'approve', 'reject', 'audit', 'serve', 'help']);

const USAGE = `Usage: scholardevclaw approvals <subcommand> [args]

Subcommands:
  list                                        List pending approval gates
  approve <integrationId> <phase> [--notes X] Approve a phase
  reject <integrationId> <phase> [--reason X] Reject a phase
  audit [integrationId]                       Print audit timeline
  serve [--port N] [--auth-key KEY]           Start the HTTP API
  help                                        Show this message

Phase must be an integer in 0..99.`;

/**
 * Pure argv parser. Returns a discriminated subcommand record. Throws
 * an Error with a user-facing message on parse failures so the CLI can
 * print it to stderr and exit 1.
 */
export function parseApprovalsArgs(
  argv: string[],
):
  | { kind: 'help' }
  | { kind: 'list' }
  | { kind: 'decide'; action: 'approved' | 'rejected'; integrationId: string; phase: number; notes?: string }
  | { kind: 'audit'; integrationId?: string }
  | { kind: 'serve'; port: number; authKey: string } {
  if (argv.length === 0 || argv[0] === 'help' || argv[0] === '--help' || argv[0] === '-h') {
    return { kind: 'help' };
  }

  const sub = argv[0];
  if (!SUBCOMMANDS.has(sub)) {
    throw new Error(`Unknown subcommand: ${sub}\n\n${USAGE}`);
  }

  if (sub === 'list') {
    return { kind: 'list' };
  }

  if (sub === 'approve' || sub === 'reject') {
    const integrationId = argv[1];
    const phaseRaw = argv[2];
    if (!integrationId) {
      throw new Error(`Missing <integrationId> for '${sub}'`);
    }
    if (!phaseRaw) {
      throw new Error(`Missing <phase> for '${sub}'`);
    }
    if (!/^[0-9]+$/.test(phaseRaw)) {
      throw new Error(`Phase must be a non-negative integer, got: ${phaseRaw}`);
    }
    const phase = Number.parseInt(phaseRaw, 10);
    if (!Number.isInteger(phase) || phase < 0 || phase > 99) {
      throw new Error(`Phase must be in 0..99, got: ${phase}`);
    }
    const notes = extractFlag(argv, ['--notes', '-n']);
    const reason = extractFlag(argv, ['--reason', '-r']);
    const combinedNotes = notes ?? reason;
    return {
      kind: 'decide',
      action: sub === 'approve' ? 'approved' : 'rejected',
      integrationId,
      phase,
      ...(combinedNotes ? { notes: combinedNotes } : {}),
    };
  }

  if (sub === 'audit') {
    return argv[1] ? { kind: 'audit', integrationId: argv[1] } : { kind: 'audit' };
  }

  if (sub === 'serve') {
    const portStr = extractFlag(argv, ['--port', '-p']);
    const authKeyFlag = extractFlag(argv, ['--auth-key', '-k']);
    const authKey = authKeyFlag ?? process.env.SCHOLARDEVCLAW_APPROVAL_AUTH_KEY ?? '';
    if (!authKey) {
      throw new Error(
        'serve requires an auth key. Pass --auth-key <key> or set SCHOLARDEVCLAW_APPROVAL_AUTH_KEY.',
      );
    }
    const port = portStr ? Number.parseInt(portStr, 10) : 8088;
    if (!Number.isInteger(port) || port < 0 || port > 65535) {
      throw new Error(`Invalid --port: ${portStr}`);
    }
    return { kind: 'serve', port, authKey };
  }

  // Should be unreachable.
  throw new Error(`Unhandled subcommand: ${sub}`);
}

function extractFlag(argv: string[], flags: string[]): string | undefined {
  for (let i = 0; i < argv.length; i += 1) {
    if (flags.includes(argv[i]) && i + 1 < argv.length) {
      return argv[i + 1];
    }
  }
  return undefined;
}

/**
 * Run the parsed command. Returns a structured result so callers can
 * decide whether to exit. The function is total — every branch either
 * returns or throws.
 */
export async function runApprovalsCommand(
  argv: string[],
  deps: ApprovalCliDeps,
): Promise<ApprovalCliResult> {
  const output = deps.output ?? ((line: string) => process.stdout.write(`${line}\n`));
  let parsed;
  try {
    parsed = parseApprovalsArgs(argv);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    output(message);
    return { exitCode: 1, subcommand: 'parse_error' };
  }

  if (parsed.kind === 'help') {
    output(USAGE);
    return { exitCode: 0, subcommand: 'help' };
  }

  if (parsed.kind === 'list') {
    return await runList(deps.audit, output);
  }

  if (parsed.kind === 'decide') {
    return await runDecide(deps.audit, output, parsed);
  }

  if (parsed.kind === 'audit') {
    return await runAudit(deps.audit, output, parsed);
  }

  if (parsed.kind === 'serve') {
    return await runServe(deps, output, parsed);
  }

  // Unreachable, but keep TS happy.
  return { exitCode: 1, subcommand: 'unknown' };
}

// ---------------------------------------------------------------------
// Subcommand handlers
// ---------------------------------------------------------------------

async function runList(
  audit: ApprovalAudit,
  output: OutputFn,
): Promise<ApprovalCliResult> {
  try {
    const pending = await audit.listPending();
    if (pending.length === 0) {
      output('No integrations are currently awaiting approval.');
    } else {
      output(`Pending approvals (${pending.length}):`);
      for (const entry of pending) {
        output(`  ${formatPendingRow(entry)}`);
      }
    }
    return { exitCode: 0, subcommand: 'list' };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    output(`Error: ${message}`);
    return { exitCode: 2, subcommand: 'list' };
  }
}

async function runDecide(
  audit: ApprovalAudit,
  output: OutputFn,
  parsed: Extract<ReturnType<typeof parseApprovalsArgs>, { kind: 'decide' }>,
): Promise<ApprovalCliResult> {
  const verb = parsed.action === 'approved' ? 'Approving' : 'Rejecting';
  output(`${verb} ${parsed.integrationId} phase ${parsed.phase}...`);
  const [result] = await audit.decide([
    {
      integrationId: parsed.integrationId,
      phase: parsed.phase,
      action: parsed.action,
      ...(parsed.notes ? { notes: parsed.notes } : {}),
    },
  ]);
  if (!result.ok) {
    output(`Failed: ${result.error ?? 'unknown error'}`);
    return { exitCode: 2, subcommand: parsed.action };
  }
  output('Done.');
  return { exitCode: 0, subcommand: parsed.action };
}

async function runAudit(
  audit: ApprovalAudit,
  output: OutputFn,
  parsed: Extract<ReturnType<typeof parseApprovalsArgs>, { kind: 'audit' }>,
): Promise<ApprovalCliResult> {
  if (!parsed.integrationId) {
    // Without an integrationId, show the pending queue as a proxy for
    // "current state". A more complete implementation would walk the
    // recent runs from RunStore, but that adds a dependency we don't
    // need for the v1 CLI.
    output('No integrationId provided. Showing pending approval queue:');
    return await runList(audit, output);
  }

  try {
    const trail = await audit.getIntegrationAudit(parsed.integrationId);
    output(`Audit for ${trail.integration._id} (${trail.integration.repoUrl}):`);
    output(`  status=${trail.integration.status}  phase=${trail.integration.currentPhase}`);
    output(
      `  summary: approved=${trail.summary.approved} rejected=${trail.summary.rejected} pending=${trail.summary.pending}`,
    );
    if (trail.entries.length === 0) {
      output('  (no audit entries)');
    } else {
      for (const entry of trail.entries) {
        output(`  ${formatAuditLine(entry)}`);
      }
    }
    return { exitCode: 0, subcommand: 'audit' };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    output(`Error: ${message}`);
    return { exitCode: 2, subcommand: 'audit' };
  }
}

async function runServe(
  deps: ApprovalCliDeps,
  output: OutputFn,
  parsed: Extract<ReturnType<typeof parseApprovalsArgs>, { kind: 'serve' }>,
): Promise<ApprovalCliResult> {
  const factory = deps.createServer ?? ((audit, authKey) => new ApprovalServer({ authKey }, { audit }));
  const server = factory(deps.audit, parsed.authKey);
  const start = deps.startServer ?? (async (s, port) => s.start(port));
  try {
    const port = await start(server, parsed.port);
    output(`Approval server listening on http://127.0.0.1:${port}`);
    logger.info('Approval CLI: serve started', { port });
    // Block forever — caller is expected to SIGINT/SIGTERM.
    await new Promise<void>(() => undefined);
    return { exitCode: 0, subcommand: 'serve' };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    output(`Error: ${message}`);
    return { exitCode: 2, subcommand: 'serve' };
  }
}

// Exported for testability; not part of the public CLI surface.
export const __testing = { USAGE };
