/**
 * Approval Audit Aggregator
 *
 * Read-side aggregation layer for the approval system. Pulls together:
 * - The pending-approval queue (integrations in `awaiting_approval` status)
 * - The full audit trail for a single integration (status changes + decision
 *   history with timestamps and notes)
 * - The local RunStore approval records (filesystem-backed per-run snapshots)
 *
 * The aggregator is intentionally separate from `ConvexClientWrapper` so it
 * can be unit-tested with stub clients and so the HTTP server and CLI can
 * share the same read-side logic.
 *
 * Pattern: the aggregator takes its dependencies (a convex client and an
 * optional RunStore) at construction time, so callers can swap in fakes.
 */

import type { ConvexClientWrapper, Integration, Approval } from './convex.js';
import type { RunStore, ApprovalRecord } from '../utils/run-store.js';
import { logger } from '../utils/logger.js';

export interface PendingApprovalEntry {
  /** Integration id (Convex document id). */
  integrationId: string;
  /** Repo URL the integration is targeting. */
  repoUrl: string;
  /** Phase the integration is blocked at (1..6). */
  currentPhase: number;
  /** Mode the integration is running in. */
  mode: Integration['mode'];
  /** Human-readable reason the integration is awaiting approval. */
  awaitingReason?: string;
  /** Reasons a guardrail policy triggered. Empty unless guardrail mode. */
  guardrailReasons: string[];
  /** Confidence score from completed phases (0..1, may be undefined). */
  confidence?: number;
  /** Unix timestamp (ms) of the most recent update. */
  updatedAt: number;
  /** When the integration was created. */
  createdAt: number;
}

export interface AuditEntry {
  /** Unix timestamp (ms) when the event was recorded. */
  timestamp: number;
  /** Phase the event applies to (1..6). May be 0 for integration-level events. */
  phase: number;
  /** Categorical kind of event. */
  kind: 'integration_created' | 'phase_completed' | 'awaiting_approval' | 'approved' | 'rejected' | 'integration_failed';
  /** Free-form details (decision notes, error message, etc.). */
  details: string;
  /** Action result, for approval events. */
  action?: 'approved' | 'rejected';
}

export interface IntegrationAudit {
  integration: Integration;
  entries: AuditEntry[];
  /** Convenience count of approved / rejected / pending phase decisions. */
  summary: {
    approved: number;
    rejected: number;
    pending: number;
  };
}

export interface ApprovalAuditOptions {
  /** Optional RunStore; when present, local run snapshots are merged in. */
  runStore?: RunStore;
}

/**
 * Pure formatter for an `AuditEntry` as a one-line human-readable string.
 * Exported for direct unit testing.
 */
export function formatAuditLine(entry: AuditEntry): string {
  const when = new Date(entry.timestamp).toISOString();
  const phaseLabel = entry.phase > 0 ? `phase=${entry.phase}` : 'phase=-';
  const action = entry.action ? ` action=${entry.action}` : '';
  const details = entry.details ? ` ${entry.details}` : '';
  return `[${when}] ${phaseLabel} ${entry.kind}${action}${details}`;
}

/**
 * Pure formatter for a `PendingApprovalEntry` row. Returns a single
 * space-separated line suitable for a CLI table.
 */
export function formatPendingRow(entry: PendingApprovalEntry): string {
  const ts = new Date(entry.updatedAt).toISOString();
  const reason = entry.awaitingReason
    ? entry.awaitingReason.length > 40
      ? entry.awaitingReason.slice(0, 37) + '...'
      : entry.awaitingReason
    : '-';
  const guardrails = entry.guardrailReasons.length > 0
    ? ` (${entry.guardrailReasons.length} guardrail)`
    : '';
  return `${entry.integrationId}  phase=${entry.currentPhase}  mode=${entry.mode}  updated=${ts}  reason=${reason}${guardrails}  repo=${entry.repoUrl}`;
}

export class ApprovalAudit {
  private readonly convex: ConvexClientWrapper;
  private readonly runStore?: RunStore;

  constructor(convex: ConvexClientWrapper, options: ApprovalAuditOptions = {}) {
    this.convex = convex;
    this.runStore = options.runStore;
  }

  /**
   * List every integration currently blocked on an approval gate, sorted
   * oldest-first so the most-stale queue entries surface first.
   */
  async listPending(): Promise<PendingApprovalEntry[]> {
    const integrations = await this.convex.listIntegrations('awaiting_approval');
    return integrations
      .map((integration) => this.toPendingEntry(integration))
      .sort((a, b) => a.updatedAt - b.updatedAt);
  }

  /**
   * Build the full audit trail for a single integration. The Convex
   * approvals table holds the source of truth for decision records; the
   * integration document provides lifecycle events.
   */
  async getIntegrationAudit(integrationId: string): Promise<IntegrationAudit> {
    const integration = await this.convex.getIntegration(integrationId);
    if (!integration) {
      throw new Error(`Integration not found: ${integrationId}`);
    }

    const convexApprovals = await this.convex.listApprovals(integrationId);
    const entries = this.mergeEntries(integration, convexApprovals);
    const summary = this.summarizeApprovals(convexApprovals, integration);

    return { integration, entries, summary };
  }

  /**
   * Pull approval records from the local RunStore for a given runId.
   * Returns an empty array if the run doesn't exist or has no approvals.
   */
  async getRunApprovals(runId: string): Promise<ApprovalRecord[]> {
    if (!this.runStore) {
      return [];
    }
    try {
      return await this.runStore.getApprovals(runId);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      logger.warn('Failed to read run approvals', { runId, error: message });
      return [];
    }
  }

  /**
   * Batch decisions: apply each one in sequence. Failures do not stop
   * the rest. Returns one result per input decision (in the same order).
   */
  async decide(
    decisions: Array<{
      integrationId: string;
      phase: number;
      action: 'approved' | 'rejected';
      notes?: string;
    }>,
  ): Promise<Array<{ ok: boolean; error?: string }>> {
    const results: Array<{ ok: boolean; error?: string }> = [];
    for (const decision of decisions) {
      try {
        await this.convex.createApproval(
          decision.integrationId,
          decision.phase,
          decision.action,
          decision.notes,
        );
        results.push({ ok: true });
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        logger.error('Approval decision failed', { ...decision, error: message });
        results.push({ ok: false, error: message });
      }
    }
    return results;
  }

  private toPendingEntry(integration: Integration): PendingApprovalEntry {
    return {
      integrationId: integration._id,
      repoUrl: integration.repoUrl,
      currentPhase: integration.currentPhase,
      mode: integration.mode,
      awaitingReason: integration.awaitingReason,
      guardrailReasons: integration.guardrailReasons ?? [],
      confidence: integration.confidence,
      updatedAt: integration.updatedAt,
      createdAt: integration.createdAt,
    };
  }

  private mergeEntries(integration: Integration, approvals: Approval[]): AuditEntry[] {
    const entries: AuditEntry[] = [];

    // Integration-level event: creation.
    entries.push({
      timestamp: integration.createdAt,
      phase: 0,
      kind: 'integration_created',
      details: `repo=${integration.repoUrl} mode=${integration.mode}`,
    });

    // Per-phase approval events.
    for (const approval of approvals) {
      entries.push({
        timestamp: approval.createdAt,
        phase: approval.phase,
        kind: approval.action,
        action: approval.action,
        details: approval.notes ? `notes="${approval.notes}"` : '',
      });
    }

    // Integration-level failure event.
    if (integration.status === 'failed' && integration.errorMessage) {
      entries.push({
        timestamp: integration.updatedAt,
        phase: integration.currentPhase,
        kind: 'integration_failed',
        details: integration.errorMessage,
      });
    }

    // Sort by timestamp ascending for a coherent timeline.
    entries.sort((a, b) => a.timestamp - b.timestamp);
    return entries;
  }

  private summarizeApprovals(
    approvals: Approval[],
    integration: Integration,
  ): IntegrationAudit['summary'] {
    let approved = 0;
    let rejected = 0;
    for (const approval of approvals) {
      if (approval.action === 'approved') approved += 1;
      else if (approval.action === 'rejected') rejected += 1;
    }
    // Pending = 1 if currently awaiting, otherwise 0.
    const pending = integration.status === 'awaiting_approval' ? 1 : 0;
    return { approved, rejected, pending };
  }
}
