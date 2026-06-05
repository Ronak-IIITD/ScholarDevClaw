import { ConvexHttpClient } from 'convex/browser';
import type {
  MappingResult,
  PatchResult,
  RepoAnalysisResult,
  ResearchSpecResult,
  ValidationResult,
} from '../bridges/python-subprocess.js';
import type { Phase6Report } from '../phases/phase6-report.js';
import { logger } from '../utils/logger.js';
import { config } from '../utils/config.js';

const CONVEX_AUTH_ENV_KEY = 'SCHOLARDEVCLAW_CONVEX_AUTH_KEY';

export type IntegrationStatus =
  | 'pending'
  | 'phase1_analyzing'
  | 'phase2_extracting'
  | 'phase3_mapping'
  | 'phase4_patching'
  | 'phase5_validating'
  | 'phase6_reporting'
  | 'awaiting_approval'
  | 'completed'
  | 'failed';

export type ExecutionMode = 'step_approval' | 'autonomous';

export type PhaseResultField =
  | 'phase1Result'
  | 'phase2Result'
  | 'phase3Result'
  | 'phase4Result'
  | 'phase5Result'
  | 'phase6Result';

export type PersistedPhaseResult =
  | RepoAnalysisResult
  | ResearchSpecResult
  | MappingResult
  | PatchResult
  | ValidationResult
  | Phase6Report;

export interface Integration {
  _id: string;
  repoUrl: string;
  paperUrl?: string;
  paperPdfPath?: string;
  status: IntegrationStatus;
  mode: ExecutionMode;
  yoloMode?: boolean;
  currentPhase: number;
  phase1Result?: RepoAnalysisResult;
  phase2Result?: ResearchSpecResult;
  phase3Result?: MappingResult;
  phase4Result?: PatchResult;
  phase5Result?: ValidationResult;
  phase6Result?: Phase6Report;
  confidence?: number;
  createdAt: number;
  updatedAt: number;
  errorMessage?: string;
  awaitingReason?: string;
  guardrailReasons?: string[];
  retryCount: number;
  branchName?: string;
}

export interface IntegrationCreate {
  repoUrl: string;
  paperUrl?: string;
  paperPdfPath?: string;
  mode?: ExecutionMode;
  yoloMode?: boolean;
}

export interface Approval {
  _id: string;
  integrationId: string;
  phase: number;
  action: 'approved' | 'rejected';
  notes?: string;
  createdAt: number;
}

export interface RetryOptions {
  /** Maximum number of attempts (default 3). */
  maxAttempts?: number;
  /** Base delay in ms (default 250). */
  baseDelayMs?: number;
  /** Cap delay in ms (default 4000). */
  maxDelayMs?: number;
  /** Override the default classification of "is this error transient?" */
  isTransient?: (err: unknown) => boolean;
}

export interface ApprovalWaitOptions {
  /** Total timeout in ms (default 30 minutes). */
  timeoutMs?: number;
  /** Initial polling interval in ms (default 2000). */
  initialIntervalMs?: number;
  /** Max polling interval when backing off (default 10000). */
  maxIntervalMs?: number;
  /** Multiplier per backoff step (default 1.5). */
  backoffMultiplier?: number;
}

const DEFAULT_RETRY: Required<RetryOptions> = {
  maxAttempts: 3,
  baseDelayMs: 250,
  maxDelayMs: 4000,
  isTransient: defaultIsTransient,
};

const DEFAULT_APPROVAL: Required<ApprovalWaitOptions> = {
  timeoutMs: 30 * 60 * 1000,
  initialIntervalMs: 2000,
  maxIntervalMs: 10000,
  backoffMultiplier: 1.5,
};

/** Minimum interval between polls to prevent tight-looping. */
const MIN_POLL_INTERVAL_MS = 1;

/**
 * Default classification of transient errors. Network failures and
 * 5xx/rate-limit responses are considered transient; auth and 4xx are not.
 */
export function defaultIsTransient(err: unknown): boolean {
  if (!err) return false;
  const message =
    err instanceof Error
      ? err.message.toLowerCase()
      : typeof err === 'string'
        ? err.toLowerCase()
        : '';
  if (!message) return false;
  if (message.includes('network') || message.includes('econnreset') || message.includes('etimedout')) {
    return true;
  }
  if (message.includes('rate limit') || message.includes('429')) {
    return true;
  }
  if (/\b5\d{2}\b/.test(message)) {
    return true;
  }
  if (message.includes('timeout') || message.includes('aborted')) {
    return true;
  }
  return false;
}

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

interface CacheEntry<T> {
  value: T;
  expiresAt: number;
}

export class ConvexClientWrapper {
  private client: ConvexHttpClient;
  private authKey: string;
  private queryCache = new Map<string, CacheEntry<unknown>>();
  /** Default TTL for the in-process query cache. */
  private queryCacheTtlMs = 1000;
  /** Pending mutation sequencing map (best-effort ordering for the same id). */
  private mutationChain = new Map<string, Promise<unknown>>();

  private getPhaseResultField(phase: number): PhaseResultField {
    switch (phase) {
      case 1:
        return 'phase1Result';
      case 2:
        return 'phase2Result';
      case 3:
        return 'phase3Result';
      case 4:
        return 'phase4Result';
      case 5:
        return 'phase5Result';
      case 6:
        return 'phase6Result';
      default:
        throw new Error(`Unsupported phase result field for phase ${phase}`);
    }
  }

  private getAuthArgs(): { authKey: string } {
    if (!this.authKey) {
      throw new Error(`${CONVEX_AUTH_ENV_KEY} is not configured`);
    }
    return { authKey: this.authKey };
  }

  private async callMutation<T = unknown>(
    name: string,
    args: Record<string, unknown>,
    options: RetryOptions = {},
    orderingKey?: string,
  ): Promise<T> {
    const execute = () => this.executeMutation<T>(name, args);
    const run = this.runWithRetry(execute, options);
    if (!orderingKey) {
      return run;
    }
    // Sequence mutations targeting the same key (e.g. integration id) to
    // avoid racing on the same document. The previous mutation in the chain
    // is awaited so we preserve relative ordering.
    const previous = this.mutationChain.get(orderingKey) ?? Promise.resolve();
    const next = previous.then(() => run, () => run);
    // Swallow rejection on the chain itself so a single failure doesn't
    // poison subsequent ordered mutations.
    this.mutationChain.set(
      orderingKey,
      next.catch(() => undefined),
    );
    return next;
  }

  private async executeMutation<T>(name: string, args: Record<string, unknown>): Promise<T> {
    const client = this.client as unknown as {
      mutation: (fn: string, payload: Record<string, unknown>) => Promise<T>;
    };
    return await client.mutation(name, { ...args, ...this.getAuthArgs() });
  }

  private async callQuery<T = unknown>(
    name: string,
    args: Record<string, unknown> = {},
    options: RetryOptions = {},
    cacheKey?: string,
  ): Promise<T> {
    if (cacheKey) {
      const cached = this.queryCache.get(cacheKey);
      if (cached && cached.expiresAt > Date.now()) {
        return cached.value as T;
      }
    }
    const result = await this.runWithRetry(
      () => this.executeQuery<T>(name, args),
      options,
    );
    if (cacheKey) {
      this.queryCache.set(cacheKey, {
        value: result,
        expiresAt: Date.now() + this.queryCacheTtlMs,
      });
    }
    return result;
  }

  private async executeQuery<T>(name: string, args: Record<string, unknown>): Promise<T> {
    const client = this.client as unknown as {
      query: (fn: string, payload: Record<string, unknown>) => Promise<T>;
    };
    return await client.query(name, { ...args, ...this.getAuthArgs() });
  }

  private async runWithRetry<T>(fn: () => Promise<T>, options: RetryOptions): Promise<T> {
    const cfg = { ...DEFAULT_RETRY, ...options };
    let attempt = 0;
    let lastError: unknown;
    while (attempt < cfg.maxAttempts) {
      try {
        return await fn();
      } catch (err) {
        lastError = err;
        if (!cfg.isTransient(err) || attempt === cfg.maxAttempts - 1) {
          throw err;
        }
        const delay = Math.min(
          cfg.maxDelayMs,
          cfg.baseDelayMs * 2 ** attempt,
        );
        logger.warn('Convex call failed, retrying', {
          attempt: attempt + 1,
          maxAttempts: cfg.maxAttempts,
          delayMs: delay,
          error: err instanceof Error ? err.message : String(err),
        });
        await sleep(delay);
        attempt += 1;
      }
    }
    throw lastError;
  }

  /** Clear the in-process query cache (useful in tests). */
  clearQueryCache(): void {
    this.queryCache.clear();
  }

  constructor(deploymentUrl?: string) {
    const url = deploymentUrl || config.convex.deploymentUrl;
    if (!url) {
      throw new Error('Convex deployment URL not configured');
    }
    this.authKey = process.env[CONVEX_AUTH_ENV_KEY] || '';
    this.client = new ConvexHttpClient(url);
    logger.info('Convex client initialized', { deploymentUrl: url });
  }

  async createIntegration(input: IntegrationCreate): Promise<string> {
    const id = await this.callMutation<string>(
      'integrations:create',
      {
        repoUrl: input.repoUrl,
        paperUrl: input.paperUrl,
        paperPdfPath: input.paperPdfPath,
        mode: input.mode || 'step_approval',
      },
      {},
      `integration-create:${input.repoUrl}`,
    );
    logger.info('Created integration', { id });
    return id;
  }

  async getIntegration(id: string): Promise<Integration | null> {
    return await this.callQuery<Integration | null>(
      'integrations:get',
      { id },
      {},
      `integration:${id}`,
    );
  }

  async listIntegrations(status?: IntegrationStatus): Promise<Integration[]> {
    const cacheKey = status ? `integrations:byStatus:${status}` : 'integrations:all';
    return await this.callQuery<Integration[]>(
      'integrations:listByStatus',
      { status },
      {},
      cacheKey,
    );
  }

  async updateStatus(
    id: string,
    status: IntegrationStatus,
    phase?: number,
    details?: { awaitingReason?: string; guardrailReasons?: string[] },
  ): Promise<void> {
    await this.callMutation(
      'integrations:updateStatus',
      {
        id,
        status,
        currentPhase: phase,
        awaitingReason: details?.awaitingReason,
        guardrailReasons: details?.guardrailReasons,
        updatedAt: Date.now(),
      },
      {},
      `integration:${id}`,
    );
    this.invalidateIntegration(id);
    logger.info('Updated integration status', { id, status, phase, ...details });
  }

  async savePhaseResult(id: string, phase: number, result: PersistedPhaseResult): Promise<void> {
    const field = this.getPhaseResultField(phase);
    await this.callMutation(
      'integrations:savePhaseResult',
      {
        id,
        payload: {
          field,
          result,
        },
        updatedAt: Date.now(),
      },
      {},
      `integration:${id}`,
    );
    this.invalidateIntegration(id);
  }

  async setConfidence(id: string, confidence: number): Promise<void> {
    await this.callMutation(
      'integrations:setConfidence',
      {
        id,
        confidence,
        updatedAt: Date.now(),
      },
      {},
      `integration:${id}`,
    );
    this.invalidateIntegration(id);
  }

  async setError(id: string, errorMessage: string, retry?: RetryOptions): Promise<void> {
    await this.callMutation(
      'integrations:setError',
      {
        id,
        errorMessage,
        updatedAt: Date.now(),
      },
      retry ?? {},
      `integration:${id}`,
    );
    this.invalidateIntegration(id);
  }

  /**
   * Save multiple log messages in one batched mutation. Useful for high-volume
   * logging where individual saves would be wasteful.
   */
  async saveLogBatch(
    id: string,
    messages: string[],
    options: { timestamps?: number[] } = {},
  ): Promise<void> {
    if (messages.length === 0) return;
    const now = Date.now();
    const entries = messages.map((message, idx) => ({
      message,
      timestamp: options.timestamps?.[idx] ?? now,
    }));
    await this.callMutation('integrations:saveLogBatch', {
      id,
      entries,
    });
  }

  async saveLog(id: string, message: string): Promise<void> {
    await this.callMutation('integrations:saveLog', {
      id,
      message,
      timestamp: Date.now(),
    });
  }

  async listLogs(id: string): Promise<any[]> {
    return await this.callQuery('integrations:getLogs', { id }, {}, `logs:${id}`);
  }

  async waitForApproval(
    id: string,
    phase: number,
    options: ApprovalWaitOptions = {},
  ): Promise<boolean> {
    logger.info('Waiting for approval', { id, phase });

    const cfg = { ...DEFAULT_APPROVAL, ...options };
    const startedAt = Date.now();
    let interval = cfg.initialIntervalMs;

    return new Promise((resolve) => {
      let settled = false;
      const finish = (result: boolean) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        resolve(result);
      };

      const tick = async () => {
        try {
          // Bypass the short-TTL query cache for polling so we always observe
          // the latest state. The cache is a per-process optimization; the
          // poll loop is explicitly polling for changes.
          this.queryCache.delete(`integration:${id}`);
          this.queryCache.delete(`approvals:${id}`);
          const integration = await this.getIntegration(id);

          if (!integration) {
            finish(false);
            return;
          }

          if (integration.status === 'failed') {
            finish(false);
            return;
          }

          const approvals = await this.listApprovals(id);
          const phaseApprovals = approvals.filter((approval) => approval.phase === phase);

          if (phaseApprovals.some((approval) => approval.action === 'rejected')) {
            finish(false);
            return;
          }

          if (phaseApprovals.some((approval) => approval.action === 'approved')) {
            finish(true);
            return;
          }

          if (integration.status !== 'awaiting_approval') {
            finish(false);
            return;
          }

          if (Date.now() - startedAt > cfg.timeoutMs) {
            logger.warn('Approval wait timed out', { id, phase, timeoutMs: cfg.timeoutMs });
            finish(false);
            return;
          }

          // Exponential backoff up to the configured cap. Always respect the
          // minimum interval so a backoff multiplier on a small base can't
          // create a tight loop.
          interval = Math.max(
            MIN_POLL_INTERVAL_MS,
            Math.min(cfg.maxIntervalMs, Math.round(interval * cfg.backoffMultiplier)),
          );
          timer = setTimeout(tick, interval);
        } catch (error) {
          logger.error('Approval polling failed', {
            id,
            phase,
            error: error instanceof Error ? error.message : String(error),
          });
          finish(false);
        }
      };

      let timer: ReturnType<typeof setTimeout> = setTimeout(
        tick,
        Math.max(MIN_POLL_INTERVAL_MS, cfg.initialIntervalMs),
      );
    });
  }

  async createApproval(
    integrationId: string,
    phase: number,
    action: 'approved' | 'rejected',
    notes?: string,
  ): Promise<void> {
    await this.callMutation(
      'integrations:createApproval',
      {
        integrationId,
        phase,
        action,
        notes,
      },
      {},
      `integration:${integrationId}`,
    );
    this.invalidateIntegration(integrationId);
    this.queryCache.delete(`approvals:${integrationId}`);
    logger.info('Created approval record', { integrationId, phase, action });
  }

  async listApprovals(integrationId: string): Promise<Approval[]> {
    return await this.callQuery<Approval[]>(
      'integrations:listApprovals',
      { integrationId },
      {},
      `approvals:${integrationId}`,
    );
  }

  private invalidateIntegration(id: string): void {
    this.queryCache.delete(`integration:${id}`);
    this.queryCache.delete(`logs:${id}`);
    this.queryCache.delete(`approvals:${id}`);
  }
}
