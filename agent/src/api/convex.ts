import { ConvexHttpClient } from 'convex/browser';
import { logger } from '../utils/logger.js';
import { config } from '../utils/config.js';

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

export interface Integration {
  _id: string;
  repoUrl: string;
  paperUrl?: string;
  paperPdfPath?: string;
  status: IntegrationStatus;
  mode: ExecutionMode;
  currentPhase: number;
  phase1Result?: unknown;
  phase2Result?: unknown;
  phase3Result?: unknown;
  phase4Result?: unknown;
  phase5Result?: unknown;
  phase6Result?: unknown;
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
}

export interface Approval {
  _id: string;
  integrationId: string;
  phase: number;
  action: 'approved' | 'rejected';
  notes?: string;
  createdAt: number;
}

export class ConvexClientWrapper {
  private client: ConvexHttpClient;

  private async callMutation<T = unknown>(name: string, args: Record<string, unknown>): Promise<T> {
    const client = this.client as unknown as {
      mutation: (fn: string, payload: Record<string, unknown>) => Promise<T>;
    };
    return await client.mutation(name, args);
  }

  private async callQuery<T = unknown>(name: string, args: Record<string, unknown> = {}): Promise<T> {
    const client = this.client as unknown as {
      query: (fn: string, payload: Record<string, unknown>) => Promise<T>;
    };
    return await client.query(name, args);
  }

  constructor(deploymentUrl?: string) {
    const url = deploymentUrl || config.convex.deploymentUrl;
    if (!url) {
      throw new Error('Convex deployment URL not configured');
    }
    this.client = new ConvexHttpClient(url);
    logger.info('Convex client initialized', { deploymentUrl: url });
  }

  async createIntegration(input: IntegrationCreate): Promise<string> {
    const id = await this.callMutation<string>('integrations:create', {
      repoUrl: input.repoUrl,
      paperUrl: input.paperUrl,
      paperPdfPath: input.paperPdfPath,
      mode: input.mode || 'step_approval',
    });
    logger.info('Created integration', { id });
    return id;
  }

  async getIntegration(id: string): Promise<Integration | null> {
    return await this.callQuery<Integration | null>('integrations:get', { id });
  }

  async listIntegrations(status?: IntegrationStatus): Promise<Integration[]> {
    if (status) {
      return await this.callQuery<Integration[]>('integrations:listByStatus', { status });
    }
    return await this.callQuery<Integration[]>('integrations:list');
  }

  async updateStatus(
    id: string,
    status: IntegrationStatus,
    phase?: number,
    details?: { awaitingReason?: string; guardrailReasons?: string[] },
  ): Promise<void> {
    await this.callMutation('integrations:updateStatus', {
      id,
      status,
      currentPhase: phase,
      awaitingReason: details?.awaitingReason,
      guardrailReasons: details?.guardrailReasons,
      updatedAt: Date.now(),
    });
    logger.info('Updated integration status', { id, status, phase, ...details });
  }

  async savePhaseResult(id: string, phase: number, result: unknown): Promise<void> {
    const field = `phase${phase}Result`;
    await this.callMutation('integrations:savePhaseResult', {
      id,
      field,
      result,
      updatedAt: Date.now(),
    });
  }

  async setConfidence(id: string, confidence: number): Promise<void> {
    await this.callMutation('integrations:setConfidence', {
      id,
      confidence,
      updatedAt: Date.now(),
    });
  }

  async setError(id: string, errorMessage: string): Promise<void> {
    await this.callMutation('integrations:setError', {
      id,
      errorMessage,
      status: 'failed',
      updatedAt: Date.now(),
    });
  }

  async incrementRetry(id: string): Promise<number> {
    const result = await this.callMutation<{ retryCount: number }>('integrations:incrementRetry', {
      id,
    });
    return result.retryCount;
  }

  async waitForApproval(id: string, phase: number): Promise<boolean> {
    logger.info('Waiting for approval', { id, phase });

    const startedAt = Date.now();
    const timeoutMs = 30 * 60 * 1000;

    return new Promise((resolve) => {
      const checkInterval = setInterval(async () => {
        const integration = await this.getIntegration(id);

        if (!integration) {
          clearInterval(checkInterval);
          resolve(false);
          return;
        }

        if (integration.status === 'failed') {
          clearInterval(checkInterval);
          resolve(false);
          return;
        }

        if (integration.status !== 'awaiting_approval') {
          resolve(true);
          clearInterval(checkInterval);
          return;
        }

        if (Date.now() - startedAt > timeoutMs) {
          logger.warn('Approval wait timed out', { id, phase, timeoutMs });
          clearInterval(checkInterval);
          resolve(false);
        }
      }, 5000);
    });
  }

  async createApproval(
    integrationId: string,
    phase: number,
    action: 'approved' | 'rejected',
    notes?: string,
  ): Promise<void> {
    await this.callMutation('integrations:createApproval', {
      integrationId,
      phase,
      action,
      notes,
    });
    logger.info('Created approval record', { integrationId, phase, action });
  }

  async listApprovals(integrationId: string): Promise<Approval[]> {
    return await this.callQuery<Approval[]>('integrations:listApprovals', { integrationId });
  }
}
