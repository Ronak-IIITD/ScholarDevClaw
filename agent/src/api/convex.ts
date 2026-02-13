import { ConvexClient } from 'convex';
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
  retryCount: number;
  branchName?: string;
}

export interface IntegrationCreate {
  repoUrl: string;
  paperUrl?: string;
  paperPdfPath?: string;
  mode?: ExecutionMode;
}

export class ConvexClientWrapper {
  private client: ConvexClient;

  constructor(deploymentUrl?: string) {
    const url = deploymentUrl || config.convex.deploymentUrl;
    if (!url) {
      throw new Error('Convex deployment URL not configured');
    }
    this.client = new ConvexClient(url);
    logger.info('Convex client initialized', { deploymentUrl: url });
  }

  async createIntegration(input: IntegrationCreate): Promise<string> {
    const id = await this.client.mutation('integrations:create', {
      repoUrl: input.repoUrl,
      paperUrl: input.paperUrl,
      paperPdfPath: input.paperPdfPath,
      mode: input.mode || 'step_approval',
    });
    logger.info('Created integration', { id });
    return id;
  }

  async getIntegration(id: string): Promise<Integration | null> {
    return await this.client.query('integrations:get', { id });
  }

  async listIntegrations(status?: IntegrationStatus): Promise<Integration[]> {
    if (status) {
      return await this.client.query('integrations:listByStatus', { status });
    }
    return await this.client.query('integrations:list');
  }

  async updateStatus(id: string, status: IntegrationStatus, phase?: number): Promise<void> {
    await this.client.mutation('integrations:updateStatus', {
      id,
      status,
      currentPhase: phase,
      updatedAt: Date.now(),
    });
    logger.info('Updated integration status', { id, status, phase });
  }

  async savePhaseResult(id: string, phase: number, result: unknown): Promise<void> {
    const field = `phase${phase}Result`;
    await this.client.mutation('integrations:savePhaseResult', {
      id,
      field,
      result,
      updatedAt: Date.now(),
    });
  }

  async setConfidence(id: string, confidence: number): Promise<void> {
    await this.client.mutation('integrations:setConfidence', {
      id,
      confidence,
      updatedAt: Date.now(),
    });
  }

  async setError(id: string, errorMessage: string): Promise<void> {
    await this.client.mutation('integrations:setError', {
      id,
      errorMessage,
      status: 'failed',
      updatedAt: Date.now(),
    });
  }

  async incrementRetry(id: string): Promise<number> {
    const result = await this.client.mutation('integrations:incrementRetry', {
      id,
    });
    return result.retryCount;
  }

  async waitForApproval(id: string, phase: number): Promise<boolean> {
    logger.info('Waiting for approval', { id, phase });
    
    return new Promise((resolve) => {
      const checkInterval = setInterval(async () => {
        const integration = await this.getIntegration(id);
        
        if (!integration) {
          clearInterval(checkInterval);
          resolve(false);
          return;
        }

        if (integration.status === 'awaiting_approval') {
          resolve(true);
          clearInterval(checkInterval);
          return;
        }

        if (integration.status === 'failed') {
          resolve(false);
          clearInterval(checkInterval);
          return;
        }
      }, 5000);
    });
  }
}
