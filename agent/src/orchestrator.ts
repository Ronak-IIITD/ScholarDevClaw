import { logger } from './utils/logger.js';
import { config } from './utils/config.js';
import { PythonSubprocessBridge, PythonHttpBridge } from './bridges/python-bridge.js';
import type {
  MappingResult,
  PatchResult,
  RepoAnalysisResult,
  ResearchSpecResult,
  ValidationResult,
} from './bridges/python-subprocess.js';
import { ConvexClientWrapper, type IntegrationCreate } from './api/convex.js';
import { GitHubClient } from './api/github.js';
import * as Phase1 from './phases/phase1-repo.js';
import * as Phase2 from './phases/phase2-research.js';
import * as Phase3 from './phases/phase3-mapping.js';
import * as Phase4 from './phases/phase4-patch.js';
import * as Phase5 from './phases/phase5-validation.js';
import * as Phase6 from './phases/phase6-report.js';

export class ScholarDevClawOrchestrator {
  private bridge: PythonSubprocessBridge | PythonHttpBridge;
  private convex?: ConvexClientWrapper;
  private github?: GitHubClient;

  constructor(useHttp: boolean = true) {
    if (useHttp) {
      this.bridge = new PythonHttpBridge(config.python.coreApiUrl);
    } else {
      this.bridge = new PythonSubprocessBridge(config.python.subprocessCommand);
    }
  }

  async initialize(): Promise<void> {
    if (config.convex.deploymentUrl) {
      this.convex = new ConvexClientWrapper();
    }

    if (config.github.token) {
      this.github = new GitHubClient();
    }

    logger.info('ScholarDevClaw orchestrator initialized');
  }

  async runIntegration(input: IntegrationCreate): Promise<void> {
    const { repoUrl, paperUrl, paperPdfPath, mode } = input;
    const executionMode = mode || config.execution.defaultMode as 'step_approval' | 'autonomous';

    logger.info('Starting integration', { repoUrl, paperUrl, mode: executionMode });

    const context: {
      repoPath: string;
      paperSource: string;
      sourceType: 'pdf' | 'arxiv';
      repoAnalysis?: RepoAnalysisResult;
      researchSpec?: ResearchSpecResult;
      mapping?: MappingResult;
      patch?: PatchResult;
      validation?: ValidationResult;
    } = {
      repoPath: repoUrl,
      paperSource: paperUrl || paperPdfPath || '',
      sourceType: paperUrl ? 'arxiv' as const : 'pdf' as const,
    };

    const phaseResults: Record<number, unknown> = {};

    try {
      // Phase 1: Repository Intelligence
      logger.info('Starting Phase 1...');
      const phase1Result = await Phase1.executePhase1(this.bridge, context);
      
      if (!phase1Result.success) {
        throw new Error(`Phase 1 failed: ${phase1Result.error}`);
      }
      
      phaseResults[1] = phase1Result.data;
      context.repoAnalysis = phase1Result.data;

      if (executionMode === 'step_approval') {
        await this.waitForApproval(1);
      }

      // Phase 2: Research Intelligence
      logger.info('Starting Phase 2...');
      const phase2Result = await Phase2.executePhase2(this.bridge, context);
      
      if (!phase2Result.success) {
        throw new Error(`Phase 2 failed: ${phase2Result.error}`);
      }
      
      phaseResults[2] = phase2Result.data;
      context.researchSpec = phase2Result.data;

      if (executionMode === 'step_approval') {
        await this.waitForApproval(2);
      }

      // Phase 3: Mapping Engine
      logger.info('Starting Phase 3...');
      const phase3Result = await Phase3.executePhase3(this.bridge, context);
      
      if (!phase3Result.success) {
        throw new Error(`Phase 3 failed: ${phase3Result.error}`);
      }
      
      phaseResults[3] = phase3Result.data;
      context.mapping = phase3Result.data;

      if (executionMode === 'step_approval') {
        await this.waitForApproval(3);
      }

      // Phase 4: Patch Generation
      logger.info('Starting Phase 4...');
      const phase4Result = await Phase4.executePhase4(this.bridge, context);
      
      if (!phase4Result.success) {
        throw new Error(`Phase 4 failed: ${phase4Result.error}`);
      }
      
      phaseResults[4] = phase4Result.data;
      context.patch = phase4Result.data;

      if (executionMode === 'step_approval') {
        await this.waitForApproval(4);
      }

      // Phase 5: Validation
      logger.info('Starting Phase 5...');
      const phase5Result = await Phase5.executePhase5(this.bridge, context, repoUrl);
      
      if (!phase5Result.success) {
        if (this.shouldRetry()) {
          logger.warn('Phase 5 failed, will retry...');
          return;
        }
        throw new Error(`Phase 5 failed: ${phase5Result.error}`);
      }
      
      phaseResults[5] = phase5Result.data;
      context.validation = phase5Result.data;

      // Phase 6: Report Generation
      logger.info('Starting Phase 6...');
      const phase6Result = await Phase6.executePhase6(context);
      
      if (!phase6Result.success) {
        throw new Error(`Phase 6 failed: ${phase6Result.error}`);
      }
      
      phaseResults[6] = phase6Result.data;

      logger.info('Integration completed successfully', {
        recommendation: phase6Result.data.recommendation.action,
      });

    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      logger.error('Integration failed', { error: message });
      throw err;
    }
  }

  private async waitForApproval(phase: number): Promise<void> {
    logger.info(`Waiting for approval after Phase ${phase}...`);
    
    await new Promise((resolve) => setTimeout(resolve, 1000));
    
    logger.info('Approval received (simulated for now)');
  }

  private shouldRetry(): boolean {
    return false;
  }
}
