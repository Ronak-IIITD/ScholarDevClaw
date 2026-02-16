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
import {
  ConvexClientWrapper,
  type Integration,
  type IntegrationCreate,
} from './api/convex.js';
import { GitHubClient } from './api/github.js';
import { RunStore, type RunSnapshot, type ApprovalRecord } from './utils/run-store.js';
import {
  evaluateMappingGuardrails,
  evaluateValidationGuardrails,
} from './utils/guardrails.js';
import * as Phase1 from './phases/phase1-repo.js';
import * as Phase2 from './phases/phase2-research.js';
import * as Phase3 from './phases/phase3-mapping.js';
import * as Phase4 from './phases/phase4-patch.js';
import * as Phase5 from './phases/phase5-validation.js';
import * as Phase6 from './phases/phase6-report.js';

type ExecutionMode = 'step_approval' | 'autonomous';

type OrchestrationContext = {
  repoPath: string;
  paperSource: string;
  sourceType: 'pdf' | 'arxiv';
  repoAnalysis?: RepoAnalysisResult;
  researchSpec?: ResearchSpecResult;
  mapping?: MappingResult;
  patch?: PatchResult;
  validation?: ValidationResult;
};

type RunIntegrationOptions = {
  runId?: string;
  integrationId?: string;
  startPhase?: number;
  retryCount?: number;
  phaseResults?: Record<number, unknown>;
  contextOverrides?: Partial<OrchestrationContext>;
};

export class ScholarDevClawOrchestrator {
  private bridge: PythonSubprocessBridge | PythonHttpBridge;
  private convex?: ConvexClientWrapper;
  private github?: GitHubClient;
  private runStore: RunStore;

  constructor(useHttp: boolean = true) {
    if (useHttp) {
      this.bridge = new PythonHttpBridge(config.python.coreApiUrl);
    } else {
      this.bridge = new PythonSubprocessBridge(config.python.subprocessCommand);
    }
    this.runStore = new RunStore();
  }

  async initialize(): Promise<void> {
    await this.runStore.initialize();

    if (config.convex.deploymentUrl) {
      this.convex = new ConvexClientWrapper();
    }

    if (config.github.token) {
      this.github = new GitHubClient();
    }

    logger.info('ScholarDevClaw orchestrator initialized');
  }

  async runIntegration(input: IntegrationCreate, options: RunIntegrationOptions = {}): Promise<void> {
    const { repoUrl, paperUrl, paperPdfPath, mode } = input;
    const executionMode = (mode || config.execution.defaultMode) as ExecutionMode;
    const runId = options.runId || this.generateRunId();
    const startPhase = options.startPhase || 1;
    const existingSnapshot = options.runId ? await this.runStore.get(options.runId) : null;
    const retryCount = options.retryCount ?? existingSnapshot?.retryCount ?? 0;

    let integrationId = options.integrationId;
    if (!integrationId && this.convex) {
      integrationId = await this.convex.createIntegration(input);
    }

    logger.info('Starting integration', {
      runId,
      integrationId,
      repoUrl,
      paperUrl,
      mode: executionMode,
      startPhase,
      retryCount,
    });

    const context: OrchestrationContext = {
      repoPath: repoUrl,
      paperSource: paperUrl || paperPdfPath || '',
      sourceType: paperUrl ? 'arxiv' as const : 'pdf' as const,
      ...options.contextOverrides,
    };

    const phaseResults: Record<number, unknown> = { ...(options.phaseResults || {}) };

    await this.saveSnapshot({
      runId,
      integrationId,
      input,
      mode: executionMode,
      status: startPhase > 1 ? 'running' : 'pending',
      currentPhase: Math.max(0, startPhase - 1),
      retryCount,
      phaseResults,
      context,
    });

    try {
      if (startPhase <= 1) {
        logger.info('Starting Phase 1...');
        await this.updateExternalPhase(integrationId, 1, 'phase1_analyzing');
        const result = await Phase1.executePhase1(this.bridge, context);
        if (!result.success || !result.data) {
          throw new Error(`Phase 1 failed: ${result.error || 'No data'}`);
        }
        context.repoAnalysis = result.data;
        phaseResults[1] = result.data;
        await this.finalizePhase(
          runId,
          integrationId,
          input,
          executionMode,
          context,
          phaseResults,
          1,
          result.data,
          retryCount,
        );
        if (executionMode === 'step_approval') {
          await this.waitForApproval(
            runId,
            integrationId,
            input,
            executionMode,
            context,
            phaseResults,
            1,
            retryCount,
          );
        }
      }

      if (startPhase <= 2) {
        if (!context.repoAnalysis) {
          throw new Error('Cannot run Phase 2 without phase1 repo analysis context');
        }
        logger.info('Starting Phase 2...');
        await this.updateExternalPhase(integrationId, 2, 'phase2_extracting');
        const result = await Phase2.executePhase2(this.bridge, {
          ...context,
          repoAnalysis: context.repoAnalysis,
        });
        if (!result.success || !result.data) {
          throw new Error(`Phase 2 failed: ${result.error || 'No data'}`);
        }
        context.researchSpec = result.data;
        phaseResults[2] = result.data;
        await this.finalizePhase(
          runId,
          integrationId,
          input,
          executionMode,
          context,
          phaseResults,
          2,
          result.data,
          retryCount,
        );
        if (executionMode === 'step_approval') {
          await this.waitForApproval(
            runId,
            integrationId,
            input,
            executionMode,
            context,
            phaseResults,
            2,
            retryCount,
          );
        }
      }

      if (startPhase <= 3) {
        if (!context.repoAnalysis || !context.researchSpec) {
          throw new Error('Cannot run Phase 3 without phase1+phase2 context');
        }
        logger.info('Starting Phase 3...');
        await this.updateExternalPhase(integrationId, 3, 'phase3_mapping');
        const result = await Phase3.executePhase3(this.bridge, {
          ...context,
          repoAnalysis: context.repoAnalysis,
          researchSpec: context.researchSpec,
        });
        if (!result.success || !result.data) {
          throw new Error(`Phase 3 failed: ${result.error || 'No data'}`);
        }
        context.mapping = result.data;
        phaseResults[3] = result.data;
        await this.finalizePhase(
          runId,
          integrationId,
          input,
          executionMode,
          context,
          phaseResults,
          3,
          result.data,
          retryCount,
        );
        if (executionMode === 'step_approval') {
          await this.waitForApproval(
            runId,
            integrationId,
            input,
            executionMode,
            context,
            phaseResults,
            3,
            retryCount,
          );
        }

        const mappingGuardrails = evaluateMappingGuardrails(
          result.data,
          config.execution.guardrails.mappingMinConfidence,
        );
        if (mappingGuardrails.triggered) {
          await this.enforceGuardrailApproval(
            runId,
            integrationId,
            input,
            executionMode,
            context,
            phaseResults,
            3,
            retryCount,
            mappingGuardrails.reasons,
          );
        }
      }

      if (startPhase <= 4) {
        if (!context.mapping || !context.repoAnalysis || !context.researchSpec) {
          throw new Error('Cannot run Phase 4 without phase3 mapping context');
        }
        logger.info('Starting Phase 4...');
        await this.updateExternalPhase(integrationId, 4, 'phase4_patching');
        const result = await Phase4.executePhase4(this.bridge, {
          ...context,
          repoAnalysis: context.repoAnalysis,
          researchSpec: context.researchSpec,
          mapping: context.mapping,
        });
        if (!result.success || !result.data) {
          throw new Error(`Phase 4 failed: ${result.error || 'No data'}`);
        }
        this.enforcePatchSafety(result.data);
        context.patch = result.data;
        phaseResults[4] = result.data;
        await this.finalizePhase(
          runId,
          integrationId,
          input,
          executionMode,
          context,
          phaseResults,
          4,
          result.data,
          retryCount,
        );
        if (executionMode === 'step_approval') {
          await this.waitForApproval(
            runId,
            integrationId,
            input,
            executionMode,
            context,
            phaseResults,
            4,
            retryCount,
          );
        }
      }

      if (startPhase <= 5) {
        if (!context.patch || !context.mapping || !context.repoAnalysis || !context.researchSpec) {
          throw new Error('Cannot run Phase 5 without phase4 patch context');
        }
        logger.info('Starting Phase 5...');
        await this.updateExternalPhase(integrationId, 5, 'phase5_validating');
        const result = await Phase5.executePhase5(this.bridge, {
          ...context,
          repoAnalysis: context.repoAnalysis,
          researchSpec: context.researchSpec,
          mapping: context.mapping,
          patch: context.patch,
        }, repoUrl);
        if (!result.success || !result.data) {
          const retried = await this.retryPhaseWithBackoff(
            5,
            result.error || 'Unknown phase 5 error',
            retryCount,
            runId,
            integrationId,
            input,
            executionMode,
            context,
            phaseResults,
          );
          if (retried) {
            return;
          }
          throw new Error(`Phase 5 failed: ${result.error || 'No data'}`);
        }
        context.validation = result.data;
        phaseResults[5] = result.data;
        await this.finalizePhase(
          runId,
          integrationId,
          input,
          executionMode,
          context,
          phaseResults,
          5,
          result.data,
          retryCount,
        );

        const validationGuardrails = evaluateValidationGuardrails(
          result.data,
          config.execution.guardrails.validationMinSpeedup,
          config.execution.guardrails.validationMaxLossChangePct,
        );
        if (validationGuardrails.triggered) {
          await this.enforceGuardrailApproval(
            runId,
            integrationId,
            input,
            executionMode,
            context,
            phaseResults,
            5,
            retryCount,
            validationGuardrails.reasons,
          );
        }
      }

      if (startPhase <= 6) {
        if (!context.validation || !context.patch || !context.mapping || !context.repoAnalysis || !context.researchSpec) {
          throw new Error('Cannot run Phase 6 without phase5 validation context');
        }
        logger.info('Starting Phase 6...');
        await this.updateExternalPhase(integrationId, 6, 'phase6_reporting');
        const result = await Phase6.executePhase6({
          ...context,
          repoAnalysis: context.repoAnalysis,
          researchSpec: context.researchSpec,
          mapping: context.mapping,
          patch: context.patch,
          validation: context.validation,
        });
        if (!result.success || !result.data) {
          throw new Error(`Phase 6 failed: ${result.error || 'No data'}`);
        }
        phaseResults[6] = result.data;
        await this.finalizePhase(
          runId,
          integrationId,
          input,
          executionMode,
          context,
          phaseResults,
          6,
          result.data,
          retryCount,
        );

        logger.info('Integration completed successfully', {
          runId,
          recommendation: result.data.recommendation.action,
        });

        await this.saveSnapshot({
          runId,
          integrationId,
          input,
          mode: executionMode,
          status: 'completed',
          currentPhase: 6,
          retryCount,
          phaseResults,
          context,
        });

        if (integrationId && this.convex) {
          await this.convex.updateStatus(integrationId, 'completed', 6);
        }
      }

    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      logger.error('Integration failed', { runId, error: message });

      await this.saveSnapshot({
        runId,
        integrationId,
        input,
        mode: executionMode,
        status: 'failed',
        currentPhase: this.highestCompletedPhase(phaseResults),
        retryCount,
        lastErrorPhase: startPhase,
        phaseResults,
        context,
        errorMessage: message,
      });

      if (integrationId && this.convex) {
        await this.convex.setError(integrationId, message);
      }

      throw err;
    }
  }

  async resumeRun(runId: string): Promise<boolean> {
    const snapshot = await this.runStore.get(runId);
    if (!snapshot) {
      logger.warn('Cannot resume run: snapshot not found', { runId });
      return false;
    }

    if (snapshot.status === 'completed') {
      logger.info('Run already completed; skipping resume', { runId });
      return true;
    }

    if (snapshot.status === 'failed') {
      logger.warn('Run is marked failed; skipping auto-resume', { runId });
      return false;
    }

    await this.runIntegration(
      {
        repoUrl: snapshot.repoUrl,
        paperUrl: snapshot.paperUrl,
        paperPdfPath: snapshot.paperPdfPath,
        mode: snapshot.mode,
      },
      {
        runId: snapshot.runId,
        integrationId: snapshot.integrationId,
        startPhase: Math.max(1, snapshot.currentPhase + 1),
        phaseResults: snapshot.phaseResults,
        retryCount: snapshot.retryCount,
        contextOverrides: snapshot.context as Partial<OrchestrationContext>,
      },
    );

    return true;
  }

  async processPendingWork(): Promise<void> {
    const localPending = await this.runStore.listByStatus(['pending', 'running', 'awaiting_approval']);
    for (const snapshot of localPending) {
      logger.info('Resuming local pending run', {
        runId: snapshot.runId,
        status: snapshot.status,
        currentPhase: snapshot.currentPhase,
      });
      try {
        await this.resumeRun(snapshot.runId);
      } catch (err) {
        logger.error('Failed to resume local run', {
          runId: snapshot.runId,
          error: err instanceof Error ? err.message : 'Unknown error',
        });
      }
    }

    if (!this.convex) {
      return;
    }

    const statuses: Array<
      'pending' | 'phase1_analyzing' | 'phase2_extracting' | 'phase3_mapping' | 'phase4_patching' | 'phase5_validating' | 'phase6_reporting'
    > = [
      'pending',
      'phase1_analyzing',
      'phase2_extracting',
      'phase3_mapping',
      'phase4_patching',
      'phase5_validating',
      'phase6_reporting',
    ];

    const seen = new Set<string>();
    for (const status of statuses) {
      const integrations = await this.convex.listIntegrations(status);
      for (const integration of integrations) {
        if (seen.has(integration._id)) {
          continue;
        }
        seen.add(integration._id);

        const runId = `convex-${integration._id}`;
        const existing = await this.runStore.get(runId);
        if (existing?.status === 'completed') {
          continue;
        }

        const phaseResults = this.phaseResultsFromIntegration(integration);
        const contextOverrides = this.contextFromPhaseResults(
          integration.repoUrl,
          integration.paperUrl || integration.paperPdfPath || '',
          integration.paperUrl ? 'arxiv' : 'pdf',
          phaseResults,
        );

        await this.runIntegration(
          {
            repoUrl: integration.repoUrl,
            paperUrl: integration.paperUrl,
            paperPdfPath: integration.paperPdfPath,
            mode: integration.mode,
          },
          {
            runId,
            integrationId: integration._id,
            startPhase: Math.max(1, (integration.currentPhase || 0) + 1),
            phaseResults,
            retryCount: integration.retryCount || 0,
            contextOverrides,
          },
        );
      }
    }
  }

  private phaseResultsFromIntegration(integration: Integration): Record<number, unknown> {
    const phaseResults: Record<number, unknown> = {};
    if (integration.phase1Result !== undefined) phaseResults[1] = integration.phase1Result;
    if (integration.phase2Result !== undefined) phaseResults[2] = integration.phase2Result;
    if (integration.phase3Result !== undefined) phaseResults[3] = integration.phase3Result;
    if (integration.phase4Result !== undefined) phaseResults[4] = integration.phase4Result;
    if (integration.phase5Result !== undefined) phaseResults[5] = integration.phase5Result;
    if (integration.phase6Result !== undefined) phaseResults[6] = integration.phase6Result;
    return phaseResults;
  }

  private contextFromPhaseResults(
    repoPath: string,
    paperSource: string,
    sourceType: 'pdf' | 'arxiv',
    phaseResults: Record<number, unknown>,
  ): Partial<OrchestrationContext> {
    return {
      repoPath,
      paperSource,
      sourceType,
      repoAnalysis: phaseResults[1] as RepoAnalysisResult | undefined,
      researchSpec: phaseResults[2] as ResearchSpecResult | undefined,
      mapping: phaseResults[3] as MappingResult | undefined,
      patch: phaseResults[4] as PatchResult | undefined,
      validation: phaseResults[5] as ValidationResult | undefined,
    };
  }

  private async waitForApproval(
    runId: string,
    integrationId: string | undefined,
    input: IntegrationCreate,
    mode: ExecutionMode,
    context: OrchestrationContext,
    phaseResults: Record<number, unknown>,
    phase: number,
    retryCount: number,
    reason?: string,
    guardrailReasons?: string[],
  ): Promise<void> {
    logger.info(`Waiting for approval after Phase ${phase}...`, { runId });

    await this.saveSnapshot({
      runId,
      integrationId,
      input,
      mode,
      status: 'awaiting_approval',
      currentPhase: phase,
      retryCount,
      awaitingReason: reason,
      guardrailReasons,
      phaseResults,
      context,
    });

    let approved = false;

    if (integrationId && this.convex) {
      await this.convex.updateStatus(integrationId, 'awaiting_approval', phase, {
        awaitingReason: reason,
        guardrailReasons,
      });
      approved = await this.convex.waitForApproval(integrationId, phase);
      if (!approved) {
        await this.runStore.addApproval(runId, phase, 'rejected', reason);
        if (this.convex) {
          await this.convex.createApproval(integrationId, phase, 'rejected', reason);
        }
        throw new Error(`Approval rejected or timed out for phase ${phase}`);
      }
      await this.runStore.addApproval(runId, phase, 'approved');
      if (this.convex) {
        await this.convex.createApproval(integrationId, phase, 'approved');
      }
      await this.convex.updateStatus(integrationId, 'pending', phase);
    } else {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      await this.runStore.addApproval(runId, phase, 'approved');
    }

    logger.info('Approval received (or simulated)');
  }

  private async finalizePhase(
    runId: string,
    integrationId: string | undefined,
    input: IntegrationCreate,
    mode: ExecutionMode,
    context: OrchestrationContext,
    phaseResults: Record<number, unknown>,
    phase: number,
    data: unknown,
    retryCount: number,
  ): Promise<void> {
    if (integrationId && this.convex) {
      await this.convex.savePhaseResult(integrationId, phase, data);
    }

    await this.saveSnapshot({
      runId,
      integrationId,
      input,
      mode,
      status: 'running',
      currentPhase: phase,
      retryCount,
      awaitingReason: undefined,
      guardrailReasons: [],
      phaseResults,
      context,
    });
  }

  private async enforceGuardrailApproval(
    runId: string,
    integrationId: string | undefined,
    input: IntegrationCreate,
    mode: ExecutionMode,
    context: OrchestrationContext,
    phaseResults: Record<number, unknown>,
    phase: number,
    retryCount: number,
    reasons: string[],
  ): Promise<void> {
    const reason = `Guardrail policy triggered at phase ${phase}`;
    logger.warn(reason, {
      runId,
      phase,
      reasons,
    });

    await this.waitForApproval(
      runId,
      integrationId,
      input,
      mode,
      context,
      phaseResults,
      phase,
      retryCount,
      reason,
      reasons,
    );
  }

  private enforcePatchSafety(patch: PatchResult): void {
    const branchName = (patch.branchName || '').trim();
    const protectedBranches = new Set(['main', 'master', 'develop', 'dev', 'release']);

    if (!branchName) {
      throw new Error('Phase 4 safety check failed: patch branchName is empty');
    }
    if (protectedBranches.has(branchName.toLowerCase())) {
      throw new Error(`Phase 4 safety check failed: unsafe target branch '${branchName}'`);
    }
    if (!branchName.startsWith('integration/')) {
      throw new Error(
        `Phase 4 safety check failed: branch '${branchName}' must start with 'integration/'`,
      );
    }

    const newFiles = patch.newFiles?.length || 0;
    const transformations = patch.transformations?.length || 0;
    if (newFiles === 0 && transformations === 0) {
      throw new Error('Phase 4 safety check failed: patch has no generated changes');
    }
  }

  private async retryPhaseWithBackoff(
    phase: number,
    errorMessage: string,
    retryCount: number,
    runId: string,
    integrationId: string | undefined,
    input: IntegrationCreate,
    mode: ExecutionMode,
    context: OrchestrationContext,
    phaseResults: Record<number, unknown>,
  ): Promise<boolean> {
    const nextRetry = retryCount + 1;
    if (nextRetry > config.execution.maxRetries) {
      logger.error('Retry budget exhausted', {
        runId,
        phase,
        retryCount,
        maxRetries: config.execution.maxRetries,
        error: errorMessage,
      });
      return false;
    }

    const delayMs = this.computeRetryDelayMs(nextRetry);
    logger.warn('Scheduling deterministic retry', {
      runId,
      phase,
      nextRetry,
      delayMs,
      error: errorMessage,
    });

    if (integrationId && this.convex) {
      await this.convex.incrementRetry(integrationId);
    }

    await this.saveSnapshot({
      runId,
      integrationId,
      input,
      mode,
      status: 'running',
      currentPhase: Math.max(phase - 1, 0),
      retryCount: nextRetry,
      lastErrorPhase: phase,
      phaseResults,
      context,
      errorMessage,
    });

    await new Promise((resolve) => setTimeout(resolve, delayMs));

    await this.runIntegration(input, {
      runId,
      integrationId,
      startPhase: phase,
      retryCount: nextRetry,
      phaseResults,
      contextOverrides: context,
    });

    return true;
  }

  private computeRetryDelayMs(retryAttempt: number): number {
    const normalizedAttempt = Math.max(1, retryAttempt);
    const baseMs = 1000;
    const maxMs = 30000;
    return Math.min(baseMs * (2 ** (normalizedAttempt - 1)), maxMs);
  }

  private async updateExternalPhase(
    integrationId: string | undefined,
    phase: number,
    status:
      | 'phase1_analyzing'
      | 'phase2_extracting'
      | 'phase3_mapping'
      | 'phase4_patching'
      | 'phase5_validating'
      | 'phase6_reporting',
  ): Promise<void> {
    if (integrationId && this.convex) {
      await this.convex.updateStatus(integrationId, status, phase);
    }
  }

  private highestCompletedPhase(phaseResults: Record<number, unknown>): number {
    const phases = Object.keys(phaseResults)
      .map((item) => Number(item))
      .filter((item) => Number.isFinite(item));
    return phases.length ? Math.max(...phases) : 0;
  }

  private generateRunId(): string {
    const stamp = Date.now().toString(36);
    const random = Math.random().toString(36).slice(2, 8);
    return `run-${stamp}-${random}`;
  }

  private async saveSnapshot(args: {
    runId: string;
    integrationId?: string;
    input: IntegrationCreate;
    mode: ExecutionMode;
    status: RunSnapshot['status'];
    currentPhase: number;
    retryCount: number;
    lastErrorPhase?: number;
    awaitingReason?: string;
    guardrailReasons?: string[];
    phaseResults: Record<number, unknown>;
    context: OrchestrationContext;
    errorMessage?: string;
  }): Promise<void> {
    const existing = await this.runStore.get(args.runId);
    const snapshot: RunSnapshot = {
      runId: args.runId,
      integrationId: args.integrationId,
      repoUrl: args.input.repoUrl,
      paperUrl: args.input.paperUrl,
      paperPdfPath: args.input.paperPdfPath,
      mode: args.mode,
      status: args.status,
      currentPhase: args.currentPhase,
      retryCount: args.retryCount,
      lastErrorPhase: args.lastErrorPhase,
      awaitingReason: args.awaitingReason,
      guardrailReasons: args.guardrailReasons,
      phaseResults: args.phaseResults,
      context: args.context,
      createdAt: existing?.createdAt || new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      errorMessage: args.errorMessage,
      approvals: existing?.approvals || [],
    };
    await this.runStore.save(snapshot);
  }
}
