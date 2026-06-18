/**
 * Phase 5: Validation — TypeScript-orchestrated parallel validation.
 *
 * Previously this was a thin wrapper over a single Python call. Now the
 * TypeScript orchestrator drives each validation step individually, running
 * independent steps in parallel via Promise.allSettled to reduce wall-clock
 * time by ~1.5–2×.
 *
 * Execution plan:
 *   1. validate_artifacts + check_policy (parallel, both fast)
 *   2. run_tests (must pass before benchmarks)
 *   3. [optional] heal_patch + re-run tests (up to 2 attempts)
 *   4. run_benchmark + run_training(baseline) + run_training(variant) (parallel)
 *   5. numerical_correctness + regression_snapshot + diff_readability (parallel)
 *   6. aggregate all results
 */

import { logger } from '../utils/logger.js';
import type { PythonHttpBridge } from '../bridges/python-http.js';
import type { Phase5Context, PhaseResult } from './types.js';
import type { PatchResult, StructuredObject, ValidationComparison, ValidationResult } from '../bridges/python-subprocess.js';
import type { PythonBridge } from '../bridges/python-bridge.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface StepResult {
  step: string;
  passed: boolean;
  data?: unknown;
  error?: string;
  durationMs: number;
}

interface AggregatedValidation extends ValidationResult {
  steps: StepResult[];
  totalDurationMs: number;
}

// ---------------------------------------------------------------------------
// Progress callback
// ---------------------------------------------------------------------------

export type ValidationProgressCallback = (step: string, status: 'running' | 'completed' | 'failed', detail?: string) => void;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function timedStep(name: string, fn: () => Promise<unknown>): Promise<StepResult> {
  const t0 = Date.now();
  try {
    const data = await fn();
    return { step: name, passed: true, data, durationMs: Date.now() - t0 };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { step: name, passed: false, error: msg, durationMs: Date.now() - t0 };
  }
}

function extractBool(result: StepResult, key: string): boolean {
  // Unwrap PhaseResult.data if present (the timedStep wraps the bridge response)
  const raw = result.data as Record<string, unknown> | undefined;
  const d = raw && typeof raw === 'object' && 'data' in raw
    ? (raw.data as Record<string, unknown> | undefined)
    : raw;
  if (d && typeof d === 'object' && key in d) {
    return Boolean(d[key]);
  }
  return result.passed;
}

function extractString(result: StepResult, key: string): string {
  const raw = result.data as Record<string, unknown> | undefined;
  const d = raw && typeof raw === 'object' && 'data' in raw
    ? (raw.data as Record<string, unknown> | undefined)
    : raw;
  if (d && typeof d === 'object' && key in d) {
    return String(d[key] ?? '');
  }
  return '';
}

// ---------------------------------------------------------------------------
// Main orchestration
// ---------------------------------------------------------------------------

const MAX_HEAL_ATTEMPTS = 2;

export async function executePhase5(
  bridge: PythonBridge,
  context: Phase5Context,
  repoPath: string,
  onProgress?: ValidationProgressCallback,
): Promise<PhaseResult<AggregatedValidation>> {
  const t0 = Date.now();
  const steps: StepResult[] = [];
  const patch = context.patch!;

  logger.info('=== Phase 5: Validation (TS-orchestrated) ===');

  try {
    // Check if bridge supports sub-step orchestration (PythonHttpBridge)
    const httpBridge = bridge as PythonHttpBridge;
    const hasSubSteps = typeof httpBridge.validateArtifacts === 'function';

    if (!hasSubSteps) {
      // Fallback: use monolithic validate() for non-HTTP bridges
      logger.info('Bridge does not support sub-step validation, using monolithic validate()');
      onProgress?.('validate', 'running');
      const result = await bridge.validate(patch, repoPath);
      onProgress?.('validate', result.success ? 'completed' : 'failed');

      if (!result.success) {
        return { success: false, error: result.error };
      }

      const validation = result.data as ValidationResult;
      return {
        success: true,
        data: { ...validation, steps: [{ step: 'validate', passed: validation.passed, durationMs: 0 }], totalDurationMs: 0 } as AggregatedValidation,
        confidence: validation.passed ? 95 : 50,
      };
    }
    // ------------------------------------------------------------------
    // Step 1: artifacts + policy check (parallel)
    // ------------------------------------------------------------------
    onProgress?.('artifacts', 'running');
    onProgress?.('policy', 'running');

    const [artifactsResult, policyResult] = await Promise.allSettled([
      timedStep('artifacts', () => httpBridge.validateArtifacts(patch)),
      timedStep('policy', () => httpBridge.checkPolicy()),
    ]);

    const artifacts = artifactsResult.status === 'fulfilled' ? artifactsResult.value : { step: 'artifacts', passed: false, error: 'Promise rejected', durationMs: 0 };
    const policy = policyResult.status === 'fulfilled' ? policyResult.value : { step: 'policy', passed: false, error: 'Promise rejected', durationMs: 0 };

    steps.push(artifacts, policy);
    onProgress?.('artifacts', extractBool(artifacts, 'passed') ? 'completed' : 'failed');
    onProgress?.('policy', extractBool(policy, 'blocked') ? 'failed' : 'completed');

    if (!extractBool(artifacts, 'passed')) {
      return buildResult(steps, t0, {
        passed: false,
        stage: 'artifacts',
        logs: extractString(artifacts, 'logs'),
        error: extractString(artifacts, 'error') || 'Artifact validation failed',
      });
    }

    if (extractBool(policy, 'blocked')) {
      return buildResult(steps, t0, {
        passed: false,
        stage: 'policy',
        logs: extractString(policy, 'logs'),
        error: extractString(policy, 'error') || 'Execution policy blocked validation',
      });
    }

    // ------------------------------------------------------------------
    // Step 2: run tests
    // ------------------------------------------------------------------
    onProgress?.('tests', 'running');
    let testResult = await timedStep('tests', () => httpBridge.runTests(repoPath));
    steps.push(testResult);
    onProgress?.('tests', testResult.passed ? 'completed' : 'failed');

    // ------------------------------------------------------------------
    // Step 3: healing loop (if tests failed)
    // ------------------------------------------------------------------
    let healAttempts = 0;
    while (!extractBool(testResult, 'passed') && healAttempts < MAX_HEAL_ATTEMPTS) {
      healAttempts++;
      logger.info(`Healing attempt ${healAttempts}/${MAX_HEAL_ATTEMPTS}`);
      onProgress?.('heal', 'running', `Attempt ${healAttempts}`);

      const healResult = await timedStep(`heal_${healAttempts}`, () =>
        httpBridge.healPatch(
          patch,
          {
            passed: extractBool(testResult, 'passed'),
            stage: extractString(testResult, 'stage') || 'tests',
            logs: extractString(testResult, 'logs'),
            error: extractString(testResult, 'error'),
          },
          context.mapping ?? null,
          repoPath,
        ),
      );
      steps.push(healResult);
      onProgress?.('heal', healResult.passed ? 'completed' : 'failed', `Attempt ${healAttempts}`);

      if (!healResult.passed) {
        logger.warn(`Healing attempt ${healAttempts} failed: ${healResult.error}`);
        break;
      }

      // Update patch with healed version
      const healedRaw = healResult.data as Record<string, unknown> | undefined;
      const healed = healedRaw && typeof healedRaw === 'object' && 'data' in healedRaw
        ? (healedRaw.data as PatchResult | undefined)
        : (healedRaw as PatchResult | undefined);
      if (healed) {
        Object.assign(patch, healed);
      }

      // Re-run tests
      onProgress?.('tests', 'running', 'Re-testing after heal');
      testResult = await timedStep('tests_retest', () => httpBridge.runTests(repoPath));
      steps.push(testResult);
      onProgress?.('tests', extractBool(testResult, 'passed') ? 'completed' : 'failed', 'After heal');

      if (extractBool(testResult, 'passed')) {
        logger.info(`Healing successful after ${healAttempts} attempt(s)`);
        break;
      }
    }

    if (!extractBool(testResult, 'passed')) {
      return buildResult(steps, t0, {
        passed: false,
        stage: 'tests',
        logs: extractString(testResult, 'logs'),
        error: extractString(testResult, 'error') || 'Tests failed after healing attempts',
      });
    }

    // ------------------------------------------------------------------
    // Step 4: benchmark + training baseline + training variant (parallel)
    // ------------------------------------------------------------------
    onProgress?.('benchmark', 'running');
    onProgress?.('training_baseline', 'running');
    onProgress?.('training_variant', 'running');

    const [benchmarkResult, baselineResult, variantResult] = await Promise.allSettled([
      timedStep('benchmark', () => httpBridge.runBenchmark(repoPath)),
      timedStep('training_baseline', () => httpBridge.runTraining(repoPath, { useVariant: false })),
      timedStep('training_variant', () => httpBridge.runTraining(repoPath, { useVariant: true })),
    ]);

    const benchmark = benchmarkResult.status === 'fulfilled' ? benchmarkResult.value : { step: 'benchmark', passed: false, error: 'Promise rejected', durationMs: 0 };
    const baseline = baselineResult.status === 'fulfilled' ? baselineResult.value : { step: 'training_baseline', passed: false, error: 'Promise rejected', durationMs: 0 };
    const variant = variantResult.status === 'fulfilled' ? variantResult.value : { step: 'training_variant', passed: false, error: 'Promise rejected', durationMs: 0 };

    steps.push(benchmark, baseline, variant);
    onProgress?.('benchmark', benchmark.passed ? 'completed' : 'failed');
    onProgress?.('training_baseline', baseline.passed ? 'completed' : 'failed');
    onProgress?.('training_variant', variant.passed ? 'completed' : 'failed');

    // ------------------------------------------------------------------
    // Step 5: numerical correctness + regression snapshot + readability (parallel)
    // ------------------------------------------------------------------
    onProgress?.('correctness', 'running');
    onProgress?.('regression', 'running');
    onProgress?.('readability', 'running');

    const hasArtifacts = Boolean(patch.newFiles?.length || patch.transformations?.length);

    const analysisPromises = [
      timedStep('correctness', () => httpBridge.runNumericalCorrectness(patch)),
      timedStep('regression', () => httpBridge.runRegressionSnapshot(patch)),
      timedStep('readability', () => httpBridge.scoreDiffReadability(patch)),
    ];

    const analysisResults = await Promise.allSettled(analysisPromises);

    const correctness = analysisResults[0]?.status === 'fulfilled' ? analysisResults[0].value : { step: 'correctness', passed: false, error: 'Promise rejected', durationMs: 0 };
    const regression = analysisResults[1]?.status === 'fulfilled' ? analysisResults[1].value : { step: 'regression', passed: false, error: 'Promise rejected', durationMs: 0 };
    const readability = analysisResults[2]?.status === 'fulfilled' ? analysisResults[2].value : { step: 'readability', passed: false, error: 'Promise rejected', durationMs: 0 };

    steps.push(correctness, regression, readability);
    onProgress?.('correctness', correctness.passed ? 'completed' : 'failed');
    onProgress?.('regression', regression.passed ? 'completed' : 'failed');
    onProgress?.('readability', readability.passed ? 'completed' : 'failed');

    // ------------------------------------------------------------------
    // Step 6: aggregate results
    // ------------------------------------------------------------------
    // Unwrap PhaseResult.data for all step results
    const unwrap = (r: StepResult) => {
      const raw = r.data as Record<string, unknown> | undefined;
      return raw && typeof raw === 'object' && 'data' in raw
        ? (raw.data as Record<string, unknown> | undefined)
        : raw;
    };

    const benchmarkData = unwrap(benchmark);
    const baselineData = unwrap(baseline);
    const variantData = unwrap(variant);

    const comparison: ValidationComparison = {
      ...(benchmarkData?.comparison as ValidationComparison ?? {}),
    };

    if (baselineData && variantData && baselineData.status === 'completed' && variantData.status === 'completed') {
      const baseTps = Number(baselineData.tokens_per_second) || 0;
      const varTps = Number(variantData.tokens_per_second) || 0;
      if (baseTps > 0) {
        comparison.speedup = Math.round((varTps / baseTps) * 10000) / 10000;
      }

      const baseLoss = Number(baselineData.loss) || 0;
      const varLoss = Number(variantData.loss) || 0;
      if (baseLoss !== 0) {
        comparison.loss_change = Math.round(((varLoss - baseLoss) / Math.abs(baseLoss)) * 10000) / 100;
      }
    }

    if (hasArtifacts) {
      const correctnessData = unwrap(correctness) as StructuredObject | undefined;
      const regressionData = unwrap(regression) as StructuredObject | undefined;
      const readabilityData = unwrap(readability) as StructuredObject | undefined;

      if (correctnessData) comparison.numerical_correctness = correctnessData;
      if (regressionData) comparison.regression_snapshot = regressionData;
      if (readabilityData) comparison.diff_readability = readabilityData;
    }

    // Determine overall pass/fail
    const testsPassed = extractBool(testResult, 'passed');
    const benchmarkPassed = benchmark.passed;
    const correctnessPassed = hasArtifacts
      ? extractBool(correctness, 'passed') || (correctness.data as Record<string, unknown>)?.status === 'skipped'
      : true;
    const regressionPassed = hasArtifacts
      ? extractBool(regression, 'passed') || (regression.data as Record<string, unknown>)?.status === 'skipped'
      : true;

    const overallPassed = testsPassed && benchmarkPassed && correctnessPassed && regressionPassed;

    let failedStage = 'full_validation';
    if (!testsPassed) failedStage = extractString(testResult, 'stage') || 'tests';
    else if (!correctnessPassed) failedStage = 'numerical_correctness';
    else if (!regressionPassed) failedStage = 'regression_snapshot';

    // Build metrics
    const baselineMetrics = baselineData && baselineData.status === 'completed'
      ? { loss: Number(baselineData.loss), perplexity: Number(baselineData.perplexity), tokensPerSecond: Number(baselineData.tokens_per_second), memoryMb: Number(baselineData.memory_mb) }
      : undefined;
    const newMetrics = variantData && variantData.status === 'completed'
      ? { loss: Number(variantData.loss), perplexity: Number(variantData.perplexity), tokensPerSecond: Number(variantData.tokens_per_second), memoryMb: Number(variantData.memory_mb) }
      : undefined;

    // Merge logs from all steps
    const allLogs = steps
      .map((s) => `[${s.step}] ${s.passed ? 'PASS' : 'FAIL'} (${s.durationMs}ms)${s.error ? `: ${s.error}` : ''}`)
      .join('\n');

    return buildResult(steps, t0, {
      passed: overallPassed,
      stage: overallPassed ? 'full_validation' : failedStage,
      baselineMetrics: baselineMetrics as ValidationResult['baselineMetrics'],
      newMetrics: newMetrics as ValidationResult['newMetrics'],
      comparison,
      logs: allLogs,
      error: overallPassed ? undefined : `Validation failed at stage: ${failedStage}`,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    logger.error('Phase 5 error', { error: message });
    return { success: false, error: message };
  }
}

function buildResult(
  steps: StepResult[],
  t0: number,
  validation: Omit<AggregatedValidation, 'steps' | 'totalDurationMs'>,
): PhaseResult<AggregatedValidation> {
  const totalDurationMs = Date.now() - t0;
  logger.info(`Phase 5 completed in ${totalDurationMs}ms`, {
    passed: validation.passed,
    stage: validation.stage,
    steps: steps.length,
  });

  return {
    success: true,
    data: { ...validation, steps, totalDurationMs } as AggregatedValidation,
    confidence: validation.passed ? 95 : 50,
  };
}

export function validatePhase5Input(context: Phase5Context): string | null {
  if (!context.patch) {
    return 'Patch is required (complete Phase 4)';
  }
  return null;
}
