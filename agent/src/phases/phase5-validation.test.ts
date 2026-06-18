import { describe, it, expect, vi, beforeEach } from 'vitest';
import { executePhase5, validatePhase5Input } from './phase5-validation.js';
import type { Phase5Context } from './types.js';
import type { PatchResult, ValidationResult } from '../bridges/python-subprocess.js';

describe('phase5-validation', () => {
  describe('validatePhase5Input', () => {
    it('should return null for valid input', () => {
      const context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv' as const,
        patch: { newFiles: [], transformations: [], branchName: 'test' } as PatchResult,
      };
      expect(validatePhase5Input(context)).toBeNull();
    });

    it('should return error for missing patch', () => {
      const context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv' as const,
        patch: undefined,
      };
      expect(validatePhase5Input(context)).toBe('Patch is required (complete Phase 4)');
    });
  });

  describe('executePhase5 — HTTP bridge (sub-step orchestration)', () => {
    let mockBridge: Record<string, ReturnType<typeof vi.fn>>;

    beforeEach(() => {
      mockBridge = {
        validateArtifacts: vi.fn().mockResolvedValue({
          success: true,
          data: { passed: true, stage: 'artifacts', logs: 'OK' },
        }),
        checkPolicy: vi.fn().mockResolvedValue({
          success: true,
          data: { blocked: false, passed: true, stage: 'policy', warning: null },
        }),
        runTests: vi.fn().mockResolvedValue({
          success: true,
          data: { passed: true, stage: 'tests', logs: 'All tests passed' },
        }),
        runBenchmark: vi.fn().mockResolvedValue({
          success: true,
          data: { passed: true, stage: 'benchmark', comparison: { speedup: 1.04 } },
        }),
        runTraining: vi.fn().mockResolvedValue({
          success: true,
          data: { status: 'completed', loss: 2.3, perplexity: 11.5, tokens_per_second: 5200, memory_mb: 2100, runtime_seconds: 1.5 },
        }),
        runNumericalCorrectness: vi.fn().mockResolvedValue({
          success: true,
          data: { status: 'skipped', reason: 'No algorithm key' },
        }),
        runRegressionSnapshot: vi.fn().mockResolvedValue({
          success: true,
          data: { status: 'passed', passed: true, files_checked: [], removed_symbols: [], signature_changes: [] },
        }),
        scoreDiffReadability: vi.fn().mockResolvedValue({
          success: true,
          data: { score: 5, source: 'heuristic', rationale: 'Clean diff' },
        }),
        healPatch: vi.fn(),
      };
    });

    const context: Phase5Context = {
      repoPath: '/home/user/repo',
      paperSource: 'https://arxiv.org/pdf/1234.5678',
      sourceType: 'arxiv',
      patch: { newFiles: [], transformations: [], branchName: 'test' } as PatchResult,
    };

    it('should return success when all steps pass', async () => {
      const result = await executePhase5(mockBridge as any, context, '/home/user/repo');

      expect(result.success).toBe(true);
      expect(result.data?.passed).toBe(true);
      expect(result.data?.stage).toBe('full_validation');
      expect(result.confidence).toBe(95);

      // Verify all sub-steps were called
      expect(mockBridge.validateArtifacts).toHaveBeenCalledOnce();
      expect(mockBridge.checkPolicy).toHaveBeenCalledOnce();
      expect(mockBridge.runTests).toHaveBeenCalled();
      expect(mockBridge.runBenchmark).toHaveBeenCalled();
      expect(mockBridge.runTraining).toHaveBeenCalledTimes(2); // baseline + variant
      expect(mockBridge.runNumericalCorrectness).toHaveBeenCalledOnce();
      expect(mockBridge.runRegressionSnapshot).toHaveBeenCalledOnce();
      expect(mockBridge.scoreDiffReadability).toHaveBeenCalledOnce();
    });

    it('should include step results and timing', async () => {
      const result = await executePhase5(mockBridge as any, context, '/home/user/repo');

      expect(result.data?.steps).toBeDefined();
      expect(result.data?.steps.length).toBeGreaterThan(0);
      expect(result.data?.totalDurationMs).toBeGreaterThanOrEqual(0);

      const stepNames = result.data?.steps.map((s) => s.step) ?? [];
      expect(stepNames).toContain('artifacts');
      expect(stepNames).toContain('policy');
      expect(stepNames).toContain('tests');
      expect(stepNames).toContain('benchmark');
    });

    it('should aggregate metrics from training steps', async () => {
      // First call = baseline, second call = variant
      mockBridge.runTraining
        .mockResolvedValueOnce({
          success: true,
          data: { status: 'completed', loss: 2.5, perplexity: 12.1, tokens_per_second: 5000, memory_mb: 2048, runtime_seconds: 1.0 },
        })
        .mockResolvedValueOnce({
          success: true,
          data: { status: 'completed', loss: 2.3, perplexity: 11.5, tokens_per_second: 5200, memory_mb: 2100, runtime_seconds: 1.5 },
        });

      const result = await executePhase5(mockBridge as any, context, '/home/user/repo');

      expect(result.data?.passed).toBe(true);
      expect(result.data?.baselineMetrics?.loss).toBe(2.5);
      expect(result.data?.baselineMetrics?.tokensPerSecond).toBe(5000);
      expect(result.data?.newMetrics?.loss).toBe(2.3);
      expect(result.data?.newMetrics?.tokensPerSecond).toBe(5200);
      expect(result.data?.comparison?.speedup).toBe(1.04);
    });

    it('should fail when artifact validation fails', async () => {
      mockBridge.validateArtifacts.mockResolvedValue({
        success: true,
        data: { passed: false, stage: 'artifacts', error: 'Syntax error' },
      });

      const result = await executePhase5(mockBridge as any, context, '/home/user/repo');

      expect(result.success).toBe(true);
      expect(result.data?.passed).toBe(false);
      expect(result.data?.stage).toBe('artifacts');
    });

    it('should fail when tests fail after healing', async () => {
      mockBridge.runTests
        .mockResolvedValueOnce({
          success: true,
          data: { passed: false, stage: 'tests', logs: 'AssertionError', error: 'Tests failed' },
        })
        .mockResolvedValue({
          success: true,
          data: { passed: false, stage: 'tests', logs: 'Still failing', error: 'Tests failed' },
        });
      mockBridge.healPatch.mockResolvedValue({
        success: true,
        data: { newFiles: [], transformations: [], branchName: 'healed' },
      });

      const result = await executePhase5(mockBridge as any, context, '/home/user/repo');

      expect(result.success).toBe(true);
      expect(result.data?.passed).toBe(false);
      expect(result.data?.stage).toBe('tests');
      expect(mockBridge.healPatch).toHaveBeenCalled();
    });

    it('should succeed after healing fixes tests', async () => {
      mockBridge.runTests
        .mockResolvedValueOnce({
          success: true,
          data: { passed: false, stage: 'tests', logs: 'fail', error: 'Tests failed' },
        })
        .mockResolvedValue({
          success: true,
          data: { passed: true, stage: 'tests', logs: 'pass' },
        });
      mockBridge.healPatch.mockResolvedValue({
        success: true,
        data: { newFiles: [], transformations: [], branchName: 'healed' },
      });

      const result = await executePhase5(mockBridge as any, context, '/home/user/repo');

      expect(result.success).toBe(true);
      expect(result.data?.passed).toBe(true);
    });

    it('should call onProgress for each step', async () => {
      const progressFn = vi.fn();

      await executePhase5(mockBridge as any, context, '/home/user/repo', progressFn);

      expect(progressFn).toHaveBeenCalledWith('artifacts', 'running');
      expect(progressFn).toHaveBeenCalledWith('artifacts', 'completed');
      expect(progressFn).toHaveBeenCalledWith('tests', 'running');
      expect(progressFn).toHaveBeenCalledWith('benchmark', 'running');
      expect(progressFn).toHaveBeenCalledWith('training_baseline', 'running');
      expect(progressFn).toHaveBeenCalledWith('training_variant', 'running');
    });
  });

  describe('executePhase5 — subprocess bridge (fallback)', () => {
    it('should fall back to monolithic validate() for non-HTTP bridges', async () => {
      const mockBridge = {
        validate: vi.fn().mockResolvedValue({
          success: true,
          data: {
            passed: true,
            stage: 'full_validation',
            baselineMetrics: { loss: 2.5, perplexity: 12.1, tokensPerSecond: 5000, memoryMb: 2048 },
            newMetrics: { loss: 2.3, perplexity: 11.5, tokensPerSecond: 5200, memoryMb: 2100 },
            comparison: { lossChange: -0.08, speedup: 1.04, passed: true },
          } as ValidationResult,
        }),
      };

      const context: Phase5Context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        patch: { newFiles: [], transformations: [], branchName: 'test' } as PatchResult,
      };

      const result = await executePhase5(mockBridge as any, context, '/home/user/repo');

      expect(result.success).toBe(true);
      expect(result.data?.passed).toBe(true);
      expect(mockBridge.validate).toHaveBeenCalledOnce();
    });
  });
});
