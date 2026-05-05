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

  describe('executePhase5', () => {
    const mockBridge = {
      validate: vi.fn(),
    };

    beforeEach(() => {
      vi.clearAllMocks();
    });

    it('should return success with validation passed', async () => {
      const mockValidation: ValidationResult = {
        passed: true,
        stage: 'full_validation',
        baselineMetrics: { loss: 2.5, perplexity: 12.1, tokensPerSecond: 5000, memoryMb: 2048 },
        newMetrics: { loss: 2.3, perplexity: 11.5, tokensPerSecond: 5200, memoryMb: 2100 },
        comparison: { lossChange: -0.08, speedup: 1.04, passed: true },
      };

      mockBridge.validate.mockResolvedValue({
        success: true,
        data: mockValidation,
      });

      const context: Phase5Context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        patch: { newFiles: [], transformations: [], branchName: 'test' } as PatchResult,
      };

      const result = await executePhase5(mockBridge as any, context, '/home/user/repo');

      expect(result.success).toBe(true);
      expect(result.data).toEqual(mockValidation);
      expect(result.confidence).toBe(95);
    });

    it('should return success with validation failed', async () => {
      const mockValidation: ValidationResult = {
        passed: false,
        stage: 'unit_tests',
      };

      mockBridge.validate.mockResolvedValue({
        success: true,
        data: mockValidation,
      });

      const context: Phase5Context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        patch: { newFiles: [], transformations: [], branchName: 'test' } as PatchResult,
      };

      const result = await executePhase5(mockBridge as any, context, '/home/user/repo');

      expect(result.success).toBe(true);
      expect(result.data?.passed).toBe(false);
      expect(result.confidence).toBe(50);
    });

    it('should return error when validation fails', async () => {
      mockBridge.validate.mockResolvedValue({
        success: false,
        error: 'Validation runner crashed',
      });

      const context: Phase5Context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        patch: { newFiles: [], transformations: [], branchName: 'test' } as PatchResult,
      };

      const result = await executePhase5(mockBridge as any, context, '/home/user/repo');

      expect(result.success).toBe(false);
      expect(result.error).toBe('Validation runner crashed');
    });

    it('should return error when exception is thrown', async () => {
      mockBridge.validate.mockRejectedValue(new Error('Docker not available'));

      const context: Phase5Context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        patch: { newFiles: [], transformations: [], branchName: 'test' } as PatchResult,
      };

      const result = await executePhase5(mockBridge as any, context, '/home/user/repo');

      expect(result.success).toBe(false);
      expect(result.error).toBe('Docker not available');
    });
  });
});