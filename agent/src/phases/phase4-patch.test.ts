import { describe, it, expect, vi, beforeEach } from 'vitest';
import { executePhase4, validatePhase4Input } from './phase4-patch.js';
import type { Phase4Context } from './types.js';
import type { MappingResult, PatchResult } from '../bridges/python-subprocess.js';

describe('phase4-patch', () => {
  describe('validatePhase4Input', () => {
    it('should return null for valid input with targets', () => {
      const context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv' as const,
        mapping: {
          targets: [{ file: 'src/attention.py', line: 42, currentCode: '', replacementRequired: true }],
          strategy: 'incremental',
          confidence: 85,
        } as MappingResult,
      };
      expect(validatePhase4Input(context)).toBeNull();
    });

    it('should return error for missing mapping', () => {
      const context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv' as const,
        mapping: undefined,
      };
      expect(validatePhase4Input(context)).toBe('Mapping result is required (complete Phase 3)');
    });

    it('should return error for empty targets', () => {
      const context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv' as const,
        mapping: {
          targets: [],
          strategy: 'none',
          confidence: 100,
        } as MappingResult,
      };
      expect(validatePhase4Input(context)).toBe('No insertion targets found - cannot generate patch');
    });
  });

  describe('executePhase4', () => {
    const mockBridge = {
      generatePatch: vi.fn(),
    };

    beforeEach(() => {
      vi.clearAllMocks();
    });

    it('should return success with patch data', async () => {
      const mockPatch: PatchResult = {
        newFiles: [
          { path: 'src/attention_new.py', content: 'def attention(): pass' },
        ],
        transformations: [
          { file: 'src/attention.py', original: 'def attention(x):', modified: 'def attention(x, n_heads=8):', changes: [] },
        ],
        branchName: 'sdc/attention-enhancement',
      };

      mockBridge.generatePatch.mockResolvedValue({
        success: true,
        data: mockPatch,
      });

      const context: Phase4Context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        mapping: {
          targets: [{ file: 'src/attention.py', line: 42, currentCode: '', replacementRequired: true }],
          strategy: 'incremental',
          confidence: 85,
        },
      };

      const result = await executePhase4(mockBridge as any, context);

      expect(result.success).toBe(true);
      expect(result.data).toEqual(mockPatch);
      expect(result.confidence).toBe(90);
    });

    it('should return error when patch generation fails', async () => {
      mockBridge.generatePatch.mockResolvedValue({
        success: false,
        error: 'Failed to generate patch - syntax errors in template',
      });

      const context: Phase4Context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        mapping: {
          targets: [{ file: 'src/attention.py', line: 42, currentCode: '', replacementRequired: true }],
          strategy: 'incremental',
          confidence: 85,
        },
      };

      const result = await executePhase4(mockBridge as any, context);

      expect(result.success).toBe(false);
      expect(result.error).toBe('Failed to generate patch - syntax errors in template');
    });

    it('should return error when exception is thrown', async () => {
      mockBridge.generatePatch.mockRejectedValue(new Error('LLM API rate limit'));

      const context: Phase4Context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        mapping: {
          targets: [{ file: 'src/attention.py', line: 42, currentCode: '', replacementRequired: true }],
          strategy: 'incremental',
          confidence: 85,
        },
      };

      const result = await executePhase4(mockBridge as any, context);

      expect(result.success).toBe(false);
      expect(result.error).toBe('LLM API rate limit');
    });

    it('should handle patch with no new files', async () => {
      const mockPatch: PatchResult = {
        newFiles: [],
        transformations: [
          { file: 'src/model.py', original: 'class Model:', modified: 'class Model(nn.Module):', changes: [] },
        ],
        branchName: 'sdc/model-update',
      };

      mockBridge.generatePatch.mockResolvedValue({
        success: true,
        data: mockPatch,
      });

      const context: Phase4Context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        mapping: {
          targets: [{ file: 'src/model.py', line: 10, currentCode: 'class Model:', replacementRequired: true }],
          strategy: 'transformation',
          confidence: 80,
        },
      };

      const result = await executePhase4(mockBridge as any, context);

      expect(result.success).toBe(true);
      expect(result.data?.newFiles).toHaveLength(0);
      expect(result.data?.transformations).toHaveLength(1);
    });
  });
});