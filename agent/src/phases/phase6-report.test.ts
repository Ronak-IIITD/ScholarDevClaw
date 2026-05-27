import { describe, it, expect, vi, beforeEach } from 'vitest';
import { executePhase6, validatePhase6Input } from './phase6-report.js';
import type { Phase6Context } from './types.js';
import type { RepoAnalysisResult, ResearchSpecResult, MappingResult, PatchResult, ValidationResult } from '../bridges/python-subprocess.js';

describe('phase6-report', () => {
  describe('validatePhase6Input', () => {
    it('should return null for valid input', () => {
      const context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv' as const,
        validation: { passed: true, stage: 'full' } as ValidationResult,
      };
      expect(validatePhase6Input(context)).toBeNull();
    });

    it('should return error for missing validation', () => {
      const context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv' as const,
        validation: undefined,
      };
      expect(validatePhase6Input(context)).toBe('Validation result is required (complete Phase 5)');
    });
  });

  describe('executePhase6', () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    it('should return success with full report when validation passed', async () => {
      const context: Phase6Context = {
        repoPath: 'https://github.com/test/repo',
        paperSource: 'https://arxiv.org/pdf/1706.03762',
        sourceType: 'arxiv',
        repoAnalysis: { repoName: 'test-repo' } as RepoAnalysisResult,
        researchSpec: {
          paper: { title: 'Attention Is All You Need' },
          algorithm: { name: 'Multi-Head Attention' },
        } as ResearchSpecResult,
        mapping: { targets: [], strategy: 'test', confidence: 85 } as MappingResult,
        patch: {
          newFiles: [{ path: 'src/attention.py', content: '' }],
          transformations: [{ file: 'src/model.py', original: '', modified: '', changes: [] }],
          branchName: 'sdc/test',
        } as PatchResult,
        validation: {
          passed: true,
          stage: 'full_validation',
          comparison: { speedup: 1.15, lossChange: -0.05, passed: true },
        } as ValidationResult,
      };

      const result = await executePhase6(context);

      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
      expect(result.data?.summary.status).toBe('completed');
      expect(result.data?.recommendation.action).toBe('approve');
      expect(result.data?.recommendation.confidence).toBe(95);
      expect(result.confidence).toBe(100);
    });

    it('should return review recommendation when speedup is low', async () => {
      const context: Phase6Context = {
        repoPath: 'https://github.com/test/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        validation: {
          passed: true,
          stage: 'full_validation',
          comparison: { speedup: 1.02, lossChange: 0, passed: true },
        } as ValidationResult,
      };

      const result = await executePhase6(context);

      expect(result.success).toBe(true);
      expect(result.data?.recommendation.action).toBe('review');
    });

    it('should return reject recommendation when validation failed', async () => {
      const context: Phase6Context = {
        repoPath: 'https://github.com/test/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        validation: {
          passed: false,
          stage: 'unit_tests',
        } as ValidationResult,
      };

      const result = await executePhase6(context);

      expect(result.success).toBe(true);
      expect(result.data?.recommendation.action).toBe('reject');
      expect(result.data?.summary.status).toBe('needs_review');
    });

    it('should add risk notes for performance regression', async () => {
      const context: Phase6Context = {
        repoPath: 'https://github.com/test/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        mapping: { confidence: 85 } as MappingResult,
        validation: {
          passed: true,
          stage: 'full',
          comparison: { speedup: 0.95, lossChange: 0, passed: true },
        } as ValidationResult,
      };

      const result = await executePhase6(context);

      expect(result.success).toBe(true);
      expect(result.data?.riskNotes).toContain('Performance regression detected - review required');
    });

    it('should add risk notes for low mapping confidence', async () => {
      const context: Phase6Context = {
        repoPath: 'https://github.com/test/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        mapping: { confidence: 60 } as MappingResult,
        validation: { passed: true, stage: 'full' } as ValidationResult,
      };

      const result = await executePhase6(context);

      expect(result.success).toBe(true);
      expect(result.data?.riskNotes).toContain('Low mapping confidence - manual review recommended');
    });

    it('should handle missing optional context fields', async () => {
      const context: Phase6Context = {
        repoPath: 'https://github.com/test/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        validation: { passed: true, stage: 'full' } as ValidationResult,
      };

      const result = await executePhase6(context);

      expect(result.success).toBe(true);
      expect(result.data?.metadata.algorithm).toBe('Unknown');
      expect(result.data?.summary.changesMade).toBe(0);
    });

    it('should generate correct diff preview', async () => {
      const context: Phase6Context = {
        repoPath: 'https://github.com/test/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        patch: {
          newFiles: [{ path: 'src/new.py', content: '' }],
          transformations: [{ file: 'src/existing.py', original: '', modified: '', changes: [] }],
          branchName: 'test',
        } as PatchResult,
        validation: { passed: true, stage: 'full' } as ValidationResult,
      };

      const result = await executePhase6(context);

      expect(result.success).toBe(true);
      expect(result.data?.diffPreview).toContain('src/new.py');
      expect(result.data?.diffPreview).toContain('src/existing.py');
    });
  });
});
