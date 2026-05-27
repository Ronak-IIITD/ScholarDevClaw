import { describe, it, expect, vi, beforeEach } from 'vitest';
import { executePhase3, validatePhase3Input } from './phase3-mapping.js';
import type { Phase3Context } from './types.js';
import type { RepoAnalysisResult, ResearchSpecResult, MappingResult } from '../bridges/python-subprocess.js';

describe('phase3-mapping', () => {
  describe('validatePhase3Input', () => {
    it('should return null for valid input', () => {
      const context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv' as const,
        repoAnalysis: { repoName: 'test' } as RepoAnalysisResult,
        researchSpec: { paper: { title: 'Test' } } as ResearchSpecResult,
      };
      expect(validatePhase3Input(context)).toBeNull();
    });

    it('should return error for missing repoAnalysis', () => {
      const context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv' as const,
        repoAnalysis: undefined,
        researchSpec: { paper: { title: 'Test' } } as ResearchSpecResult,
      };
      expect(validatePhase3Input(context)).toBe('Repository analysis is required (complete Phase 1)');
    });

    it('should return error for missing researchSpec', () => {
      const context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv' as const,
        repoAnalysis: { repoName: 'test' } as RepoAnalysisResult,
        researchSpec: undefined,
      };
      expect(validatePhase3Input(context)).toBe('Research specification is required (complete Phase 2)');
    });
  });

  describe('executePhase3', () => {
    const mockBridge = {
      mapArchitecture: vi.fn(),
    };

    beforeEach(() => {
      vi.clearAllMocks();
    });

    it('should return success with mapping data', async () => {
      const mockMapping: MappingResult = {
        targets: [
          { file: 'src/attention.py', line: 42, currentCode: 'def attention(x):', replacementRequired: true },
          { file: 'src/model.py', line: 100, currentCode: 'class Model:', replacementRequired: false },
        ],
        strategy: 'incremental_modification',
        confidence: 85,
      };

      mockBridge.mapArchitecture.mockResolvedValue({
        success: true,
        data: mockMapping,
      });

      const context: Phase3Context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        repoAnalysis: {} as RepoAnalysisResult,
        researchSpec: {} as ResearchSpecResult,
      };

      const result = await executePhase3(mockBridge as any, context);

      expect(result.success).toBe(true);
      expect(result.data).toEqual(mockMapping);
      expect(result.confidence).toBe(85);
    });

    it('should return error when mapping fails', async () => {
      mockBridge.mapArchitecture.mockResolvedValue({
        success: false,
        error: 'Failed to analyze repository architecture',
      });

      const context: Phase3Context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        repoAnalysis: {} as RepoAnalysisResult,
        researchSpec: {} as ResearchSpecResult,
      };

      const result = await executePhase3(mockBridge as any, context);

      expect(result.success).toBe(false);
      expect(result.error).toBe('Failed to analyze repository architecture');
    });

    it('should return error when exception is thrown', async () => {
      mockBridge.mapArchitecture.mockRejectedValue(new Error('Network timeout'));

      const context: Phase3Context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        repoAnalysis: {} as RepoAnalysisResult,
        researchSpec: {} as ResearchSpecResult,
      };

      const result = await executePhase3(mockBridge as any, context);

      expect(result.success).toBe(false);
      expect(result.error).toBe('Network timeout');
    });

    it('should handle empty targets list', async () => {
      const mockMapping: MappingResult = {
        targets: [],
        strategy: 'no_changes_needed',
        confidence: 100,
      };

      mockBridge.mapArchitecture.mockResolvedValue({
        success: true,
        data: mockMapping,
      });

      const context: Phase3Context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        repoAnalysis: {} as RepoAnalysisResult,
        researchSpec: {} as ResearchSpecResult,
      };

      const result = await executePhase3(mockBridge as any, context);

      expect(result.success).toBe(true);
      expect(result.data?.targets).toHaveLength(0);
      expect(result.confidence).toBe(100);
    });
  });
});
