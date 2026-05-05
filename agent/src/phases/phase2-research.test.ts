import { describe, it, expect, vi, beforeEach } from 'vitest';
import { executePhase2, validatePhase2Input } from './phase2-research.js';
import type { Phase2Context } from './types.js';
import type { RepoAnalysisResult, ResearchSpecResult } from '../bridges/python-subprocess.js';

describe('phase2-research', () => {
  describe('validatePhase2Input', () => {
    it('should return null for valid input', () => {
      const context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv' as const,
        repoAnalysis: {} as RepoAnalysisResult,
      };
      expect(validatePhase2Input(context)).toBeNull();
    });

    it('should return error for missing paperSource', () => {
      const context = {
        repoPath: '/home/user/repo',
        paperSource: '',
        sourceType: 'pdf' as const,
        repoAnalysis: {} as RepoAnalysisResult,
      };
      expect(validatePhase2Input(context)).toBe('Paper source is required');
    });

    it('should return error for missing sourceType', () => {
      const context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: '' as any,
        repoAnalysis: {} as RepoAnalysisResult,
      };
      expect(validatePhase2Input(context)).toBe('Source type (pdf or arxiv) is required');
    });
  });

  describe('executePhase2', () => {
    const mockBridge = {
      extractResearch: vi.fn(),
    };

    beforeEach(() => {
      vi.clearAllMocks();
    });

    it('should return success with data when extraction succeeds', async () => {
      const mockSpec: ResearchSpecResult = {
        paper: {
          title: 'Attention Is All You Need',
          authors: ['Vaswani et al.'],
          arxiv: '1706.03762',
          year: 2017,
        },
        algorithm: {
          name: 'Multi-Head Attention',
          description: 'Self-attention mechanism',
          formula: 'Attention(Q,K,V) = softmax(QK^T / sqrt(d))V',
        },
        implementation: {
          moduleName: 'attention',
          parentClass: 'Module',
          parameters: ['query', 'key', 'value'],
          codeTemplate: 'def forward(self, query, key, value): ...',
        },
        changes: {
          type: 'modify',
          targetPattern: 'class Attention',
          insertionPoints: ['forward method'],
        },
      };

      mockBridge.extractResearch.mockResolvedValue({
        success: true,
        data: mockSpec,
      });

      const context: Phase2Context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1706.03762',
        sourceType: 'arxiv',
        repoAnalysis: {} as RepoAnalysisResult,
      };

      const result = await executePhase2(mockBridge as any, context);

      expect(result.success).toBe(true);
      expect(result.data).toEqual(mockSpec);
      // 60 (base) + 15 (arxiv) + 15 (codeTemplate) + 10 (targetPattern) = 100
      expect(result.confidence).toBe(100);
    });

    it('should return error when extraction fails', async () => {
      mockBridge.extractResearch.mockResolvedValue({
        success: false,
        error: 'Failed to parse PDF',
      });

      const context: Phase2Context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        repoAnalysis: {} as RepoAnalysisResult,
      };

      const result = await executePhase2(mockBridge as any, context);

      expect(result.success).toBe(false);
      expect(result.error).toBe('Failed to parse PDF');
    });

    it('should return error when exception is thrown', async () => {
      mockBridge.extractResearch.mockRejectedValue(new Error('Network error'));

      const context: Phase2Context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        repoAnalysis: {} as RepoAnalysisResult,
      };

      const result = await executePhase2(mockBridge as any, context);

      expect(result.success).toBe(false);
      expect(result.error).toBe('Network error');
    });

    it('should calculate confidence correctly for minimal spec', async () => {
      const mockSpec: ResearchSpecResult = {
        paper: {
          title: 'Test Paper',
          authors: ['Author'],
          year: 2020,
        },
        algorithm: {
          name: 'Test Algorithm',
          description: 'Test description',
        },
        implementation: {
          moduleName: 'test',
          parentClass: 'Module',
          parameters: [],
          codeTemplate: '',
        },
        changes: {
          type: 'add',
          targetPattern: '',
          insertionPoints: [],
        },
      };

      mockBridge.extractResearch.mockResolvedValue({
        success: true,
        data: mockSpec,
      });

      const context: Phase2Context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        repoAnalysis: {} as RepoAnalysisResult,
      };

      const result = await executePhase2(mockBridge as any, context);

      // 60 (base) + 0 (no arxiv) + 0 (no codeTemplate) + 0 (no targetPattern) = 60
      expect(result.confidence).toBe(60);
    });

    it('should calculate confidence correctly with arxiv only', async () => {
      const mockSpec: ResearchSpecResult = {
        paper: {
          title: 'Test Paper',
          authors: ['Author'],
          arxiv: '1234.5678',
          year: 2020,
        },
        algorithm: {
          name: 'Test Algorithm',
          description: 'Test description',
        },
        implementation: {
          moduleName: 'test',
          parentClass: 'Module',
          parameters: [],
          codeTemplate: '',
        },
        changes: {
          type: 'add',
          targetPattern: '',
          insertionPoints: [],
        },
      };

      mockBridge.extractResearch.mockResolvedValue({
        success: true,
        data: mockSpec,
      });

      const context: Phase2Context = {
        repoPath: '/home/user/repo',
        paperSource: 'https://arxiv.org/pdf/1234.5678',
        sourceType: 'arxiv',
        repoAnalysis: {} as RepoAnalysisResult,
      };

      const result = await executePhase2(mockBridge as any, context);

      // 60 (base) + 15 (arxiv) = 75
      expect(result.confidence).toBe(75);
    });
  });
});