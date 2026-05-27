import { describe, it, expect, vi, beforeEach } from 'vitest';
import { executePhase1, validatePhase1Input } from './phase1-repo.js';
import type { Phase1Context, PhaseResult } from './types.js';
import type { RepoAnalysisResult } from '../bridges/python-subprocess.js';

describe('phase1-repo', () => {
  describe('validatePhase1Input', () => {
    it('should return null for valid GitHub URL', () => {
      const context = { repoPath: 'https://github.com/test/repo', paperSource: '', sourceType: 'pdf' as const };
      expect(validatePhase1Input(context)).toBeNull();
    });

    it('should return null for valid local path', () => {
      const context = { repoPath: '/home/user/repo', paperSource: '', sourceType: 'pdf' as const };
      expect(validatePhase1Input(context)).toBeNull();
    });

    it('should return error for missing repoPath', () => {
      const context = { repoPath: '', paperSource: '', sourceType: 'pdf' as const };
      expect(validatePhase1Input(context)).toBe('Repository path is required');
    });

    it('should return error for invalid path format', () => {
      const context = { repoPath: 'invalid-path', paperSource: '', sourceType: 'pdf' as const };
      expect(validatePhase1Input(context)).toBe('Invalid repository path format');
    });

    it('should accept .git suffix', () => {
      const context = { repoPath: 'https://github.com/test/repo.git', paperSource: '', sourceType: 'pdf' as const };
      expect(validatePhase1Input(context)).toBeNull();
    });
  });

  describe('executePhase1', () => {
    const mockBridge = {
      analyzeRepo: vi.fn(),
    };

    beforeEach(() => {
      vi.clearAllMocks();
    });

    it('should return success with data when bridge returns success', async () => {
      const mockAnalysis: RepoAnalysisResult = {
        repoName: 'test-repo',
        architecture: {
          models: [
            { name: 'Model1', file: 'models/model1.py', line: 10, parent: '', components: {} },
            { name: 'Model2', file: 'models/model2.py', line: 20, parent: '', components: {} },
          ],
          trainingLoop: { file: 'train.py', line: 5, optimizer: 'adam', lossFn: 'crossentropy' },
        },
        dependencies: {},
        testSuite: {
          runner: 'pytest',
          testFiles: ['test_main.py'],
        },
      };

      mockBridge.analyzeRepo.mockResolvedValue({
        success: true,
        data: mockAnalysis,
      });

      const context: Phase1Context = {
        repoPath: 'https://github.com/test/repo',
        paperSource: '',
        sourceType: 'pdf',
      };

      const result = await executePhase1(mockBridge as any, context);

      expect(result.success).toBe(true);
      expect(result.data).toEqual(mockAnalysis);
      expect(result.confidence).toBe(100); // 50 + 20 (models) + 15 (training) + 15 (tests)
    });

    it('should return error when bridge fails', async () => {
      mockBridge.analyzeRepo.mockResolvedValue({
        success: false,
        error: 'Failed to clone repository',
      });

      const context: Phase1Context = {
        repoPath: 'https://github.com/test/repo',
        paperSource: '',
        sourceType: 'pdf',
      };

      const result = await executePhase1(mockBridge as any, context);

      expect(result.success).toBe(false);
      expect(result.error).toBe('Failed to clone repository');
    });

    it('should return error when exception is thrown', async () => {
      mockBridge.analyzeRepo.mockRejectedValue(new Error('Network error'));

      const context: Phase1Context = {
        repoPath: 'https://github.com/test/repo',
        paperSource: '',
        sourceType: 'pdf',
      };

      const result = await executePhase1(mockBridge as any, context);

      expect(result.success).toBe(false);
      expect(result.error).toBe('Network error');
    });

    it('should calculate confidence correctly for minimal analysis', async () => {
      const mockAnalysis: RepoAnalysisResult = {
        repoName: 'test-repo',
        architecture: {
          models: [],
        },
        dependencies: {},
        testSuite: {
          runner: 'none',
          testFiles: [],
        },
      };

      mockBridge.analyzeRepo.mockResolvedValue({
        success: true,
        data: mockAnalysis,
      });

      const context: Phase1Context = {
        repoPath: 'https://github.com/test/repo',
        paperSource: '',
        sourceType: 'pdf',
      };

      const result = await executePhase1(mockBridge as any, context);

      expect(result.confidence).toBe(50); // Base only
    });

    it('should calculate confidence correctly with all factors', async () => {
      const mockAnalysis: RepoAnalysisResult = {
        repoName: 'test-repo',
        architecture: {
          models: [{ name: 'Model1', file: 'model.py', line: 1, parent: '', components: {} }],
          trainingLoop: { file: 'train.py', line: 1, optimizer: 'adam', lossFn: 'mse' },
        },
        dependencies: {},
        testSuite: {
          runner: 'pytest',
          testFiles: ['test_main.py'],
        },
      };

      mockBridge.analyzeRepo.mockResolvedValue({
        success: true,
        data: mockAnalysis,
      });

      const context: Phase1Context = {
        repoPath: 'https://github.com/test/repo',
        paperSource: '',
        sourceType: 'pdf',
      };

      const result = await executePhase1(mockBridge as any, context);

      expect(result.confidence).toBe(100); // Max capped
    });
  });
});
