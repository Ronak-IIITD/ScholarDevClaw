import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { config } from './config.js';

// Set up fetch mock at module level using vi.hoisted to ensure it's available
const { mockFetch } = vi.hoisted(() => ({
  mockFetch: vi.fn(),
}));
vi.stubGlobal('fetch', mockFetch);

import { AgentTools } from './tools.js';

describe('AgentTools', () => {
  const mockBridge = {
    analyzeRepo: vi.fn(),
    extractResearch: vi.fn(),
    mapArchitecture: vi.fn(),
    generatePatch: vi.fn(),
    validate: vi.fn(),
    healthCheck: vi.fn(),
  };

  let tools: AgentTools;

  beforeEach(() => {
    vi.clearAllMocks();
    tools = new AgentTools(mockBridge as any);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('runAnalyze', () => {
    it('returns success result from bridge', async () => {
      mockBridge.analyzeRepo.mockResolvedValue({ success: true, data: { files: [] } });
      const result = await tools.runAnalyze('/repo');
      expect(result.success).toBe(true);
      expect(result.data).toEqual({ files: [] });
      expect(result.duration).toBeGreaterThanOrEqual(0);
      expect(mockBridge.analyzeRepo).toHaveBeenCalledWith('/repo');
    });

    it('returns error result when bridge throws', async () => {
      mockBridge.analyzeRepo.mockRejectedValue(new Error('connection refused'));
      const result = await tools.runAnalyze('/repo');
      expect(result.success).toBe(false);
      expect(result.error).toBe('connection refused');
    });

    it('returns error for non-Error throws', async () => {
      mockBridge.analyzeRepo.mockRejectedValue('string error');
      const result = await tools.runAnalyze('/repo');
      expect(result.success).toBe(false);
      expect(result.error).toBe('Unknown error');
    });
  });

  describe('runResearch', () => {
    it('calls bridge.extractResearch with correct args', async () => {
      mockBridge.extractResearch.mockResolvedValue({ success: true, data: { specs: [] } });
      const result = await tools.runResearch('paper.pdf', 'pdf');
      expect(result.success).toBe(true);
      expect(mockBridge.extractResearch).toHaveBeenCalledWith('paper.pdf', 'pdf');
    });
  });

  describe('runMapping', () => {
    it('calls bridge.mapArchitecture', async () => {
      const analysis = { files: ['a.py'] };
      const spec = { improvements: [] };
      mockBridge.mapArchitecture.mockResolvedValue({ success: true, data: { map: {} } });
      const result = await tools.runMapping(analysis as any, spec as any);
      expect(result.success).toBe(true);
      expect(mockBridge.mapArchitecture).toHaveBeenCalledWith(analysis, spec);
    });
  });

  describe('runGenerate', () => {
    it('calls bridge.generatePatch', async () => {
      const mapping = { targets: [] };
      mockBridge.generatePatch.mockResolvedValue({ success: true, data: { patches: [] } });
      const result = await tools.runGenerate(mapping as any, '/repo');
      expect(result.success).toBe(true);
      expect(mockBridge.generatePatch).toHaveBeenCalledWith(mapping, '/repo');
    });
  });

  describe('runValidate', () => {
    it('calls bridge.validate', async () => {
      const patch = { branchName: 'integration/test' };
      mockBridge.validate.mockResolvedValue({ success: true, data: { passed: true } });
      const result = await tools.runValidate(patch as any, '/repo');
      expect(result.success).toBe(true);
      expect(mockBridge.validate).toHaveBeenCalledWith(patch, '/repo');
    });
  });

  describe('healthCheck', () => {
    it('returns true when bridge is healthy', async () => {
      mockBridge.healthCheck.mockResolvedValue(true);
      const result = await tools.healthCheck();
      expect(result).toBe(true);
    });

    it('returns false when bridge throws', async () => {
      mockBridge.healthCheck.mockRejectedValue(new Error('down'));
      const result = await tools.healthCheck();
      expect(result).toBe(false);
    });
  });

  describe('HTTP endpoint methods', () => {
    beforeEach(() => {
      mockFetch.mockReset();
    });

    it('runPlanner calls /planner/run', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ result: 'ok' }),
        status: 200,
      });

      const result = await tools.runPlanner('/repo', 3);
      expect(result.success).toBe(true);
      expect(result.data).toEqual({ result: 'ok' });
      expect(mockFetch).toHaveBeenCalledWith(
        `${config.python.coreApiUrl}/planner/run`,
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('/repo'),
        }),
      );
    });

    it('runPlanner sets default maxSpecs to 5', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ result: 'ok' }),
        status: 200,
      });

      await tools.runPlanner('/repo');
      const callBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(callBody.maxSpecs).toBe(5);
    });

    it('runCritic calls /critic/run', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ result: 'ok' }),
        status: 200,
      });

      const result = await tools.runCritic('/repo', 'spec-1');
      expect(result.success).toBe(true);
    });

    it('getContext calls /context/run', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ result: 'ok' }),
        status: 200,
      });

      const result = await tools.getContext('/repo', 'analyze');
      expect(result.success).toBe(true);
    });

    it('runExperiment calls /experiment/run', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ result: 'ok' }),
        status: 200,
      });

      const result = await tools.runExperiment('/repo', 'spec-1', 2);
      expect(result.success).toBe(true);
    });

    it('handles fetch failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
      });

      const result = await tools.runPlanner('/repo');
      expect(result.success).toBe(false);
      expect(result.error).toBe('HTTP 500');
    });

    it('handles fetch throw', async () => {
      mockFetch.mockRejectedValueOnce(new Error('network error'));

      const result = await tools.runCritic('/repo');
      expect(result.success).toBe(false);
      expect(result.error).toBe('network error');
    });
  });
});
