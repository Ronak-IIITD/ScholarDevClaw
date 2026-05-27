import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  createAnalyzeNode,
  createResearchNode,
  createMappingNode,
  createPatchNode,
  createValidationNode,
  createReportNode,
  createPlannerNode,
  createCriticNode,
  createExperimentNode,
  createIntegrationWorkflow,
  createExperimentWorkflow,
} from './integration.js';
import { FunctionNode } from './node.js';

describe('integration phase node creators', () => {
  const mockBridge = {
    analyzeRepo: vi.fn(),
    extractResearch: vi.fn(),
    mapArchitecture: vi.fn(),
    generatePatch: vi.fn(),
    validate: vi.fn(),
  };

  const mockState = () => ({
    workflowId: 'test',
    status: 'running',
    context: {} as Record<string, unknown>,
    nodeResults: new Map(),
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('createAnalyzeNode', () => {
    it('returns a FunctionNode with correct config', () => {
      const node = createAnalyzeNode(mockBridge as any);
      expect(node).toBeInstanceOf(FunctionNode);
      expect(node.id).toBe('phase1-analyze');
      expect(node.name).toBe('Repository Analysis');
    });

    it('executes bridge.analyzeRepo and stores result in context', async () => {
      mockBridge.analyzeRepo.mockResolvedValue({ success: true, data: { files: ['a.py'] } });
      const node = createAnalyzeNode(mockBridge as any);
      const state = mockState();

      const result = await node.execute({ repoPath: '/repo' }, state);

      expect(mockBridge.analyzeRepo).toHaveBeenCalledWith('/repo');
      expect(state.context.repoAnalysis).toEqual({ files: ['a.py'] });
      expect(result).toEqual({ files: ['a.py'] });
    });

    it('throws when bridge returns no data', async () => {
      mockBridge.analyzeRepo.mockResolvedValue({ success: true, data: null });
      const node = createAnalyzeNode(mockBridge as any);

      await expect(node.execute({ repoPath: '/repo' }, mockState())).rejects.toThrow('Phase 1 failed');
    });

    it('throws when bridge returns failure', async () => {
      mockBridge.analyzeRepo.mockResolvedValue({ success: false, error: 'connection error' });
      const node = createAnalyzeNode(mockBridge as any);

      await expect(node.execute({ repoPath: '/repo' }, mockState())).rejects.toThrow('connection error');
    });

    it('accepts custom config overrides', () => {
      const node = createAnalyzeNode(mockBridge as any, { id: 'custom-analyze', timeout: 60000 });
      expect(node.id).toBe('custom-analyze');
      expect((node as any).timeout).toBe(60000);
    });
  });

  describe('createResearchNode', () => {
    it('calls bridge.extractResearch with correct args', async () => {
      mockBridge.extractResearch.mockResolvedValue({ success: true, data: { specs: [] } });
      const node = createResearchNode(mockBridge as any);
      const state = mockState();

      await node.execute({ paperSource: 'paper.pdf', sourceType: 'pdf' }, state);
      expect(mockBridge.extractResearch).toHaveBeenCalledWith('paper.pdf', 'pdf');
      expect(state.context.researchSpec).toEqual({ specs: [] });
    });
  });

  describe('createMappingNode', () => {
    it('uses context repoAnalysis and researchSpec', async () => {
      mockBridge.mapArchitecture.mockResolvedValue({ success: true, data: { map: {} } });
      const node = createMappingNode(mockBridge as any);
      const state = mockState();
      state.context.repoAnalysis = { files: ['a.py'] };
      state.context.researchSpec = { improvements: [] };

      await node.execute({}, state);
      expect(mockBridge.mapArchitecture).toHaveBeenCalledWith(
        { files: ['a.py'] },
        { improvements: [] },
      );
    });

    it('throws when repoAnalysis is missing', async () => {
      const node = createMappingNode(mockBridge as any);
      const state = mockState();

      await expect(node.execute({}, state)).rejects.toThrow('Missing repo analysis');
    });
  });

  describe('createPatchNode', () => {
    it('calls bridge.generatePatch', async () => {
      mockBridge.generatePatch.mockResolvedValue({ success: true, data: { branchName: 'integration/test' } });
      const node = createPatchNode(mockBridge as any);
      const state = mockState();
      state.context.mapping = { targets: [] };

      await node.execute({ repoPath: '/repo' }, state);
      expect(mockBridge.generatePatch).toHaveBeenCalledWith({ targets: [] }, '/repo');
    });

    it('throws when mapping is missing', async () => {
      const node = createPatchNode(mockBridge as any);
      const state = mockState();

      await expect(node.execute({ repoPath: '/repo' }, state)).rejects.toThrow('Missing mapping');
    });
  });

  describe('createValidationNode', () => {
    it('calls bridge.validate', async () => {
      mockBridge.validate.mockResolvedValue({ success: true, data: { passed: true } });
      const node = createValidationNode(mockBridge as any);
      const state = mockState();
      state.context.patch = { branchName: 'integration/test' };

      await node.execute({ repoPath: '/repo' }, state);
      expect(mockBridge.validate).toHaveBeenCalledWith({ branchName: 'integration/test' }, '/repo');
    });

    it('throws when patch is missing', async () => {
      const node = createValidationNode(mockBridge as any);
      const state = mockState();

      await expect(node.execute({ repoPath: '/repo' }, state)).rejects.toThrow('Missing patch');
    });
  });

  describe('createReportNode', () => {
    it('assembles a report from all context phases', async () => {
      const node = createReportNode();
      const state = mockState();
      state.context.repoAnalysis = { files: [] };
      state.context.researchSpec = { specs: [] };
      state.context.mapping = { targets: [] };
      state.context.patch = { branchName: 'integration/test' };
      state.context.validation = { passed: true };

      const result = await node.execute({}, state);
      expect(result).toHaveProperty('repoAnalysis');
      expect(result).toHaveProperty('researchSpec');
      expect(result).toHaveProperty('mapping');
      expect(result).toHaveProperty('patch');
      expect(result).toHaveProperty('validation');
      expect(result).toHaveProperty('completedAt');
    });
  });

  describe('createPlannerNode', () => {
    it('makes HTTP request and stores result', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ plan: 'test' }),
      });
      vi.stubGlobal('fetch', mockFetch);

      const node = createPlannerNode(mockBridge as any);
      const state = mockState();

      await node.execute({ repoPath: '/repo', maxSpecs: 3 }, state);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/planner/run'),
        expect.objectContaining({ method: 'POST' }),
      );
      expect(state.context.plannerResult).toEqual({ plan: 'test' });

      vi.unstubAllGlobals();
    });

    it('defaults maxSpecs to 5', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });
      vi.stubGlobal('fetch', mockFetch);

      const node = createPlannerNode(mockBridge as any);
      const state = mockState();
      await node.execute({ repoPath: '/repo' }, state);

      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.maxSpecs).toBe(5);

      vi.unstubAllGlobals();
    });
  });

  describe('createCriticNode', () => {
    it('makes HTTP request and stores result', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ payload: { issue_count: 0 } }),
      });
      vi.stubGlobal('fetch', mockFetch);

      const node = createCriticNode(mockBridge as any);
      const state = mockState();
      state.context.patch = { branchName: 'integration/test' };

      await node.execute({ repoPath: '/repo' }, state);
      expect(state.context.criticResult).toBeDefined();

      vi.unstubAllGlobals();
    });

    it('extracts issues when present', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ payload: { issue_count: 2, issues: ['bug1', 'bug2'] } }),
      });
      vi.stubGlobal('fetch', mockFetch);

      const node = createCriticNode(mockBridge as any);
      const state = mockState();
      state.context.patch = { branchName: 'integration/test' };

      await node.execute({ repoPath: '/repo' }, state);
      expect(state.context.criticIssues).toEqual(['bug1', 'bug2']);

      vi.unstubAllGlobals();
    });

    it('throws on HTTP error', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: false,
        statusText: 'Server Error',
      });
      vi.stubGlobal('fetch', mockFetch);

      const node = createCriticNode(mockBridge as any);
      const state = mockState();
      state.context.patch = {};

      await expect(node.execute({ repoPath: '/repo' }, state)).rejects.toThrow('Critic failed');

      vi.unstubAllGlobals();
    });
  });

  describe('createExperimentNode', () => {
    it('makes HTTP request with correct params', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ results: [] }),
      });
      vi.stubGlobal('fetch', mockFetch);

      const node = createExperimentNode(mockBridge as any);
      const state = mockState();

      await node.execute({ repoPath: '/repo', spec: 'test-spec', variants: 2 }, state);
      const body = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(body.spec).toBe('test-spec');
      expect(body.variants).toBe(2);

      vi.unstubAllGlobals();
    });
  });
});

describe('createIntegrationWorkflow', () => {
  const mockBridge = {} as any;

  it('creates a standard 6-phase workflow', () => {
    const nodes = createIntegrationWorkflow(mockBridge);
    const nodeIds = nodes.map(n => n.id);
    expect(nodeIds).toContain('phase1-analyze');
    expect(nodeIds).toContain('phase2-research');
    expect(nodeIds).toContain('phase3-mapping');
    expect(nodeIds).toContain('phase4-patch');
    expect(nodeIds).toContain('phase5-validation');
    expect(nodeIds).toContain('phase6-report');
  });

  it('includes critic node when includeCritic is true', () => {
    const nodes = createIntegrationWorkflow(mockBridge, { includeCritic: true });
    const nodeIds = nodes.map(n => n.id);
    expect(nodeIds).toContain('critic');
  });

  it('does not include planner by default', () => {
    const nodes = createIntegrationWorkflow(mockBridge);
    const nodeIds = nodes.map(n => n.id);
    expect(nodeIds).not.toContain('planner');
  });
});

describe('createExperimentWorkflow', () => {
  const mockBridge = {} as any;

  it('creates workflow with experiment node', () => {
    const nodes = createExperimentWorkflow(mockBridge);
    expect(nodes).toHaveLength(1);
    expect(nodes[0].id).toBe('experiment');
  });

  it('includes critic when requested', () => {
    const nodes = createExperimentWorkflow(mockBridge, { includeCritic: true });
    expect(nodes).toHaveLength(2);
    expect(nodes[1].id).toBe('critic');
  });
});
