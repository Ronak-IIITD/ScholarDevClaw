import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { DynamicWorkflowBuilder, createQuickWorkflow } from './builder.js';
import { FunctionNode, ConditionalNode } from './node.js';

describe('DynamicWorkflowBuilder', () => {
  const mockBridge = { analyzeRepo: vi.fn(), extractResearch: vi.fn() };
  let builder: DynamicWorkflowBuilder;

  beforeEach(() => {
    vi.clearAllMocks();
    builder = new DynamicWorkflowBuilder({ bridge: mockBridge as any });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('starts with empty nodes and default context', () => {
    expect(builder.getContext()).toEqual({});
  });

  describe('addNodeDefinition', () => {
    it('adds a function node definition', () => {
      builder.addNodeDefinition({
        id: 'analyze',
        type: 'function',
        config: { id: 'analyze', name: 'Analyze' },
      });

      const nodes = builder.build();
      expect(nodes).toHaveLength(1);
      expect(nodes[0].id).toBe('analyze');
      expect(nodes[0]).toBeInstanceOf(FunctionNode);
    });

    it('returns self for chaining', () => {
      const result = builder.addNodeDefinition({
        id: 'a', type: 'function', config: { id: 'a', name: 'A' },
      });
      expect(result).toBe(builder);
    });
  });

  describe('addNodeDefinitions', () => {
    it('adds multiple definitions at once', () => {
      builder.addNodeDefinitions([
        { id: 'a', type: 'function', config: { id: 'a', name: 'A' } },
        { id: 'b', type: 'function', config: { id: 'b', name: 'B' } },
      ]);

      expect(builder.build()).toHaveLength(2);
    });
  });

  describe('addEdge', () => {
    it('adds a dependency from one node to another', () => {
      builder.addNodeDefinition({
        id: 'b', type: 'function', config: { id: 'b', name: 'B' },
      });

      builder.addEdge('b', 'a');

      // Rebuild and verify
      const nodes = builder.build();
      const nodeB = nodes.find(n => n.id === 'b');
      expect(nodeB?.dependencies).toContain('a');
    });
  });

  describe('addEdges', () => {
    it('adds multiple edges', () => {
      builder.addNodeDefinitions([
        { id: 'a', type: 'function', config: { id: 'a', name: 'A' } },
        { id: 'b', type: 'function', config: { id: 'b', name: 'B' } },
      ]);

      builder.addEdges([{ from: 'b', to: 'a' }]);
      const nodes = builder.build();
      const nodeB = nodes.find(n => n.id === 'b');
      expect(nodeB?.dependencies).toContain('a');
    });
  });

  describe('build', () => {
    it('returns all created nodes', () => {
      builder.addNodeDefinitions([
        { id: 'a', type: 'function', config: { id: 'a', name: 'A' } },
        { id: 'b', type: 'function', config: { id: 'b', name: 'B' } },
      ]);

      const nodes = builder.build();
      expect(nodes).toHaveLength(2);
      expect(nodes.map(n => n.id)).toEqual(['a', 'b']);
    });

    it('handles empty definitions', () => {
      expect(builder.build()).toEqual([]);
    });

    it('returns null for unknown node types', () => {
      builder.addNodeDefinition({
        id: 'bad',
        type: 'nonexistent' as any,
        config: { id: 'bad', name: 'Bad' },
      });

      // Should just skip unknown types
      const nodes = builder.build();
      // The createNode returns null for unknown types, so it won't be added
      expect(nodes).toHaveLength(0);
    });
  });

  describe('setContext', () => {
    it('sets a context value and returns self', () => {
      const result = builder.setContext('repoPath', '/test');
      expect(result).toBe(builder);
      expect(builder.getContext()).toEqual({ repoPath: '/test' });
    });
  });

  describe('fromDefinition', () => {
    it('builds nodes from a workflow definition', () => {
      const nodes = DynamicWorkflowBuilder.fromDefinition(
        {
          id: 'wf-1',
          name: 'Test Workflow',
          nodes: [
            { id: 'a', type: 'function', config: { id: 'a', name: 'A' } },
            { id: 'b', type: 'function', config: { id: 'b', name: 'B', dependencies: ['a'] } },
          ],
          edges: [{ from: 'b', to: 'a' }],
        },
        { bridge: mockBridge as any },
      );

      expect(nodes).toHaveLength(2);
      const nodeB = nodes.find(n => n.id === 'b');
      expect(nodeB?.dependencies).toContain('a');
    });
  });
});

describe('createQuickWorkflow', () => {
  const mockBridge = { analyzeRepo: vi.fn(), extractResearch: vi.fn(), healthCheck: vi.fn() };

  it('creates nodes for specified steps', () => {
    const nodes = createQuickWorkflow(mockBridge as any, ['analyze', 'research']);
    expect(nodes).toHaveLength(2);
    expect(nodes[0].id).toBe('analyze');
    expect(nodes[1].id).toBe('research');
  });

  it('chains nodes with dependencies', () => {
    const nodes = createQuickWorkflow(mockBridge as any, ['analyze', 'research']);
    expect(nodes[0].dependencies).toContain('research');
  });

  it('returns empty array for empty steps', () => {
    const nodes = createQuickWorkflow(mockBridge as any, []);
    expect(nodes).toEqual([]);
  });

  it('skips unknown steps', () => {
    const nodes = createQuickWorkflow(mockBridge as any, ['analyze', 'unknown-step']);
    expect(nodes).toHaveLength(1);
    expect(nodes[0].id).toBe('analyze');
  });

  it('handles single step', () => {
    const nodes = createQuickWorkflow(mockBridge as any, ['analyze']);
    expect(nodes).toHaveLength(1);
  });
});
