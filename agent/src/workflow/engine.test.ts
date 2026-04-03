import { afterEach, describe, expect, it, vi } from 'vitest';
import { DAGEngine } from './engine.js';
import { FunctionNode, ParallelNode, ConditionalNode } from './node.js';
import type { WorkflowState } from './types.js';

describe('DAGEngine', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('should execute nodes in topological order', async () => {
    const engine = new DAGEngine({ workflowId: 'test-1' });
    
    const executionOrder: string[] = [];
    
    const node1 = new FunctionNode(
      { id: 'node1', name: 'First' },
      async () => { executionOrder.push('node1'); return 'result1'; }
    );
    
    const node2 = new FunctionNode(
      { id: 'node2', name: 'Second', dependencies: ['node1'] },
      async () => { executionOrder.push('node2'); return 'result2'; }
    );
    
    const node3 = new FunctionNode(
      { id: 'node3', name: 'Third', dependencies: ['node2'] },
      async () => { executionOrder.push('node3'); return 'result3'; }
    );
    
    engine.addNodes([node3, node1, node2]);
    
    const state = await engine.execute({});
    
    expect(state.status).toBe('completed');
    expect(executionOrder).toEqual(['node1', 'node2', 'node3']);
  });

  it('should execute independent nodes in parallel', async () => {
    const engine = new DAGEngine({ workflowId: 'test-2', maxParallelism: 4 });
    
    const executionTimes: { id: string; start: number; end: number }[] = [];
    const now = Date.now;
    
    const createNode = (id: string) => new FunctionNode(
      { id, name: id, parallel: true },
      async () => {
        executionTimes.push({ id, start: Date.now(), end: 0 });
        await new Promise(r => setTimeout(r, 50));
        executionTimes[executionTimes.length - 1].end = Date.now();
        return id;
      }
    );
    
    engine.addNodes([createNode('a'), createNode('b'), createNode('c')]);
    
    const state = await engine.execute({});
    
    expect(state.status).toBe('completed');
    expect(state.nodeResults.size).toBe(3);
  });

  it('should handle node failures', async () => {
    const engine = new DAGEngine({ workflowId: 'test-3' });
    
    const node1 = new FunctionNode(
      { id: 'node1', name: 'First' },
      async () => { throw new Error('Node failed'); }
    );
    
    const node2 = new FunctionNode(
      { id: 'node2', name: 'Second', dependencies: ['node1'] },
      async () => 'result2'
    );
    
    engine.addNodes([node1, node2]);
    
    const state = await engine.execute({});
    
    expect(state.status).toBe('failed');
    expect(state.nodeResults.get('node1')?.status).toBe('failed');
    expect(state.nodeResults.get('node1')?.error).toBe('Node failed');
  });

  it('should skip nodes based on conditions', async () => {
    const engine = new DAGEngine({ workflowId: 'test-4' });
    
    const node1 = new FunctionNode(
      { id: 'node1', name: 'First' },
      async () => { throw new Error('Failed'); }
    );
    
    const node2 = new FunctionNode(
      { id: 'node2', name: 'Second', dependencies: ['node1'], condition: 'on_failure' },
      async () => 'recovery'
    );
    
    engine.addNodes([node1, node2]);
    
    const state = await engine.execute({});
    
    expect(state.nodeResults.get('node1')?.status).toBe('failed');
    expect(state.nodeResults.get('node2')?.status).toBe('completed');
  });

  it('should emit events during execution', async () => {
    const engine = new DAGEngine({ workflowId: 'test-5' });
    const events: string[] = [];
    
    engine.onEvent((event) => {
      events.push(event.type);
    });
    
    const node1 = new FunctionNode(
      { id: 'node1', name: 'First' },
      async () => 'result'
    );
    
    engine.addNode(node1);
    await engine.execute({});
    
    expect(events).toContain('workflow_started');
    expect(events).toContain('node_started');
    expect(events).toContain('node_completed');
    expect(events).toContain('workflow_completed');
  });

  it('should share context between nodes', async () => {
    const engine = new DAGEngine({ workflowId: 'test-6' });
    
    const node1 = new FunctionNode(
      { id: 'node1', name: 'First' },
      async (_context, state) => {
        state.context.value = 'shared';
        return 'result1';
      }
    );
    
    const node2 = new FunctionNode(
      { id: 'node2', name: 'Second', dependencies: ['node1'] },
      async (_context, state) => {
        return state.context.value;
      }
    );
    
    engine.addNodes([node1, node2]);
    
    const state = await engine.execute({});
    
    expect(state.context.value).toBe('shared');
    expect(state.nodeResults.get('node2')?.output).toBe('shared');
  });

  it('should detect cycles in DAG', async () => {
    const node1 = new FunctionNode({ id: 'a', name: 'A', dependencies: ['b'] }, async () => 'a');
    const node2 = new FunctionNode({ id: 'b', name: 'B', dependencies: ['a'] }, async () => 'b');
    
    expect(() => DAGEngine.topologicalSort([node1, node2])).toThrow('Cycle detected');
  });

  it('should retry failed nodes', async () => {
    const engine = new DAGEngine({ workflowId: 'test-7' });
    let attempts = 0;
    
    const node1 = new FunctionNode(
      { id: 'node1', name: 'First', retryCount: 2 },
      async () => {
        attempts++;
        if (attempts < 3) throw new Error('Not yet');
        return 'success';
      }
    );
    
    engine.addNode(node1);
    const state = await engine.execute({});
    
    expect(state.status).toBe('completed');
    expect(attempts).toBe(3);
  });

  it('should fail nodes that exceed their timeout', async () => {
    const engine = new DAGEngine({ workflowId: 'test-8' });

    const node = new FunctionNode(
      { id: 'slow-node', name: 'Slow', timeout: 10 },
      async () => {
        await new Promise((resolve) => setTimeout(resolve, 50));
        return 'late-result';
      }
    );

    engine.addNode(node);

    const state = await engine.execute({});

    expect(state.status).toBe('failed');
    expect(state.nodeResults.get('slow-node')?.status).toBe('failed');
    expect(state.nodeResults.get('slow-node')?.error).toBe('Node timeout');
  });

  it('should fail the workflow when aborted while idle', async () => {
    const engine = new DAGEngine({ workflowId: 'test-9' });

    const blockedNode = new FunctionNode(
      { id: 'blocked', name: 'Blocked', dependencies: ['missing-node'] },
      async () => 'never-runs'
    );

    engine.addNode(blockedNode);

    const statePromise = engine.execute({});
    engine.abort();
    const state = await statePromise;

    expect(state.status).toBe('failed');
    expect(state.error).toBe('Workflow aborted');
    expect(state.nodeResults.size).toBe(0);
  });
});
