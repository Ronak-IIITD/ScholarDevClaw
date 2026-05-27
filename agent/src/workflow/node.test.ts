import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  WorkflowNode,
  FunctionNode,
  ParallelNode,
  ConditionalNode,
} from './node.js';
import type { WorkflowState } from './types.js';

describe('WorkflowNode', () => {
  it('cannot be instantiated directly (abstract)', () => {
    expect(WorkflowNode).toHaveProperty('prototype');
    // Verify it's abstract by checking execute is not implemented
    expect(WorkflowNode.prototype.execute).toBeUndefined();
  });

  describe('constructor defaults', () => {
    it('sets default values for optional config', () => {
      class TestNode extends WorkflowNode {
        async execute() { return null; }
      }
      const node = new TestNode({ id: 'test', name: 'Test' });

      expect(node.id).toBe('test');
      expect(node.name).toBe('Test');
      expect(node.description).toBe('');
      expect(node.dependencies).toEqual([]);
      expect(node.condition).toBe('on_success');
      expect(node.timeout).toBe(300000);
      expect(node.retryCount).toBe(0);
      expect(node.parallel).toBe(true);
      expect(node.status).toBe('pending');
      expect(node.output).toBeNull();
      expect(node.error).toBeNull();
      expect(node.startedAt).toBeNull();
      expect(node.completedAt).toBeNull();
    });

    it('overrides default values with provided config', () => {
      class TestNode extends WorkflowNode {
        async execute() { return null; }
      }
      const node = new TestNode({
        id: 'custom',
        name: 'Custom',
        description: 'A custom node',
        dependencies: ['dep1'],
        condition: 'on_failure',
        timeout: 5000,
        retryCount: 3,
        parallel: false,
      });

      expect(node.id).toBe('custom');
      expect(node.description).toBe('A custom node');
      expect(node.dependencies).toEqual(['dep1']);
      expect(node.condition).toBe('on_failure');
      expect(node.timeout).toBe(5000);
      expect(node.retryCount).toBe(3);
      expect(node.parallel).toBe(false);
    });
  });

  describe('canExecute', () => {
    it('returns true when no dependencies', () => {
      class TestNode extends WorkflowNode {
        async execute() { return null; }
      }
      const node = new TestNode({ id: 'test', name: 'Test' });
      const state = { nodeResults: new Map() } as unknown as WorkflowState;

      expect(node.canExecute(state)).toBe(true);
    });

    it('returns true when all dependencies completed', () => {
      class TestNode extends WorkflowNode {
        async execute() { return null; }
      }
      const node = new TestNode({ id: 'test', name: 'Test', dependencies: ['dep1'] });
      const state = {
        nodeResults: new Map([
          ['dep1', { nodeId: 'dep1', status: 'completed', output: 'ok' }],
        ]),
      } as unknown as WorkflowState;

      expect(node.canExecute(state)).toBe(true);
    });

    it('returns false when a dependency has not completed', () => {
      class TestNode extends WorkflowNode {
        async execute() { return null; }
      }
      const node = new TestNode({ id: 'test', name: 'Test', dependencies: ['dep1'] });
      const state = {
        nodeResults: new Map(),
      } as unknown as WorkflowState;

      expect(node.canExecute(state)).toBe(false);
    });

    it('returns true with condition=always even if dependency failed', () => {
      class TestNode extends WorkflowNode {
        async execute() { return null; }
      }
      const node = new TestNode({ id: 'test', name: 'Test', dependencies: ['dep1'], condition: 'always' });
      const state = {
        nodeResults: new Map([
          ['dep1', { nodeId: 'dep1', status: 'failed', output: null }],
        ]),
      } as unknown as WorkflowState;

      expect(node.canExecute(state)).toBe(true);
    });

    it('returns true with condition=on_failure and dependency failed', () => {
      class TestNode extends WorkflowNode {
        async execute() { return null; }
      }
      const node = new TestNode({ id: 'test', name: 'Test', dependencies: ['dep1'], condition: 'on_failure' });
      const state = {
        nodeResults: new Map([
          ['dep1', { nodeId: 'dep1', status: 'failed', output: null }],
        ]),
      } as unknown as WorkflowState;

      expect(node.canExecute(state)).toBe(true);
    });
  });

  describe('shouldSkip', () => {
    it('returns false when condition is always', () => {
      class TestNode extends WorkflowNode {
        async execute() { return null; }
      }
      const node = new TestNode({ id: 'test', name: 'Test', dependencies: ['dep1'], condition: 'always' });
      const state = {
        nodeResults: new Map([
          ['dep1', { nodeId: 'dep1', status: 'failed', output: null }],
        ]),
      } as unknown as WorkflowState;

      expect(node.shouldSkip(state)).toBe(false);
    });

    it('returns true when condition=on_success and dependency failed', () => {
      class TestNode extends WorkflowNode {
        async execute() { return null; }
      }
      const node = new TestNode({ id: 'test', name: 'Test', dependencies: ['dep1'], condition: 'on_success' });
      const state = {
        nodeResults: new Map([
          ['dep1', { nodeId: 'dep1', status: 'failed', output: null }],
        ]),
      } as unknown as WorkflowState;

      expect(node.shouldSkip(state)).toBe(true);
    });

    it('returns false when condition=on_success and dependency completed', () => {
      class TestNode extends WorkflowNode {
        async execute() { return null; }
      }
      const node = new TestNode({ id: 'test', name: 'Test', dependencies: ['dep1'], condition: 'on_success' });
      const state = {
        nodeResults: new Map([
          ['dep1', { nodeId: 'dep1', status: 'completed', output: 'ok' }],
        ]),
      } as unknown as WorkflowState;

      expect(node.shouldSkip(state)).toBe(false);
    });

    it('returns true when condition=on_failure and dependency succeeded', () => {
      class TestNode extends WorkflowNode {
        async execute() { return null; }
      }
      const node = new TestNode({ id: 'test', name: 'Test', dependencies: ['dep1'], condition: 'on_failure' });
      const state = {
        nodeResults: new Map([
          ['dep1', { nodeId: 'dep1', status: 'completed', output: 'ok' }],
        ]),
      } as unknown as WorkflowState;

      expect(node.shouldSkip(state)).toBe(true);
    });
  });

  describe('toResult', () => {
    it('returns NodeResult with node metadata', () => {
      class TestNode extends WorkflowNode {
        async execute() { return null; }
      }
      const node = new TestNode({ id: 'my-node', name: 'My Node' });
      node.status = 'completed';
      node.output = 'done';
      node.startedAt = '2026-01-01T00:00:00.000Z';
      node.completedAt = '2026-01-01T00:01:00.000Z';

      const result = node.toResult();
      expect(result.nodeId).toBe('my-node');
      expect(result.status).toBe('completed');
      expect(result.output).toBe('done');
      expect(result.duration).toBe(60000);
    });

    it('returns undefined duration when dates are missing', () => {
      class TestNode extends WorkflowNode {
        async execute() { return null; }
      }
      const node = new TestNode({ id: 'test', name: 'Test' });
      const result = node.toResult();
      expect(result.duration).toBeUndefined();
    });
  });
});

describe('FunctionNode', () => {
  it('executes the provided function', async () => {
    const fn = vi.fn().mockResolvedValue('result');
    const node = new FunctionNode({ id: 'fn', name: 'Fn' }, fn);

    const output = await node.execute({}, { workflowId: 'w1', status: 'running', context: {}, nodeResults: new Map() });
    expect(output).toBe('result');
    expect(fn).toHaveBeenCalled();
  });

  it('passes context and state to the executor', async () => {
    const fn = vi.fn().mockResolvedValue('ok');
    const node = new FunctionNode({ id: 'fn', name: 'Fn' }, fn);
    const context = { key: 'val' };
    const state = { workflowId: 'w1', status: 'running', context: {}, nodeResults: new Map() };

    await node.execute(context, state);
    expect(fn).toHaveBeenCalledWith(context, state);
  });
});

describe('ParallelNode', () => {
  it('executes all child nodes concurrently', async () => {
    const fn1 = vi.fn().mockResolvedValue('a');
    const fn2 = vi.fn().mockResolvedValue('b');

    const child1 = new FunctionNode({ id: 'c1', name: 'C1' }, fn1);
    const child2 = new FunctionNode({ id: 'c2', name: 'C2' }, fn2);

    const node = new ParallelNode({ id: 'parallel', name: 'Parallel' }, [child1, child2]);
    const state = { workflowId: 'w1', status: 'running', context: {}, nodeResults: new Map() };

    const result = await node.execute({}, state);
    expect(result).toEqual(['a', 'b']);
    expect(fn1).toHaveBeenCalled();
    expect(fn2).toHaveBeenCalled();
  });

  it('handles empty children array', async () => {
    const node = new ParallelNode({ id: 'empty', name: 'Empty' }, []);
    const state = { workflowId: 'w1', status: 'running', context: {}, nodeResults: new Map() };

    const result = await node.execute({}, state);
    expect(result).toEqual([]);
  });
});

describe('ConditionalNode', () => {
  it('executes trueNode when condition is true', async () => {
    const trueFn = vi.fn().mockResolvedValue('true-branch');
    const falseFn = vi.fn().mockResolvedValue('false-branch');

    const trueNode = new FunctionNode({ id: 't', name: 'True' }, trueFn);
    const falseNode = new FunctionNode({ id: 'f', name: 'False' }, falseFn);

    const node = new ConditionalNode({ id: 'cond', name: 'Cond' }, () => true, trueNode, falseNode);
    const state = { workflowId: 'w1', status: 'running', context: {}, nodeResults: new Map() };

    const result = await node.execute({}, state);
    expect(result).toBe('true-branch');
    expect(trueFn).toHaveBeenCalled();
    expect(falseFn).not.toHaveBeenCalled();
  });

  it('executes falseNode when condition is false with falseNode', async () => {
    const trueFn = vi.fn().mockResolvedValue('true-branch');
    const falseFn = vi.fn().mockResolvedValue('false-branch');

    const trueNode = new FunctionNode({ id: 't', name: 'True' }, trueFn);
    const falseNode = new FunctionNode({ id: 'f', name: 'False' }, falseFn);

    const node = new ConditionalNode({ id: 'cond', name: 'Cond' }, () => false, trueNode, falseNode);
    const state = { workflowId: 'w1', status: 'running', context: {}, nodeResults: new Map() };

    const result = await node.execute({}, state);
    expect(result).toBe('false-branch');
    expect(trueFn).not.toHaveBeenCalled();
    expect(falseFn).toHaveBeenCalled();
  });

  it('returns null when condition is false and no falseNode', async () => {
    const trueFn = vi.fn().mockResolvedValue('true-branch');

    const trueNode = new FunctionNode({ id: 't', name: 'True' }, trueFn);

    const node = new ConditionalNode({ id: 'cond', name: 'Cond' }, () => false, trueNode, null);
    const state = { workflowId: 'w1', status: 'running', context: {}, nodeResults: new Map() };

    const result = await node.execute({}, state);
    expect(result).toBeNull();
    expect(trueFn).not.toHaveBeenCalled();
  });

  it('passes context and state to condition function', async () => {
    const conditionFn = vi.fn().mockReturnValue(true);
    const trueFn = vi.fn().mockResolvedValue('ok');
    const trueNode = new FunctionNode({ id: 't', name: 'True' }, trueFn);

    const node = new ConditionalNode({ id: 'cond', name: 'Cond' }, conditionFn, trueNode);
    const context = { data: 42 };
    const state = { workflowId: 'w1', status: 'running', context: {}, nodeResults: new Map() };

    await node.execute(context, state);
    expect(conditionFn).toHaveBeenCalledWith(context, state);
  });
});
