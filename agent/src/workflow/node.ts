import type { NodeStatus, NodeCondition, NodeResult, WorkflowEvent, WorkflowState } from './types.js';

export type NodeExecutor = (context: Record<string, unknown>, state: WorkflowState) => Promise<unknown>;

export interface NodeConfig {
  id: string;
  name: string;
  description?: string;
  dependencies?: string[];
  condition?: NodeCondition;
  timeout?: number;
  retryCount?: number;
  parallel?: boolean;
}

export abstract class WorkflowNode {
  id: string;
  name: string;
  description: string;
  dependencies: string[];
  condition: NodeCondition;
  timeout: number;
  retryCount: number;
  parallel: boolean;
  status: NodeStatus = 'pending';
  output: unknown = null;
  error: string | null = null;
  startedAt: string | null = null;
  completedAt: string | null = null;

  constructor(config: NodeConfig) {
    this.id = config.id;
    this.name = config.name;
    this.description = config.description || '';
    this.dependencies = config.dependencies || [];
    this.condition = config.condition || 'on_success';
    this.timeout = config.timeout || 300000;
    this.retryCount = config.retryCount || 0;
    this.parallel = config.parallel ?? true;
  }

  abstract execute(context: Record<string, unknown>, state: WorkflowState): Promise<unknown>;

  canExecute(state: WorkflowState): boolean {
    if (this.dependencies.length === 0) {
      return true;
    }

    for (const depId of this.dependencies) {
      const depResult = state.nodeResults.get(depId);
      if (!depResult) return false;
      if (depResult.status !== 'completed') {
        if (this.condition === 'always') continue;
        if (this.condition === 'on_failure' && depResult.status === 'failed') continue;
        return false;
      }
    }
    return true;
  }

  shouldSkip(state: WorkflowState): boolean {
    if (this.condition === 'always') return false;

    for (const depId of this.dependencies) {
      const depResult = state.nodeResults.get(depId);
      if (!depResult) continue;

      if (this.condition === 'on_success' && depResult.status === 'failed') {
        return true;
      }
      if (this.condition === 'on_failure' && depResult.status === 'completed') {
        return true;
      }
    }
    return false;
  }

  toResult(): NodeResult {
    return {
      nodeId: this.id,
      status: this.status,
      output: this.output,
      error: this.error || undefined,
      duration: this.startedAt && this.completedAt
        ? new Date(this.completedAt).getTime() - new Date(this.startedAt).getTime()
        : undefined,
      startedAt: this.startedAt || undefined,
      completedAt: this.completedAt || undefined,
    };
  }
}

export class FunctionNode extends WorkflowNode {
  private executor: NodeExecutor;

  constructor(config: NodeConfig, executor: NodeExecutor) {
    super(config);
    this.executor = executor;
  }

  async execute(context: Record<string, unknown>, state: WorkflowState): Promise<unknown> {
    return this.executor(context, state);
  }
}

export class ParallelNode extends WorkflowNode {
  private nodes: WorkflowNode[];

  constructor(config: NodeConfig, nodes: WorkflowNode[]) {
    super(config);
    this.nodes = nodes;
  }

  async execute(context: Record<string, unknown>, state: WorkflowState): Promise<unknown> {
    const results = await Promise.all(
      this.nodes.map(node => node.execute(context, state))
    );
    return results;
  }
}

export class ConditionalNode extends WorkflowNode {
  private conditionFn: (context: Record<string, unknown>, state: WorkflowState) => boolean;
  private trueNode: WorkflowNode;
  private falseNode: WorkflowNode | null;

  constructor(
    config: NodeConfig,
    conditionFn: (context: Record<string, unknown>, state: WorkflowState) => boolean,
    trueNode: WorkflowNode,
    falseNode: WorkflowNode | null = null
  ) {
    super(config);
    this.conditionFn = conditionFn;
    this.trueNode = trueNode;
    this.falseNode = falseNode;
  }

  async execute(context: Record<string, unknown>, state: WorkflowState): Promise<unknown> {
    if (this.conditionFn(context, state)) {
      return this.trueNode.execute(context, state);
    } else if (this.falseNode) {
      return this.falseNode.execute(context, state);
    }
    return null;
  }
}