import type { WorkflowConfig, WorkflowState, WorkflowEvent, NodeResult } from './types.js';
import { WorkflowNode } from './node.js';
import { logger } from '../utils/logger.js';
import { v4 as uuidv4 } from 'uuid';

export type EventCallback = (event: WorkflowEvent) => void;

export class DAGEngine {
  private nodes: Map<string, WorkflowNode> = new Map();
  private config: WorkflowConfig;
  private state: WorkflowState;
  private eventCallbacks: EventCallback[] = [];
  private abortController: AbortController | null = null;

  constructor(config: WorkflowConfig) {
    this.config = {
      maxParallelism: 4,
      timeout: 600000,
      retryFailed: false,
      retryCount: 0,
      ...config,
    };

    this.state = {
      workflowId: config.workflowId,
      status: 'pending',
      nodeResults: new Map(),
      context: {},
    };
  }

  addNode(node: WorkflowNode): void {
    this.nodes.set(node.id, node);
  }

  addNodes(nodes: WorkflowNode[]): void {
    for (const node of nodes) {
      this.addNode(node);
    }
  }

  onEvent(callback: EventCallback): void {
    this.eventCallbacks.push(callback);
  }

  private emit(event: WorkflowEvent): void {
    for (const callback of this.eventCallbacks) {
      callback(event);
    }
  }

  private getReadyNodes(): WorkflowNode[] {
    const ready: WorkflowNode[] = [];

    for (const node of this.nodes.values()) {
      if (node.status !== 'pending') continue;
      if (node.shouldSkip(this.state)) {
        node.status = 'skipped';
        this.state.nodeResults.set(node.id, node.toResult());
        continue;
      }
      if (node.canExecute(this.state)) {
        ready.push(node);
      }
    }

    return ready;
  }

  private async executeNode(node: WorkflowNode): Promise<NodeResult> {
    node.status = 'running';
    node.startedAt = new Date().toISOString();

    this.emit({
      type: 'node_started',
      nodeId: node.id,
      status: 'running',
      timestamp: node.startedAt,
    });

    let attempts = 0;
    const maxAttempts = Math.max(1, node.retryCount + 1);

    while (attempts < maxAttempts) {
      try {
        const timeoutPromise = new Promise<never>((_, reject) => {
          setTimeout(() => reject(new Error('Node timeout')), node.timeout);
        });

        const output = await Promise.race([
          node.execute(this.state.context, this.state),
          timeoutPromise,
        ]);

        node.output = output;
        node.status = 'completed';
        node.completedAt = new Date().toISOString();

        const result = node.toResult();
        this.state.nodeResults.set(node.id, result);

        this.emit({
          type: 'node_completed',
          nodeId: node.id,
          status: 'completed',
          output,
          timestamp: node.completedAt,
          progress: this.calculateProgress(),
        });

        return result;
      } catch (error) {
        attempts++;
        node.error = error instanceof Error ? error.message : String(error);

        if (attempts >= maxAttempts) {
          node.status = 'failed';
          node.completedAt = new Date().toISOString();

          const result = node.toResult();
          this.state.nodeResults.set(node.id, result);

          this.emit({
            type: 'node_failed',
            nodeId: node.id,
            status: 'failed',
            error: node.error,
            timestamp: node.completedAt,
          });

          return result;
        }

        logger.warn(`Node ${node.id} failed, retrying (${attempts}/${maxAttempts})`);
        await new Promise(resolve => setTimeout(resolve, 1000 * attempts));
      }
    }

    return node.toResult();
  }

  private calculateProgress(): number {
    const total = this.nodes.size;
    const completed = Array.from(this.nodes.values()).filter(
      n => n.status === 'completed' || n.status === 'failed' || n.status === 'skipped'
    ).length;
    return total > 0 ? Math.round((completed / total) * 100) : 0;
  }

  async execute(initialContext: Record<string, unknown> = {}): Promise<WorkflowState> {
    this.abortController = new AbortController();
    this.state.context = { ...initialContext };
    this.state.status = 'running';
    this.state.startedAt = new Date().toISOString();

    this.emit({
      type: 'workflow_started',
      timestamp: this.state.startedAt,
    });

    try {
      while (true) {
        if (this.abortController.signal.aborted) {
          this.state.status = 'failed';
          this.state.error = 'Workflow aborted';
          break;
        }

        const readyNodes = this.getReadyNodes();
        
        if (readyNodes.length === 0) {
          const allDone = Array.from(this.nodes.values()).every(
            n => n.status !== 'pending' && n.status !== 'running'
          );
          if (allDone) break;

          await new Promise(resolve => setTimeout(resolve, 100));
          continue;
        }

        const parallelNodes = readyNodes.filter(n => n.parallel);
        const sequentialNodes = readyNodes.filter(n => !n.parallel);

        if (parallelNodes.length > 0) {
          const batch = parallelNodes.slice(0, this.config.maxParallelism || 4);
          await Promise.all(batch.map(node => this.executeNode(node)));
        }

        for (const node of sequentialNodes) {
          await this.executeNode(node);
        }
      }

      const hasFailed = Array.from(this.nodes.values()).some(n => n.status === 'failed');
      this.state.status = hasFailed ? 'failed' : 'completed';
    } catch (error) {
      this.state.status = 'failed';
      this.state.error = error instanceof Error ? error.message : String(error);
    }

    this.state.completedAt = new Date().toISOString();

    this.emit({
      type: 'workflow_completed',
      timestamp: this.state.completedAt,
      status: this.state.status,
    });

    return this.state;
  }

  abort(): void {
    this.abortController?.abort();
  }

  getState(): WorkflowState {
    return this.state;
  }

  getResults(): Map<string, NodeResult> {
    return this.state.nodeResults;
  }

  static topologicalSort(nodes: WorkflowNode[]): string[] {
    const visited = new Set<string>();
    const result: string[] = [];
    const temp = new Set<string>();

    function visit(nodeId: string): void {
      if (temp.has(nodeId)) {
        throw new Error(`Cycle detected in DAG at node: ${nodeId}`);
      }
      if (visited.has(nodeId)) return;

      temp.add(nodeId);
      const node = nodes.find(n => n.id === nodeId);
      if (node) {
        for (const dep of node.dependencies) {
          visit(dep);
        }
      }
      temp.delete(nodeId);
      visited.add(nodeId);
      result.push(nodeId);
    }

    for (const node of nodes) {
      visit(node.id);
    }

    return result;
  }
}