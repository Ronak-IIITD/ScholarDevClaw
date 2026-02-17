export type NodeStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';

export type NodeCondition = 'on_success' | 'on_failure' | 'always';

export interface NodeResult {
  nodeId: string;
  status: NodeStatus;
  output?: unknown;
  error?: string;
  duration?: number;
  startedAt?: string;
  completedAt?: string;
}

export interface WorkflowEvent {
  type: 'node_started' | 'node_completed' | 'node_failed' | 'workflow_started' | 'workflow_completed';
  nodeId?: string;
  status?: NodeStatus;
  output?: unknown;
  error?: string;
  timestamp: string;
  progress?: number;
}

export interface WorkflowState {
  workflowId: string;
  status: NodeStatus;
  startedAt?: string;
  completedAt?: string;
  nodeResults: Map<string, NodeResult>;
  context: Record<string, unknown>;
  error?: string;
}

export interface WorkflowConfig {
  workflowId: string;
  maxParallelism?: number;
  timeout?: number;
  retryFailed?: boolean;
  retryCount?: number;
}