import { WorkflowEvent, WorkflowState, NodeResult } from './types.js';
import { logger } from '../utils/logger.js';

export interface WorkflowMetrics {
  workflowId: string;
  startedAt: string;
  completedAt?: string;
  duration?: number;
  status: string;
  nodeMetrics: Map<string, NodeMetrics>;
  totalNodes: number;
  completedNodes: number;
  failedNodes: number;
  skippedNodes: number;
}

export interface NodeMetrics {
  nodeId: string;
  name: string;
  status: string;
  startedAt?: string;
  completedAt?: string;
  duration?: number;
  retryCount: number;
  error?: string;
}

export interface WorkflowMonitor {
  onWorkflowStart(workflowId: string): void;
  onWorkflowComplete(workflowId: string, status: string, duration: number): void;
  onNodeStart(nodeId: string, name: string): void;
  onNodeComplete(nodeId: string, status: string, duration: number, error?: string): void;
  getMetrics(workflowId: string): WorkflowMetrics | null;
  getAllMetrics(): WorkflowMetrics[];
  getActiveWorkflows(): string[];
}

export class MetricsCollector implements WorkflowMonitor {
  private metrics: Map<string, WorkflowMetrics> = new Map();
  private activeWorkflows: Set<string> = new Set();

  onWorkflowStart(workflowId: string): void {
    this.activeWorkflows.add(workflowId);
    this.metrics.set(workflowId, {
      workflowId,
      startedAt: new Date().toISOString(),
      status: 'running',
      nodeMetrics: new Map(),
      totalNodes: 0,
      completedNodes: 0,
      failedNodes: 0,
      skippedNodes: 0,
    });
    logger.info(`[METRICS] Workflow started: ${workflowId}`);
  }

  onWorkflowComplete(workflowId: string, status: string, duration: number): void {
    const metrics = this.metrics.get(workflowId);
    if (metrics) {
      metrics.status = status;
      metrics.completedAt = new Date().toISOString();
      metrics.duration = duration;
    }
    this.activeWorkflows.delete(workflowId);
    logger.info(`[METRICS] Workflow completed: ${workflowId} in ${duration}ms - ${status}`);
  }

  onNodeStart(nodeId: string, name: string): void {
    for (const metrics of this.metrics.values()) {
      if (this.activeWorkflows.has(metrics.workflowId)) {
        metrics.totalNodes++;
        metrics.nodeMetrics.set(nodeId, {
          nodeId,
          name,
          status: 'running',
          startedAt: new Date().toISOString(),
          retryCount: 0,
        });
        break;
      }
    }
    logger.debug(`[METRICS] Node started: ${nodeId}`);
  }

  onNodeComplete(nodeId: string, status: string, duration: number, error?: string): void {
    for (const metrics of this.metrics.values()) {
      const nodeMetrics = metrics.nodeMetrics.get(nodeId);
      if (nodeMetrics) {
        nodeMetrics.status = status;
        nodeMetrics.completedAt = new Date().toISOString();
        nodeMetrics.duration = duration;
        nodeMetrics.error = error;

        if (status === 'completed') metrics.completedNodes++;
        else if (status === 'failed') metrics.failedNodes++;
        else if (status === 'skipped') metrics.skippedNodes++;
        break;
      }
    }
    logger.debug(`[METRICS] Node completed: ${nodeId} in ${duration}ms - ${status}`);
  }

  getMetrics(workflowId: string): WorkflowMetrics | null {
    return this.metrics.get(workflowId) || null;
  }

  getAllMetrics(): WorkflowMetrics[] {
    return Array.from(this.metrics.values());
  }

  getActiveWorkflows(): string[] {
    return Array.from(this.activeWorkflows);
  }

  getSummary(): {
    totalWorkflows: number;
    activeWorkflows: number;
    completedWorkflows: number;
    failedWorkflows: number;
    averageDuration: number;
    successRate: number;
  } {
    const all = this.getAllMetrics();
    const completed = all.filter(m => m.status === 'completed');
    const failed = all.filter(m => m.status === 'failed');
    const durations = completed.filter(m => m.duration).map(m => m.duration!);
    const avgDuration = durations.length > 0 
      ? durations.reduce((a, b) => a + b, 0) / durations.length 
      : 0;

    return {
      totalWorkflows: all.length,
      activeWorkflows: this.getActiveWorkflows().length,
      completedWorkflows: completed.length,
      failedWorkflows: failed.length,
      averageDuration: avgDuration,
      successRate: all.length > 0 ? (completed.length / all.length) * 100 : 0,
    };
  }

  clear(): void {
    this.metrics.clear();
    this.activeWorkflows.clear();
  }
}

export class EventDrivenMonitor {
  private collector: MetricsCollector;

  constructor(collector: MetricsCollector) {
    this.collector = collector;
  }

  attachToWorkflow(workflow: { onEvent: (callback: (event: WorkflowEvent) => void) => void }): void {
    workflow.onEvent((event) => {
      if (event.type === 'workflow_started') {
        this.collector.onWorkflowStart(event.timestamp);
      } else if (event.type === 'workflow_completed') {
        const metrics = this.collector.getMetrics(event.timestamp);
        const duration = metrics?.startedAt 
          ? new Date(event.timestamp).getTime() - new Date(metrics.startedAt).getTime()
          : 0;
        this.collector.onWorkflowComplete(event.timestamp, event.status || 'completed', duration);
      } else if (event.type === 'node_started') {
        this.collector.onNodeStart(event.nodeId || '', event.status || 'running');
      } else if (event.type === 'node_completed' || event.type === 'node_failed') {
        const metrics = this.collector.getMetrics(event.timestamp);
        let duration = 0;
        if (metrics) {
          const nodeMetrics = metrics.nodeMetrics.get(event.nodeId || '');
          if (nodeMetrics?.startedAt) {
            duration = new Date(event.timestamp).getTime() - new Date(nodeMetrics.startedAt).getTime();
          }
        }
        this.collector.onNodeComplete(
          event.nodeId || '', 
          event.status === 'failed' ? 'failed' : 'completed',
          duration,
          event.error
        );
      }
    });
  }
}
