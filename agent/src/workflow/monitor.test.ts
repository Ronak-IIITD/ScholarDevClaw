import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { MetricsCollector, EventDrivenMonitor } from './monitor.js';

describe('MetricsCollector', () => {
  let collector: MetricsCollector;

  beforeEach(() => {
    collector = new MetricsCollector();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('onWorkflowStart', () => {
    it('creates metrics entry and adds workflow to active set', () => {
      collector.onWorkflowStart('wf-1');
      const metrics = collector.getMetrics('wf-1');
      expect(metrics).not.toBeNull();
      expect(metrics!.workflowId).toBe('wf-1');
      expect(metrics!.status).toBe('running');
      expect(metrics!.startedAt).toBeDefined();
      expect(collector.getActiveWorkflows()).toContain('wf-1');
    });

    it('initializes all counters to zero', () => {
      collector.onWorkflowStart('wf-1');
      const metrics = collector.getMetrics('wf-1')!;
      expect(metrics.totalNodes).toBe(0);
      expect(metrics.completedNodes).toBe(0);
      expect(metrics.failedNodes).toBe(0);
      expect(metrics.skippedNodes).toBe(0);
    });
  });

  describe('onWorkflowComplete', () => {
    it('sets status, completedAt, duration, and removes from active', () => {
      vi.useFakeTimers();
      collector.onWorkflowStart('wf-1');

      vi.advanceTimersByTime(5000);
      collector.onWorkflowComplete('wf-1', 'completed', 5000);

      const metrics = collector.getMetrics('wf-1')!;
      expect(metrics.status).toBe('completed');
      expect(metrics.completedAt).toBeDefined();
      expect(metrics.duration).toBe(5000);
      expect(collector.getActiveWorkflows()).not.toContain('wf-1');
      vi.useRealTimers();
    });
  });

  describe('onNodeStart', () => {
    it('creates node metrics entry and increments totalNodes', () => {
      collector.onWorkflowStart('wf-1');
      collector.onNodeStart('n1', 'Node 1');

      const metrics = collector.getMetrics('wf-1')!;
      expect(metrics.totalNodes).toBe(1);
      expect(metrics.nodeMetrics.get('n1')?.status).toBe('running');
      expect(metrics.nodeMetrics.get('n1')?.name).toBe('Node 1');
    });
  });

  describe('onNodeComplete', () => {
    it('updates node status and increments counters', () => {
      collector.onWorkflowStart('wf-1');
      collector.onNodeStart('n1', 'Node 1');
      collector.onNodeComplete('n1', 'completed', 1000);

      const metrics = collector.getMetrics('wf-1')!;
      expect(metrics.nodeMetrics.get('n1')?.status).toBe('completed');
      expect(metrics.nodeMetrics.get('n1')?.duration).toBe(1000);
      expect(metrics.completedNodes).toBe(1);
    });

    it('increments failedNodes on failure', () => {
      collector.onWorkflowStart('wf-1');
      collector.onNodeStart('n1', 'Node 1');
      collector.onNodeComplete('n1', 'failed', 500, 'error message');

      const metrics = collector.getMetrics('wf-1')!;
      expect(metrics.failedNodes).toBe(1);
      expect(metrics.nodeMetrics.get('n1')?.error).toBe('error message');
    });

    it('increments skippedNodes on skip', () => {
      collector.onWorkflowStart('wf-1');
      collector.onNodeStart('n1', 'Node 1');
      collector.onNodeComplete('n1', 'skipped', 0);

      const metrics = collector.getMetrics('wf-1')!;
      expect(metrics.skippedNodes).toBe(1);
    });
  });

  describe('getMetrics', () => {
    it('returns null for unknown workflow', () => {
      expect(collector.getMetrics('unknown')).toBeNull();
    });
  });

  describe('getAllMetrics', () => {
    it('returns all workflow metrics', () => {
      collector.onWorkflowStart('wf-1');
      collector.onWorkflowStart('wf-2');
      const all = collector.getAllMetrics();
      expect(all).toHaveLength(2);
    });
  });

  describe('getSummary', () => {
    it('returns summary with zeros when no workflows', () => {
      const summary = collector.getSummary();
      expect(summary.totalWorkflows).toBe(0);
      expect(summary.activeWorkflows).toBe(0);
      expect(summary.completedWorkflows).toBe(0);
      expect(summary.failedWorkflows).toBe(0);
      expect(summary.averageDuration).toBe(0);
      expect(summary.successRate).toBe(0);
    });

    it('calculates summary correctly', () => {
      collector.onWorkflowStart('wf-1');
      collector.onWorkflowStart('wf-2');
      collector.onWorkflowStart('wf-3');
      collector.onWorkflowComplete('wf-1', 'completed', 1000);
      collector.onWorkflowComplete('wf-2', 'completed', 2000);
      collector.onWorkflowComplete('wf-3', 'failed', 500);

      const summary = collector.getSummary();
      expect(summary.totalWorkflows).toBe(3);
      expect(summary.activeWorkflows).toBe(0);
      expect(summary.completedWorkflows).toBe(2);
      expect(summary.failedWorkflows).toBe(1);
      expect(summary.averageDuration).toBe(1500);
      expect(summary.successRate).toBeCloseTo(66.67, 1);
    });
  });

  describe('clear', () => {
    it('removes all metrics and active workflows', () => {
      collector.onWorkflowStart('wf-1');
      collector.clear();
      expect(collector.getAllMetrics()).toHaveLength(0);
      expect(collector.getActiveWorkflows()).toHaveLength(0);
    });
  });
});

describe('EventDrivenMonitor', () => {
  it('attaches event handlers to workflow', () => {
    const collector = new MetricsCollector();
    const eventMonitor = new EventDrivenMonitor(collector);

    const onEventSpy = vi.fn();
    const eventHandler = { onEvent: onEventSpy };
    eventMonitor.attachToWorkflow(eventHandler as any);

    expect(onEventSpy).toHaveBeenCalledWith(expect.any(Function));
  });

  it('handles workflow_started event via attached callback', () => {
    const collector = new MetricsCollector();
    const eventMonitor = new EventDrivenMonitor(collector);

    let capturedCallback: Function = () => {};
    const eventHandler = {
      onEvent: vi.fn().mockImplementation((cb: Function) => {
        capturedCallback = cb;
      }),
    };

    eventMonitor.attachToWorkflow(eventHandler as any);

    // Simulate event
    capturedCallback({ type: 'workflow_started', timestamp: 'wf-start-1' });

    // The event uses timestamp as workflowId, so it creates a metrics entry with that ID
    expect(collector.getActiveWorkflows()).toContain('wf-start-1');
  });

  it('handles node_started event via callback', () => {
    const collector = new MetricsCollector();
    const eventMonitor = new EventDrivenMonitor(collector);

    let capturedCallback: Function = () => {};
    const eventHandler = {
      onEvent: vi.fn().mockImplementation((cb: Function) => {
        capturedCallback = cb;
      }),
    };

    eventMonitor.attachToWorkflow(eventHandler as any);

    // Set up workflow first
    capturedCallback({ type: 'workflow_started', timestamp: 'wf-1' });

    // Now node started
    capturedCallback({ type: 'node_started', timestamp: 'wf-1', nodeId: 'n1', status: 'running' });

    const metrics = collector.getMetrics('wf-1')!;
    expect(metrics.totalNodes).toBe(1);
  });
});
