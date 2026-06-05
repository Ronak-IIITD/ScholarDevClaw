import { describe, it, expect, vi, beforeEach } from 'vitest';
import { PhaseObserver, createPhaseObserver } from './phase-observer.js';

describe('PhaseObserver', () => {
  let observer: PhaseObserver;
  let progressEvents: any[];

  beforeEach(() => {
    progressEvents = [];
    observer = new PhaseObserver('test-run-id', 'test-integration-id', (event) => {
      progressEvents.push(event);
    });
  });

  it('should track phase start', () => {
    observer.startPhase(1, 'Test Phase');

    const timings = observer.getTimings();
    expect(timings).toHaveLength(1);
    expect(timings[0].phase).toBe(1);
    expect(timings[0].name).toBe('Test Phase');
    expect(timings[0].status).toBe('running');
    expect(timings[0].startTime).toBeGreaterThan(0);
  });

  it('should track phase completion', () => {
    observer.startPhase(1, 'Test Phase');
    observer.completePhase(1, 'Test Phase', { data: 'result' });

    const timings = observer.getTimings();
    expect(timings[0].status).toBe('completed');
    expect(timings[0].endTime).toBeDefined();
    expect(timings[0].durationMs).toBeDefined();
    expect(timings[0].durationMs).toBeGreaterThanOrEqual(0);
  });

  it('should track phase failure', () => {
    observer.startPhase(1, 'Test Phase');
    observer.failPhase(1, 'Test Phase', new Error('Something went wrong'));

    const timings = observer.getTimings();
    expect(timings[0].status).toBe('failed');
    expect(timings[0].error).toBe('Something went wrong');
    expect(timings[0].endTime).toBeDefined();
  });

  it('should track phase retry', () => {
    observer.startPhase(1, 'Test Phase');
    observer.retryPhase(1, 'Test Phase', 2, 'Temporary error');

    expect(progressEvents).toHaveLength(2); // start + retry
    expect(progressEvents[1].type).toBe('phase_retry');
    expect(progressEvents[1].data?.attempt).toBe(2);
  });

  it('should emit progress events', () => {
    observer.startPhase(1, 'Test Phase');
    observer.progress(1, 'Test Phase', { step: 'sub-step-1' });

    expect(progressEvents).toHaveLength(2); // start + progress
    expect(progressEvents[1].type).toBe('phase_progress');
    expect(progressEvents[1].data?.step).toBe('sub-step-1');
  });

  it('should provide execution summary', () => {
    observer.startPhase(1, 'Phase 1');
    observer.completePhase(1, 'Phase 1');
    observer.startPhase(2, 'Phase 2');
    observer.failPhase(2, 'Phase 2', 'Error');

    const summary = observer.getSummary();
    expect(summary.totalPhases).toBe(2);
    expect(summary.completedPhases).toBe(1);
    expect(summary.failedPhases).toBe(1);
    expect(summary.totalDurationMs).toBeGreaterThanOrEqual(0);
  });

  it('should handle multiple phases', () => {
    observer.startPhase(1, 'Phase 1');
    observer.completePhase(1, 'Phase 1');
    observer.startPhase(2, 'Phase 2');
    observer.completePhase(2, 'Phase 2');
    observer.startPhase(3, 'Phase 3');
    observer.completePhase(3, 'Phase 3');

    const timings = observer.getTimings();
    expect(timings).toHaveLength(3);
    expect(timings.every((t) => t.status === 'completed')).toBe(true);
  });
});

describe('createPhaseObserver', () => {
  it('should create observer with convex integration', async () => {
    const mockConvex = {
      saveLog: vi.fn().mockResolvedValue(undefined),
    };

    const observer = createPhaseObserver('run-1', 'int-1', mockConvex);
    observer.startPhase(1, 'Test Phase');

    // Wait for async log
    await new Promise((resolve) => setTimeout(resolve, 10));

    expect(mockConvex.saveLog).toHaveBeenCalled();
    const logCall = mockConvex.saveLog.mock.calls[0];
    expect(logCall[0]).toBe('int-1');
    expect(logCall[1]).toContain('Phase 1 (Test Phase)');
    expect(logCall[1]).toContain('STARTED');
  });

  it('should work without convex client', () => {
    const observer = createPhaseObserver('run-1');
    expect(() => observer.startPhase(1, 'Test Phase')).not.toThrow();
  });
});
