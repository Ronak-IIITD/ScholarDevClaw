/**
 * Phase execution observability utilities.
 * Provides structured logging, timing, and progress tracking for phase execution.
 */

import { logger } from '../utils/logger.js';

export interface PhaseTiming {
  phase: number;
  name: string;
  startTime: number;
  endTime?: number;
  durationMs?: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  error?: string;
}

export interface PhaseProgressEvent {
  type: 'phase_start' | 'phase_progress' | 'phase_complete' | 'phase_error' | 'phase_retry';
  phase: number;
  phaseName: string;
  timestamp: string;
  data?: Record<string, unknown>;
  durationMs?: number;
  error?: string;
}

export interface PhaseExecutionSummary {
  totalPhases: number;
  completedPhases: number;
  failedPhases: number;
  totalDurationMs: number;
  phaseTimings: PhaseTiming[];
}

/**
 * Phase observability tracker for a single integration run.
 */
export class PhaseObserver {
  private timings: PhaseTiming[] = [];
  private runId: string;
  private integrationId?: string;
  private onProgress?: (event: PhaseProgressEvent) => void;

  constructor(runId: string, integrationId?: string, onProgress?: (event: PhaseProgressEvent) => void) {
    this.runId = runId;
    this.integrationId = integrationId;
    this.onProgress = onProgress;
  }

  /**
   * Record the start of a phase.
   */
  startPhase(phase: number, name: string): void {
    const timing: PhaseTiming = {
      phase,
      name,
      startTime: Date.now(),
      status: 'running',
    };
    this.timings.push(timing);

    this.emitProgress({
      type: 'phase_start',
      phase,
      phaseName: name,
      timestamp: new Date().toISOString(),
    });

    logger.info(`Phase ${phase} (${name}) started`, {
      runId: this.runId,
      integrationId: this.integrationId,
      phase,
      phaseName: name,
    });
  }

  /**
   * Record progress within a phase (e.g., sub-steps).
   */
  progress(phase: number, name: string, data: Record<string, unknown> = {}): void {
    this.emitProgress({
      type: 'phase_progress',
      phase,
      phaseName: name,
      timestamp: new Date().toISOString(),
      data,
    });

    logger.debug(`Phase ${phase} (${name}) progress`, {
      runId: this.runId,
      integrationId: this.integrationId,
      phase,
      phaseName: name,
      ...data,
    });
  }

  /**
   * Record successful completion of a phase.
   */
  completePhase(phase: number, name: string, result?: unknown): void {
    const timing = this.timings.find((t) => t.phase === phase);
    if (timing) {
      timing.endTime = Date.now();
      timing.durationMs = timing.endTime - timing.startTime;
      timing.status = 'completed';
    }

    this.emitProgress({
      type: 'phase_complete',
      phase,
      phaseName: name,
      timestamp: new Date().toISOString(),
      durationMs: timing?.durationMs,
      data: { result: result ? 'present' : 'none' },
    });

    logger.info(`Phase ${phase} (${name}) completed`, {
      runId: this.runId,
      integrationId: this.integrationId,
      phase,
      phaseName: name,
      durationMs: timing?.durationMs,
    });
  }

  /**
   * Record phase failure.
   */
  failPhase(phase: number, name: string, error: Error | string): void {
    const timing = this.timings.find((t) => t.phase === phase);
    if (timing) {
      timing.endTime = Date.now();
      timing.durationMs = timing.endTime - timing.startTime;
      timing.status = 'failed';
      timing.error = error instanceof Error ? error.message : error;
    }

    this.emitProgress({
      type: 'phase_error',
      phase,
      phaseName: name,
      timestamp: new Date().toISOString(),
      durationMs: timing?.durationMs,
      error: error instanceof Error ? error.message : error,
    });

    logger.error(`Phase ${phase} (${name}) failed`, {
      runId: this.runId,
      integrationId: this.integrationId,
      phase,
      phaseName: name,
      durationMs: timing?.durationMs,
      error: error instanceof Error ? error.message : error,
    });
  }

  /**
   * Record a retry attempt for a phase.
   */
  retryPhase(phase: number, name: string, attempt: number, error: string): void {
    this.emitProgress({
      type: 'phase_retry',
      phase,
      phaseName: name,
      timestamp: new Date().toISOString(),
      data: { attempt, error },
    });

    logger.warn(`Phase ${phase} (${name}) retry attempt ${attempt}`, {
      runId: this.runId,
      integrationId: this.integrationId,
      phase,
      phaseName: name,
      attempt,
      error,
    });
  }

  /**
   * Get all phase timings.
   */
  getTimings(): PhaseTiming[] {
    return [...this.timings];
  }

  /**
   * Get execution summary.
   */
  getSummary(): PhaseExecutionSummary {
    const completed = this.timings.filter((t) => t.status === 'completed').length;
    const failed = this.timings.filter((t) => t.status === 'failed').length;
    const totalDuration = this.timings.reduce((sum, t) => sum + (t.durationMs || 0), 0);

    return {
      totalPhases: this.timings.length,
      completedPhases: completed,
      failedPhases: failed,
      totalDurationMs: totalDuration,
      phaseTimings: [...this.timings],
    };
  }

  /**
   * Emit a progress event to the callback if registered.
   */
  private emitProgress(event: PhaseProgressEvent): void {
    if (this.onProgress) {
      try {
        this.onProgress(event);
      } catch (e) {
        logger.error('Phase progress callback error', { error: e });
      }
    }
  }
}

/**
 * Create a phase observer with Convex integration for real-time UI updates.
 */
export function createPhaseObserver(
  runId: string,
  integrationId?: string,
  convexClient?: { saveLog: (id: string, log: string) => Promise<void> }
): PhaseObserver {
  return new PhaseObserver(runId, integrationId, async (event) => {
    if (!convexClient || !integrationId) return;

    const logLine = formatPhaseEvent(event);
    try {
      await convexClient.saveLog(integrationId, logLine);
    } catch (e) {
      logger.error('Failed to save phase event to Convex', { error: e });
    }
  });
}

/**
 * Format a phase event as a log line for Convex.
 */
function formatPhaseEvent(event: PhaseProgressEvent): string {
  const prefix = `[${event.timestamp}] Phase ${event.phase} (${event.phaseName})`;
  switch (event.type) {
    case 'phase_start':
      return `${prefix} ▶ STARTED`;
    case 'phase_progress':
      return `${prefix} ⚡ ${JSON.stringify(event.data)}`;
    case 'phase_complete':
      return `${prefix} ✓ COMPLETED (${event.durationMs}ms)`;
    case 'phase_error':
      return `${prefix} ✗ FAILED: ${event.error}`;
    case 'phase_retry':
      return `${prefix} ↻ RETRY attempt ${event.data?.attempt}: ${event.data?.error}`;
    default:
      return `${prefix} ${event.type}`;
  }
}

/**
 * Decorator to add observability to a phase execution function.
 */
export function withPhaseObservability<T extends (...args: unknown[]) => Promise<unknown>>(
  phase: number,
  name: string,
  observer: PhaseObserver,
  fn: T
): T {
  return (async (...args: unknown[]) => {
    observer.startPhase(phase, name);
    try {
      const result = await fn(...args);
      observer.completePhase(phase, name, result);
      return result;
    } catch (error) {
      observer.failPhase(phase, name, error instanceof Error ? error : String(error));
      throw error;
    }
  }) as T;
}
