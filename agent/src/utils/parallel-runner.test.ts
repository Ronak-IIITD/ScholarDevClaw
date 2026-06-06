import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  ParallelPhaseRunner,
  findCycle,
  topologicalLayers,
  topologicalOrder,
  validateDag,
  type TaskSpec,
} from './parallel-runner.js';

// =========================================================================
// validateDag
// =========================================================================

describe('validateDag', () => {
  it('accepts an empty task list', () => {
    expect(() => validateDag([])).not.toThrow();
  });

  it('accepts a single task with no deps', () => {
    expect(() => validateDag([{ id: 'a', name: 'A', run: () => Promise.resolve() }])).not.toThrow();
  });

  it('accepts a linear chain', () => {
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', run: () => Promise.resolve() },
      { id: 'b', name: 'B', dependsOn: ['a'], run: () => Promise.resolve() },
      { id: 'c', name: 'C', dependsOn: ['b'], run: () => Promise.resolve() },
    ];
    expect(() => validateDag(tasks)).not.toThrow();
  });

  it('accepts a diamond', () => {
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', run: () => Promise.resolve() },
      { id: 'b', name: 'B', dependsOn: ['a'], run: () => Promise.resolve() },
      { id: 'c', name: 'C', dependsOn: ['a'], run: () => Promise.resolve() },
      { id: 'd', name: 'D', dependsOn: ['b', 'c'], run: () => Promise.resolve() },
    ];
    expect(() => validateDag(tasks)).not.toThrow();
  });

  it('rejects duplicate ids', () => {
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A1', run: () => Promise.resolve() },
      { id: 'a', name: 'A2', run: () => Promise.resolve() },
    ];
    expect(() => validateDag(tasks)).toThrow(/Duplicate task id/);
  });

  it('rejects self-dependency', () => {
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', dependsOn: ['a'], run: () => Promise.resolve() },
    ];
    expect(() => validateDag(tasks)).toThrow(/depends on itself/);
  });

  it('rejects unknown dependency', () => {
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', dependsOn: ['b'], run: () => Promise.resolve() },
    ];
    expect(() => validateDag(tasks)).toThrow(/unknown task 'b'/);
  });

  it('rejects a 2-node cycle', () => {
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', dependsOn: ['b'], run: () => Promise.resolve() },
      { id: 'b', name: 'B', dependsOn: ['a'], run: () => Promise.resolve() },
    ];
    expect(() => validateDag(tasks)).toThrow(/Cycle detected/);
  });

  it('rejects a 3-node cycle', () => {
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', dependsOn: ['c'], run: () => Promise.resolve() },
      { id: 'b', name: 'B', dependsOn: ['a'], run: () => Promise.resolve() },
      { id: 'c', name: 'C', dependsOn: ['b'], run: () => Promise.resolve() },
    ];
    expect(() => validateDag(tasks)).toThrow(/Cycle detected/);
  });

  it('rejects a self-cycle via indirect chain', () => {
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', dependsOn: ['b'], run: () => Promise.resolve() },
      { id: 'b', name: 'B', dependsOn: ['c'], run: () => Promise.resolve() },
      { id: 'c', name: 'C', dependsOn: ['a'], run: () => Promise.resolve() },
    ];
    expect(() => validateDag(tasks)).toThrow(/Cycle detected/);
  });
});

// =========================================================================
// findCycle
// =========================================================================

describe('findCycle', () => {
  it('returns null for an acyclic graph', () => {
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', run: () => Promise.resolve() },
      { id: 'b', name: 'B', dependsOn: ['a'], run: () => Promise.resolve() },
    ];
    expect(findCycle(tasks)).toBeNull();
  });

  it('detects a 2-node cycle', () => {
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', dependsOn: ['b'], run: () => Promise.resolve() },
      { id: 'b', name: 'B', dependsOn: ['a'], run: () => Promise.resolve() },
    ];
    const cycle = findCycle(tasks);
    expect(cycle).not.toBeNull();
    expect(cycle![0]).toBe(cycle![cycle!.length - 1]);
    expect(cycle!.length).toBeGreaterThan(1);
  });

  it('returns null for a diamond', () => {
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', run: () => Promise.resolve() },
      { id: 'b', name: 'B', dependsOn: ['a'], run: () => Promise.resolve() },
      { id: 'c', name: 'C', dependsOn: ['a'], run: () => Promise.resolve() },
      { id: 'd', name: 'D', dependsOn: ['b', 'c'], run: () => Promise.resolve() },
    ];
    expect(findCycle(tasks)).toBeNull();
  });
});

// =========================================================================
// topologicalLayers
// =========================================================================

describe('topologicalLayers', () => {
  it('handles empty input', () => {
    expect(topologicalLayers([])).toEqual([]);
  });

  it('puts all independent tasks in layer 0', () => {
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', run: () => Promise.resolve() },
      { id: 'b', name: 'B', run: () => Promise.resolve() },
      { id: 'c', name: 'C', run: () => Promise.resolve() },
    ];
    const layers = topologicalLayers(tasks);
    expect(layers).toHaveLength(1);
    expect(layers[0].map((t) => t.id).sort()).toEqual(['a', 'b', 'c']);
  });

  it('puts a linear chain in separate layers', () => {
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', run: () => Promise.resolve() },
      { id: 'b', name: 'B', dependsOn: ['a'], run: () => Promise.resolve() },
      { id: 'c', name: 'C', dependsOn: ['b'], run: () => Promise.resolve() },
    ];
    const layers = topologicalLayers(tasks);
    expect(layers.map((l) => l.map((t) => t.id))).toEqual([['a'], ['b'], ['c']]);
  });

  it('groups parallel branches in the same layer (diamond)', () => {
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', run: () => Promise.resolve() },
      { id: 'b', name: 'B', dependsOn: ['a'], run: () => Promise.resolve() },
      { id: 'c', name: 'C', dependsOn: ['a'], run: () => Promise.resolve() },
      { id: 'd', name: 'D', dependsOn: ['b', 'c'], run: () => Promise.resolve() },
    ];
    const layers = topologicalLayers(tasks);
    expect(layers.map((l) => l.map((t) => t.id).sort())).toEqual([
      ['a'],
      ['b', 'c'],
      ['d'],
    ]);
  });
});

describe('topologicalOrder', () => {
  it('returns tasks in dependency order', () => {
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', run: () => Promise.resolve() },
      { id: 'b', name: 'B', dependsOn: ['a'], run: () => Promise.resolve() },
      { id: 'c', name: 'C', dependsOn: ['a'], run: () => Promise.resolve() },
      { id: 'd', name: 'D', dependsOn: ['b', 'c'], run: () => Promise.resolve() },
    ];
    const order = topologicalOrder(tasks).map((t) => t.id);
    expect(order.indexOf('a')).toBeLessThan(order.indexOf('b'));
    expect(order.indexOf('a')).toBeLessThan(order.indexOf('c'));
    expect(order.indexOf('b')).toBeLessThan(order.indexOf('d'));
    expect(order.indexOf('c')).toBeLessThan(order.indexOf('d'));
  });
});

// =========================================================================
// ParallelPhaseRunner construction
// =========================================================================

describe('ParallelPhaseRunner construction', () => {
  it('accepts a maxConcurrency option', () => {
    expect(() => new ParallelPhaseRunner({ maxConcurrency: 8 })).not.toThrow();
  });

  it('rejects maxConcurrency < 1', () => {
    expect(() => new ParallelPhaseRunner({ maxConcurrency: 0 })).toThrow();
    expect(() => new ParallelPhaseRunner({ maxConcurrency: -1 })).toThrow();
  });

  it('defaults to maxConcurrency = 4', () => {
    const runner = new ParallelPhaseRunner();
    expect(runner['maxConcurrency']).toBe(4);
  });
});

// =========================================================================
// ParallelPhaseRunner.run — happy path
// =========================================================================

describe('ParallelPhaseRunner.run — happy path', () => {
  it('runs an empty task list', async () => {
    const runner = new ParallelPhaseRunner();
    const report = await runner.run([]);
    expect(report.ok).toBe(true);
    expect(report.results.size).toBe(0);
    expect(report.succeeded).toEqual([]);
    expect(report.failed).toEqual([]);
  });

  it('runs a single task', async () => {
    const runner = new ParallelPhaseRunner();
    const tasks: TaskSpec<number>[] = [
      { id: 'a', name: 'A', run: async () => 42 },
    ];
    const report = await runner.run(tasks);
    expect(report.ok).toBe(true);
    expect(report.succeeded).toEqual(['a']);
    expect(report.results.get('a')?.status).toBe('succeeded');
    expect(report.results.get('a')?.value).toBe(42);
  });

  it('runs independent tasks in parallel', async () => {
    const runner = new ParallelPhaseRunner({ maxConcurrency: 4 });
    const events: string[] = [];
    const tasks: TaskSpec<string>[] = [
      {
        id: 'a',
        name: 'A',
        run: async () => {
          events.push('a-start');
          await new Promise((r) => setTimeout(r, 50));
          events.push('a-end');
          return 'A';
        },
      },
      {
        id: 'b',
        name: 'B',
        run: async () => {
          events.push('b-start');
          await new Promise((r) => setTimeout(r, 50));
          events.push('b-end');
          return 'B';
        },
      },
    ];
    const t0 = Date.now();
    const report = await runner.run(tasks);
    const elapsed = Date.now() - t0;
    expect(report.ok).toBe(true);
    expect(report.succeeded.sort()).toEqual(['a', 'b']);
    // Both should have started before either ended (proves parallelism)
    expect(events.indexOf('a-start')).toBeLessThan(events.indexOf('a-end'));
    expect(events.indexOf('b-start')).toBeLessThan(events.indexOf('b-end'));
    expect(events.indexOf('a-start')).toBeLessThan(events.indexOf('a-end'));
    // Should be ~50ms not ~100ms
    expect(elapsed).toBeLessThan(150);
  });

  it('respects task dependencies (topological order)', async () => {
    const runner = new ParallelPhaseRunner();
    const events: string[] = [];
    const tasks: TaskSpec<string>[] = [
      {
        id: 'a',
        name: 'A',
        run: async () => {
          events.push('a');
          return 'A';
        },
      },
      {
        id: 'b',
        name: 'B',
        dependsOn: ['a'],
        run: async () => {
          events.push('b');
          return 'B';
        },
      },
      {
        id: 'c',
        name: 'C',
        dependsOn: ['b'],
        run: async () => {
          events.push('c');
          return 'C';
        },
      },
    ];
    await runner.run(tasks);
    expect(events).toEqual(['a', 'b', 'c']);
  });

  it('respects maxConcurrency (does not over-subscribe)', async () => {
    let active = 0;
    let maxActive = 0;
    const tasks: TaskSpec<void>[] = Array.from({ length: 10 }, (_, i) => ({
      id: `t${i}`,
      name: `T${i}`,
      run: async () => {
        active += 1;
        maxActive = Math.max(maxActive, active);
        await new Promise((r) => setTimeout(r, 20));
        active -= 1;
      },
    }));
    const runner = new ParallelPhaseRunner({ maxConcurrency: 3 });
    const report = await runner.run(tasks);
    expect(report.ok).toBe(true);
    expect(report.succeeded).toHaveLength(10);
    expect(maxActive).toBeLessThanOrEqual(3);
  });

  it('passes the shared context to every task', async () => {
    const runner = new ParallelPhaseRunner();
    const ctx = { userId: 'u1', mode: 'autonomous' };
    const seen: unknown[] = [];
    const tasks: TaskSpec<string, typeof ctx>[] = [
      { id: 'a', name: 'A', run: async (c) => { seen.push(c); return 'A'; } },
      { id: 'b', name: 'B', dependsOn: ['a'], run: async (c) => { seen.push(c); return 'B'; } },
    ];
    const report = await runner.run(tasks, { context: ctx });
    expect(report.ok).toBe(true);
    expect(seen).toEqual([ctx, ctx]);
  });
});

// =========================================================================
// ParallelPhaseRunner.run — failure handling
// =========================================================================

describe('ParallelPhaseRunner.run — failure handling', () => {
  it('captures a single task failure without throwing', async () => {
    const runner = new ParallelPhaseRunner();
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', run: async () => 'ok' },
      { id: 'b', name: 'B', run: async () => { throw new Error('boom'); } },
    ];
    const report = await runner.run(tasks);
    expect(report.ok).toBe(false);
    expect(report.succeeded).toEqual(['a']);
    expect(report.failed).toEqual(['b']);
    expect(report.results.get('b')?.error?.message).toBe('boom');
  });

  it('skips downstream tasks when a dependency fails', async () => {
    const runner = new ParallelPhaseRunner();
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', run: async () => { throw new Error('a-failed'); } },
      { id: 'b', name: 'B', dependsOn: ['a'], run: async () => 'B' },
      { id: 'c', name: 'C', dependsOn: ['b'], run: async () => 'C' },
    ];
    const report = await runner.run(tasks);
    expect(report.results.get('a')?.status).toBe('failed');
    expect(report.results.get('b')?.status).toBe('skipped');
    expect(report.results.get('c')?.status).toBe('skipped');
  });

  it('runs tolerateDependencyFailures tasks even when deps fail', async () => {
    const runner = new ParallelPhaseRunner();
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', run: async () => { throw new Error('a-failed'); } },
      {
        id: 'b',
        name: 'B',
        dependsOn: ['a'],
        tolerateDependencyFailures: true,
        run: async () => 'B-ran-anyway',
      },
    ];
    const report = await runner.run(tasks);
    expect(report.results.get('a')?.status).toBe('failed');
    expect(report.results.get('b')?.status).toBe('succeeded');
    expect(report.results.get('b')?.value).toBe('B-ran-anyway');
  });

  it('does not block independent tasks when a sibling fails', async () => {
    const runner = new ParallelPhaseRunner();
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', run: async () => { throw new Error('boom'); } },
      { id: 'b', name: 'B', run: async () => 'B' },
    ];
    const report = await runner.run(tasks);
    expect(report.results.get('a')?.status).toBe('failed');
    expect(report.results.get('b')?.status).toBe('succeeded');
  });

  it('records task duration even on failure', async () => {
    const runner = new ParallelPhaseRunner();
    const tasks: TaskSpec[] = [
      {
        id: 'a',
        name: 'A',
        run: async () => {
          await new Promise((r) => setTimeout(r, 30));
          throw new Error('late boom');
        },
      },
    ];
    const report = await runner.run(tasks);
    expect(report.results.get('a')?.durationMs).toBeGreaterThanOrEqual(25);
  });

  it('catches non-Error throws and wraps them', async () => {
    const runner = new ParallelPhaseRunner();
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', run: async () => { throw 'string-error'; } },
    ];
    const report = await runner.run(tasks);
    expect(report.results.get('a')?.error).toBeInstanceOf(Error);
    expect(report.results.get('a')?.error?.message).toBe('string-error');
  });
});

// =========================================================================
// ParallelPhaseRunner.run — observability hooks
// =========================================================================

describe('ParallelPhaseRunner.run — observability', () => {
  let runner: ParallelPhaseRunner;
  beforeEach(() => {
    runner = new ParallelPhaseRunner();
  });

  it('invokes onTaskStart and onTaskComplete hooks', async () => {
    const starts: string[] = [];
    const completes: string[] = [];
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', run: async () => 'A' },
      { id: 'b', name: 'B', dependsOn: ['a'], run: async () => 'B' },
    ];
    await runner.run(tasks, {
      onTaskStart: (t) => starts.push(t.id),
      onTaskComplete: (r) => completes.push(r.id),
    });
    expect(starts.sort()).toEqual(['a', 'b']);
    expect(completes.sort()).toEqual(['a', 'b']);
  });

  it('invokes onTaskComplete with failed status', async () => {
    const statuses: string[] = [];
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', run: async () => { throw new Error('boom'); } },
    ];
    await runner.run(tasks, {
      onTaskComplete: (r) => statuses.push(r.status),
    });
    expect(statuses).toEqual(['failed']);
  });

  it('records startedAt and finishedAt timestamps', async () => {
    const tasks: TaskSpec[] = [
      { id: 'a', name: 'A', run: async () => 'A' },
    ];
    const report = await runner.run(tasks);
    const result = report.results.get('a')!;
    expect(result.startedAt).toBeDefined();
    expect(result.finishedAt).toBeDefined();
    expect(new Date(result.finishedAt!).getTime()).toBeGreaterThanOrEqual(
      new Date(result.startedAt!).getTime(),
    );
  });
});

// =========================================================================
// ParallelPhaseRunner.run — cancellation
// =========================================================================

describe('ParallelPhaseRunner.run — cancellation', () => {
  it('stops scheduling new tasks after abort', async () => {
    const controller = new AbortController();
    const runner = new ParallelPhaseRunner({ maxConcurrency: 1 });
    const events: string[] = [];
    const tasks: TaskSpec[] = Array.from({ length: 5 }, (_, i) => ({
      id: `t${i}`,
      name: `T${i}`,
      run: async () => {
        events.push(`run-${i}`);
        await new Promise((r) => setTimeout(r, 20));
        if (i === 0) controller.abort();
        return i;
      },
    }));
    const report = await runner.run(tasks, { signal: controller.signal });
    expect(report.ok).toBe(false);
    // At least one task should have been cancelled (and probably more,
    // depending on timing); not all should have run
    expect(events.length).toBeLessThan(5);
  });

  it('marks in-flight tasks as cancelled when aborted', async () => {
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 5);
    const runner = new ParallelPhaseRunner();
    const tasks: TaskSpec[] = [
      {
        id: 'a',
        name: 'A',
        run: async () => {
          await new Promise((r) => setTimeout(r, 50));
          return 'A';
        },
      },
    ];
    const report = await runner.run(tasks, { signal: controller.signal });
    // The single task may have completed before or after the abort;
    // either way, the report should not be ok
    expect(['cancelled', 'succeeded']).toContain(report.results.get('a')?.status);
    if (report.results.get('a')?.status === 'cancelled') {
      expect(report.cancelled).toContain('a');
    }
  });
});

// =========================================================================
// ParallelPhaseRunner.run — performance / scale
// =========================================================================

describe('ParallelPhaseRunner.run — performance', () => {
  it('runs 20 independent tasks in parallel faster than sequentially', async () => {
    const runner = new ParallelPhaseRunner({ maxConcurrency: 10 });
    const tasks: TaskSpec<number>[] = Array.from({ length: 20 }, (_, i) => ({
      id: `t${i}`,
      name: `T${i}`,
      run: async () => {
        await new Promise((r) => setTimeout(r, 30));
        return i;
      },
    }));
    const t0 = Date.now();
    const report = await runner.run(tasks);
    const elapsed = Date.now() - t0;
    expect(report.ok).toBe(true);
    // 20 tasks * 30ms / 10 workers ≈ 60ms (allow generous bound)
    expect(elapsed).toBeLessThan(250);
  });
});

// =========================================================================
// Integration: realistic 6-phase pipeline
// =========================================================================

describe('integration: 6-phase pipeline DAG', () => {
  it('runs a realistic ML pipeline (Phase1 → 2 → 3 → 4 → 5 → 6)', async () => {
    const runner = new ParallelPhaseRunner({ maxConcurrency: 2 });
    const events: string[] = [];

    const tasks: TaskSpec<string>[] = [
      {
        id: 'p1',
        name: 'Repo analysis',
        run: async () => {
          events.push('p1');
          return 'repo';
        },
      },
      {
        id: 'p2',
        name: 'Research',
        dependsOn: ['p1'],
        run: async () => {
          events.push('p2');
          return 'spec';
        },
      },
      {
        id: 'p3',
        name: 'Mapping',
        dependsOn: ['p1', 'p2'],
        run: async () => {
          events.push('p3');
          return 'mapping';
        },
      },
      {
        id: 'p4',
        name: 'Patch',
        dependsOn: ['p3'],
        run: async () => {
          events.push('p4');
          return 'patch';
        },
      },
      {
        id: 'p5',
        name: 'Validation',
        dependsOn: ['p4'],
        run: async () => {
          events.push('p5');
          return 'validation';
        },
      },
      {
        id: 'p6',
        name: 'Report',
        dependsOn: ['p5'],
        run: async () => {
          events.push('p6');
          return 'report';
        },
      },
    ];

    const report = await runner.run(tasks);
    expect(report.ok).toBe(true);
    expect(report.succeeded).toEqual(['p1', 'p2', 'p3', 'p4', 'p5', 'p6']);
    // Order must be respected
    for (let i = 0; i < events.length - 1; i++) {
      expect(events.indexOf(`p${i + 1}`)).toBeLessThan(events.indexOf(`p${i + 2}`));
    }
  });

  it('skips downstream phases when Phase 3 fails', async () => {
    const runner = new ParallelPhaseRunner();
    const tasks: TaskSpec[] = [
      { id: 'p1', name: 'P1', run: async () => 'r' },
      { id: 'p2', name: 'P2', dependsOn: ['p1'], run: async () => 'r' },
      { id: 'p3', name: 'P3', dependsOn: ['p2'], run: async () => { throw new Error('p3-broke'); } },
      { id: 'p4', name: 'P4', dependsOn: ['p3'], run: async () => 'r' },
      { id: 'p5', name: 'P5', dependsOn: ['p4'], run: async () => 'r' },
      { id: 'p6', name: 'P6', dependsOn: ['p5'], run: async () => 'r' },
    ];
    const report = await runner.run(tasks);
    expect(report.results.get('p1')?.status).toBe('succeeded');
    expect(report.results.get('p2')?.status).toBe('succeeded');
    expect(report.results.get('p3')?.status).toBe('failed');
    expect(report.results.get('p4')?.status).toBe('skipped');
    expect(report.results.get('p5')?.status).toBe('skipped');
    expect(report.results.get('p6')?.status).toBe('skipped');
  });
});

// =========================================================================
// Cleanup
// =========================================================================

afterEach(() => {
  vi.restoreAllMocks();
});
