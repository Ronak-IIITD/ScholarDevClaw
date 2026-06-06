/**
 * Parallel Phase Runner
 *
 * Executes a DAG of named tasks with explicit ``dependsOn`` relationships
 * using a bounded worker pool. Tasks whose dependencies are all met run
 * concurrently up to ``maxConcurrency``. When a dependency fails, its
 * downstream tasks are marked ``skipped`` unless they opt in via
 * ``tolerateDependencyFailures: true``.
 *
 * The runner is deliberately generic — it knows nothing about ML
 * pipelines, Convex, the Python bridge, or the orchestrator. The
 * orchestrator (or any future caller) is responsible for translating its
 * concrete phase definitions into ``TaskSpec`` objects and interpreting
 * the returned ``TaskResult`` map.
 *
 * Design properties:
 *
 * - Pure functions for the algorithmic bits (``validateDag``,
 *   ``topologicalLayers``, ``findCycle``) so they're easy to unit-test
 *   without spinning up the runner.
 * - The runner itself is stateless across runs; you can reuse one
 *   instance for multiple ``run()`` calls.
 * - Cancellation is cooperative via the standard ``AbortSignal``;
 *   in-flight tasks continue to completion (their results are dropped).
 * - The runner never throws on a single task failure; instead the
 *   affected ``TaskResult.status`` is ``'failed'`` and the report
 *   collects all outcomes. The runner only throws on programmer errors
 *   (invalid DAG, unknown dependency, duplicate ID, etc.).
 */

export type TaskId = string;

export interface TaskSpec<TResult = unknown, TContext = unknown> {
  id: TaskId;
  name: string;
  /** IDs of tasks that must complete (succeed or fail) before this task runs. */
  dependsOn?: TaskId[];
  /**
   * If ``true``, this task will still run even when one of its
   * dependencies has failed. Default: ``false`` (downstream tasks are
   * skipped on dep failure).
   */
  tolerateDependencyFailures?: boolean;
  /** The work. Receives the shared ``TContext`` and may return any value. */
  run: (context: TContext) => Promise<TResult>;
}

export type TaskStatus = 'succeeded' | 'failed' | 'skipped' | 'cancelled';

export interface TaskResult<TResult = unknown> {
  id: TaskId;
  name: string;
  status: TaskStatus;
  value?: TResult;
  error?: Error;
  durationMs: number;
  startedAt?: string;
  finishedAt?: string;
}

export interface ParallelRunnerOptions<TResult = unknown, TContext = unknown> {
  /** Maximum number of tasks to run concurrently. Default: ``4``. */
  maxConcurrency?: number;
  /**
   * Optional abort signal. When aborted, no new tasks are started and
   * already-running tasks are reported as ``cancelled`` once they
   * complete (their result value is dropped).
   */
  signal?: AbortSignal;
  /** Optional context value passed to every task's ``run()``. */
  context?: TContext;
  /** Called when a task starts. */
  onTaskStart?: (task: TaskSpec<TResult, TContext>) => void;
  /** Called when a task reaches a terminal status (succeeded / failed / skipped / cancelled). */
  onTaskComplete?: (result: TaskResult<TResult>) => void;
}

export interface ParallelRunnerReport<TResult = unknown> {
  results: Map<TaskId, TaskResult<TResult>>;
  succeeded: TaskId[];
  failed: TaskId[];
  skipped: TaskId[];
  cancelled: TaskId[];
  totalDurationMs: number;
  /**
   * True only when the run completed without failures, skips,
   * cancellations, or an abort signal — a run that was aborted is
   * not ``ok`` even if all started tasks happened to succeed.
   */
  ok: boolean;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export class ParallelPhaseRunner {
  private readonly maxConcurrency: number;

  constructor(options: { maxConcurrency?: number } = {}) {
    if (options.maxConcurrency !== undefined && options.maxConcurrency < 1) {
      throw new Error('maxConcurrency must be >= 1');
    }
    this.maxConcurrency = options.maxConcurrency ?? 4;
  }

  /**
   * Execute ``tasks`` respecting their ``dependsOn`` graph.
   *
   * Throws on programmer errors (duplicate ID, unknown dependency, cycle).
   * Does NOT throw on individual task failures — see ``ParallelRunnerReport``.
   */
  async run<TResult = unknown, TContext = unknown>(
    tasks: TaskSpec<TResult, TContext>[],
    options: ParallelRunnerOptions<TResult, TContext> = {},
  ): Promise<ParallelRunnerReport<TResult>> {
    validateDag(tasks);

    const concurrency = options.maxConcurrency ?? this.maxConcurrency;
    if (concurrency < 1) {
      throw new Error('maxConcurrency must be >= 1');
    }

    const t0 = Date.now();
    const results = new Map<TaskId, TaskResult<TResult>>();
    const completed = new Set<TaskId>();
    const failed = new Set<TaskId>();
    const cancelled = new Set<TaskId>();

    // Build adjacency lists.
    const byId = new Map<TaskId, TaskSpec<TResult, TContext>>();
    for (const task of tasks) {
      byId.set(task.id, task);
    }

    // Pre-compute dependents for each task so we can walk the DAG cheaply.
    const dependents = new Map<TaskId, TaskId[]>();
    for (const task of tasks) {
      for (const dep of task.dependsOn ?? []) {
        if (!dependents.has(dep)) {
          dependents.set(dep, []);
        }
        dependents.get(dep)!.push(task.id);
      }
    }

    // Initialize the ready queue with all tasks that have no deps.
    const ready: TaskId[] = [];
    for (const task of tasks) {
      if (!task.dependsOn || task.dependsOn.length === 0) {
        ready.push(task.id);
      }
    }

    let activeCount = 0;
    // Cooperative completion signaling. Each in-flight task calls
    // ``signalCompletion`` when it finishes, which resolves one
    // pending waiter. We use this rather than ``Promise.race`` over
    // the in-flight promises because race resolves immediately on
    // any already-settled promise, leading to a busy loop that
    // re-schedules without ever yielding.
    const pendingResolvers: Array<() => void> = [];
    const waitForNextCompletion = (): Promise<void> => {
      return new Promise<void>((resolve) => {
        pendingResolvers.push(resolve);
      });
    };
    const signalCompletion = (): void => {
      const r = pendingResolvers.shift();
      if (r) r();
    };

    const tryReleaseDependents = (finishedId: TaskId): void => {
      for (const dependentId of dependents.get(finishedId) ?? []) {
        const dep = byId.get(dependentId);
        if (!dep) continue;

        const upstream = dep.dependsOn ?? [];
        const allDone = upstream.every((id) => completed.has(id) || cancelled.has(id));
        if (!allDone) continue;

        // A dep is treated as a "failure" for downstream purposes when
        // its result is either 'failed' or 'skipped' (skipped propagates
        // because it means an ancestor failed and this task never ran).
        // Tasks with tolerateDependencyFailures: true still run.
        const hasFailure = upstream.some((id) => {
          const r = results.get(id);
          return r?.status === 'failed' || r?.status === 'skipped';
        });
        if (hasFailure && !dep.tolerateDependencyFailures) {
          const now = new Date().toISOString();
          const result: TaskResult<TResult> = {
            id: dep.id,
            name: dep.name,
            status: 'skipped',
            durationMs: 0,
            finishedAt: now,
          };
          results.set(dep.id, result);
          completed.add(dep.id);
          // A skipped task also "completes" the dependency for the purposes
          // of unblocking further downstream tasks; whether those run
          // depends on the tolerateDependencyFailures flag on each task.
          options.onTaskComplete?.(result);
          // Walk further downstream — they need to see this task as done.
          tryReleaseDependents(dep.id);
          continue;
        }

        ready.push(dep.id);
      }
    };

    // Cooperative cancellation: when aborted, no new tasks start, and
    // tasks already in flight are reported as cancelled when they finish.
    const isAborted = (): boolean => options.signal?.aborted === true;

    // The main scheduling loop. We pull up to ``concurrency`` tasks off the
    // ready queue at a time and run them concurrently. We resolve only
    // when the ready queue is empty AND no tasks are running.
    //
    // Implementation note: rather than a polling loop, we use a promise
    // per in-flight task. When a task resolves, we synchronously fill its
    // slot with another ready task. The outer while-loop terminates when
    // ready is empty and there are no active tasks.
    const runTask = async (taskId: TaskId): Promise<void> => {
      const task = byId.get(taskId);
      if (!task) {
        signalCompletion();
        return;
      }

      if (isAborted()) {
        const now = new Date().toISOString();
        const result: TaskResult<TResult> = {
          id: task.id,
          name: task.name,
          status: 'cancelled',
          durationMs: 0,
          finishedAt: now,
        };
        results.set(task.id, result);
        completed.add(task.id);
        cancelled.add(task.id);
        options.onTaskComplete?.(result);
        tryReleaseDependents(task.id);
        signalCompletion();
        return;
      }

      const startedAt = new Date().toISOString();
      const startMs = Date.now();
      options.onTaskStart?.(task);
      activeCount += 1;

      let result: TaskResult<TResult>;
      try {
        const value = await task.run(options.context as TContext);
        // If the abort signal fired while we were running, treat this
        // task as cancelled (the result value is dropped — the runner
        // is cooperatively cancellable).
        const abortedDuringRun = isAborted();
        result = {
          id: task.id,
          name: task.name,
          status: abortedDuringRun ? 'cancelled' : 'succeeded',
          value: abortedDuringRun ? undefined : value,
          durationMs: Date.now() - startMs,
          startedAt,
          finishedAt: new Date().toISOString(),
        };
      } catch (err) {
        result = {
          id: task.id,
          name: task.name,
          status: isAborted() ? 'cancelled' : 'failed',
          error: err instanceof Error ? err : new Error(String(err)),
          durationMs: Date.now() - startMs,
          startedAt,
          finishedAt: new Date().toISOString(),
        };
      }

      results.set(task.id, result);
      completed.add(task.id);
      if (result.status === 'failed') failed.add(task.id);
      if (result.status === 'cancelled') cancelled.add(task.id);
      options.onTaskComplete?.(result);
      activeCount -= 1;
      tryReleaseDependents(task.id);
      signalCompletion();
    };

    // Drive the schedule.
    //
    // We pull up to ``concurrency`` tasks off the ready queue at a time
    // and run them concurrently. The outer loop terminates when the
    // ready queue is empty AND no tasks are running. We use a
    // completion-callback pattern (signalCompletion) to wait for the
    // next task to finish; ``Promise.race`` over the in-flight array
    // would resolve immediately on any already-settled promise and
    // spin the event loop.
    while (ready.length > 0 || activeCount > 0) {
      // Fill worker slots. Don't schedule new tasks if aborted — the
      // post-loop cleanup marks unstarted tasks as cancelled.
      while (activeCount < concurrency && ready.length > 0 && !isAborted()) {
        const nextId = ready.shift()!;
        // runTask increments activeCount synchronously before its first
        // await, so the next iteration of this inner loop sees the
        // updated count and respects the concurrency bound.
        void runTask(nextId);
      }

      if (activeCount === 0) {
        // Nothing is running. Either we're done, or we just aborted
        // and have unstarted tasks waiting. The post-loop cleanup
        // handles the latter.
        break;
      }

      // Wait for the next task to finish.
      await waitForNextCompletion();
    }

    // Mark any tasks left in the ready queue (typically because of
    // abort) as cancelled. We walk downstream via tryReleaseDependents
    // so that tasks blocked on these also end up cancelled rather
    // than orphaned in the queue.
    while (ready.length > 0) {
      const pendingId = ready.shift()!;
      const task = byId.get(pendingId)!;
      const cancelledResult: TaskResult<TResult> = {
        id: task.id,
        name: task.name,
        status: 'cancelled',
        durationMs: 0,
        finishedAt: new Date().toISOString(),
      };
      results.set(task.id, cancelledResult);
      completed.add(task.id);
      cancelled.add(task.id);
      options.onTaskComplete?.(cancelledResult);
      tryReleaseDependents(task.id);
    }

    const succeeded: TaskId[] = [];
    const failedIds: TaskId[] = [];
    const skipped: TaskId[] = [];
    const cancelledIds: TaskId[] = [];
    for (const [id, result] of results) {
      if (result.status === 'succeeded') succeeded.push(id);
      else if (result.status === 'failed') failedIds.push(id);
      else if (result.status === 'skipped') skipped.push(id);
      else if (result.status === 'cancelled') cancelledIds.push(id);
    }

    return {
      results,
      succeeded,
      failed: failedIds,
      skipped,
      cancelled: cancelledIds,
      totalDurationMs: Date.now() - t0,
      // ok is true only when the run completed without failures, skips,
      // cancellations, or an abort signal — a run that was aborted is
      // not ok even if all started tasks happened to succeed.
      ok: !isAborted() && failedIds.length === 0 && cancelledIds.length === 0,
    };
  }
}

// ---------------------------------------------------------------------------
// Pure helpers (exported for direct unit testing)
// ---------------------------------------------------------------------------

/**
 * Validate the DAG of tasks. Throws on:
 * - duplicate IDs
 * - self-dependency
 * - unknown dependency
 * - cycle
 */
export function validateDag<TResult = unknown, TContext = unknown>(
  tasks: TaskSpec<TResult, TContext>[],
): void {
  const seen = new Set<TaskId>();
  const ids = new Set<TaskId>();
  for (const task of tasks) {
    if (seen.has(task.id)) {
      throw new Error(`Duplicate task id: ${task.id}`);
    }
    seen.add(task.id);
    ids.add(task.id);
  }
  for (const task of tasks) {
    for (const dep of task.dependsOn ?? []) {
      if (dep === task.id) {
        throw new Error(`Task '${task.id}' depends on itself`);
      }
      if (!ids.has(dep)) {
        throw new Error(`Task '${task.id}' depends on unknown task '${dep}'`);
      }
    }
  }
  const cycle = findCycle(tasks);
  if (cycle) {
    throw new Error(`Cycle detected in task DAG: ${cycle.join(' -> ')}`);
  }
}

/**
 * Find a cycle in the task DAG using iterative DFS. Returns the cycle as
 * a list of task IDs in dependency order, or ``null`` if acyclic.
 */
export function findCycle<TResult = unknown, TContext = unknown>(
  tasks: TaskSpec<TResult, TContext>[],
): TaskId[] | null {
  const adj = new Map<TaskId, TaskId[]>();
  for (const task of tasks) {
    adj.set(task.id, task.dependsOn ?? []);
  }

  const WHITE = 0;
  const GRAY = 1;
  const BLACK = 2;
  const color = new Map<TaskId, number>();
  for (const id of adj.keys()) color.set(id, WHITE);

  const stack: TaskId[] = [];
  const stackSet = new Set<TaskId>();

  const visit = (id: TaskId): TaskId[] | null => {
    color.set(id, GRAY);
    stack.push(id);
    stackSet.add(id);
    for (const dep of adj.get(id) ?? []) {
      const c = color.get(dep) ?? WHITE;
      if (c === GRAY) {
        // Found a back-edge; extract the cycle from the stack.
        const cycleStart = stack.indexOf(dep);
        return stack.slice(cycleStart).concat([dep]);
      }
      if (c === WHITE) {
        const result = visit(dep);
        if (result) return result;
      }
    }
    stack.pop();
    stackSet.delete(id);
    color.set(id, BLACK);
    return null;
  };

  for (const id of adj.keys()) {
    if ((color.get(id) ?? WHITE) === WHITE) {
      const result = visit(id);
      if (result) return result;
    }
  }
  return null;
}

/**
 * Group tasks into layers where each layer can run in parallel with
 * other tasks in the same layer. Layer 0 has no dependencies; layer N
 * depends only on tasks in earlier layers.
 */
export function topologicalLayers<TResult = unknown, TContext = unknown>(
  tasks: TaskSpec<TResult, TContext>[],
): TaskSpec<TResult, TContext>[][] {
  validateDag(tasks);
  const layerOf = new Map<TaskId, number>();
  const layers: TaskSpec<TResult, TContext>[][] = [];

  for (const task of tasks) {
    const deps = task.dependsOn ?? [];
    const myLayer = deps.length === 0
      ? 0
      : Math.max(...deps.map((d) => (layerOf.get(d) ?? 0) + 1));
    layerOf.set(task.id, myLayer);
    if (!layers[myLayer]) layers[myLayer] = [];
    layers[myLayer].push(task);
  }
  return layers;
}

/**
 * Return tasks in a valid topological execution order. Useful for
 * callers that want to stream execution (e.g. logging each layer).
 */
export function topologicalOrder<TResult = unknown, TContext = unknown>(
  tasks: TaskSpec<TResult, TContext>[],
): TaskSpec<TResult, TContext>[] {
  return topologicalLayers(tasks).flat();
}
