import { mkdtemp } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import { describe, expect, it } from 'vitest';

import { RunStore, type RunSnapshot } from './run-store.js';

describe('RunStore', () => {
  it('saves and loads snapshots', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'sdc-run-store-'));
    const store = new RunStore(dir);

    const snapshot: RunSnapshot = {
      runId: 'run-1',
      repoUrl: '/tmp/repo',
      mode: 'autonomous',
      status: 'running',
      currentPhase: 2,
      phaseResults: { 1: { ok: true } },
      context: { repoPath: '/tmp/repo' },
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    await store.save(snapshot);

    const loaded = await store.get('run-1');
    expect(loaded).not.toBeNull();
    expect(loaded?.runId).toBe('run-1');
    expect(loaded?.currentPhase).toBe(2);
  });

  it('filters snapshots by status', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'sdc-run-store-'));
    const store = new RunStore(dir);

    const now = new Date().toISOString();

    await store.save({
      runId: 'run-a',
      repoUrl: '/tmp/repo-a',
      mode: 'autonomous',
      status: 'completed',
      currentPhase: 6,
      phaseResults: {},
      context: {},
      createdAt: now,
      updatedAt: now,
    });

    await store.save({
      runId: 'run-b',
      repoUrl: '/tmp/repo-b',
      mode: 'step_approval',
      status: 'running',
      currentPhase: 3,
      phaseResults: {},
      context: {},
      createdAt: now,
      updatedAt: now,
    });

    const running = await store.listByStatus(['running']);
    expect(running).toHaveLength(1);
    expect(running[0]?.runId).toBe('run-b');
  });
});
