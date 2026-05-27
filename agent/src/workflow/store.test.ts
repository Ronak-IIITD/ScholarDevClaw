import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { existsSync } from 'fs';
import { readFile, writeFile, mkdir, unlink, readdir } from 'fs/promises';

import { WorkflowStore, ResumableWorkflow } from './store.js';

// Mock fs/promises and fs
vi.mock('fs/promises');
vi.mock('fs');

// Set up mockGetState before the engine mock
const mockGetState = vi.fn();
vi.mock('./engine.js', () => ({
  DAGEngine: vi.fn().mockImplementation(() => ({
    getState: mockGetState,
  })),
}));

describe('WorkflowStore', () => {
  let store: WorkflowStore;

  beforeEach(() => {
    vi.clearAllMocks();
    store = new WorkflowStore('/tmp/test-store');
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('save', () => {
    it('ensures directory exists before writing', async () => {
      (existsSync as any).mockReturnValue(false);
      (writeFile as any).mockResolvedValue(undefined);
      (mkdir as any).mockResolvedValue(undefined);

      await store.save('wf-1', { key: 'config' }, {
        workflowId: 'wf-1',
        status: 'running',
        startedAt: '2026-01-01T00:00:00.000Z',
        context: {},
        nodeResults: new Map(),
      } as any);

      expect(mkdir).toHaveBeenCalledWith('/tmp/test-store', { recursive: true });
    });

    it('writes a JSON snapshot file', async () => {
      (existsSync as any).mockReturnValue(true);
      (writeFile as any).mockResolvedValue(undefined);

      await store.save('wf-1', { key: 'val' }, {
        workflowId: 'wf-1',
        status: 'running',
        context: { repo: '/test' },
        nodeResults: new Map([['node1', { nodeId: 'node1', status: 'completed', output: 'result' }]]),
      } as any);

      expect(writeFile).toHaveBeenCalledWith(
        '/tmp/test-store/wf-1.json',
        expect.any(String),
      );

      const writtenJson = JSON.parse((writeFile as any).mock.calls[0][1]);
      expect(writtenJson.workflowId).toBe('wf-1');
      expect(writtenJson.status).toBe('running');
      expect(writtenJson.state.context).toEqual({ repo: '/test' });
    });
  });

  describe('load', () => {
    it('returns null when file does not exist', async () => {
      (existsSync as any).mockReturnValue(false);

      const result = await store.load('nonexistent');
      expect(result).toBeNull();
    });

    it('returns parsed snapshot when file exists', async () => {
      (existsSync as any).mockReturnValue(true);
      (readFile as any).mockResolvedValue(JSON.stringify({
        workflowId: 'wf-1',
        status: 'completed',
        config: {},
        state: { workflowId: 'wf-1', status: 'completed', context: {} },
        nodeResults: {},
        savedAt: '2026-01-01T00:00:00.000Z',
      }));

      const result = await store.load('wf-1');
      expect(result).not.toBeNull();
      expect(result!.workflowId).toBe('wf-1');
      expect(result!.status).toBe('completed');
    });

    it('returns null when JSON parse fails', async () => {
      (existsSync as any).mockReturnValue(true);
      (readFile as any).mockResolvedValue('invalid json');

      const result = await store.load('wf-1');
      expect(result).toBeNull();
    });
  });

  describe('list', () => {
    it('returns empty array when readdir fails', async () => {
      (existsSync as any).mockReturnValue(true);
      (readdir as any).mockRejectedValue(new Error('permission denied'));

      const result = await store.list();
      expect(result).toEqual([]);
    });

    it('returns sorted snapshots', async () => {
      (existsSync as any).mockReturnValue(true);
      (readdir as any).mockResolvedValue(['wf-1.json', 'wf-2.json']);
      (readFile as any).mockImplementation((path: string) => {
        if (path.includes('wf-1')) {
          return JSON.stringify({ workflowId: 'wf-1', savedAt: '2026-01-02T00:00:00.000Z' });
        }
        return JSON.stringify({ workflowId: 'wf-2', savedAt: '2026-01-01T00:00:00.000Z' });
      });

      const result = await store.list();
      expect(result).toHaveLength(2);
      expect(result[0].workflowId).toBe('wf-1');
      expect(result[1].workflowId).toBe('wf-2');
    });

    it('skips non-JSON files', async () => {
      (existsSync as any).mockReturnValue(true);
      (readdir as any).mockResolvedValue(['wf-1.json', 'notes.txt']);
      (readFile as any).mockImplementation((path: string) => {
        if (path.endsWith('.json')) {
          return JSON.stringify({ workflowId: 'wf-1', savedAt: '2026-01-01T00:00:00.000Z' });
        }
        return 'not json';
      });

      const result = await store.list();
      expect(result).toHaveLength(1);
    });
  });

  describe('delete', () => {
    it('deletes the JSON file', async () => {
      (existsSync as any).mockReturnValue(true);
      (unlink as any).mockResolvedValue(undefined);

      await store.delete('wf-1');
      expect(unlink).toHaveBeenCalledWith('/tmp/test-store/wf-1.json');
    });

    it('does not throw when file does not exist', async () => {
      (existsSync as any).mockReturnValue(false);

      await expect(store.delete('wf-1')).resolves.toBeUndefined();
    });
  });

  describe('getByStatus', () => {
    it('filters by status', async () => {
      (existsSync as any).mockReturnValue(true);
      (readdir as any).mockResolvedValue(['wf-1.json', 'wf-2.json']);
      (readFile as any).mockImplementation((path: string) => {
        if (path.includes('wf-1')) {
          return JSON.stringify({ workflowId: 'wf-1', status: 'running', savedAt: '2026-01-01T00:00:00.000Z' });
        }
        return JSON.stringify({ workflowId: 'wf-2', status: 'completed', savedAt: '2026-01-02T00:00:00.000Z' });
      });

      const result = await store.getByStatus('running');
      expect(result).toHaveLength(1);
      expect(result[0].workflowId).toBe('wf-1');
    });
  });

  describe('getRecent', () => {
    it('returns up to limit items', async () => {
      (existsSync as any).mockReturnValue(true);
      (readdir as any).mockResolvedValue(['wf-1.json', 'wf-2.json', 'wf-3.json']);
      (readFile as any).mockImplementation((path: string) => {
        const id = path.includes('wf-1') ? 'wf-1' : path.includes('wf-2') ? 'wf-2' : 'wf-3';
        return JSON.stringify({ workflowId: id, savedAt: '2026-01-01T00:00:00.000Z' });
      });

      const result = await store.getRecent(2);
      expect(result).toHaveLength(2);
    });
  });

  describe('validateWorkflowId (internal)', () => {
    it('allows valid IDs', async () => {
      (existsSync as any).mockReturnValue(false);
      await expect(store.load('valid-id')).resolves.toBeNull();
      await expect(store.load('test_123')).resolves.toBeNull();
      await expect(store.load('a.b')).resolves.toBeNull();
    });

    it('rejects IDs with slashes', async () => {
      await expect(store.load('../etc/passwd')).rejects.toThrow('invalid characters');
    });

    it('rejects empty IDs', async () => {
      await expect(store.load('')).rejects.toThrow('must be a non-empty string');
    });

    it('rejects IDs with special characters', async () => {
      await expect(store.load('test$id')).rejects.toThrow('disallowed characters');
    });
  });
});

describe('ResumableWorkflow', () => {
  let store: WorkflowStore;

  beforeEach(async () => {
    vi.clearAllMocks();
    store = new WorkflowStore('/tmp/test-store');
    mockGetState.mockReturnValue({
      workflowId: 'wf-1',
      status: 'running',
      context: {},
      nodeResults: new Map(),
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('save', () => {
    it('gets state from engine and saves to store', async () => {
      (existsSync as any).mockReturnValue(true);
      (writeFile as any).mockResolvedValue(undefined);

      const rw = new ResumableWorkflow({ getState: mockGetState } as any, store, 'wf-1');
      await rw.save();

      expect(mockGetState).toHaveBeenCalled();
      expect(writeFile).toHaveBeenCalled();
    });
  });

  describe('resume', () => {
    it('returns null when snapshot not found', async () => {
      (existsSync as any).mockReturnValue(false);

      const result = await ResumableWorkflow.resume(
        { getState: mockGetState } as any,
        store,
        'nonexistent',
        [],
      );
      expect(result).toBeNull();
    });

    it('restores node results from snapshot', async () => {
      (existsSync as any).mockReturnValue(true);
      (readFile as any).mockResolvedValue(JSON.stringify({
        workflowId: 'wf-1',
        status: 'running',
        config: {},
        state: {
          workflowId: 'wf-1',
          status: 'running',
          startedAt: '2026-01-01T00:00:00.000Z',
          context: { repo: '/test' },
        },
        nodeResults: {
          node1: { nodeId: 'node1', status: 'completed', output: 'result1' },
        },
        savedAt: '2026-01-01T00:00:00.000Z',
      }));

      const fakeNode = { id: 'node1' };
      const result = await ResumableWorkflow.resume(
        { getState: mockGetState } as any,
        store,
        'wf-1',
        [fakeNode as any],
      );

      expect(result).not.toBeNull();
      expect(result!.state.context).toEqual({ repo: '/test' });
      expect(result!.nodeResults.get('node1')?.output).toBe('result1');
    });
  });
});
