import { beforeEach, describe, expect, it, vi } from 'vitest';

const queryMock = vi.fn();
const mutationMock = vi.fn();

vi.mock('convex/browser', () => ({
  ConvexHttpClient: class {
    public query = queryMock;
    public mutation = mutationMock;
  },
}));

vi.mock('../utils/logger.js', () => ({
  logger: {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

vi.mock('../utils/config.js', () => ({
  config: {
    convex: { deploymentUrl: 'https://example.convex.cloud' },
  },
}));

import { ConvexClientWrapper, defaultIsTransient } from './convex.js';

describe('ConvexClientWrapper auth and approval semantics', () => {
  beforeEach(() => {
    process.env.SCHOLARDEVCLAW_CONVEX_AUTH_KEY = 'convex-secret';
    queryMock.mockReset();
    mutationMock.mockReset();
    vi.useRealTimers();
  });

  it('adds convex auth key to every query and mutation payload', async () => {
    const client = new ConvexClientWrapper('https://deployment.example');

    queryMock.mockResolvedValueOnce(null);
    mutationMock.mockResolvedValueOnce(undefined);

    await client.getIntegration('integration-id' as string);
    await client.updateStatus('integration-id', 'pending');

    expect(queryMock).toHaveBeenCalledWith('integrations:get', {
      id: 'integration-id',
      authKey: 'convex-secret',
    });
    expect(mutationMock).toHaveBeenCalledWith(
      'integrations:updateStatus',
      expect.objectContaining({
        id: 'integration-id',
        status: 'pending',
        authKey: 'convex-secret',
      }),
    );
  });

  it('wraps typed phase results in the savePhaseResult payload contract', async () => {
    const client = new ConvexClientWrapper('https://deployment.example');

    mutationMock.mockResolvedValueOnce(undefined);

    await client.savePhaseResult('integration-id', 6, {
      metadata: {
        integrationId: 'integration-id',
        repoUrl: '/repo',
        paper: 'Attention Is All You Need',
        algorithm: 'FlashAttention',
        createdAt: '2026-05-25T00:00:00.000Z',
      },
      summary: {
        status: 'completed',
        confidence: 95,
        changesMade: 2,
        filesModified: ['src/model.py'],
        newFiles: ['src/flash_attention.py'],
      },
      whatChanged: 'Replaced attention blocks.',
      why: 'Improves throughput.',
      observedImpact: {
        metricsComparison: {
          speedup: 1.2,
          numerical_correctness: {
            status: 'passed',
          },
        },
        meetsExpectations: true,
      },
      riskNotes: [],
      diffPreview: 'Modified: src/model.py',
      testResults: {
        unitTestsPassed: true,
        benchmarkResults: {
          speedup: 1.2,
        },
      },
      recommendation: {
        action: 'approve',
        confidence: 95,
        notes: 'Ready for integration.',
      },
    });

    expect(mutationMock).toHaveBeenCalledWith(
      'integrations:savePhaseResult',
      expect.objectContaining({
        id: 'integration-id',
        payload: {
          field: 'phase6Result',
          result: expect.objectContaining({
            diffPreview: 'Modified: src/model.py',
            recommendation: expect.objectContaining({ action: 'approve' }),
          }),
        },
        authKey: 'convex-secret',
      }),
    );
  });

  it('waitForApproval only resolves true on explicit approved action for phase', async () => {
    vi.useFakeTimers();
    const client = new ConvexClientWrapper('https://deployment.example');

    queryMock.mockImplementation(async (name: string) => {
      if (name === 'integrations:get') {
        return { status: 'awaiting_approval' };
      }
      if (name === 'integrations:listApprovals') {
        return [{ phase: 3, action: 'approved' }];
      }
      return null;
    });

    const pending = client.waitForApproval('integration-id', 3);
    await vi.advanceTimersByTime(5000);

    await expect(pending).resolves.toBe(true);
  });

  it('waitForApproval resolves false when status changes without approval', async () => {
    vi.useFakeTimers();
    const client = new ConvexClientWrapper('https://deployment.example');

    queryMock.mockImplementation(async (name: string) => {
      if (name === 'integrations:get') {
        return { status: 'completed' };
      }
      if (name === 'integrations:listApprovals') {
        return [];
      }
      return null;
    });

    const pending = client.waitForApproval('integration-id', 2);
    await vi.advanceTimersByTime(5000);

    await expect(pending).resolves.toBe(false);
  });

  it('waitForApproval resolves false when rejected approval exists', async () => {
    vi.useFakeTimers();
    const client = new ConvexClientWrapper('https://deployment.example');

    queryMock.mockImplementation(async (name: string) => {
      if (name === 'integrations:get') {
        return { status: 'awaiting_approval' };
      }
      if (name === 'integrations:listApprovals') {
        return [{ phase: 2, action: 'rejected' }];
      }
      return null;
    });

    const pending = client.waitForApproval('integration-id', 2);
    await vi.advanceTimersByTime(5000);

    await expect(pending).resolves.toBe(false);
  });

  describe('retry behavior', () => {
    it('retries transient errors and eventually succeeds', async () => {
      const client = new ConvexClientWrapper('https://deployment.example');

      mutationMock
        .mockRejectedValueOnce(new Error('Network unstable'))
        .mockRejectedValueOnce(new Error('Network unstable'))
        .mockResolvedValueOnce('ok');

      const result = await client.setError('integration-id', 'boom');
      expect(result).toBeUndefined();
      expect(mutationMock).toHaveBeenCalledTimes(3);
    });

    it('does not retry permanent errors', async () => {
      const client = new ConvexClientWrapper('https://deployment.example');

      mutationMock.mockRejectedValue(new Error('Unauthorized: bad token'));

      await expect(
        client.setError('integration-id', 'boom'),
      ).rejects.toThrow('Unauthorized');
      expect(mutationMock).toHaveBeenCalledTimes(1);
    });

    it('gives up after maxAttempts on persistent transient errors', async () => {
      const client = new ConvexClientWrapper('https://deployment.example');

      mutationMock.mockRejectedValue(new Error('ETIMEDOUT'));

      await expect(
        client.setError('integration-id', 'boom'),
      ).rejects.toThrow('ETIMEDOUT');
      expect(mutationMock).toHaveBeenCalledTimes(3); // default maxAttempts
    });

    it('respects custom maxAttempts', async () => {
      const client = new ConvexClientWrapper('https://deployment.example');

      mutationMock.mockRejectedValue(new Error('rate limit 429'));

      await expect(
        client.setError('integration-id', 'boom', { maxAttempts: 2 }),
      ).rejects.toThrow('rate limit');
      expect(mutationMock).toHaveBeenCalledTimes(2);
    });
  });

  describe('query caching', () => {
    it('returns cached getIntegration within TTL', async () => {
      const client = new ConvexClientWrapper('https://deployment.example');
      const integration = { _id: 'i1', status: 'pending' } as any;

      queryMock.mockResolvedValue(integration);

      const first = await client.getIntegration('i1');
      const second = await client.getIntegration('i1');

      expect(first).toBe(integration);
      expect(second).toBe(integration);
      expect(queryMock).toHaveBeenCalledTimes(1);
    });

    it('refetches after cache invalidation from mutation', async () => {
      const client = new ConvexClientWrapper('https://deployment.example');

      queryMock
        .mockResolvedValueOnce({ _id: 'i1', status: 'pending' })
        .mockResolvedValueOnce({ _id: 'i1', status: 'completed' });

      mutationMock.mockResolvedValue(undefined);

      const first = await client.getIntegration('i1');
      expect(first).not.toBeNull();
      expect(first!.status).toBe('pending');

      await client.setError('i1', 'finished');

      const second = await client.getIntegration('i1');
      expect(second).not.toBeNull();
      expect(second!.status).toBe('completed');
      expect(queryMock).toHaveBeenCalledTimes(2);
    });

    it('clearQueryCache forces a refetch', async () => {
      const client = new ConvexClientWrapper('https://deployment.example');
      const a = { _id: 'i1' } as any;
      const b = { _id: 'i1' } as any;

      queryMock.mockResolvedValueOnce(a).mockResolvedValueOnce(b);

      await client.getIntegration('i1');
      client.clearQueryCache();
      await client.getIntegration('i1');

      expect(queryMock).toHaveBeenCalledTimes(2);
    });
  });

  describe('saveLogBatch', () => {
    it('sends all messages in one mutation', async () => {
      const client = new ConvexClientWrapper('https://deployment.example');
      mutationMock.mockResolvedValueOnce(undefined);

      await client.saveLogBatch('integration-id', ['a', 'b', 'c']);

      expect(mutationMock).toHaveBeenCalledWith(
        'integrations:saveLogBatch',
        expect.objectContaining({
          id: 'integration-id',
          entries: expect.arrayContaining([
            expect.objectContaining({ message: 'a' }),
            expect.objectContaining({ message: 'b' }),
            expect.objectContaining({ message: 'c' }),
          ]),
        }),
      );
    });

    it('skips the mutation when no messages are provided', async () => {
      const client = new ConvexClientWrapper('https://deployment.example');
      await client.saveLogBatch('integration-id', []);
      expect(mutationMock).not.toHaveBeenCalled();
    });

    it('honors explicit per-message timestamps', async () => {
      const client = new ConvexClientWrapper('https://deployment.example');
      mutationMock.mockResolvedValueOnce(undefined);

      await client.saveLogBatch('integration-id', ['a', 'b'], {
        timestamps: [100, 200],
      });

      expect(mutationMock).toHaveBeenCalledWith(
        'integrations:saveLogBatch',
        expect.objectContaining({
          entries: [
            { message: 'a', timestamp: 100 },
            { message: 'b', timestamp: 200 },
          ],
        }),
      );
    });
  });

  describe('approval polling with backoff', () => {
    it('uses exponential backoff for consecutive polls', async () => {
      vi.useFakeTimers();
      const client = new ConvexClientWrapper('https://deployment.example');

      let polls = 0;
      queryMock.mockImplementation(async (name: string) => {
        if (name === 'integrations:get') {
          polls += 1;
          if (polls >= 3) {
            return { status: 'awaiting_approval' };
          }
          return { status: 'awaiting_approval' };
        }
        if (name === 'integrations:listApprovals') {
          if (polls >= 3) {
            return [{ phase: 1, action: 'approved' }];
          }
          return [];
        }
        return null;
      });

      const pending = client.waitForApproval('integration-id', 1, {
        initialIntervalMs: 50,
        maxIntervalMs: 500,
        backoffMultiplier: 2,
        timeoutMs: 60000,
      });

      // Drive the fake clock a single large step, matching the pattern used
      // by other passing tests in this file.
      await vi.advanceTimersByTimeAsync(2000);

      await expect(pending).resolves.toBe(true);
      expect(polls).toBeGreaterThanOrEqual(3);
    }, 10000);

    it('honors timeout', async () => {
      vi.useFakeTimers();
      const client = new ConvexClientWrapper('https://deployment.example');

      queryMock.mockImplementation(async (name: string) => {
        if (name === 'integrations:get') {
          return { status: 'awaiting_approval' };
        }
        if (name === 'integrations:listApprovals') {
          return [];
        }
        return null;
      });

      const pending = client.waitForApproval('integration-id', 1, {
        initialIntervalMs: 100,
        maxIntervalMs: 1000,
        backoffMultiplier: 1.5,
        timeoutMs: 250,
      });

      await vi.advanceTimersByTimeAsync(5000);

      await expect(pending).resolves.toBe(false);
    });
  });

  describe('defaultIsTransient', () => {
    it('classifies network errors as transient', () => {
      expect(defaultIsTransient(new Error('ECONNRESET'))).toBe(true);
      expect(defaultIsTransient(new Error('Network unreachable'))).toBe(true);
      expect(defaultIsTransient(new Error('request timeout'))).toBe(true);
    });

    it('classifies 5xx and rate limits as transient', () => {
      expect(defaultIsTransient(new Error('Internal Server Error 500'))).toBe(true);
      expect(defaultIsTransient(new Error('rate limit exceeded'))).toBe(true);
      expect(defaultIsTransient(new Error('429 Too Many Requests'))).toBe(true);
    });

    it('classifies auth/validation errors as permanent', () => {
      expect(defaultIsTransient(new Error('Unauthorized'))).toBe(false);
      expect(defaultIsTransient(new Error('Bad request: invalid field'))).toBe(false);
      expect(defaultIsTransient(null)).toBe(false);
    });
  });
});
