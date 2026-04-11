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

import { ConvexClientWrapper } from './convex.js';

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
    await vi.advanceTimersByTimeAsync(5000);

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
    await vi.advanceTimersByTimeAsync(5000);

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
    await vi.advanceTimersByTimeAsync(5000);

    await expect(pending).resolves.toBe(false);
  });
});
