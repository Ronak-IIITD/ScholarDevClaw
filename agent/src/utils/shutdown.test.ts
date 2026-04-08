import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { GracefulShutdown, ShutdownError } from './shutdown.js';

vi.mock('./logger.js', () => ({
  logger: {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  },
}));

describe('GracefulShutdown', () => {
  let shutdown: GracefulShutdown;

  beforeEach(() => {
    shutdown = new GracefulShutdown(100);
    // Prevent process.exit from killing test runner
    vi.spyOn(process, 'exit').mockImplementation((() => {}) as any);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('registers and calls handlers in priority order', async () => {
    const calls: string[] = [];
    shutdown.registerHandler('low', () => {
      calls.push('low');
    }, 0);
    shutdown.registerHandler('high', () => {
      calls.push('high');
    }, 10);

    await shutdown.shutdown('test');

    expect(calls).toEqual(['high', 'low']);
  });

  it('is idempotent — second shutdown returns same promise', async () => {
    const handler = vi.fn();
    shutdown.registerHandler('once', handler);

    const p1 = shutdown.shutdown('first');
    const p2 = shutdown.shutdown('second');

    expect(p1).toStrictEqual(p2);
    await p1;
    expect(handler).toHaveBeenCalledTimes(1);
  });

  it('checkShutdown throws after shutdown starts', async () => {
    shutdown.registerHandler('block', () => {});

    const promise = shutdown.shutdown('maintenance');

    expect(() => shutdown.checkShutdown()).toThrow(ShutdownError);
    await promise;
  });

  it('handles handler errors without aborting shutdown', async () => {
    const good = vi.fn();
    const bad = vi.fn().mockImplementation(() => {
      throw new Error('boom');
    });

    shutdown.registerHandler('bad', bad);
    shutdown.registerHandler('good', good);

    await shutdown.shutdown('test');

    expect(bad).toHaveBeenCalled();
    expect(good).toHaveBeenCalled();
  });

  it('respects timeout and stops handler loop', async () => {
    // Use a very short shutdown timeout (10ms) so the loop exits quickly
    const fastShutdown = new GracefulShutdown(10);
    vi.spyOn(process, 'exit').mockImplementation((() => {}) as any);

    const slow = vi.fn().mockImplementation(async () => {
      // This will be interrupted by the 5000ms handler race timeout,
      // but the shutdown timeout (10ms) should stop the loop first
      await new Promise((r) => setTimeout(r, 10000));
    });
    const fast = vi.fn();

    fastShutdown.registerHandler('slow', slow);
    fastShutdown.registerHandler('fast', fast);

    await fastShutdown.shutdown('timeout');

    // The slow handler should not complete before timeout
    expect(fast).not.toHaveBeenCalled();
  });
});

describe('withShutdownGuard', () => {
  it('throws when shutdown is active', async () => {
    const { GracefulShutdown: GS } = await import('./shutdown.js');
    const localManager = new GS(1000);
    vi.spyOn(process, 'exit').mockImplementation((() => {}) as any);

    localManager.registerHandler('noop', () => {});
    await localManager.shutdown('test');

    // Test checkShutdown behavior directly (withShutdownGuard uses module singleton)
    expect(() => localManager.checkShutdown()).toThrow(ShutdownError);
  });
});
