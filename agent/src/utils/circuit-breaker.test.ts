import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  CircuitBreaker,
  CircuitBreakerRegistry,
  CircuitOpenError,
  CircuitState,
  circuitRegistry,
  withCircuitBreaker,
} from './circuit-breaker.js';

describe('CircuitBreaker', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it('starts in Closed state', () => {
    const cb = new CircuitBreaker('test');
    expect(cb.isClosed()).toBe(true);
    expect(cb.isOpen()).toBe(false);
    expect(cb.getStats().state).toBe(CircuitState.Closed);
  });

  it('executes function successfully when closed', async () => {
    const cb = new CircuitBreaker('test');
    const result = await cb.call(async () => 'success');
    expect(result).toBe('success');
  });

  it('tracks success stats', async () => {
    const cb = new CircuitBreaker('test');
    await cb.call(async () => 'ok');
    const stats = cb.getStats();
    expect(stats.totalCalls).toBe(1);
    expect(stats.totalSuccesses).toBe(1);
    expect(stats.totalFailures).toBe(0);
  });

  it('tracks failure stats', async () => {
    const cb = new CircuitBreaker('test', 3);
    try {
      await cb.call(async () => { throw new Error('boom'); });
    } catch { /* expected */ }
    const stats = cb.getStats();
    expect(stats.totalCalls).toBe(1);
    expect(stats.totalFailures).toBe(1);
    expect(stats.totalSuccesses).toBe(0);
    expect(stats.lastFailureMessage).toBe('boom');
    expect(stats.lastFailureTime).toBeGreaterThan(0);
  });

  it('throws CircuitOpenError when circuit is open', async () => {
    const cb = new CircuitBreaker('test', 2);
    for (let i = 0; i < 2; i++) {
      try { await cb.call(async () => { throw new Error('fail'); }); }
      catch { /* expected */ }
    }

    await expect(cb.call(async () => 'should not run')).rejects.toThrow(CircuitOpenError);
  });

  it('rejects with message containing circuit name and last failure', async () => {
    const cb = new CircuitBreaker('my-circuit', 2);
    for (let i = 0; i < 2; i++) {
      try { await cb.call(async () => { throw new Error('timeout'); }); }
      catch { /* expected */ }
    }

    try {
      await cb.call(async () => 'x');
    } catch (err) {
      expect(err).toBeInstanceOf(CircuitOpenError);
      expect((err as CircuitOpenError).message).toContain('my-circuit');
      expect((err as CircuitOpenError).message).toContain('timeout');
    }
  });

  it('transitions to HalfOpen after recovery timeout', async () => {
    const cb = new CircuitBreaker('test', 2, 5000);
    for (let i = 0; i < 2; i++) {
      try { await cb.call(async () => { throw new Error('fail'); }); }
      catch { /* expected */ }
    }

    expect(cb.getStats().state).toBe(CircuitState.Open);

    vi.advanceTimersByTime(5000);
    cb.isOpen(); // triggers maybeTransition

    expect(cb.getStats().state).toBe(CircuitState.HalfOpen);
  });

  it('resets circuit after enough successes in half-open', async () => {
    const cb = new CircuitBreaker('test', 2, 1000, 2);
    // Trip the circuit
    for (let i = 0; i < 2; i++) {
      try { await cb.call(async () => { throw new Error('fail'); }); }
      catch { /* expected */ }
    }

    // Recovery timeout
    vi.advanceTimersByTime(1000);

    // First half-open success
    await cb.call(async () => 'ok1');
    expect(cb.getStats().state).toBe(CircuitState.HalfOpen);

    // Second half-open success resets
    await cb.call(async () => 'ok2');
    expect(cb.getStats().state).toBe(CircuitState.Closed);
  });

  it('re-trips on failure in half-open state', async () => {
    const cb = new CircuitBreaker('test', 2, 1000);
    for (let i = 0; i < 2; i++) {
      try { await cb.call(async () => { throw new Error('fail'); }); }
      catch { /* expected */ }
    }

    vi.advanceTimersByTime(1000);

    // Half-open call fails
    try { await cb.call(async () => { throw new Error('half-open fail'); }); }
    catch { /* expected */ }

    expect(cb.getStats().state).toBe(CircuitState.Open);
  });

  it('closes after enough half-open successes with max calls limit', async () => {
    const cb = new CircuitBreaker('test', 2, 1000, 1);
    for (let i = 0; i < 2; i++) {
      try { await cb.call(async () => { throw new Error('fail'); }); }
      catch { /* expected */ }
    }

    vi.advanceTimersByTime(1000);

    // Single success in half-open with halfOpenMaxCalls=1 should close
    await cb.call(async () => 'ok');
    expect(cb.getStats().state).toBe(CircuitState.Closed);
  });

  it('forceOpen trips the circuit immediately', () => {
    const cb = new CircuitBreaker('test');
    expect(cb.isClosed()).toBe(true);

    cb.forceOpen();
    expect(cb.isOpen()).toBe(true);
    expect(cb.getStats().state).toBe(CircuitState.Open);
  });

  it('forceClose resets the circuit immediately', () => {
    const cb = new CircuitBreaker('test', 2);
    // Trip it first - must use try/catch since call is async
    // We'll test forceClose independently
    cb.forceOpen();
    expect(cb.isOpen()).toBe(true);

    cb.forceClose();
    expect(cb.isClosed()).toBe(true);
    expect(cb.getStats().failureCount).toBe(0);
  });

  it('getStats returns correct CircuitStats shape', () => {
    const cb = new CircuitBreaker('stats-test', 3, 10000);
    const stats = cb.getStats();

    expect(stats).toHaveProperty('state');
    expect(stats).toHaveProperty('failureCount');
    expect(stats).toHaveProperty('successCount');
    expect(stats).toHaveProperty('lastFailureTime');
    expect(stats).toHaveProperty('lastFailureMessage');
    expect(stats).toHaveProperty('openedAt');
    expect(stats).toHaveProperty('totalCalls');
    expect(stats).toHaveProperty('totalFailures');
    expect(stats).toHaveProperty('totalSuccesses');
    expect(stats.state).toBe(CircuitState.Closed);
  });

  it('does not execute the wrapped function when circuit is open', async () => {
    const cb = new CircuitBreaker('test', 1);
    try { await cb.call(async () => { throw new Error('fail'); }); }
    catch { /* expected */ }

    const fn = vi.fn(async () => 'should not be called');
    await expect(cb.call(fn)).rejects.toThrow(CircuitOpenError);
    expect(fn).not.toHaveBeenCalled();
  });

  it('handles zero failure threshold', async () => {
    const cb = new CircuitBreaker('zero', 0);
    try { await cb.call(async () => { throw new Error('fail'); }); }
    catch { /* expected */ }
    expect(cb.isOpen()).toBe(true);
  });
});

describe('CircuitBreakerRegistry', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('creates a new circuit breaker', () => {
    const registry = new CircuitBreakerRegistry();
    const cb = registry.getOrCreate('my-service', 3, 5000);
    expect(cb).toBeInstanceOf(CircuitBreaker);
    expect(cb.isClosed()).toBe(true);
  });

  it('returns existing circuit breaker for same name', () => {
    const registry = new CircuitBreakerRegistry();
    const cb1 = registry.getOrCreate('same', 3, 5000);
    const cb2 = registry.getOrCreate('same', 10, 9999);
    expect(cb1).toBe(cb2);
  });

  it('get returns undefined for unknown name', () => {
    const registry = new CircuitBreakerRegistry();
    expect(registry.get('nonexistent')).toBeUndefined();
  });

  it('get returns existing breaker', () => {
    const registry = new CircuitBreakerRegistry();
    registry.getOrCreate('known');
    expect(registry.get('known')).toBeDefined();
  });

  it('getAllStats returns stats for all breakers', () => {
    const registry = new CircuitBreakerRegistry();
    registry.getOrCreate('a');
    registry.getOrCreate('b');
    const stats = registry.getAllStats();
    expect(Object.keys(stats)).toEqual(['a', 'b']);
    expect(stats['a'].state).toBe(CircuitState.Closed);
    expect(stats['b'].state).toBe(CircuitState.Closed);
  });

  it('getHealth reports healthy when no circuits are open', () => {
    const registry = new CircuitBreakerRegistry();
    registry.getOrCreate('a');
    registry.getOrCreate('b');
    const health = registry.getHealth();
    expect(health.healthy).toBe(true);
    expect(health.totalCircuits).toBe(2);
    expect(health.openCircuits).toEqual([]);
  });

  it('getHealth reports unhealthy when circuits are open', async () => {
    const registry = new CircuitBreakerRegistry();
    const cb = registry.getOrCreate('faulty', 1);
    try { await cb.call(async () => { throw new Error('fail'); }); }
    catch { /* expected */ }

    const health = registry.getHealth();
    expect(health.healthy).toBe(false);
    expect(health.openCircuits).toContain('faulty');
  });
});

describe('withCircuitBreaker', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('executes function via registry and returns result', async () => {
    const result = await withCircuitBreaker('util-fn', async () => 'done');
    expect(result).toBe('done');
  });

  it('throws CircuitOpenError when circuit is open', async () => {
    // Trip the circuit (5 failures by default)
    for (let i = 0; i < 5; i++) {
      try { await withCircuitBreaker('fail-fn', async () => { throw new Error(`fail ${i}`); }); }
      catch { /* expected */ }
    }

    await expect(
      withCircuitBreaker('fail-fn', async () => 'should not run')
    ).rejects.toThrow(CircuitOpenError);
  });

  it('accepts custom failureThreshold and recoveryTimeout options', async () => {
    const result = await withCircuitBreaker(
      'custom-fn',
      async () => 'custom-ok',
      { failureThreshold: 10, recoveryTimeout: 60000 }
    );
    expect(result).toBe('custom-ok');
  });
});
