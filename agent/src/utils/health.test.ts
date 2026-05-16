import { describe, it, expect } from 'vitest';
import { HealthChecker, LivenessProbe, ReadinessProbe } from './health.js';

describe('HealthChecker', () => {
  it('reports memory usage accurately', () => {
    const checker = new HealthChecker('test');
    const status = checker.runCheck('memory');

    expect(status.name).toBe('memory');
    expect(status.healthy).toBe(true);
    expect(status.message).toContain('Heap:');
    expect(status.details?.heapUsedMB).toBeDefined();
    expect(status.details?.heapTotalMB).toBeDefined();
    expect(status.details?.percentUsed).toBeDefined();
  });

  it('reports event loop lag with async measurement', () => {
    const checker = new HealthChecker('test');

    // First call: should report "calibrating" since no measurement exists yet
    const firstCall = checker.runCheck('event_loop');
    expect(firstCall.name).toBe('event_loop');
    expect(firstCall.details?.checkCount).toBe(1);

    // Second call: should have started measurement
    const secondCall = checker.runCheck('event_loop');
    expect(secondCall.details?.checkCount).toBe(2);
    expect(secondCall.healthy).toBe(true);
    // Lag should be either null (first measurement in flight) or a reasonable number
    const lag = secondCall.details?.lagMs as number | null;
    expect(lag === null || (typeof lag === 'number' && lag >= 0)).toBe(true);
  });

  it('checks python bridge asynchronously', () => {
    const checker = new HealthChecker('test');

    // First call: should report "checking" since no measurement exists yet
    const firstCall = checker.runCheck('python_bridge');
    expect(firstCall.name).toBe('python_bridge');
    expect(firstCall.details?.checked).toBe(false);
    expect(firstCall.message).toContain('checking');

    // Default mode is HTTP, which will fail to connect to localhost:8000
    // but should still complete the async check eventually
  });

  it('runAllChecks returns complete health report', () => {
    const checker = new HealthChecker('test');
    const health = checker.runAllChecks();

    expect(health.checks.length).toBe(3);
    expect(health.overallHealthy).toBeDefined();
    expect(health.uptimeSeconds).toBeGreaterThanOrEqual(0);
    expect(health.version).toBe('test');
    expect(health.nodeVersion).toBe(process.version);
    expect(health.platform).toBe(process.platform);
  });

  it('runQuickCheck uses critical checks only', () => {
    const checker = new HealthChecker('test');
    const result = checker.runQuickCheck();
    expect(typeof result).toBe('boolean');
  });

  it('returns false for unknown check name', () => {
    const checker = new HealthChecker('test');
    const status = checker.runCheck('nonexistent');

    expect(status.healthy).toBe(false);
    expect(status.message).toContain('Unknown check');
  });

  it('registers custom checks', () => {
    const checker = new HealthChecker('test');
    checker.registerCheck('custom', () => ({
      name: 'custom',
      healthy: true,
      message: 'custom check passed',
      timestamp: new Date().toISOString(),
    }));

    const status = checker.runCheck('custom');
    expect(status.healthy).toBe(true);
    expect(status.message).toBe('custom check passed');
  });
});

describe('LivenessProbe', () => {
  it('starts alive', () => {
    const probe = new LivenessProbe(30000);
    expect(probe.isAlive()).toBe(true);
  });

  it('reports death after timeout', () => {
    const probe = new LivenessProbe(0);
    expect(probe.isAlive()).toBe(false);
  });

  it('check returns structured output', () => {
    const probe = new LivenessProbe(30000);
    probe.heartbeat();

    const result = probe.check();
    expect(result.alive).toBe(true);
    expect(result.lastHeartbeat).toBeDefined();
    expect(result.secondsSinceHeartbeat).toBeGreaterThanOrEqual(0);
  });
});

describe('ReadinessProbe', () => {
  it('starts ready', () => {
    const probe = new ReadinessProbe();
    expect(probe.isReady()).toBe(true);
  });

  it('can be marked not ready with reason', () => {
    const probe = new ReadinessProbe();
    probe.setReady(false, 'initializing');

    expect(probe.isReady()).toBe(false);
    expect(probe.check().reasons).toContain('initializing');
  });

  it('clears reasons when marked ready again', () => {
    const probe = new ReadinessProbe();
    probe.setReady(false, 'initializing');
    probe.setReady(true);

    expect(probe.isReady()).toBe(true);
    expect(probe.check().reasons).toHaveLength(0);
  });
});
