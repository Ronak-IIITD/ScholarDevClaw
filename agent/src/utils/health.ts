import { logger } from './logger.js';

export interface HealthStatus {
  name: string;
  healthy: boolean;
  message: string;
  details?: Record<string, unknown>;
  timestamp: string;
}

export interface SystemHealth {
  overallHealthy: boolean;
  checks: HealthStatus[];
  uptimeSeconds: number;
  version: string;
  nodeVersion: string;
  platform: string;
}

export class HealthChecker {
  private startTime: number;
  private checks: Map<string, () => HealthStatus> = new Map();
  private version: string;

  constructor(version: string = '1.0.0') {
    this.startTime = Date.now();
    this.version = version;
    this.registerDefaultChecks();
  }

  private registerDefaultChecks(): void {
    this.registerCheck('memory', this.checkMemory.bind(this));
    this.registerCheck('event_loop', this.checkEventLoop.bind(this));
    this.registerCheck('python_bridge', this.checkPythonBridge.bind(this));
  }

  registerCheck(name: string, checkFn: () => HealthStatus): void {
    this.checks.set(name, checkFn);
  }

  runCheck(name: string): HealthStatus {
    const checkFn = this.checks.get(name);
    if (!checkFn) {
      return {
        name,
        healthy: false,
        message: `Unknown check: ${name}`,
        timestamp: new Date().toISOString(),
      };
    }

    try {
      return checkFn();
    } catch (error) {
      return {
        name,
        healthy: false,
        message: `Check failed: ${error}`,
        timestamp: new Date().toISOString(),
      };
    }
  }

  runAllChecks(): SystemHealth {
    const results: HealthStatus[] = [];

    for (const name of this.checks.keys()) {
      results.push(this.runCheck(name));
    }

    const allHealthy = results.every(r => r.healthy);

    return {
      overallHealthy: allHealthy,
      checks: results,
      uptimeSeconds: (Date.now() - this.startTime) / 1000,
      version: this.version,
      nodeVersion: process.version,
      platform: process.platform,
    };
  }

  runQuickCheck(): boolean {
    const criticalChecks = ['memory', 'event_loop'];
    return criticalChecks.every(name => this.runCheck(name).healthy);
  }

  private checkMemory(): HealthStatus {
    const memUsage = process.memoryUsage();
    const heapUsedMB = memUsage.heapUsed / 1024 / 1024;
    const heapTotalMB = memUsage.heapTotal / 1024 / 1024;
    const rssMB = memUsage.rss / 1024 / 1024;
    const percentUsed = (heapUsedMB / heapTotalMB) * 100;

    const healthy = percentUsed < 90;
    let message = `Heap: ${heapUsedMB.toFixed(1)}MB / ${heapTotalMB.toFixed(1)}MB (${percentUsed.toFixed(1)}%)`;

    if (percentUsed > 80) {
      message += ' [WARNING: High memory usage]';
    }

    return {
      name: 'memory',
      healthy,
      message,
      details: {
        heapUsedMB: Math.round(heapUsedMB * 100) / 100,
        heapTotalMB: Math.round(heapTotalMB * 100) / 100,
        rssMB: Math.round(rssMB * 100) / 100,
        percentUsed: Math.round(percentUsed * 100) / 100,
      },
      timestamp: new Date().toISOString(),
    };
  }

  // --- Event loop lag measurement (async-tracked, sync-reported) ---
  private _lastEventLoopLagMs = -1;
  private _eventLoopCheckCount = 0;
  private _measuringEventLoop = false;

  private _measureEventLoopLag(): void {
    if (this._measuringEventLoop) return;
    this._measuringEventLoop = true;

    const start = process.hrtime.bigint();
    setImmediate(() => {
      const elapsed = Number(process.hrtime.bigint() - start) / 1e6; // ms
      // Exponential moving average to smooth outliers
      if (this._lastEventLoopLagMs < 0) {
        this._lastEventLoopLagMs = elapsed;
      } else {
        this._lastEventLoopLagMs = this._lastEventLoopLagMs * 0.7 + elapsed * 0.3;
      }
      this._measuringEventLoop = false;
    });
  }

  private checkEventLoop(): HealthStatus {
    // Trigger a fresh measurement each time this is called
    this._measureEventLoopLag();
    this._eventLoopCheckCount++;

    // Report the most recent measurement (will be -1 on first call, improving over time)
    const lagMs = this._lastEventLoopLagMs < 0 ? 0 : Math.round(this._lastEventLoopLagMs * 10) / 10;

    // Thresholds: <10ms good, <50ms warning, >50ms degraded
    const healthy = this._lastEventLoopLagMs < 0 || lagMs < 50;
    const isWarning = lagMs >= 10 && lagMs < 50;
    const isDegraded = lagMs >= 50;

    let message: string;
    if (this._lastEventLoopLagMs < 0) {
      message = 'Event loop: calibrating (measurements start on next call)';
    } else if (isDegraded) {
      message = `Event loop lag: ${lagMs.toFixed(1)}ms [DEGRADED]`;
    } else if (isWarning) {
      message = `Event loop lag: ${lagMs.toFixed(1)}ms [WARNING]`;
    } else {
      message = `Event loop lag: ${lagMs.toFixed(1)}ms`;
    }

    return {
      name: 'event_loop',
      healthy,
      message,
      details: { lagMs: this._lastEventLoopLagMs < 0 ? null : lagMs, checkCount: this._eventLoopCheckCount },
      timestamp: new Date().toISOString(),
    };
  }

  // --- Python bridge connectivity check ---
  private _bridgeHealthChecked = false;
  private _bridgeHealthy = false;
  private _bridgeMessage = 'Not yet checked';
  private _bridgeDetails: Record<string, unknown> = {};

  private async _checkPythonBridgeAsync(): Promise<void> {
    const httpMode = (process.env.CORE_BRIDGE_MODE || 'http') !== 'subprocess';
    const coreUrl = process.env.CORE_API_URL || 'http://localhost:8000';

    if (httpMode) {
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 3000);
        const resp = await fetch(`${coreUrl}/health`, { signal: controller.signal });
        clearTimeout(timer);

        if (resp.ok) {
          const data = (await resp.json()) as Record<string, unknown>;
          this._bridgeHealthy = true;
          this._bridgeMessage = `Python bridge healthy (HTTP ${coreUrl}, v${data.version ?? '?'})`;
          this._bridgeDetails = { mode: 'http', status: resp.status, ...data };
        } else {
          this._bridgeHealthy = false;
          this._bridgeMessage = `Python bridge returned ${resp.status} (${coreUrl})`;
          this._bridgeDetails = { mode: 'http', status: resp.status };
        }
      } catch (err: unknown) {
        this._bridgeHealthy = false;
        this._bridgeMessage = `Python bridge unreachable (${coreUrl}): ${err instanceof Error ? err.message : String(err)}`;
        this._bridgeDetails = { mode: 'http' };
      }
    } else {
      // Subprocess mode: check if a Python process exists and is responsive
      // We can't easily test the subprocess synchronously, so check if Python is available on PATH
      try {
        const { spawnSync } = await import('child_process');
        const pythonCmd = process.env.PYTHON_COMMAND || 'python3';
        const result = spawnSync(pythonCmd, ['-c', 'import sys; print(sys.version)'], {
          timeout: 3000,
          encoding: 'utf8',
        });

        if (result.status === 0 && result.stdout) {
          this._bridgeHealthy = true;
          this._bridgeMessage = `Python bridge available (${pythonCmd}: ${result.stdout.trim().split('\n')[0]})`;
          this._bridgeDetails = { mode: 'subprocess', pythonVersion: result.stdout.trim() };
        } else {
          this._bridgeHealthy = false;
          this._bridgeMessage = `Python command failed (${pythonCmd}): ${result.stderr?.trim() || result.error?.message || 'unknown error'}`;
          this._bridgeDetails = { mode: 'subprocess', exitCode: result.status };
        }
      } catch (err: unknown) {
        this._bridgeHealthy = false;
        this._bridgeMessage = `Python bridge check failed: ${err instanceof Error ? err.message : String(err)}`;
        this._bridgeDetails = { mode: 'subprocess' };
      }
    }
    this._bridgeHealthChecked = true;
  }

  private checkPythonBridge(): HealthStatus {
    // Trigger async check (results appear on next call)
    void this._checkPythonBridgeAsync();

    const details = this._bridgeDetails;
    return {
      name: 'python_bridge',
      healthy: this._bridgeHealthChecked ? this._bridgeHealthy : true, // First call: assume healthy until verified
      message: this._bridgeHealthChecked ? this._bridgeMessage : 'Python bridge: checking (async)...',
      details: {
        ...details,
        checked: this._bridgeHealthChecked,
      },
      timestamp: new Date().toISOString(),
    };
  }

  isHealthy(): boolean {
    return this.runQuickCheck();
  }

  getUptime(): number {
    return (Date.now() - this.startTime) / 1000;
  }
}

export class LivenessProbe {
  private lastHeartbeat: number;
  private timeoutMs: number;

  constructor(timeoutMs: number = 30000) {
    this.timeoutMs = timeoutMs;
    this.lastHeartbeat = Date.now();
  }

  heartbeat(): void {
    this.lastHeartbeat = Date.now();
  }

  isAlive(): boolean {
    return Date.now() - this.lastHeartbeat < this.timeoutMs;
  }

  check(): { alive: boolean; lastHeartbeat: string; secondsSinceHeartbeat: number } {
    return {
      alive: this.isAlive(),
      lastHeartbeat: new Date(this.lastHeartbeat).toISOString(),
      secondsSinceHeartbeat: (Date.now() - this.lastHeartbeat) / 1000,
    };
  }
}

export class ReadinessProbe {
  private ready: boolean = true;
  private reasons: string[] = [];

  setReady(ready: boolean, reason?: string): void {
    this.ready = ready;
    if (reason && !ready) {
      this.reasons.push(reason);
    } else if (ready) {
      this.reasons = [];
    }
  }

  isReady(): boolean {
    return this.ready;
  }

  check(): { ready: boolean; reasons: string[] } {
    return {
      ready: this.ready,
      reasons: this.reasons,
    };
  }
}

export const healthChecker = new HealthChecker();
export const livenessProbe = new LivenessProbe();
export const readinessProbe = new ReadinessProbe();
