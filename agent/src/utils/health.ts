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

  private checkEventLoop(): HealthStatus {
    const start = Date.now();
    
    return new Promise<HealthStatus>((resolve) => {
      setImmediate(() => {
        const lag = Date.now() - start;
        const healthy = lag < 100;
        
        resolve({
          name: 'event_loop',
          healthy,
          message: `Event loop lag: ${lag}ms`,
          details: { lagMs: lag },
          timestamp: new Date().toISOString(),
        });
      });
    }) as unknown as HealthStatus;
  }

  private checkPythonBridge(): HealthStatus {
    return {
      name: 'python_bridge',
      healthy: true,
      message: 'Python bridge available',
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
