export enum CircuitState {
  Closed = 'closed',
  Open = 'open',
  HalfOpen = 'half_open',
}

export interface CircuitStats {
  state: CircuitState;
  failureCount: number;
  successCount: number;
  lastFailureTime: number | null;
  lastFailureMessage: string | null;
  openedAt: number | null;
  totalCalls: number;
  totalFailures: number;
  totalSuccesses: number;
}

export class CircuitOpenError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CircuitOpenError';
  }
}

export class CircuitBreaker {
  private state: CircuitState = CircuitState.Closed;
  private failureCount = 0;
  private successCount = 0;
  private lastFailureTime: number | null = null;
  private lastFailureMessage: string | null = null;
  private openedAt: number | null = null;
  private halfOpenCalls = 0;
  private totalCalls = 0;
  private totalFailures = 0;
  private totalSuccesses = 0;

  constructor(
    private readonly name: string,
    private readonly failureThreshold: number = 5,
    private readonly recoveryTimeout: number = 30000,
    private readonly halfOpenMaxCalls: number = 3
  ) {}

  async call<T>(fn: () => Promise<T>): Promise<T> {
    this.maybeTransition();

    if (this.state === CircuitState.Open) {
      throw new CircuitOpenError(
        `Circuit '${this.name}' is open. Last failure: ${this.lastFailureMessage || 'unknown'}`
      );
    }

    if (this.state === CircuitState.HalfOpen) {
      if (this.halfOpenCalls >= this.halfOpenMaxCalls) {
        throw new CircuitOpenError(
          `Circuit '${this.name}' is in half-open state with max calls reached`
        );
      }
      this.halfOpenCalls++;
    }

    this.totalCalls++;

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure(error instanceof Error ? error.message : String(error));
      throw error;
    }
  }

  private maybeTransition(): void {
    if (this.state === CircuitState.Open && this.openedAt !== null) {
      if (Date.now() - this.openedAt >= this.recoveryTimeout) {
        this.state = CircuitState.HalfOpen;
        this.halfOpenCalls = 0;
      }
    }
  }

  private onSuccess(): void {
    this.successCount++;
    this.totalSuccesses++;

    if (this.state === CircuitState.HalfOpen) {
      if (this.successCount >= this.halfOpenMaxCalls) {
        this.reset();
      }
    }
  }

  private onFailure(message: string): void {
    this.failureCount++;
    this.totalFailures++;
    this.lastFailureTime = Date.now();
    this.lastFailureMessage = message;

    if (this.state === CircuitState.HalfOpen) {
      this.trip();
    } else if (this.failureCount >= this.failureThreshold) {
      this.trip();
    }
  }

  private trip(): void {
    this.state = CircuitState.Open;
    this.openedAt = Date.now();
  }

  private reset(): void {
    this.state = CircuitState.Closed;
    this.failureCount = 0;
    this.successCount = 0;
    this.openedAt = null;
  }

  forceOpen(): void {
    this.trip();
  }

  forceClose(): void {
    this.reset();
  }

  getStats(): CircuitStats {
    return {
      state: this.state,
      failureCount: this.failureCount,
      successCount: this.successCount,
      lastFailureTime: this.lastFailureTime,
      lastFailureMessage: this.lastFailureMessage,
      openedAt: this.openedAt,
      totalCalls: this.totalCalls,
      totalFailures: this.totalFailures,
      totalSuccesses: this.totalSuccesses,
    };
  }

  isOpen(): boolean {
    this.maybeTransition();
    return this.state === CircuitState.Open;
  }

  isClosed(): boolean {
    this.maybeTransition();
    return this.state === CircuitState.Closed;
  }
}

export class CircuitBreakerRegistry {
  private breakers: Map<string, CircuitBreaker> = new Map();

  getOrCreate(
    name: string,
    failureThreshold: number = 5,
    recoveryTimeout: number = 30000
  ): CircuitBreaker {
    let breaker = this.breakers.get(name);
    if (!breaker) {
      breaker = new CircuitBreaker(name, failureThreshold, recoveryTimeout);
      this.breakers.set(name, breaker);
    }
    return breaker;
  }

  get(name: string): CircuitBreaker | undefined {
    return this.breakers.get(name);
  }

  getAllStats(): Record<string, CircuitStats> {
    const stats: Record<string, CircuitStats> = {};
    for (const [name, breaker] of this.breakers) {
      stats[name] = breaker.getStats();
    }
    return stats;
  }

  getHealth(): { healthy: boolean; totalCircuits: number; openCircuits: string[] } {
    const stats = this.getAllStats();
    const openCircuits = Object.entries(stats)
      .filter(([, s]) => s.state === CircuitState.Open)
      .map(([name]) => name);

    return {
      healthy: openCircuits.length === 0,
      totalCircuits: Object.keys(stats).length,
      openCircuits,
    };
  }
}

export const circuitRegistry = new CircuitBreakerRegistry();

export function withCircuitBreaker<T>(
  name: string,
  fn: () => Promise<T>,
  options?: { failureThreshold?: number; recoveryTimeout?: number }
): Promise<T> {
  const breaker = circuitRegistry.getOrCreate(
    name,
    options?.failureThreshold,
    options?.recoveryTimeout
  );
  return breaker.call(fn);
}
