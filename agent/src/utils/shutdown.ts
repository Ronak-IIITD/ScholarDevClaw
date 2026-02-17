import { logger } from './logger.js';

export interface ShutdownHandler {
  name: string;
  handler: () => Promise<void> | void;
  priority: number;
}

export class GracefulShutdown {
  private handlers: ShutdownHandler[] = [];
  private isShuttingDown: boolean = false;
  private shutdownReason: string = '';
  private timeoutMs: number;
  private shutdownPromise: Promise<void> | null = null;

  constructor(timeoutMs: number = 30000) {
    this.timeoutMs = timeoutMs;
    this.setupSignalHandlers();
  }

  private setupSignalHandlers(): void {
    process.on('SIGTERM', () => this.handleSignal('SIGTERM'));
    process.on('SIGINT', () => this.handleSignal('SIGINT'));
    
    process.on('beforeExit', () => {
      if (!this.isShuttingDown) {
        this.shutdown('Process exit');
      }
    });

    process.on('uncaughtException', (error) => {
      logger.error('Uncaught exception', { error: error.message });
      this.shutdown('Uncaught exception');
    });

    process.on('unhandledRejection', (reason) => {
      logger.error('Unhandled rejection', { reason: String(reason) });
    });
  }

  private handleSignal(signal: string): void {
    logger.info(`Received signal: ${signal}`);
    this.shutdown(`Signal: ${signal}`);
  }

  registerHandler(name: string, handler: () => Promise<void> | void, priority: number = 0): void {
    this.handlers.push({ name, handler, priority });
    this.handlers.sort((a, b) => b.priority - a.priority);
  }

  async shutdown(reason: string = 'Requested'): Promise<void> {
    if (this.isShuttingDown) {
      return this.shutdownPromise || Promise.resolve();
    }

    this.isShuttingDown = true;
    this.shutdownReason = reason;

    logger.info(`Initiating graceful shutdown: ${reason}`);

    this.shutdownPromise = this.executeShutdown();
    return this.shutdownPromise;
  }

  private async executeShutdown(): Promise<void> {
    const startTime = Date.now();
    const timeoutTime = startTime + this.timeoutMs;
    let handlersCalled = 0;

    for (const { name, handler } of this.handlers) {
      if (Date.now() > timeoutTime) {
        logger.error('Shutdown timeout exceeded');
        break;
      }

      try {
        await Promise.race([
          Promise.resolve(handler()),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Handler timeout')), 5000)
          ),
        ]);
        handlersCalled++;
        logger.debug(`Shutdown handler completed: ${name}`);
      } catch (error) {
        logger.error(`Shutdown handler failed: ${name}`, { error: String(error) });
      }
    }

    const duration = Date.now() - startTime;
    logger.info(`Shutdown complete: ${handlersCalled} handlers called in ${duration}ms`);

    process.exit(0);
  }

  isShuttingDownNow(): boolean {
    return this.isShuttingDown;
  }

  getReason(): string {
    return this.shutdownReason;
  }

  checkShutdown(): void {
    if (this.isShuttingDown) {
      throw new ShutdownError(`System is shutting down: ${this.shutdownReason}`);
    }
  }
}

export class ShutdownError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ShutdownError';
  }
}

export const shutdownManager = new GracefulShutdown();

export function withShutdownGuard<T>(fn: () => Promise<T>): Promise<T> {
  shutdownManager.checkShutdown();
  return fn();
}

export function setupGracefulShutdown(): void {
  shutdownManager.registerHandler('log-shutdown', async () => {
    logger.info('Application shutting down');
  }, 1000);
}
