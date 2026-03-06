export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  message: string;
  context?: Record<string, unknown>;
}

class Logger {
  private logs: LogEntry[] = [];
  private minLevel: LogLevel;
  // SECURITY: Cap in-memory log buffer to prevent unbounded memory growth
  private static readonly MAX_LOG_ENTRIES = 10_000;

  private levelPriority: Record<LogLevel, number> = {
    debug: 0,
    info: 1,
    warn: 2,
    error: 3,
  };

  constructor(minLevel: LogLevel = 'info') {
    this.minLevel = minLevel;
  }

  private shouldLog(level: LogLevel): boolean {
    return this.levelPriority[level] >= this.levelPriority[this.minLevel];
  }

  private log(level: LogLevel, message: string, context?: Record<string, unknown>): void {
    if (!this.shouldLog(level)) return;

    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      context,
    };

    this.logs.push(entry);
    // SECURITY: Evict oldest entries when buffer exceeds cap (ring buffer behavior)
    if (this.logs.length > Logger.MAX_LOG_ENTRIES) {
      this.logs = this.logs.slice(-Logger.MAX_LOG_ENTRIES);
    }

    const color = this.getColor(level);
    const prefix = `[${entry.timestamp}] ${level.toUpperCase()}`;
    console.log(`${color}${prefix}${this.resetColor()}: ${message}`);
    if (context) {
      console.log(JSON.stringify(context, null, 2));
    }
  }

  private getColor(level: LogLevel): string {
    const colors = {
      debug: '\x1b[36m',   // cyan
      info: '\x1b[32m',    // green
      warn: '\x1b[33m',    // yellow
      error: '\x1b[31m',   // red
    };
    return colors[level];
  }

  private resetColor(): string {
    return '\x1b[0m';
  }

  debug(message: string, context?: Record<string, unknown>): void {
    this.log('debug', message, context);
  }

  info(message: string, context?: Record<string, unknown>): void {
    this.log('info', message, context);
  }

  warn(message: string, context?: Record<string, unknown>): void {
    this.log('warn', message, context);
  }

  error(message: string, context?: Record<string, unknown>): void {
    this.log('error', message, context);
  }

  getLogs(): LogEntry[] {
    return [...this.logs];
  }

  clear(): void {
    this.logs = [];
  }
}

export const logger = new Logger();
