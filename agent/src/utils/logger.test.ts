import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// Note: Logger class is not exported from logger.ts (only the singleton is)
import { logger } from './logger.js';
import type { LogLevel } from './logger.js';

// Helper to create fresh logger instances by accessing the constructor through the singleton
function createLogger(minLevel: string = 'info') {
  // The LogEntry must match what the logger produces
  return {
    log: new (Object.getPrototypeOf(logger).constructor)(minLevel),
    LoggerCtor: Object.getPrototypeOf(logger).constructor,
  };
}

describe('Logger', () => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let logSpy: any;

  beforeEach(() => {
    logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('level filtering', () => {
    it('respects minLevel by filtering debug when minLevel is info', () => {
      const { log } = createLogger('info');
      log.debug('should not appear');
      expect(logSpy).not.toHaveBeenCalled();
    });

    it('passes info when minLevel is info', () => {
      const { log } = createLogger('info');
      log.info('info message');
      expect(logSpy).toHaveBeenCalled();
    });

    it('passes warn when minLevel is info', () => {
      const { log } = createLogger('info');
      log.warn('warn message');
      expect(logSpy).toHaveBeenCalled();
    });

    it('passes error when minLevel is info', () => {
      const { log } = createLogger('info');
      log.error('error message');
      expect(logSpy).toHaveBeenCalled();
    });

    it('passes debug when minLevel is debug', () => {
      const { log } = createLogger('debug');
      log.debug('debug message');
      expect(logSpy).toHaveBeenCalled();
    });
  });

  describe('log entry format', () => {
    it('includes timestamp in output', () => {
      const { log } = createLogger('info');
      log.info('test');
      const output = logSpy.mock.calls[0][0] as string;
      expect(output).toMatch(/\[\d{4}-\d{2}-\d{2}T/);
    });

    it('includes level prefix in output', () => {
      const { log } = createLogger('info');
      log.info('hello');
      const output = logSpy.mock.calls[0][0] as string;
      expect(output).toContain('INFO');
    });

    it('includes message text', () => {
      const { log } = createLogger('info');
      log.info('hello world');
      const output = logSpy.mock.calls[0][0] as string;
      expect(output).toContain('hello world');
    });

    it('writes context to a second console.log call', () => {
      const { log } = createLogger('info');
      log.info('msg', { key: 'val' });
      expect(logSpy.mock.calls[1][0]).toContain('key');
      expect(logSpy.mock.calls[1][0]).toContain('val');
    });
  });

  describe('getLogs and clear', () => {
    it('getLogs returns stored entries', () => {
      const { log } = createLogger('info');
      log.info('a');
      log.warn('b');
      const entries = log.getLogs();
      expect(entries).toHaveLength(2);
      expect(entries[0].message).toBe('a');
      expect(entries[1].message).toBe('b');
    });

    it('getLogs returns a copy', () => {
      const { log } = createLogger('info');
      log.info('test');
      const entries = log.getLogs();
      entries.pop();
      expect(log.getLogs()).toHaveLength(1);
    });

    it('clear empties all entries', () => {
      const { log } = createLogger('info');
      log.info('a');
      log.clear();
      expect(log.getLogs()).toHaveLength(0);
    });

    it('entry stores level and context', () => {
      const { log } = createLogger('info');
      log.warn('warning', { code: 123 });
      const entry = log.getLogs()[0];
      expect(entry.level).toBe('warn');
      expect(entry.context).toEqual({ code: 123 });
    });
  });

  describe('safeReplacer', () => {
    it('serializes BigInt as string', () => {
      const { log } = createLogger('info');
      log.info('bigint', { val: BigInt(42) });
      const output = logSpy.mock.calls[1][0] as string;
      expect(output).toContain('"42"');
    });

    it('serializes Function as "[Function]"', () => {
      const { log } = createLogger('info');
      log.info('fn', { val: () => 1 });
      const output = logSpy.mock.calls[1][0] as string;
      expect(output).toContain('[Function]');
    });

    it('serializes Symbol as string', () => {
      const { log } = createLogger('info');
      log.info('sym', { val: Symbol('foo') });
      const output = logSpy.mock.calls[1][0] as string;
      expect(output).toContain('Symbol(foo)');
    });

    it('serializes Error as {name, message}', () => {
      const { log } = createLogger('info');
      log.info('err', { val: new Error('boom') });
      const output = logSpy.mock.calls[1][0] as string;
      expect(output).toContain('Error');
      expect(output).toContain('boom');
    });
  });

  describe('logger singleton', () => {
    it('logger has all log methods', () => {
      expect(typeof logger.info).toBe('function');
      expect(typeof logger.warn).toBe('function');
      expect(typeof logger.error).toBe('function');
      expect(typeof logger.debug).toBe('function');
    });

    it('logger.getLogs works', () => {
      expect(Array.isArray(logger.getLogs())).toBe(true);
    });
  });
});
