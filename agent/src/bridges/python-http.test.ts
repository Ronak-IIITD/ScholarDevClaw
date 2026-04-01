import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { PythonHttpBridge } from './python-http.js';

// Mock logger to avoid console noise in tests
vi.mock('../utils/logger.js', () => ({
  logger: {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

describe('PythonHttpBridge', () => {
  let bridge: PythonHttpBridge;

  beforeEach(() => {
    // 0 retries for deterministic fast tests
    bridge = new PythonHttpBridge('http://localhost:8000', 5000, 0, 100);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('sends auth header when SCHOLARDEVCLAW_API_AUTH_KEY is set', async () => {
    const original = process.env.SCHOLARDEVCLAW_API_AUTH_KEY;
    process.env.SCHOLARDEVCLAW_API_AUTH_KEY = 'test-secret';
    const authBridge = new PythonHttpBridge('http://localhost:8000');
    expect(authBridge['authToken']).toBe('test-secret');
    process.env.SCHOLARDEVCLAW_API_AUTH_KEY = original;
  });

  it('returns success on 200 response', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: { get: () => 'application/json' },
      json: () => Promise.resolve({ status: 'ok' }),
    });
    vi.stubGlobal('fetch', mockFetch);

    const result = await bridge.healthCheck();

    expect(result).toBe(true);
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/health',
      expect.objectContaining({ method: 'GET' }),
    );
  });

  it('returns false on 404 response (non-retryable)', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      headers: { get: () => 'application/json' },
      text: () => Promise.resolve('Not Found'),
    });
    vi.stubGlobal('fetch', mockFetch);

    const result = await bridge.healthCheck();

    expect(result).toBe(false);
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it('rejects non-JSON content type', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: { get: () => 'text/html' },
      text: () => Promise.resolve('<html></html>'),
    });
    vi.stubGlobal('fetch', mockFetch);

    const result = await bridge.healthCheck();

    expect(result).toBe(false);
  });
});

describe('PythonHttpBridge with retries', () => {
  it('retries on 503 and succeeds', async () => {
    vi.useFakeTimers();
    const bridge = new PythonHttpBridge('http://localhost:8000', 5000, 2, 10);
    let callCount = 0;
    const mockFetch = vi.fn().mockImplementation(() => {
      callCount++;
      if (callCount < 3) {
        return Promise.resolve({
          ok: false,
          status: 503,
          headers: { get: () => 'application/json' },
          text: () => Promise.resolve('Service Unavailable'),
        });
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        headers: { get: () => 'application/json' },
        json: () => Promise.resolve({ status: 'ok' }),
      });
    });
    vi.stubGlobal('fetch', mockFetch);

    const promise = bridge.healthCheck();
    await vi.advanceTimersByTimeAsync(100);
    const result = await promise;

    expect(result).toBe(true);
    expect(callCount).toBe(3);
    vi.useRealTimers();
  });

  it('fails after max retries on persistent 503', async () => {
    vi.useFakeTimers();
    const bridge = new PythonHttpBridge('http://localhost:8000', 5000, 2, 10);
    const mockFetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      headers: { get: () => 'application/json' },
      text: () => Promise.resolve('Service Unavailable'),
    });
    vi.stubGlobal('fetch', mockFetch);

    const promise = bridge.healthCheck();
    await vi.advanceTimersByTimeAsync(100);
    const result = await promise;

    expect(result).toBe(false);
    vi.useRealTimers();
  });
});
