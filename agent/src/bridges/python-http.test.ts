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
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    // 0 retries for deterministic fast tests
    bridge = new PythonHttpBridge('http://localhost:8000', 5000, 0, 100);
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.useRealTimers();
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
    globalThis.fetch = mockFetch as typeof fetch;

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
    globalThis.fetch = mockFetch as typeof fetch;

    const result = await bridge.healthCheck();

    expect(result).toBe(false);
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it('sends auth headers and JSON body for repository analysis', async () => {
    const original = process.env.SCHOLARDEVCLAW_API_AUTH_KEY;
    process.env.SCHOLARDEVCLAW_API_AUTH_KEY = 'bridge-secret';
    const authBridge = new PythonHttpBridge('http://localhost:8000', 5000, 0, 100);

    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: { get: () => 'application/json' },
      json: () => Promise.resolve({ status: 'ok', repo: 'analyzed' }),
    });
    globalThis.fetch = mockFetch as typeof fetch;

    const result = await authBridge.analyzeRepo('/tmp/repo');

    expect(result.success).toBe(true);
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/repo/analyze',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({
          Authorization: 'Bearer bridge-secret',
          'Content-Type': 'application/json',
        }),
        body: JSON.stringify({ repoPath: '/tmp/repo' }),
      }),
    );

    process.env.SCHOLARDEVCLAW_API_AUTH_KEY = original;
  });

  it('sends repoPath when generating patch via HTTP', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: { get: () => 'application/json' },
      json: () => Promise.resolve({ newFiles: [], transformations: [], branchName: 'integration/test' }),
    });
    globalThis.fetch = mockFetch as typeof fetch;

    const result = await bridge.generatePatch({ targets: [], strategy: 'replace', confidence: 90 }, '/tmp/repo');

    expect(result.success).toBe(true);
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/patch/generate',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          mapping: { targets: [], strategy: 'replace', confidence: 90 },
          repoPath: '/tmp/repo',
        }),
      }),
    );
  });

  it('rejects non-JSON content type', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: { get: () => 'text/html' },
      text: () => Promise.resolve('<html></html>'),
    });
    globalThis.fetch = mockFetch as typeof fetch;

    const result = await bridge.healthCheck();

    expect(result).toBe(false);
  });
});

describe('PythonHttpBridge with retries', () => {
  it('retries on 503 and succeeds', async () => {
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
    globalThis.fetch = mockFetch as typeof fetch;

    const result = await bridge.healthCheck();

    expect(result).toBe(true);
    expect(callCount).toBe(3);
  });

  it('fails after max retries on persistent 503', async () => {
    const bridge = new PythonHttpBridge('http://localhost:8000', 5000, 2, 10);
    const mockFetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      headers: { get: () => 'application/json' },
      text: () => Promise.resolve('Service Unavailable'),
    });
    globalThis.fetch = mockFetch as typeof fetch;

    const result = await bridge.healthCheck();

    expect(result).toBe(false);
  });

  it('retries on abort errors and succeeds on a later attempt', async () => {
    const bridge = new PythonHttpBridge('http://localhost:8000', 5000, 2, 10);
    let callCount = 0;
    const timeoutError = new Error('timed out');
    timeoutError.name = 'AbortError';

    const mockFetch = vi.fn().mockImplementation(() => {
      callCount++;
      if (callCount < 3) {
        return Promise.reject(timeoutError);
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        headers: { get: () => 'application/json' },
        json: () => Promise.resolve({ status: 'ok' }),
      });
    });
    globalThis.fetch = mockFetch as typeof fetch;

    const result = await bridge.healthCheck();

    expect(result).toBe(true);
    expect(callCount).toBe(3);
  });
});
