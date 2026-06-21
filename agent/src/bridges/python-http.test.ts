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

  it('posts validate payload to /validation/run', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: { get: () => 'application/json' },
      json: () =>
        Promise.resolve({
          passed: false,
          stage: 'policy',
          error: 'Docker sandbox requested but unavailable',
          logs: 'Strict execution mode requires Docker',
        }),
    });
    globalThis.fetch = mockFetch as typeof fetch;

    const patchPayload = {
      newFiles: [{ path: 'rmsnorm.py', content: 'print("ok")' }],
      transformations: [],
      branchName: 'integration/rmsnorm',
    };

    const result = await bridge.validate(patchPayload, '/tmp/repo');

    expect(result.success).toBe(true);
    expect(result.data).toEqual(
      expect.objectContaining({
        passed: false,
        stage: 'policy',
      }),
    );
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/validation/run',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ patch: patchPayload, repoPath: '/tmp/repo' }),
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

describe('PythonHttpBridge — validation sub-step methods', () => {
  let bridge: PythonHttpBridge;
  let mockFetch: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    bridge = new PythonHttpBridge('http://localhost:8000', 5000, 0, 100);
    mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: { get: () => 'application/json' },
      json: () => Promise.resolve({ passed: true, stage: 'test', logs: 'ok' }),
    });
    globalThis.fetch = mockFetch as typeof fetch;
  });

  it('validateArtifacts posts to /validation/artifacts', async () => {
    const patch = { newFiles: [{ path: 'f.py', content: 'x = 1' }], transformations: [], branchName: 'b' };
    const result = await bridge.validateArtifacts(patch as any);

    expect(result.success).toBe(true);
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/validation/artifacts',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ patch }) }),
    );
  });

  it('checkPolicy posts to /validation/policy', async () => {
    const result = await bridge.checkPolicy();

    expect(result.success).toBe(true);
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/validation/policy',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({}) }),
    );
  });

  it('runTests posts to /validation/tests with repoPath', async () => {
    const result = await bridge.runTests('/tmp/repo');

    expect(result.success).toBe(true);
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/validation/tests',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ repoPath: '/tmp/repo' }) }),
    );
  });

  it('runBenchmark posts to /validation/benchmark with repoPath', async () => {
    const result = await bridge.runBenchmark('/tmp/repo');

    expect(result.success).toBe(true);
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/validation/benchmark',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ repoPath: '/tmp/repo' }) }),
    );
  });

  it('runTraining posts to /validation/training with options', async () => {
    const result = await bridge.runTraining('/tmp/repo', { useVariant: true, iterations: 20 });

    expect(result.success).toBe(true);
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.repoPath).toBe('/tmp/repo');
    expect(body.useVariant).toBe(true);
    expect(body.iterations).toBe(20);
    expect(body.batchSize).toBe(4); // default
    expect(body.seqLen).toBe(32); // default
  });

  it('runTraining defaults all options to sensible values', async () => {
    await bridge.runTraining('/tmp/repo');

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.useVariant).toBe(false);
    expect(body.useTorch).toBe(false);
    expect(body.iterations).toBe(10);
    expect(body.batchSize).toBe(4);
    expect(body.seqLen).toBe(32);
  });

  it('runNumericalCorrectness posts to /validation/correctness', async () => {
    const patch = { newFiles: [], transformations: [], algorithmName: 'rmsnorm' };
    const result = await bridge.runNumericalCorrectness(patch as any);

    expect(result.success).toBe(true);
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/validation/correctness',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ patch }) }),
    );
  });

  it('runRegressionSnapshot posts to /validation/regression', async () => {
    const patch = { newFiles: [], transformations: [{ file: 'f.py', original: 'a', modified: 'b', changes: [] }] };
    const result = await bridge.runRegressionSnapshot(patch as any);

    expect(result.success).toBe(true);
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/validation/regression',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ patch }) }),
    );
  });

  it('scoreDiffReadability posts to /validation/readability', async () => {
    const patch = { newFiles: [], transformations: [] };
    const result = await bridge.scoreDiffReadability(patch as any);

    expect(result.success).toBe(true);
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/validation/readability',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ patch }) }),
    );
  });

  it('healPatch posts to /validation/heal with all fields', async () => {
    mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: { get: () => 'application/json' },
      json: () => Promise.resolve({ new_files: [], transformations: [], branch_name: 'healed' }),
    });
    globalThis.fetch = mockFetch as typeof fetch;

    const patch = { newFiles: [{ path: 'f.py', content: 'x = 1' }], transformations: [], branchName: 'b' };
    const testResult = { passed: false, stage: 'tests', logs: 'fail', error: 'AssertionError' };
    const mappingResult = { targets: [] };

    const result = await bridge.healPatch(patch as any, testResult, mappingResult, '/tmp/repo');

    expect(result.success).toBe(true);
    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.patch).toEqual(patch);
    expect(body.testResult).toEqual(testResult);
    expect(body.mappingResult).toEqual(mappingResult);
    expect(body.repoPath).toBe('/tmp/repo');
  });

  it('handles HTTP errors on validation sub-step methods', async () => {
    mockFetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      headers: { get: () => 'application/json' },
      text: () => Promise.resolve('Internal Server Error'),
    });
    globalThis.fetch = mockFetch as typeof fetch;

    const result = await bridge.validateArtifacts({} as any);
    expect(result.success).toBe(false);
    expect(result.error).toContain('500');
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
