import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

describe('config', () => {
  let warnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.resetModules();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
  });

  it('has default values when no env vars are set', async () => {
    vi.stubEnv('OPENCLAW_TOKEN', '');
    vi.stubEnv('CONVEX_URL', '');
    vi.stubEnv('GITHUB_TOKEN', '');
    vi.stubEnv('ANTHROPIC_API_KEY', '');
    vi.stubEnv('CORE_API_URL', '');
    vi.stubEnv('DEFAULT_MODE', '');
    vi.stubEnv('CORE_BRIDGE_MODE', 'http');
    vi.stubEnv('SCHOLARDEVCLAW_API_AUTH_KEY', 'some-key');
    vi.stubEnv('SCHOLARDEVCLAW_CORS_ORIGINS', 'http://localhost');

    const { config } = await import('./config.js');

    expect(config.openclaw.token).toBe('');
    expect(config.openclaw.apiUrl).toBe('http://localhost:3000');
    expect(config.convex.deploymentUrl).toBe('');
    expect(config.github.token).toBe('');
    expect(config.anthropic.apiKey).toBe('');
    expect(config.python.coreApiUrl).toBe('http://localhost:8000');
    expect(config.python.subprocessCommand).toBe('python3');
    expect(config.execution.defaultMode).toBe('step_approval');
    expect(config.execution.maxRetries).toBe(2);
    expect(config.execution.benchmarkTimeout).toBe(300);
  });

  it('reads OPENCLAW_TOKEN from env', async () => {
    vi.stubEnv('OPENCLAW_TOKEN', 'my-token');
    vi.stubEnv('CORE_BRIDGE_MODE', 'http');
    vi.stubEnv('SCHOLARDEVCLAW_API_AUTH_KEY', 'some-key');
    vi.stubEnv('SCHOLARDEVCLAW_CORS_ORIGINS', 'http://localhost');

    const { config } = await import('./config.js');
    expect(config.openclaw.token).toBe('my-token');
  });

  it('reads CONVEX_URL from env', async () => {
    vi.stubEnv('CONVEX_URL', 'https://my-convex.example.com');
    vi.stubEnv('CORE_BRIDGE_MODE', 'http');
    vi.stubEnv('SCHOLARDEVCLAW_API_AUTH_KEY', 'some-key');
    vi.stubEnv('SCHOLARDEVCLAW_CORS_ORIGINS', 'http://localhost');

    const { config } = await import('./config.js');
    expect(config.convex.deploymentUrl).toBe('https://my-convex.example.com');
  });

  it('reads GITHUB_TOKEN from env', async () => {
    vi.stubEnv('GITHUB_TOKEN', 'gh_token_123');
    vi.stubEnv('CORE_BRIDGE_MODE', 'http');
    vi.stubEnv('SCHOLARDEVCLAW_API_AUTH_KEY', 'some-key');
    vi.stubEnv('SCHOLARDEVCLAW_CORS_ORIGINS', 'http://localhost');

    const { config } = await import('./config.js');
    expect(config.github.token).toBe('gh_token_123');
  });

  it('reads CORE_API_URL from env', async () => {
    vi.stubEnv('CORE_API_URL', 'http://api.example.com:8080');
    vi.stubEnv('CORE_BRIDGE_MODE', 'http');
    vi.stubEnv('SCHOLARDEVCLAW_API_AUTH_KEY', 'some-key');
    vi.stubEnv('SCHOLARDEVCLAW_CORS_ORIGINS', 'http://localhost');

    const { config } = await import('./config.js');
    expect(config.python.coreApiUrl).toBe('http://api.example.com:8080');
  });

  it('reads DEFAULT_MODE from env', async () => {
    vi.stubEnv('DEFAULT_MODE', 'autonomous');
    vi.stubEnv('CORE_BRIDGE_MODE', 'http');
    vi.stubEnv('SCHOLARDEVCLAW_API_AUTH_KEY', 'some-key');
    vi.stubEnv('SCHOLARDEVCLAW_CORS_ORIGINS', 'http://localhost');

    const { config } = await import('./config.js');
    expect(config.execution.defaultMode).toBe('autonomous');
  });

  it('parses MAPPING_MIN_CONFIDENCE as number', async () => {
    vi.stubEnv('MAPPING_MIN_CONFIDENCE', '85');
    vi.stubEnv('CORE_BRIDGE_MODE', 'http');
    vi.stubEnv('SCHOLARDEVCLAW_API_AUTH_KEY', 'some-key');
    vi.stubEnv('SCHOLARDEVCLAW_CORS_ORIGINS', 'http://localhost');

    const { config } = await import('./config.js');
    expect(config.execution.guardrails.mappingMinConfidence).toBe(85);
  });

  it('parses VALIDATION_MIN_SPEEDUP as number', async () => {
    vi.stubEnv('VALIDATION_MIN_SPEEDUP', '1.5');
    vi.stubEnv('CORE_BRIDGE_MODE', 'http');
    vi.stubEnv('SCHOLARDEVCLAW_API_AUTH_KEY', 'some-key');
    vi.stubEnv('SCHOLARDEVCLAW_CORS_ORIGINS', 'http://localhost');

    const { config } = await import('./config.js');
    expect(config.execution.guardrails.validationMinSpeedup).toBe(1.5);
  });

  it('parses VALIDATION_MAX_LOSS_CHANGE_PCT as number', async () => {
    vi.stubEnv('VALIDATION_MAX_LOSS_CHANGE_PCT', '10');
    vi.stubEnv('CORE_BRIDGE_MODE', 'http');
    vi.stubEnv('SCHOLARDEVCLAW_API_AUTH_KEY', 'some-key');
    vi.stubEnv('SCHOLARDEVCLAW_CORS_ORIGINS', 'http://localhost');

    const { config } = await import('./config.js');
    expect(config.execution.guardrails.validationMaxLossChangePct).toBe(10);
  });

  it('warns when OPENCLAW_TOKEN is not set', async () => {
    vi.stubEnv('OPENCLAW_TOKEN', '');
    vi.stubEnv('CORE_BRIDGE_MODE', 'http');
    vi.stubEnv('SCHOLARDEVCLAW_API_AUTH_KEY', 'some-key');
    vi.stubEnv('SCHOLARDEVCLAW_CORS_ORIGINS', 'http://localhost');

    await import('./config.js');
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining('OPENCLAW_TOKEN'),
    );
  });

  it('warns when SCHOLARDEVCLAW_CORS_ORIGINS is not set', async () => {
    vi.stubEnv('OPENCLAW_TOKEN', 'some-token');
    vi.stubEnv('CORE_BRIDGE_MODE', 'http');
    vi.stubEnv('SCHOLARDEVCLAW_API_AUTH_KEY', 'some-key');
    vi.stubEnv('SCHOLARDEVCLAW_CORS_ORIGINS', '');

    await import('./config.js');
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining('CORS_ORIGINS'),
    );
  });
});
