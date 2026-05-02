import { describe, expect, it, vi } from 'vitest';

vi.mock('../utils/logger.js', () => ({
  logger: {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  },
}));

import { WebhookNotifier } from './webhooks.js';
import { logger } from '../utils/logger.js';


describe('WebhookNotifier logging', () => {
  it('sanitizes URL when registering webhooks', () => {
    const notifier = new WebhookNotifier();

    notifier.register('hook', {
      url: 'https://example.com/path?token=secret&x=1',
      events: ['workflow_started'],
    });

    const infoSpy = logger.info as ReturnType<typeof vi.fn>;
    const firstCallArgs = infoSpy.mock.calls[0]?.[0] as string;
    expect(firstCallArgs).toContain('https://example.com/path');
    expect(firstCallArgs).not.toContain('token=secret');
  });

  it('blocks non-https URLs except loopback http', () => {
    const notifier = new WebhookNotifier();

    expect(() =>
      notifier.register('insecure', {
        url: 'http://example.com/webhook',
        events: ['workflow_started'],
      })
    ).toThrow(/https/i);

    expect(() =>
      notifier.register('loopback', {
        url: 'http://localhost:8080/webhook',
        events: ['workflow_started'],
      })
    ).not.toThrow();
  });
});
