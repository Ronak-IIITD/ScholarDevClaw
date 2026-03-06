import { logger } from '../utils/logger.js';
import { WorkflowEvent } from './types.js';

const WEBHOOK_FETCH_TIMEOUT_MS = 10_000;
const MAX_WEBHOOKS = 50;

/**
 * Validate that a URL is an acceptable webhook destination.
 * Blocks private/internal IPs and non-HTTPS URLs (except localhost for dev).
 */
function validateWebhookUrl(url: string): void {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw new Error(`Invalid webhook URL: ${url}`);
  }

  if (!['http:', 'https:'].includes(parsed.protocol)) {
    throw new Error(`Webhook URL must use http or https: ${url}`);
  }

  const hostname = parsed.hostname.toLowerCase();

  // Block private/internal hostnames (SSRF protection)
  const blockedPatterns = [
    /^127\./,
    /^10\./,
    /^172\.(1[6-9]|2\d|3[01])\./,
    /^192\.168\./,
    /^0\./,
    /^169\.254\./,
    /^::1$/,
    /^fc00:/,
    /^fe80:/,
    /^fd/,
    /^localhost$/,
    /^.*\.local$/,
    /^.*\.internal$/,
    /^metadata\./,
  ];

  for (const pattern of blockedPatterns) {
    if (pattern.test(hostname)) {
      throw new Error(`Webhook URL points to a blocked address: ${hostname}`);
    }
  }
}

export interface WebhookConfig {
  url: string;
  events: string[];
  headers?: Record<string, string>;
  retryCount?: number;
  retryDelay?: number;
}

export interface WebhookPayload {
  event: string;
  timestamp: string;
  data: Record<string, unknown>;
}

export class WebhookNotifier {
  private webhooks: Map<string, WebhookConfig> = new Map();

  register(name: string, config: WebhookConfig): void {
    validateWebhookUrl(config.url);
    if (this.webhooks.size >= MAX_WEBHOOKS) {
      throw new Error(`Maximum webhook limit (${MAX_WEBHOOKS}) reached`);
    }
    this.webhooks.set(name, config);
    logger.info(`[WEBHOOK] Registered: ${name} -> ${config.url}`);
  }

  unregister(name: string): void {
    this.webhooks.delete(name);
    logger.info(`[WEBHOOK] Unregistered: ${name}`);
  }

  async notify(event: WorkflowEvent): Promise<void> {
    const payload: WebhookPayload = {
      event: event.type,
      timestamp: event.timestamp,
      data: {
        workflowId: event.timestamp,
        nodeId: event.nodeId,
        status: event.status,
        output: event.output,
        error: event.error,
        progress: event.progress,
      },
    };

    const promises: Promise<void>[] = [];

    for (const [name, config] of this.webhooks.entries()) {
      if (!config.events.includes(event.type) && !config.events.includes('*')) {
        continue;
      }

      promises.push(this.sendWebhook(name, config, payload));
    }

    await Promise.allSettled(promises);
  }

  private async sendWebhook(
    name: string,
    config: WebhookConfig,
    payload: WebhookPayload
  ): Promise<void> {
    const maxRetries = config.retryCount || 3;
    const retryDelay = config.retryDelay || 1000;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const response = await fetch(config.url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...config.headers,
          },
          body: JSON.stringify(payload),
          signal: AbortSignal.timeout(WEBHOOK_FETCH_TIMEOUT_MS),
        });

        if (response.ok) {
          logger.debug(`[WEBHOOK] ${name}: delivered (attempt ${attempt})`);
          return;
        }

        logger.warn(`[WEBHOOK] ${name}: HTTP ${response.status} (attempt ${attempt})`);
      } catch (error) {
        logger.warn(`[WEBHOOK] ${name}: ${error} (attempt ${attempt})`);
      }

      if (attempt < maxRetries) {
        await new Promise(resolve => setTimeout(resolve, retryDelay * attempt));
      }
    }

    logger.error(`[WEBHOOK] ${name}: failed after ${maxRetries} attempts`);
  }

  list(): { name: string; url: string; events: string[] }[] {
    return Array.from(this.webhooks.entries()).map(([name, config]) => ({
      name,
      url: config.url,
      events: config.events,
      // NOTE: headers intentionally excluded to avoid leaking tokens
    }));
  }

  clear(): void {
    this.webhooks.clear();
  }
}

export const defaultWebhooks = new WebhookNotifier();

export function createSlackWebhook(webhookUrl: string): WebhookConfig {
  return {
    url: webhookUrl,
    events: ['workflow_completed', 'workflow_started', 'node_failed'],
    headers: { 'Content-Type': 'application/json' },
  };
}

export function createDiscordWebhook(webhookUrl: string): WebhookConfig {
  return {
    url: webhookUrl,
    events: ['workflow_completed', 'workflow_started', 'node_failed'],
    headers: { 'Content-Type': 'application/json' },
  };
}

export function createGitHubWebhook(repo: string, token: string): WebhookConfig {
  return {
    url: `https://api.github.com/repos/${repo}/dispatches`,
    events: ['workflow_completed'],
    headers: {
      'Authorization': `Bearer ${token}`,
      'Accept': 'application/vnd.github+json',
    },
  };
}
