import { logger } from '../utils/logger.js';
import type {
  PhaseResult,
  RepoAnalysisResult,
  ResearchSpecResult,
  MappingResult,
  PatchResult,
  ValidationResult,
} from './python-subprocess.js';

// Retryable HTTP status codes (transient server errors / rate limits)
const RETRYABLE_STATUS = new Set([429, 500, 502, 503, 504]);

// Default retry configuration
const DEFAULT_MAX_RETRIES = 2;
const DEFAULT_BASE_DELAY_MS = 1000;

export class PythonHttpBridge {
  readonly baseUrl: string;
  private timeout: number;
  private maxRetries: number;
  private baseDelayMs: number;
  private authToken: string | undefined;

  constructor(
    baseUrl: string = 'http://localhost:8000',
    timeout: number = 60000,
    maxRetries: number = DEFAULT_MAX_RETRIES,
    baseDelayMs: number = DEFAULT_BASE_DELAY_MS,
  ) {
    this.baseUrl = baseUrl;
    this.timeout = timeout;
    this.maxRetries = maxRetries;
    this.baseDelayMs = baseDelayMs;
    this.authToken = process.env.SCHOLARDEVCLAW_API_AUTH_KEY || undefined;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    }
    return headers;
  }

  private async requestWithRetry<T>(
    endpoint: string,
    method: string = 'GET',
    body?: unknown,
    attempt: number = 0,
  ): Promise<PhaseResult> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method,
        headers: this.getHeaders(),
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // Retry on transient errors
      if (RETRYABLE_STATUS.has(response.status) && attempt < this.maxRetries) {
        const delayMs = this.baseDelayMs * Math.pow(2, attempt) * (0.5 + Math.random());
        logger.warn(
          `HTTP ${response.status} on ${method} ${endpoint}, retrying in ${Math.round(delayMs)}ms (attempt ${attempt + 1}/${this.maxRetries})`,
        );
        await this.sleep(delayMs);
        return this.requestWithRetry<T>(endpoint, method, body, attempt + 1);
      }

      if (!response.ok) {
        const error = await response.text();
        logger.error(`HTTP error ${response.status}: ${error}`);
        return { success: false, error: `HTTP ${response.status}: ${error}` };
      }

      // Validate content type before parsing
      const contentType = response.headers.get('content-type') || '';
      if (!contentType.includes('application/json')) {
        const text = await response.text();
        return { success: false, error: `Unexpected content type: ${contentType}` };
      }

      const data = await response.json();
      return { success: true, data };
    } catch (err) {
      clearTimeout(timeoutId);

      // Retry on network errors
      if (attempt < this.maxRetries && err instanceof Error && err.name === 'AbortError') {
        const delayMs = this.baseDelayMs * Math.pow(2, attempt) * (0.5 + Math.random());
        logger.warn(
          `Request timeout on ${method} ${endpoint}, retrying in ${Math.round(delayMs)}ms (attempt ${attempt + 1}/${this.maxRetries})`,
        );
        await this.sleep(delayMs);
        return this.requestWithRetry<T>(endpoint, method, body, attempt + 1);
      }

      const message = err instanceof Error ? err.message : 'Unknown error';
      logger.error(`Request failed: ${message}`);
      return { success: false, error: message };
    }
  }

  private async request<T>(endpoint: string, method: string = 'GET', body?: unknown): Promise<PhaseResult> {
    return this.requestWithRetry<T>(endpoint, method, body, 0);
  }

  async analyzeRepo(repoPath: string): Promise<PhaseResult> {
    logger.info('Analyzing repository via HTTP', { repoPath });
    return this.request<RepoAnalysisResult>('/repo/analyze', 'POST', { repoPath });
  }

  async extractResearch(paperSource: string, sourceType: 'pdf' | 'arxiv'): Promise<PhaseResult> {
    logger.info('Extracting research via HTTP', { paperSource, sourceType });
    return this.request<ResearchSpecResult>('/research/extract', 'POST', {
      source: paperSource,
      sourceType,
    });
  }

  async mapArchitecture(repoAnalysis: RepoAnalysisResult, researchSpec: ResearchSpecResult): Promise<PhaseResult> {
    logger.info('Mapping architecture via HTTP');
    return this.request<MappingResult>('/mapping/map', 'POST', {
      repoAnalysis,
      researchSpec,
    });
  }

  async generatePatch(mapping: MappingResult): Promise<PhaseResult> {
    logger.info('Generating patch via HTTP');
    return this.request<PatchResult>('/patch/generate', 'POST', { mapping });
  }

  async validate(patch: PatchResult, repoPath: string): Promise<PhaseResult> {
    logger.info('Running validation via HTTP');
    return this.request<ValidationResult>('/validation/run', 'POST', {
      patch,
      repoPath,
    });
  }

  async healthCheck(): Promise<boolean> {
    const result = await this.request<{ status: string }>('/health', 'GET');
    const data = result.data as { status?: string } | undefined;
    return result.success && data?.status === 'ok';
  }
}
