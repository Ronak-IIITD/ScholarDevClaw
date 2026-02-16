import { logger } from '../utils/logger.js';
import type {
  PhaseResult,
  RepoAnalysisResult,
  ResearchSpecResult,
  MappingResult,
  PatchResult,
  ValidationResult,
} from './python-subprocess.js';

export class PythonHttpBridge {
  private baseUrl: string;
  private timeout: number;

  constructor(baseUrl: string = 'http://localhost:8000', timeout: number = 60000) {
    this.baseUrl = baseUrl;
    this.timeout = timeout;
  }

  private async request<T>(endpoint: string, method: string = 'GET', body?: unknown): Promise<PhaseResult> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method,
        headers: {
          'Content-Type': 'application/json',
        },
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const error = await response.text();
        logger.error(`HTTP error ${response.status}: ${error}`);
        return { success: false, error: `HTTP ${response.status}: ${error}` };
      }

      const data = await response.json();
      return { success: true, data };
    } catch (err) {
      clearTimeout(timeoutId);
      const message = err instanceof Error ? err.message : 'Unknown error';
      logger.error(`Request failed: ${message}`);
      return { success: false, error: message };
    }
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
