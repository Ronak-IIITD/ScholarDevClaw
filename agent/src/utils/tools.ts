import { logger } from './logger.js';
import { config } from './config.js';
import { PythonSubprocessBridge, PythonHttpBridge } from '../bridges/python-bridge.js';
import type {
  RepoAnalysisResult,
  ResearchSpecResult,
  MappingResult,
  PatchResult,
} from '../bridges/python-subprocess.js';


export interface ToolResult {
  success: boolean;
  data?: unknown;
  error?: string;
  duration?: number;
}


export class AgentTools {
  private bridge: PythonSubprocessBridge | PythonHttpBridge;

  constructor(bridge: PythonSubprocessBridge | PythonHttpBridge) {
    this.bridge = bridge;
  }

  async runAnalyze(repoPath: string): Promise<ToolResult> {
    const start = Date.now();
    try {
      const result = await this.bridge.analyzeRepo(repoPath);
      return {
        success: result.success,
        data: result.data,
        duration: Date.now() - start,
      };
    } catch (err) {
      return {
        success: false,
        error: err instanceof Error ? err.message : 'Unknown error',
        duration: Date.now() - start,
      };
    }
  }

  async runResearch(paperSource: string, sourceType: 'pdf' | 'arxiv'): Promise<ToolResult> {
    const start = Date.now();
    try {
      const result = await this.bridge.extractResearch(paperSource, sourceType);
      return {
        success: result.success,
        data: result.data,
        duration: Date.now() - start,
      };
    } catch (err) {
      return {
        success: false,
        error: err instanceof Error ? err.message : 'Unknown error',
        duration: Date.now() - start,
      };
    }
  }

  async runMapping(repoAnalysis: RepoAnalysisResult, researchSpec: ResearchSpecResult): Promise<ToolResult> {
    const start = Date.now();
    try {
      const result = await this.bridge.mapArchitecture(repoAnalysis, researchSpec);
      return {
        success: result.success,
        data: result.data,
        duration: Date.now() - start,
      };
    } catch (err) {
      return {
        success: false,
        error: err instanceof Error ? err.message : 'Unknown error',
        duration: Date.now() - start,
      };
    }
  }

  async runGenerate(mapping: MappingResult): Promise<ToolResult> {
    const start = Date.now();
    try {
      const result = await this.bridge.generatePatch(mapping);
      return {
        success: result.success,
        data: result.data,
        duration: Date.now() - start,
      };
    } catch (err) {
      return {
        success: false,
        error: err instanceof Error ? err.message : 'Unknown error',
        duration: Date.now() - start,
      };
    }
  }

  async runValidate(patch: PatchResult, repoPath: string): Promise<ToolResult> {
    const start = Date.now();
    try {
      const result = await this.bridge.validate(patch, repoPath);
      return {
        success: result.success,
        data: result.data,
        duration: Date.now() - start,
      };
    } catch (err) {
      return {
        success: false,
        error: err instanceof Error ? err.message : 'Unknown error',
        duration: Date.now() - start,
      };
    }
  }

  async runPlanner(repoPath: string, maxSpecs: number = 5): Promise<ToolResult> {
    const start = Date.now();
    try {
      const result = await this.request('/planner/run', {
        repoPath,
        maxSpecs,
      });
      return {
        success: result.success,
        data: result.data,
        duration: Date.now() - start,
      };
    } catch (err) {
      return {
        success: false,
        error: err instanceof Error ? err.message : 'Unknown error',
        duration: Date.now() - start,
      };
    }
  }

  async runCritic(repoPath: string, spec?: string): Promise<ToolResult> {
    const start = Date.now();
    try {
      const result = await this.request('/critic/run', {
        repoPath,
        spec,
      });
      return {
        success: result.success,
        data: result.data,
        duration: Date.now() - start,
      };
    } catch (err) {
      return {
        success: false,
        error: err instanceof Error ? err.message : 'Unknown error',
        duration: Date.now() - start,
      };
    }
  }

  async getContext(repoPath: string, action: string): Promise<ToolResult> {
    const start = Date.now();
    try {
      const result = await this.request('/context/run', {
        repoPath,
        action,
      });
      return {
        success: result.success,
        data: result.data,
        duration: Date.now() - start,
      };
    } catch (err) {
      return {
        success: false,
        error: err instanceof Error ? err.message : 'Unknown error',
        duration: Date.now() - start,
      };
    }
  }

  async runExperiment(
    repoPath: string,
    spec: string,
    variants: number = 3
  ): Promise<ToolResult> {
    const start = Date.now();
    try {
      const result = await this.request('/experiment/run', {
        repoPath,
        spec,
        variants,
      });
      return {
        success: result.success,
        data: result.data,
        duration: Date.now() - start,
      };
    } catch (err) {
      return {
        success: false,
        error: err instanceof Error ? err.message : 'Unknown error',
        duration: Date.now() - start,
      };
    }
  }

  private async request(endpoint: string, body: object): Promise<{ success: boolean; data?: unknown; error?: string }> {
    const response = await fetch(`${config.python.coreApiUrl}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      return { success: false, error: `HTTP ${response.status}` };
    }

    return { success: true, data: await response.json() };
  }

  async healthCheck(): Promise<boolean> {
    try {
      return await this.bridge.healthCheck();
    } catch {
      return false;
    }
  }
}
