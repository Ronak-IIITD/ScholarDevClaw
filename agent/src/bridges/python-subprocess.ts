import { spawn } from 'child_process';
import { resolve } from 'path';
import { logger } from '../utils/logger.js';

export interface PhaseResult {
  success: boolean;
  data?: unknown;
  error?: string;
}

export interface RepoAnalysisResult {
  repoName: string;
  architecture: {
    models: Array<{
      name: string;
      file: string;
      line: number;
      parent: string;
      components: Record<string, unknown>;
    }>;
    trainingLoop?: {
      file: string;
      line: number;
      optimizer: string;
      lossFn: string;
    };
  };
  dependencies: Record<string, unknown>;
  testSuite: {
    runner: string;
    testFiles: string[];
  };
}

export interface ResearchSpecResult {
  paper: {
    title: string;
    authors: string[];
    arxiv?: string;
    year: number;
  };
  algorithm: {
    name: string;
    replaces?: string;
    description: string;
    formula?: string;
  };
  implementation: {
    moduleName: string;
    parentClass: string;
    parameters: string[];
    codeTemplate: string;
  };
  changes: {
    type: string;
    targetPattern: string;
    insertionPoints: string[];
  };
}

export interface MappingResult {
  targets: Array<{
    file: string;
    line: number;
    currentCode: string;
    replacementRequired: boolean;
  }>;
  strategy: string;
  confidence: number;
}

export interface PatchResult {
  newFiles: Array<{
    path: string;
    content: string;
  }>;
  transformations: Array<{
    file: string;
    original: string;
    modified: string;
    changes: unknown[];
  }>;
  branchName: string;
}

export interface ValidationResult {
  passed: boolean;
  stage: string;
  baselineMetrics?: {
    loss: number;
    perplexity: number;
    tokensPerSecond: number;
    memoryMb: number;
  };
  newMetrics?: {
    loss: number;
    perplexity: number;
    tokensPerSecond: number;
    memoryMb: number;
  };
  comparison?: {
    lossChange: number;
    speedup: number;
    passed: boolean;
  };
  logs: string;
  error?: string;
}

export class PythonSubprocessBridge {
  private pythonCmd: string;
  private corePath: string;

  constructor(pythonCmd: string = 'python3', corePath: string = '../../core') {
    this.pythonCmd = pythonCmd;
    this.corePath = corePath;
  }

  private async runPython(script: string, args: string[] = []): Promise<PhaseResult> {
    return new Promise((resolve) => {
      const proc = spawn(this.pythonCmd, ['-m', `scholardevclaw.${script}`, ...args], {
        cwd: resolve(this.corePath),
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      let stdout = '';
      let stderr = '';

      proc.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      proc.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      proc.on('close', (code) => {
        if (code === 0) {
          try {
            const result = stdout.trim() ? JSON.parse(stdout) : {};
            resolve({ success: true, data: result });
          } catch {
            resolve({ success: true, data: stdout.trim() });
          }
        } else {
          logger.error(`Python script failed: ${stderr}`);
          resolve({ success: false, error: stderr || `Exit code: ${code}` });
        }
      });

      proc.on('error', (err) => {
        logger.error(`Failed to start Python: ${err.message}`);
        resolve({ success: false, error: err.message });
      });
    });
  }

  async analyzeRepo(repoPath: string): Promise<PhaseResult> {
    logger.info('Analyzing repository', { repoPath });
    return this.runPython('repo_intelligence.parser', [repoPath]);
  }

  async extractResearch(paperSource: string, sourceType: 'pdf' | 'arxiv' = 'pdf'): Promise<PhaseResult> {
    logger.info('Extracting research', { paperSource, sourceType });
    return this.runPython('research_intelligence.extractor', [paperSource, sourceType]);
  }

  async mapArchitecture(repoAnalysis: unknown, researchSpec: unknown): Promise<PhaseResult> {
    logger.info('Mapping architecture');
    return this.runPython('mapping.engine', []);
  }

  async generatePatch(mapping: unknown): Promise<PhaseResult> {
    logger.info('Generating patch');
    return this.runPython('patch_generation.generator', []);
  }

  async validate(patch: unknown): Promise<PhaseResult> {
    logger.info('Running validation');
    return this.runPython('validation.runner', []);
  }
}
