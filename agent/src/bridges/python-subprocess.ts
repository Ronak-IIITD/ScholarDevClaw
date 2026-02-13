import { spawn } from 'child_process';
import { resolve as pathResolve } from 'path';
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

  private getCoreWorkingDirectory(): string {
    return pathResolve(process.cwd(), this.corePath);
  }

  private async runPythonModule(modulePath: string, args: string[] = []): Promise<PhaseResult> {
    return new Promise((promiseResolve) => {
      const proc = spawn(this.pythonCmd, ['-m', modulePath, ...args], {
        cwd: this.getCoreWorkingDirectory(),
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
          const output = stdout.trim();
          const parsed = this.parseJsonFromOutput(output);
          promiseResolve({ success: true, data: parsed ?? output });
        } else {
          logger.error(`Python script failed: ${stderr}`);
          promiseResolve({ success: false, error: stderr || `Exit code: ${code}` });
        }
      });

      proc.on('error', (err) => {
        logger.error(`Failed to start Python: ${err.message}`);
        promiseResolve({ success: false, error: err.message });
      });
    });
  }

  private parseJsonFromOutput(output: string): unknown | null {
    if (!output) {
      return {};
    }

    try {
      return JSON.parse(output);
    } catch {
      const lastJsonStart = output.lastIndexOf('\n{');
      if (lastJsonStart >= 0) {
        const candidate = output.slice(lastJsonStart + 1);
        try {
          return JSON.parse(candidate);
        } catch {
          return null;
        }
      }

      return null;
    }
  }

  async analyzeRepo(repoPath: string): Promise<PhaseResult> {
    logger.info('Analyzing repository', { repoPath });
    return this.runPythonModule('scholardevclaw.cli', ['analyze', repoPath, '--output-json']);
  }

  async extractResearch(paperSource: string, sourceType: 'pdf' | 'arxiv' = 'pdf'): Promise<PhaseResult> {
    logger.info('Extracting research', { paperSource, sourceType });
    return {
      success: false,
      error:
        'Subprocess mode does not support research extraction via module CLI. Use PythonHttpBridge for end-to-end phase execution.',
    };
  }

  async mapArchitecture(repoAnalysis: unknown, researchSpec: unknown): Promise<PhaseResult> {
    logger.info('Mapping architecture');
    void repoAnalysis;
    void researchSpec;
    return {
      success: false,
      error:
        'Subprocess mode does not support mapping execution. Use PythonHttpBridge for end-to-end phase execution.',
    };
  }

  async generatePatch(mapping: unknown): Promise<PhaseResult> {
    logger.info('Generating patch');
    void mapping;
    return {
      success: false,
      error:
        'Subprocess mode does not support patch generation execution. Use PythonHttpBridge for end-to-end phase execution.',
    };
  }

  async validate(patch: unknown, repoPath?: string): Promise<PhaseResult> {
    logger.info('Running validation');
    void patch;
    void repoPath;
    return {
      success: false,
      error:
        'Subprocess mode does not support validation execution. Use PythonHttpBridge for end-to-end phase execution.',
    };
  }
}
