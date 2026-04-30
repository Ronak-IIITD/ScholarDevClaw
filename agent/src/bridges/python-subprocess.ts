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

  private static readonly SUBPROCESS_TIMEOUT_MS = 300_000; // 5 minutes

  private async runPythonModule(modulePath: string, args: string[] = []): Promise<PhaseResult> {
    return new Promise((promiseResolve) => {
      let settled = false;
      const resolve = (result: PhaseResult) => {
        if (settled) return;
        settled = true;
        clearTimeout(killTimer);
        promiseResolve(result);
      };

      const proc = spawn(this.pythonCmd, ['-m', modulePath, ...args], {
        cwd: this.getCoreWorkingDirectory(),
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      // SECURITY: Kill subprocess if it exceeds timeout to prevent hanging
      const killTimer = setTimeout(() => {
        if (!settled) {
          logger.error(`Python subprocess timed out after ${PythonSubprocessBridge.SUBPROCESS_TIMEOUT_MS}ms: ${modulePath}`);
          proc.kill('SIGKILL');
          resolve({ success: false, error: `Subprocess timed out after ${PythonSubprocessBridge.SUBPROCESS_TIMEOUT_MS}ms` });
        }
      }, PythonSubprocessBridge.SUBPROCESS_TIMEOUT_MS);

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
          resolve({ success: true, data: parsed ?? output });
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
    const args = sourceType === 'arxiv' 
      ? ['search', '--arxiv', paperSource, '--output-json']
      : ['search', '--pdf', paperSource, '--output-json'];
    return this.runPythonModule('scholardevclaw.cli', args);
  }

  async mapArchitecture(repoAnalysis: unknown, researchSpec: unknown): Promise<PhaseResult> {
    logger.info('Mapping architecture');
    // repoAnalysis and researchSpec are used to build map arguments
    const repoPath = (repoAnalysis as any)?.repoPath || '';
    const spec = (researchSpec as any)?.algorithm?.name || 'rmsnorm';
    return this.runPythonModule('scholardevclaw.cli', ['map', repoPath, spec, '--output-json']);
  }

  async generatePatch(mapping: unknown, repoPath?: string): Promise<PhaseResult> {
    logger.info('Generating patch');
    const spec = (mapping as any)?.strategy || 'rmsnorm';
    const path = repoPath || '';
    return this.runPythonModule('scholardevclaw.cli', ['generate', path, spec, '--output-json']);
  }

  async validate(patch: unknown, repoPath?: string): Promise<PhaseResult> {
    logger.info('Running validation');
    const path = repoPath || '';
    return this.runPythonModule('scholardevclaw.cli', ['validate', path, '--output-json']);
  }

  async healthCheck(): Promise<boolean> {
    try {
      const result = await this.runPythonModule('scholardevclaw.cli', ['--version']);
      return result.success;
    } catch {
      return false;
    }
  }
}
