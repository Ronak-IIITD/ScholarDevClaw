import { spawn } from 'child_process';
import { existsSync } from 'fs';
import { dirname, resolve as pathResolve } from 'path';
import { fileURLToPath } from 'url';
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

  constructor(pythonCmd: string = 'python3', corePath: string = '../core') {
    this.pythonCmd = pythonCmd;
    this.corePath = corePath;
  }

  private getCoreWorkingDirectory(): string {
    const moduleDir = dirname(fileURLToPath(import.meta.url));
    const candidates = [
      pathResolve(process.cwd(), this.corePath),
      pathResolve(process.cwd(), '../core'),
      pathResolve(process.cwd(), 'core'),
      pathResolve(moduleDir, '../../../core'),
    ];
    const found = candidates.find((candidate) =>
      existsSync(pathResolve(candidate, 'src/scholardevclaw/cli.py')),
    );
    return found ?? candidates[0];
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
    const result = await this.runPythonModule('scholardevclaw.cli', [
      'analyze',
      repoPath,
      '--output-json',
    ]);
    if (!result.success) {
      return result;
    }
    return { success: true, data: this.normalizeRepoAnalysis(result.data, repoPath) };
  }

  async searchResearch(
    query: string,
    options: {
      includeArxiv?: boolean;
      includeWeb?: boolean;
      language?: string;
      maxResults?: string;
    } = {},
  ): Promise<PhaseResult> {
    logger.info('Searching research', { query });
    const args = ['search', query, '--output-json'];
    if (options.includeArxiv) args.push('--arxiv');
    if (options.includeWeb) args.push('--web');
    if (options.language) args.push('--language', options.language);
    if (options.maxResults) args.push('--max-results', options.maxResults);
    return this.runPythonModule('scholardevclaw.cli', args);
  }

  async runCli(args: string[]): Promise<PhaseResult> {
    return this.runPythonModule('scholardevclaw.cli', args);
  }

  async extractResearch(paperSource: string, sourceType: 'pdf' | 'arxiv' = 'pdf'): Promise<PhaseResult> {
    logger.info('Extracting research', { paperSource, sourceType });
    if (sourceType === 'pdf' && paperSource.trim().toLowerCase().endsWith('.pdf')) {
      return {
        success: false,
        error:
          'Subprocess mode does not support full PDF extraction. Use PythonHttpBridge for paper PDF ingestion.',
      };
    }

    const query = paperSource.startsWith('spec:') ? paperSource.slice('spec:'.length) : paperSource;
    const result = await this.searchResearch(query || paperSource, {
      includeArxiv: sourceType === 'arxiv' && !paperSource.startsWith('spec:'),
    });
    if (!result.success) {
      return result;
    }

    const spec = this.normalizeResearchSpec(result.data, query || paperSource);
    if (!spec) {
      return {
        success: false,
        error: `No research specification found for '${paperSource}'`,
      };
    }
    return { success: true, data: spec };
  }

  async mapArchitecture(repoAnalysis: unknown, researchSpec: unknown): Promise<PhaseResult> {
    logger.info('Mapping architecture');
    // repoAnalysis and researchSpec are used to build map arguments
    const repoRecord = this.asRecord(repoAnalysis);
    const repoPath = String(repoRecord?.repoPath || repoRecord?.root_path || '');
    const spec = this.inferSpecName(this.asRecord(this.asRecord(researchSpec)?.algorithm)?.name);
    const result = await this.runPythonModule('scholardevclaw.cli', ['map', repoPath, spec, '--output-json']);
    if (!result.success) {
      return result;
    }
    return { success: true, data: this.normalizeMapping(result.data) };
  }

  async generatePatch(mapping: unknown, repoPath?: string): Promise<PhaseResult> {
    logger.info('Generating patch');
    const mappingRecord = this.asRecord(mapping);
    const spec = this.inferSpecName(mappingRecord?.research_spec || mappingRecord?.strategy);
    const path = repoPath || '';
    const result = await this.runPythonModule('scholardevclaw.cli', [
      'generate',
      path,
      spec,
      '--use-specs',
      '--output-json',
    ]);
    if (!result.success) {
      return result;
    }
    return { success: true, data: this.normalizePatch(result.data) };
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

  private asRecord(value: unknown): Record<string, unknown> | null {
    return value !== null && typeof value === 'object' && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : null;
  }

  private asStringArray(value: unknown): string[] {
    return Array.isArray(value) ? value.map((item) => String(item)) : [];
  }

  private inferSpecName(value: unknown): string {
    const text = String(value || '').trim().toLowerCase();
    const knownSpecs = [
      'rmsnorm',
      'flashattention',
      'swiglu',
      'geglu',
      'gqa',
      'rope',
      'preln',
      'alibi',
      'qknorm',
    ];
    return knownSpecs.find((spec) => text.includes(spec)) || text || 'rmsnorm';
  }

  private normalizeResearchSpec(data: unknown, fallbackName: string): ResearchSpecResult | null {
    const record = this.asRecord(data);
    if (!record) {
      return null;
    }
    if (this.asRecord(record.paper) && this.asRecord(record.algorithm)) {
      return record as unknown as ResearchSpecResult;
    }

    const localSpecs = Array.isArray(record.local_specs) ? record.local_specs : [];
    const first = this.asRecord(localSpecs[0]);
    if (!first) {
      return null;
    }

    const name = this.inferSpecName(first.name || fallbackName);
    return {
      paper: {
        title: String(first.title || name),
        authors: this.asStringArray(first.authors),
        arxiv: first.arxiv ? String(first.arxiv) : undefined,
        year: Number(first.year || 0),
      },
      algorithm: {
        name,
        replaces: first.replaces ? String(first.replaces) : undefined,
        description: String(first.description || ''),
      },
      implementation: {
        moduleName: `${name}Module`,
        parentClass: '',
        parameters: [],
        codeTemplate: '',
      },
      changes: {
        type: 'replace',
        targetPattern: String(first.replaces || name),
        insertionPoints: [],
      },
    };
  }

  private normalizeRepoAnalysis(data: unknown, fallbackPath: string): RepoAnalysisResult {
    const record = this.asRecord(data) || {};
    if (this.asRecord(record.architecture) && this.asRecord(record.testSuite)) {
      return record as unknown as RepoAnalysisResult;
    }

    const rootPath = String(record.root_path || fallbackPath);
    const repoName = rootPath.split('/').filter(Boolean).at(-1) || rootPath || 'repository';
    return {
      repoName,
      architecture: {
        models: [],
        ...(Array.isArray(record.entry_points) && record.entry_points.length > 0
          ? {
              trainingLoop: {
                file: String(record.entry_points[0]),
                line: 0,
                optimizer: '',
                lossFn: '',
              },
            }
          : {}),
      },
      dependencies: {
        languages: Array.isArray(record.languages) ? record.languages : [],
        frameworks: Array.isArray(record.frameworks) ? record.frameworks : [],
        patterns: this.asRecord(record.patterns) || {},
      },
      testSuite: {
        runner: 'pytest',
        testFiles: this.asStringArray(record.test_files),
      },
      ...(rootPath ? { root_path: rootPath } : {}),
    } as RepoAnalysisResult;
  }

  private normalizeMapping(data: unknown): MappingResult {
    const record = this.asRecord(data) || {};
    const rawTargets = Array.isArray(record.targets) ? record.targets : [];
    return {
      targets: rawTargets.map((target) => {
        const targetRecord = this.asRecord(target) || {};
        return {
          file: String(targetRecord.file || ''),
          line: Number(targetRecord.line || 0),
          currentCode: String(targetRecord.currentCode || targetRecord.current_code || ''),
          replacementRequired: Boolean(
            targetRecord.replacementRequired ?? targetRecord.replacement_required ?? false,
          ),
        };
      }),
      strategy: String(record.strategy || ''),
      confidence: Number(record.confidence || 0),
      ...(record.research_spec ? { research_spec: record.research_spec } : {}),
    } as MappingResult;
  }

  private normalizePatch(data: unknown): PatchResult {
    const record = this.asRecord(data) || {};
    const rawFiles = Array.isArray(record.newFiles)
      ? record.newFiles
      : Array.isArray(record.new_files)
        ? record.new_files
        : [];
    const rawTransformations = Array.isArray(record.transformations) ? record.transformations : [];

    return {
      newFiles: rawFiles.map((item) => {
        const file = this.asRecord(item) || {};
        return {
          path: String(file.path || ''),
          content: String(file.content || ''),
        };
      }),
      transformations: rawTransformations.map((item) => {
        const transformation = this.asRecord(item) || {};
        return {
          file: String(transformation.file || ''),
          original: String(transformation.original || ''),
          modified: String(transformation.modified || ''),
          changes: Array.isArray(transformation.changes) ? transformation.changes : [],
        };
      }),
      branchName: String(record.branchName || record.branch_name || ''),
    };
  }
}
