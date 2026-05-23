import { spawn } from 'child_process';
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { dirname, resolve as pathResolve } from 'path';
import { fileURLToPath } from 'url';
import { logger } from '../utils/logger.js';

export interface PhaseResult {
  success: boolean;
  data?: unknown;
  error?: string;
}

export interface BridgeOptions {
  /** Enable local-only mode - runs Python CLI directly without API server */
  localOnly?: boolean;
  /** Path to Python executable */
  pythonPath?: string;
  /** Path to the core module */
  corePath?: string;
  /** Timeout in ms for subprocess operations */
  timeout?: number;
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
  research_spec?: Record<string, unknown>;
  researchSpec?: Record<string, unknown>;
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
  algorithmName?: string;
  paperReference?: string;
  researchSpec?: Record<string, unknown>;
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
  logs?: string;
  error?: string;
}

export class PythonSubprocessBridge {
  private pythonCmd: string;
  private corePath: string;
  private localOnly: boolean;
  private timeout: number;

  constructor(pythonCmd: string = 'python3', corePath: string = '../core', options?: BridgeOptions) {
    this.pythonCmd = options?.pythonPath || pythonCmd;
    this.corePath = options?.corePath || corePath;
    this.localOnly = options?.localOnly ?? false;
    this.timeout = options?.timeout ?? 120000;

    if (this.localOnly) {
      logger.info('Local-only mode enabled - running Python CLI directly');
      const coreDir = pathResolve(dirname(fileURLToPath(import.meta.url)), this.corePath);
      if (!existsSync(coreDir)) {
        logger.warn(`Core directory not found at ${coreDir}. Run from the agent/ directory.`);
      }
    }
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

      const cmd = this.pythonCmd;
      const procArgs = ['-m', modulePath, ...args];
      const workDir = this.getCoreWorkingDirectory();

      logger.debug('Running Python module', { cmd, args: procArgs, cwd: workDir });
      const proc = spawn(cmd, procArgs, {
        cwd: workDir,
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

  private async withJsonPayloadFile<T>(
    prefix: string,
    payload: unknown,
    callback: (payloadPath: string) => Promise<T>,
  ): Promise<T> {
    const payloadDir = mkdtempSync(pathResolve(tmpdir(), `${prefix}-`));
    const payloadPath = pathResolve(payloadDir, 'payload.json');
    writeFileSync(payloadPath, JSON.stringify(payload), 'utf8');
    try {
      return await callback(payloadPath);
    } finally {
      rmSync(payloadDir, { recursive: true, force: true });
    }
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
    if (!mappingRecord) {
      return { success: false, error: 'Patch generation requires a mapping payload object' };
    }

    const path = repoPath || String(mappingRecord?.repoPath || mappingRecord?.root_path || '');
    if (!path) {
      return { success: false, error: 'Patch generation requires a repository path' };
    }

    const result = await this.withJsonPayloadFile('sdc-generate', mappingRecord, (payloadPath) =>
      this.runPythonModule('scholardevclaw.cli', [
        'generate',
        path,
        '__mapping_payload__',
        '--mapping-json',
        payloadPath,
        '--output-json',
      ]),
    );
    if (!result.success) {
      return result;
    }
    return { success: true, data: this.normalizePatch(result.data) };
  }

  async validate(patch: unknown, repoPath?: string): Promise<PhaseResult> {
    logger.info('Running validation');
    const patchRecord = this.asRecord(patch);
    if (!patchRecord) {
      return { success: false, error: 'Validation requires a patch payload object' };
    }

    const path = repoPath || String(patchRecord.repoPath || patchRecord.root_path || '');
    if (!path) {
      return { success: false, error: 'Validation requires a repository path' };
    }

    const result = await this.withJsonPayloadFile('sdc-validate', patchRecord, (payloadPath) =>
      this.runPythonModule('scholardevclaw.cli', [
        'validate',
        path,
        '--patch-json',
        payloadPath,
        '--output-json',
      ]),
    );
    if (!result.success) {
      return result;
    }
    return { success: true, data: this.normalizeValidationResult(result.data) };
  }

  async healthCheck(): Promise<boolean> {
    try {
      const result = await this.runPythonModule('scholardevclaw.cli', ['--version']);
      return result.success;
    } catch {
      return false;
    }
  }

  /** Check if running in local-only mode (no API server needed) */
  isLocalOnly(): boolean {
    return this.localOnly;
  }

  /** Set or disable local-only mode */
  setLocalOnly(enabled: boolean): void {
    this.localOnly = enabled;
    logger.info(`Local-only mode ${enabled ? 'enabled' : 'disabled'}`);
  }

  /** Auto-detect if we should use local mode */
  static async detectLocalMode(pythonCmd: string = 'python3'): Promise<boolean> {
    try {
      const bridge = new PythonSubprocessBridge(pythonCmd);
      const healthy = await bridge.healthCheck();
      return healthy;
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

    const researchSpec = this.asRecord(record.researchSpec ?? record.research_spec) || undefined;

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
      ...(record.algorithmName || record.algorithm_name
        ? { algorithmName: String(record.algorithmName || record.algorithm_name || '') }
        : {}),
      ...(record.paperReference || record.paper_reference
        ? { paperReference: String(record.paperReference || record.paper_reference || '') }
        : {}),
      ...(researchSpec ? { researchSpec } : {}),
    };
  }

  private normalizeValidationResult(data: unknown): ValidationResult {
    const record = this.asRecord(data) || {};
    const baselineMetrics =
      this.asRecord(record.baselineMetrics) || this.asRecord(record.baseline_metrics) || undefined;
    const newMetrics =
      this.asRecord(record.newMetrics) || this.asRecord(record.new_metrics) || undefined;
    const rawComparison = this.asRecord(record.comparison) || undefined;
    const comparison = rawComparison
      ? ({
          ...rawComparison,
          ...(rawComparison.loss_change !== undefined && rawComparison.lossChange === undefined
            ? { lossChange: Number(rawComparison.loss_change) }
            : {}),
          ...(rawComparison.speedup !== undefined ? { speedup: Number(rawComparison.speedup) } : {}),
          ...(rawComparison.passed !== undefined ? { passed: Boolean(rawComparison.passed) } : {}),
        } as ValidationResult['comparison'])
      : undefined;

    return {
      passed: Boolean(record.passed),
      stage: String(record.stage || ''),
      baselineMetrics: baselineMetrics
        ? {
            loss: Number(baselineMetrics.loss || 0),
            perplexity: Number(baselineMetrics.perplexity || 0),
            tokensPerSecond: Number(
              baselineMetrics.tokensPerSecond ?? baselineMetrics.tokens_per_second ?? 0,
            ),
            memoryMb: Number(baselineMetrics.memoryMb ?? baselineMetrics.memory_mb ?? 0),
          }
        : undefined,
      newMetrics: newMetrics
        ? {
            loss: Number(newMetrics.loss || 0),
            perplexity: Number(newMetrics.perplexity || 0),
            tokensPerSecond: Number(newMetrics.tokensPerSecond ?? newMetrics.tokens_per_second ?? 0),
            memoryMb: Number(newMetrics.memoryMb ?? newMetrics.memory_mb ?? 0),
          }
        : undefined,
      comparison,
      logs: String(record.logs || ''),
      error: record.error ? String(record.error) : undefined,
    };
  }
}
