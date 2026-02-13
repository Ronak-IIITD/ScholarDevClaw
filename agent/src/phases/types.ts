import type {
  RepoAnalysisResult,
  ResearchSpecResult,
  MappingResult,
  PatchResult,
  ValidationResult,
} from '../bridges/python-subprocess.js';

export interface Phase {
  id: number;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  confidence?: number;
  result?: unknown;
  error?: string;
}

export interface PhaseContext {
  repoPath: string;
  paperSource: string;
  sourceType: 'pdf' | 'arxiv';
}

export interface Phase1Context extends PhaseContext {
  repoAnalysis?: RepoAnalysisResult;
}

export interface Phase2Context extends PhaseContext {
  repoAnalysis: RepoAnalysisResult;
  researchSpec?: ResearchSpecResult;
}

export interface Phase3Context extends Phase2Context {
  researchSpec: ResearchSpecResult;
  mapping?: MappingResult;
}

export interface Phase4Context extends Phase3Context {
  mapping: MappingResult;
  patch?: PatchResult;
}

export interface Phase5Context extends Phase4Context {
  patch: PatchResult;
  validation?: ValidationResult;
}

export interface Phase6Context extends Phase5Context {
  validation: ValidationResult;
  report?: {
    metadata: unknown;
    summary: unknown;
    diff: string;
  };
}

export interface PhaseResult<T> {
  success: boolean;
  data?: T;
  error?: string;
  confidence?: number;
}
