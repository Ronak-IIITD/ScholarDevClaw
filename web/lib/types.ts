// Shared types for the dashboard

export interface SpecSummary {
  name: string;
  title: string;
  algorithm: string;
  category: string;
  replaces: string;
  arxiv_id: string;
  description: string;
}

export interface PipelineStepResult {
  step: string;
  status: "running" | "completed" | "failed";
  sequence: number;
  started_at: number;
  finished_at: number;
  duration_seconds: number;
  data: Record<string, unknown>;
  error: string | null;
}

export interface PipelineRunStatus {
  run_id: string;
  status: "idle" | "running" | "completed" | "failed";
  repo_path: string;
  spec_names: string[];
  steps: PipelineStepResult[];
  started_at: number;
  finished_at: number;
  total_seconds: number;
}

export interface PipelineRunRequest {
  repo_path: string;
  spec_names: string[];
  skip_validate: boolean;
  output_dir: string | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  spec_count: number;
  yolo_mode: boolean;
}

export interface WSMessage {
  type: string;
  run_id?: string;
  step?: string;
  status?: string;
  data?: Record<string, unknown>;
  duration?: number;
  sequence?: number;
  started_at?: number;
  finished_at?: number;
  error?: string;
  run?: PipelineRunStatus;
  total_seconds?: number;
}