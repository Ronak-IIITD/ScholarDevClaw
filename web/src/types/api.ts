// Types matching the backend Pydantic models in dashboard.py

export interface SpecSummary {
  name: string;
  title: string;
  algorithm: string;
  category: string;
  replaces: string;
  arxiv_id: string;
  description: string;
}

export interface PipelineRunRequest {
  repo_path: string;
  spec_names: string[];
  skip_validate: boolean;
  output_dir: string | null;
}

export interface PipelineStepResult {
  step: string;
  status: "running" | "completed" | "failed";
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

// WebSocket message types
export interface WsStepMessage {
  type: "pipeline_step";
  run_id: string;
  step: string;
  status: string;
  data?: Record<string, unknown>;
  duration?: number;
}

export interface WsCompleteMessage {
  type: "pipeline_complete";
  run_id: string;
  status: string;
  total_seconds: number;
}

export interface WsErrorMessage {
  type: "pipeline_error";
  run_id: string;
  error: string;
}

export interface WsPongMessage {
  type: "pong";
}

export type WsMessage =
  | WsStepMessage
  | WsCompleteMessage
  | WsErrorMessage
  | WsPongMessage;
