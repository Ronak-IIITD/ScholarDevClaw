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
  sequence?: number;
  started_at?: number;
  finished_at?: number;
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
  sequence?: number;
  started_at?: number;
  finished_at?: number;
  data?: Record<string, unknown>;
  duration?: number;
}

export interface WsSnapshotMessage {
  type: "pipeline_snapshot";
  run: PipelineRunStatus;
}

export interface WsAuthOkMessage {
  type: "auth_ok";
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

export interface TimelineEvent {
  id: string;
  eventType: "step" | "complete" | "error";
  status: string;
  step?: string;
  timestamp: number;
  duration_seconds?: number;
  error?: string;
}

export type WsMessage =
  | WsStepMessage
  | WsSnapshotMessage
  | WsAuthOkMessage
  | WsCompleteMessage
  | WsErrorMessage
  | WsPongMessage;
