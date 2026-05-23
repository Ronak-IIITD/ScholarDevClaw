import type {
  SpecSummary,
  PipelineRunRequest,
  PipelineRunStatus,
} from "@/types/api";

export interface FromPaperResponse {
  researchSpec: Record<string, unknown>;
  mapping: Record<string, unknown>;
  patch: {
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
  };
  validation: {
    passed: boolean;
    stage: string;
    logs?: string;
    error?: string;
  };
  schemaVersion: string;
  payloadType: string;
}

const BASE = "";

const API_BASE = (import.meta.env.VITE_API_URL || BASE).replace(/\/$/, "");

function resolveWsUrl(): string {
  const explicit = import.meta.env.VITE_WS_URL;
  if (explicit) {
    return explicit;
  }

  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  return `${protocol}//${host}/api/ws/pipeline`;
}

function getWsAuthToken(): string {
  const envToken = import.meta.env.VITE_WS_AUTH_TOKEN as string | undefined;
  if (envToken && envToken.trim()) {
    return envToken.trim();
  }
  const local = window.localStorage.getItem("scholardevclaw_api_token") || "";
  return local.trim();
}

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const token = getToken();
  const isFormData =
    typeof FormData !== "undefined" && init?.body instanceof FormData;
  const headers: Record<string, string> = {
    ...(init?.headers as Record<string, string> || {}),
  };
  if (!isFormData && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  const res = await fetch(`${API_BASE}${url}`, {
    ...init,
    headers,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

function getToken(): string | null {
  const local = window.localStorage.getItem("scholardevclaw_api_token");
  return local || null;
}

export async function getSpecs(): Promise<SpecSummary[]> {
  return fetchJSON<SpecSummary[]>("/api/specs");
}

export async function getSpec(name: string): Promise<SpecSummary> {
  return fetchJSON<SpecSummary>(`/api/specs/${encodeURIComponent(name)}`);
}

export async function getPipelineStatus(): Promise<PipelineRunStatus> {
  return fetchJSON<PipelineRunStatus>("/api/pipeline/status");
}

export async function startPipelineRun(
  request: PipelineRunRequest
): Promise<PipelineRunStatus> {
  return fetchJSON<PipelineRunStatus>("/api/pipeline/run", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function startDemoRun(): Promise<PipelineRunStatus> {
  return fetchJSON<PipelineRunStatus>("/api/demo", {
    method: "POST",
  });
}

export async function runFromPaper(
  request: { paperSource: string; sourceType: "pdf" | "arxiv"; repoPath: string }
): Promise<FromPaperResponse> {
  return fetchJSON<FromPaperResponse>("/api/from-paper", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function runFromPaperUpload(
  file: File,
  repoPath: string
): Promise<FromPaperResponse> {
  const formData = new FormData();
  formData.set("repoPath", repoPath);
  formData.set("file", file);

  return fetchJSON<FromPaperResponse>("/api/from-paper/upload", {
    method: "POST",
    body: formData,
  });
}

export async function getHealth(): Promise<{ status: string; version?: string; spec_count?: number }> {
  return fetchJSON<{ status: string; version?: string; spec_count?: number }>("/health");
}

export function createPipelineWebSocket(): WebSocket {
  const ws = new WebSocket(resolveWsUrl());
  const token = getWsAuthToken();
  if (token) {
    ws.addEventListener(
      "open",
      () => {
        ws.send(JSON.stringify({ type: "auth", token }));
      },
      { once: true }
    );
  }
  return ws;
}
