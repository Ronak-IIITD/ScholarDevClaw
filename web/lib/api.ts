// Typed API client for ScholarDevClaw backend

import type {
  SpecSummary,
  PipelineRunStatus,
  PipelineRunRequest,
  HealthResponse,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";

function authHeaders(): HeadersInit {
  const h: HeadersInit = { "Content-Type": "application/json" };
  if (API_KEY) {
    h["Authorization"] = `Bearer ${API_KEY}`;
  }
  return h;
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { ...authHeaders(), ...init?.headers },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

export const api = {
  health: () => fetchJson<HealthResponse>("/health"),

  specs: {
    list: () => fetchJson<SpecSummary[]>("/api/specs"),
    get: (name: string) => fetchJson<SpecSummary>(`/api/specs/${name}`),
  },

  pipeline: {
    status: () => fetchJson<PipelineRunStatus>("/api/pipeline/status"),
    run: (req: PipelineRunRequest) =>
      fetchJson<PipelineRunStatus>("/api/pipeline/run", {
        method: "POST",
        body: JSON.stringify(req),
      }),
    demo: () =>
      fetchJson<PipelineRunStatus>("/api/demo", { method: "POST" }),
    stream: (runId: string) => `${API_BASE}/api/pipeline/stream/${runId}`,
    ws: () => `${API_BASE.replace(/^http/, "ws")}/api/ws/pipeline`,
  },

  repo: {
    analyze: (repoPath: string) =>
      fetchJson<unknown>("/repo/analyze", {
        method: "POST",
        body: JSON.stringify({ repoPath }),
      }),
  },

  research: {
    extract: (source: string, sourceType: "pdf" | "arxiv" = "arxiv") =>
      fetchJson<unknown>("/research/extract", {
        method: "POST",
        body: JSON.stringify({ source, sourceType }),
      }),
  },

  mapping: {
    map: (repoAnalysis: unknown, researchSpec: unknown) =>
      fetchJson<unknown>("/mapping/map", {
        method: "POST",
        body: JSON.stringify({ repo_analysis: repoAnalysis, research_spec: researchSpec }),
      }),
  },

  patch: {
    generate: (mapping: unknown, repoPath: string) =>
      fetchJson<unknown>("/patch/generate", {
        method: "POST",
        body: JSON.stringify({ mapping, repoPath }),
      }),
  },

  validation: {
    run: (patch: unknown, repoPath: string) =>
      fetchJson<unknown>("/validation/run", {
        method: "POST",
        body: JSON.stringify({ patch, repoPath }),
      }),
  },

  fromPaper: (repoPath: string, paperSource: string, sourceType: "arxiv" | "pdf" = "arxiv") =>
    fetchJson<unknown>("/from-paper", {
      method: "POST",
      body: JSON.stringify({ repo_path: repoPath, paper_source: paperSource, source_type: sourceType }),
    }),
};

export { API_BASE, API_KEY };