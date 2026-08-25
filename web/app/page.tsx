"use client";

import { useEffect, useState } from "react";
import { Nav } from "@/components/Nav";
import { StatusBadge } from "@/components/StatusBadge";
import { StepTimeline } from "@/components/StepTimeline";
import { MetricCard } from "@/components/MetricCard";
import { LogFeed } from "@/components/LogFeed";
import { api } from "@/lib/api";
import { usePipelineSocket } from "@/lib/usePipelineSocket";
import type { PipelineRunStatus, PipelineRunRequest, HealthResponse, SpecSummary } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Dashboard() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [specs, setSpecs] = useState<SpecSummary[]>([]);
  const [runStatus, setRunStatus] = useState<PipelineRunStatus | null>(null);
  const [repoPath, setRepoPath] = useState("");
  const [selectedSpecs, setSelectedSpecs] = useState<string[]>([]);
  const [skipValidate, setSkipValidate] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [logLines, setLogLines] = useState<string[]>([]);

  const addLog = (line: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogLines((prev) => [...prev, `[${timestamp}] ${line}`]);
  };

  const loadHealth = async () => {
    try {
      const h = await api.health();
      setHealth(h);
    } catch (e) {
      console.error("Health check failed:", e);
    }
  };

  const loadSpecs = async () => {
    try {
      const s = await api.specs.list();
      setSpecs(s);
    } catch (e) {
      console.error("Failed to load specs:", e);
    }
  };

  const loadPipelineStatus = async () => {
    try {
      const status = await api.pipeline.status();
      setRunStatus(status);
    } catch (e) {
      console.error("Failed to load pipeline status:", e);
    }
  };

  const { connected, runStatus: wsRunStatus } = usePipelineSocket({
    onMessage: (msg) => {
      if (msg.type === "pipeline_step") {
        addLog(`[${msg.step}] ${msg.status}${msg.duration ? ` (${msg.duration}s)` : ""}`);
      } else if (msg.type === "pipeline_complete") {
        addLog(`Pipeline completed in ${msg.total_seconds}s`);
      } else if (msg.type === "pipeline_error") {
        addLog(`Pipeline failed: ${msg.error}`);
      }
    },
  });

  // Merge WS status with local state
  const effectiveRunStatus = wsRunStatus || runStatus;

  useEffect(() => {
    loadHealth();
    loadSpecs();
    loadPipelineStatus();
  }, []);

  const handleRun = async () => {
    if (!repoPath.trim()) {
      setError("Please enter a repository path");
      return;
    }
    setError(null);
    setRunning(true);
    setLogLines([]);
    addLog(`Starting pipeline for ${repoPath}`);
    if (selectedSpecs.length) addLog(`Specs: ${selectedSpecs.join(", ")}`);
    else addLog("Specs: auto-suggest");

    try {
      const req: PipelineRunRequest = {
        repo_path: repoPath,
        spec_names: selectedSpecs,
        skip_validate: skipValidate,
        output_dir: null,
      };
      const result = await api.pipeline.run(req);
      setRunStatus(result);
      addLog(`Pipeline started (run_id: ${result.run_id})`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      addLog(`Error: ${msg}`);
    } finally {
      setRunning(false);
    }
  };

  const handleDemo = async () => {
    setError(null);
    setRunning(true);
    setLogLines([]);
    addLog("Starting demo pipeline (nanoGPT + rmsnorm)");

    try {
      const result = await api.pipeline.demo();
      setRunStatus(result);
      addLog(`Demo started (run_id: ${result.run_id})`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      addLog(`Error: ${msg}`);
    } finally {
      setRunning(false);
    }
  };

  const handleRefresh = () => {
    loadPipelineStatus();
    loadHealth();
  };

  const status = effectiveRunStatus?.status || "idle";
  const currentStep = effectiveRunStatus?.steps.find((s) => s.status === "running")?.step;

  return (
    <>
      <Nav />
      <main className="main" role="main">
        <header className="page-head">
          <div className="page-eyebrow">Dashboard</div>
          <h1 className="page-title">Pipeline Control</h1>
          <p className="page-sub">
            Run the full research-to-code pipeline: analyze → suggest → map → generate → validate
          </p>
        </header>

        {/* Server health + quick metrics */}
        <div className="grid grid-4" style={{ marginBottom: "1.5rem" }}>
          <MetricCard
            label="Server"
            value={health?.status === "ok" ? "Healthy" : "Unknown"}
            sub={`v${health?.version || "—"} • ${health?.spec_count || 0} specs`}
          />
          <MetricCard
            label="Pipeline"
            value={status.charAt(0).toUpperCase() + status.slice(1)}
            sub={effectiveRunStatus?.run_id ? `Run ${effectiveRunStatus.run_id}` : "No active run"}
          />
          <MetricCard
            label="WebSocket"
            value={connected ? "Connected" : "Disconnected"}
            sub={connected ? "Live updates active" : "Polling fallback"}
          />
          <MetricCard
            label="YOLO Mode"
            value={health?.yolo_mode ? "On" : "Off"}
            sub={health?.yolo_mode ? "Auto-apply enabled" : "Manual approval required"}
          />
        </div>

        {/* Run controls */}
        <div className="card" style={{ marginBottom: "1.5rem" }}>
          <div className="card-title">
            Run Pipeline
            <StatusBadge status={status} />
          </div>

          <div className="grid grid-2" style={{ gap: "1rem", marginBottom: "1rem" }}>
            <div className="field">
              <label htmlFor="repo-path">Repository Path</label>
              <input
                id="repo-path"
                type="text"
                className="input"
                placeholder="/path/to/your/repo"
                value={repoPath}
                onChange={(e) => setRepoPath(e.target.value)}
                disabled={running}
              />
            </div>
            <div className="field">
              <label>Specs (optional)</label>
              <select
                className="input"
                multiple
                value={selectedSpecs}
                onChange={(e) => {
                  const opts = Array.from(e.target.selectedOptions).map((o) => o.value);
                  setSelectedSpecs(opts);
                }}
                disabled={running}
              >
                {specs.map((s) => (
                  <option key={s.name} value={s.name}>
                    {s.algorithm} — {s.title}
                  </option>
                ))}
              </select>
              <small className="muted mono">Hold Ctrl/Cmd to select multiple. Empty = auto-suggest.</small>
            </div>
          </div>

          <div className="field" style={{ display: "flex", alignItems: "flex-end", gap: "1rem" }}>
            <label style={{ display: "flex", alignItems: "center", gap: "0.4rem", cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={skipValidate}
                onChange={(e) => setSkipValidate(e.target.checked)}
                disabled={running}
              />
              <span className="small">Skip validation benchmarks</span>
            </label>
            <div className="row" style={{ marginLeft: "auto" }}>
              <button
                className="btn btn-ghost"
                onClick={handleRefresh}
                disabled={running}
              >
                Refresh
              </button>
              <button className="btn" onClick={handleDemo} disabled={running}>
                Run Demo
              </button>
              <button className="btn" onClick={handleRun} disabled={running || !repoPath.trim()}>
                {running ? "Starting…" : "Run Pipeline"}
              </button>
            </div>
          </div>

          {error && (
            <div className="mono" style={{ color: "var(--err)", marginTop: "0.8rem" }}>
              Error: {error}
            </div>
          )}
        </div>

        {/* Live pipeline timeline */}
        <div className="card" style={{ marginBottom: "1.5rem" }}>
          <div className="card-title">
            Live Pipeline
            <span className="right mono small">
              {effectiveRunStatus?.run_id ? `Run ${effectiveRunStatus.run_id}` : "—"}
            </span>
          </div>
          <StepTimeline steps={effectiveRunStatus?.steps || []} currentStep={currentStep} />
        </div>

        {/* Log feed */}
        <div className="card">
          <div className="card-title">
            Log Feed
            <span className="right mono small">{logLines.length} lines</span>
          </div>
          <LogFeed lines={logLines} />
        </div>
      </main>
      <footer className="footer">
        <span>ScholarDevClaw Dashboard</span>
        <span>API: {API_BASE}</span>
      </footer>
    </>
  );
}