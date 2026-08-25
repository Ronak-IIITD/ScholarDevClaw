"use client";

import { useEffect, useState } from "react";
import { Nav } from "@/components/Nav";
import { StatusBadge } from "@/components/StatusBadge";
import { StepTimeline } from "@/components/StepTimeline";
import { LogFeed } from "@/components/LogFeed";
import { api } from "@/lib/api";
import { usePipelineSocket } from "@/lib/usePipelineSocket";
import type { PipelineRunStatus } from "@/lib/types";

export default function RunPage() {
  const [runStatus, setRunStatus] = useState<PipelineRunStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [logLines, setLogLines] = useState<string[]>([]);

  const addLog = (line: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogLines((prev) => [...prev, `[${timestamp}] ${line}`]);
  };

  const loadStatus = async () => {
    try {
      const status = await api.pipeline.status();
      setRunStatus(status);
    } catch (e) {
      console.error("Failed to load pipeline status:", e);
    } finally {
      setLoading(false);
    }
  };

  const { connected, runStatus: wsRunStatus } = usePipelineSocket({
    onMessage: (msg) => {
      if (msg.type === "pipeline_step") {
        addLog(`[${msg.step}] ${msg.status}${msg.duration ? ` (${msg.duration}s)` : ""}`);
        if (msg.data) {
          Object.entries(msg.data).forEach(([k, v]) => {
            addLog(`  ${k}: ${JSON.stringify(v)}`);
          });
        }
      } else if (msg.type === "pipeline_complete") {
        addLog(`✓ Pipeline completed in ${msg.total_seconds}s`);
      } else if (msg.type === "pipeline_error") {
        addLog(`✗ Pipeline failed: ${msg.error}`);
      } else if (msg.type === "pipeline_snapshot" && msg.run) {
        addLog(`Connected to run ${msg.run.run_id}`);
      }
    },
    onOpen: () => addLog("WebSocket connected"),
    onClose: () => addLog("WebSocket disconnected"),
  });

  // Merge WS status with local state
  const effectiveRunStatus = wsRunStatus || runStatus;

  useEffect(() => {
    loadStatus();
  }, []);

  if (loading) {
    return (
      <>
        <Nav />
        <main className="main"><div className="empty">Loading pipeline status…</div></main>
        <footer className="footer"><span>ScholarDevClaw Dashboard</span></footer>
      </>
    );
  }

  const status = effectiveRunStatus?.status || "idle";
  const currentStep = effectiveRunStatus?.steps.find((s) => s.status === "running")?.step;

  return (
    <>
      <Nav />
      <main className="main" role="main">
        <header className="page-head">
          <div className="page-eyebrow">Pipeline Run</div>
          <h1 className="page-title">Live Pipeline View</h1>
          <p className="page-sub">
            Real-time progress via WebSocket. Start a run from the Dashboard or API.
          </p>
        </header>

        <div className="grid grid-4" style={{ marginBottom: "1.5rem" }}>
          <div className="metric">
            <div className="label">Status</div>
            <div className="value">
              <StatusBadge status={status} />
            </div>
            <div className="sub">{effectiveRunStatus?.run_id ? `Run ${effectiveRunStatus.run_id}` : "No active run"}</div>
          </div>
          <div className="metric">
            <div className="label">Repository</div>
            <div className="value mono small">{effectiveRunStatus?.repo_path || "—"}</div>
          </div>
          <div className="metric">
            <div className="label">Specs</div>
            <div className="value mono small">
              {effectiveRunStatus?.spec_names.length ? effectiveRunStatus.spec_names.join(", ") : "—"}
            </div>
          </div>
          <div className="metric">
            <div className="label">WebSocket</div>
            <div className="value">
              <StatusBadge status={connected ? "completed" : "idle"} label={connected ? "Connected" : "Disconnected"} />
            </div>
          </div>
        </div>

        <div className="grid grid-2" style={{ gap: "1.5rem" }}>
          {/* Timeline */}
          <div className="card" style={{ minHeight: "400px" }}>
            <div className="card-title">
              Step Timeline
              <span className="right mono small">
                {effectiveRunStatus?.total_seconds ? `${effectiveRunStatus.total_seconds.toFixed(1)}s total` : "—"}
              </span>
            </div>
            <StepTimeline steps={effectiveRunStatus?.steps || []} currentStep={currentStep} />
          </div>

          {/* Log feed */}
          <div className="card" style={{ minHeight: "400px", display: "flex", flexDirection: "column" }}>
            <div className="card-title">
              Live Log
              <span className="right mono small">{logLines.length} lines</span>
            </div>
            <LogFeed lines={logLines} style={{ flex: 1 }} />
          </div>
        </div>

        {/* Raw JSON for debugging */}
        {effectiveRunStatus && (
          <details className="card" style={{ marginTop: "1.5rem" }}>
            <summary className="card-title mono small" style={{ cursor: "pointer" }}>
              Raw Run Data (JSON)
            </summary>
            <div className="code-block mono" style={{ marginTop: "0.8rem" }}>
              {JSON.stringify(effectiveRunStatus, null, 2)}
            </div>
          </details>
        )}
      </main>
      <footer className="footer">
        <span>ScholarDevClaw Dashboard</span>
        <span>WS: {connected ? "Live" : "Offline"}</span>
      </footer>
    </>
  );
}