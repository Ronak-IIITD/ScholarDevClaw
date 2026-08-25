"use client";

import type { PipelineStepResult } from "@/lib/types";
import { StepStatusBadge } from "./StatusBadge";

interface StepTimelineProps {
  steps: PipelineStepResult[];
  currentStep?: string;
}

function formatDuration(seconds: number): string {
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(1);
  return `${m}m ${s}s`;
}

export function StepTimeline({ steps, currentStep }: StepTimelineProps) {
  if (!steps.length) {
    return (
      <div className="empty">
        No pipeline steps yet
      </div>
    );
  }

  return (
    <ul className="timeline" role="list" aria-label="Pipeline steps">
      {steps.map((step, idx) => {
        const isCurrent = currentStep && step.step === currentStep && step.status === "running";
        const isLast = idx === steps.length - 1;
        return (
          <li
            key={step.sequence}
            className={`timeline-item ${step.status} ${isCurrent ? "running" : ""}`}
            style={{ "--last": isLast ? "true" : "false" } as React.CSSProperties}
          >
            <div className="timeline-step">
              <StepStatusBadge step={step} />
              <span className="name">{step.step.replace(/:/g, " / ")}</span>
              <span className="dur">{formatDuration(step.duration_seconds)}</span>
            </div>
            {step.data && Object.keys(step.data).length > 0 && (
              <div className="timeline-detail">
                {Object.entries(step.data)
                  .filter(([, v]) => v !== undefined && v !== null && v !== "")
                  .map(([k, v]) => (
                    <div key={k}>
                      <span className="mono muted">{k}:</span>{" "}
                      {Array.isArray(v)
                        ? v.map((item, i) => (
                            <span key={i} className="mono">
                              {typeof item === "object" ? JSON.stringify(item) : String(item)}
                              {i < v.length - 1 ? ", " : ""}
                            </span>
                          ))
                        : typeof v === "object"
                        ? JSON.stringify(v)
                        : String(v)}
                    </div>
                  ))}
                {step.error && <div className="err mono">Error: {step.error}</div>}
              </div>
            )}
          </li>
        );
      })}
    </ul>
  );
}