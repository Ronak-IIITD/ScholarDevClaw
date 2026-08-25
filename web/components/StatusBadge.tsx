"use client";

import type { PipelineStepResult } from "@/lib/types";

interface StatusBadgeProps {
  status: "idle" | "running" | "completed" | "failed";
  label?: string;
  className?: string;
}

const statusConfig = {
  idle: { class: "badge-idle", label: "Idle" },
  running: { class: "badge-running", label: "Running" },
  completed: { class: "badge-ok", label: "Completed" },
  failed: { class: "badge-failed", label: "Failed" },
} as const;

export function StatusBadge({ status, label, className = "" }: StatusBadgeProps) {
  const config = statusConfig[status];
  return (
    <span className={`badge ${config.class} ${className}`}>
      <span className="dot" aria-hidden="true"></span>
      {label || config.label}
    </span>
  );
}

export function StepStatusBadge({ step }: { step: PipelineStepResult }) {
  return <StatusBadge status={step.status} label={step.status} />;
}