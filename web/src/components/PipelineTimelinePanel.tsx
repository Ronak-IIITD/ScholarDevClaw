import type { PipelineStepResult, TimelineEvent } from "@/types/api";
import StatusBadge from "./StatusBadge";

interface PipelineTimelinePanelProps {
  steps: PipelineStepResult[];
  timelineEvents: TimelineEvent[];
  status: string;
}

function stepLabel(step?: string): string {
  if (!step) return "Pipeline event";
  if (step === "analyze") return "Repository analysis";
  if (step === "suggest") return "Research suggestions";
  if (step === "specs_resolved") return "Specs resolved";
  if (step.startsWith("map:")) return `Mapping: ${step.slice(4)}`;
  if (step.startsWith("generate:")) return `Patch generation: ${step.slice(9)}`;
  if (step.startsWith("validate:")) return `Validation: ${step.slice(9)}`;
  if (step === "error") return "Pipeline error";
  return step;
}

function formatTime(tsSeconds: number): string {
  return new Date(tsSeconds * 1000).toLocaleTimeString();
}

function formatDuration(value?: number): string | null {
  if (typeof value !== "number" || Number.isNaN(value) || value <= 0) return null;
  return `${value.toFixed(2)}s`;
}

export default function PipelineTimelinePanel({
  steps,
  timelineEvents,
  status,
}: PipelineTimelinePanelProps) {
  const fallbackEvents: TimelineEvent[] = [...steps]
    .sort((a, b) => (a.sequence ?? 0) - (b.sequence ?? 0))
    .map((step, index) => ({
      id: `fallback-step-${step.sequence ?? index + 1}`,
      eventType: "step",
      status: step.status,
      step: step.step,
      timestamp: step.finished_at ?? step.started_at ?? Date.now() / 1000,
      duration_seconds: step.duration_seconds,
      error: step.error ?? undefined,
    }));

  const events = timelineEvents.length > 0 ? timelineEvents : fallbackEvents;

  return (
    <section className="rounded-xl border border-gray-800 bg-gray-900/70 p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-200">Live timeline</h3>
        <StatusBadge status={status} />
      </div>

      {events.length === 0 ? (
        <p className="mt-4 text-sm text-gray-500">Timeline will appear after the run starts.</p>
      ) : (
        <ol className="mt-4 space-y-3" aria-label="Pipeline timeline events">
          {events.map((event) => {
            const duration = formatDuration(event.duration_seconds);
            const title =
              event.eventType === "complete"
                ? "Pipeline completed"
                : event.eventType === "error"
                ? "Pipeline failed"
                : stepLabel(event.step);

            return (
              <li key={event.id} className="rounded-lg border border-gray-800 bg-gray-950/50 p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="text-sm font-medium text-gray-200">{title}</p>
                  <StatusBadge status={event.status} />
                </div>
                <p className="mt-1 text-xs text-gray-500">
                  {formatTime(event.timestamp)}
                  {duration ? ` • ${duration}` : ""}
                </p>
                {event.error && (
                  <p className="mt-2 rounded bg-red-900/20 px-2 py-1 text-xs text-red-300">
                    {event.error}
                  </p>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
