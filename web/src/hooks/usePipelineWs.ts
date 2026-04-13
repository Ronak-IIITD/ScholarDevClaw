import { useState, useEffect, useRef, useCallback } from "react";
import type {
  WsMessage,
  PipelineStepResult,
  PipelineRunStatus,
  TimelineEvent,
} from "@/types/api";
import { createPipelineWebSocket } from "@/lib/api";

/**
 * Hook to manage a WebSocket connection for real-time pipeline updates.
 * Reconnects automatically on disconnect.
 */
export function usePipelineWs() {
  const [connected, setConnected] = useState(false);
  const [liveSteps, setLiveSteps] = useState<PipelineStepResult[]>([]);
  const [pipelineStatus, setPipelineStatus] = useState<string>("idle");
  const [totalSeconds, setTotalSeconds] = useState<number>(0);
  const [timelineEvents, setTimelineEvents] = useState<TimelineEvent[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();
  const pingTimer = useRef<ReturnType<typeof setInterval>>();
  const eventCounter = useRef(0);

  const nextEventId = useCallback(() => {
    eventCounter.current += 1;
    return `evt-${Date.now()}-${eventCounter.current}`;
  }, []);

  const normalizeTimestamp = (value: unknown): number | undefined => {
    if (typeof value === "number" && Number.isFinite(value) && value > 0) {
      return value;
    }
    return undefined;
  };

  const isPipelineRunStatus = (value: unknown): value is PipelineRunStatus => {
    if (!value || typeof value !== "object") return false;
    const candidate = value as Record<string, unknown>;
    return (
      typeof candidate.run_id === "string" &&
      typeof candidate.status === "string" &&
      Array.isArray(candidate.steps)
    );
  };

  const buildEventsFromRun = useCallback((run: PipelineRunStatus): TimelineEvent[] => {
    const orderedSteps = [...(run.steps ?? [])].sort(
      (a, b) => (a.sequence ?? 0) - (b.sequence ?? 0)
    );

    const events: TimelineEvent[] = orderedSteps.map((step, index) => ({
      id: `${run.run_id || "run"}-step-${step.sequence ?? index + 1}`,
      eventType: "step",
      status: step.status,
      step: step.step,
      timestamp:
        step.finished_at ?? step.started_at ?? run.started_at ?? Date.now() / 1000,
      duration_seconds: step.duration_seconds,
      error: step.error ?? undefined,
    }));

    if (run.status === "completed") {
      events.push({
        id: `${run.run_id || "run"}-complete`,
        eventType: "complete",
        status: "completed",
        timestamp: run.finished_at || Date.now() / 1000,
        duration_seconds: run.total_seconds,
      });
    } else if (run.status === "failed") {
      const latestError = [...orderedSteps].reverse().find((s) => Boolean(s.error))?.error;
      events.push({
        id: `${run.run_id || "run"}-error`,
        eventType: "error",
        status: "failed",
        timestamp: run.finished_at || Date.now() / 1000,
        error: latestError ?? undefined,
      });
    }

    return events;
  }, []);

  const hydrateFromSnapshot = useCallback(
    (run: PipelineRunStatus) => {
      setLiveSteps(run.steps ?? []);
      setPipelineStatus(run.status ?? "idle");
      setTotalSeconds(run.total_seconds ?? 0);
      setTimelineEvents(buildEventsFromRun(run));
    },
    [buildEventsFromRun]
  );

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = createPipelineWebSocket();
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      // Ping every 25s to keep alive
      pingTimer.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "ping" }));
        }
      }, 25000);
    };

    ws.onmessage = (evt) => {
      try {
        const parsed: unknown = JSON.parse(evt.data);

        // Backward compatibility: server may send raw PipelineRunStatus snapshot.
        if (isPipelineRunStatus(parsed)) {
          hydrateFromSnapshot(parsed);
          return;
        }

        const msg = parsed as WsMessage;
        if (msg.type === "auth_ok") return;
        if (msg.type === "pong") return;

        if (msg.type === "pipeline_snapshot") {
          if (isPipelineRunStatus(msg.run)) {
            hydrateFromSnapshot(msg.run);
          }
          return;
        }

        if (msg.type === "pipeline_step") {
          setLiveSteps((prev) => {
            const existing = prev.findIndex((s) => s.step === msg.step);
            const startedAt = normalizeTimestamp(msg.started_at);
            const finishedAt = normalizeTimestamp(msg.finished_at);
            const computedDuration =
              startedAt !== undefined && finishedAt !== undefined
                ? Math.max(0, finishedAt - startedAt)
                : 0;

            const entry: PipelineStepResult = {
              step: msg.step,
              status: msg.status as PipelineStepResult["status"],
              sequence: msg.sequence,
              started_at: startedAt,
              finished_at: finishedAt,
              duration_seconds: msg.duration ?? computedDuration,
              data: msg.data ?? {},
              error: null,
            };
            if (existing >= 0) {
              const copy = [...prev];
              copy[existing] = entry;
              return copy;
            }
            return [...prev, entry];
          });

          setTimelineEvents((prev) => [
            ...prev,
            {
              id: nextEventId(),
              eventType: "step",
              status: msg.status,
              step: msg.step,
              timestamp:
                normalizeTimestamp(msg.finished_at) ??
                normalizeTimestamp(msg.started_at) ??
                Date.now() / 1000,
              duration_seconds: msg.duration,
            },
          ]);

          setPipelineStatus("running");
        } else if (msg.type === "pipeline_complete") {
          setPipelineStatus("completed");
          setTotalSeconds(msg.total_seconds);
          setTimelineEvents((prev) => [
            ...prev,
            {
              id: nextEventId(),
              eventType: "complete",
              status: msg.status,
              timestamp: Date.now() / 1000,
              duration_seconds: msg.total_seconds,
            },
          ]);
        } else if (msg.type === "pipeline_error") {
          setPipelineStatus("failed");
          setTimelineEvents((prev) => [
            ...prev,
            {
              id: nextEventId(),
              eventType: "error",
              status: "failed",
              timestamp: Date.now() / 1000,
              error: msg.error,
            },
          ]);
        }
      } catch {
        // ignore parse errors
      }
    };

    ws.onclose = () => {
      setConnected(false);
      if (pingTimer.current) clearInterval(pingTimer.current);
      // Auto-reconnect after 3s
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [hydrateFromSnapshot, nextEventId]);

  const disconnect = useCallback(() => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    if (pingTimer.current) clearInterval(pingTimer.current);
    wsRef.current?.close();
    wsRef.current = null;
    setConnected(false);
  }, []);

  const reset = useCallback(() => {
    setLiveSteps([]);
    setPipelineStatus("idle");
    setTotalSeconds(0);
    setTimelineEvents([]);
  }, []);

  useEffect(() => {
    connect();
    return disconnect;
  }, [connect, disconnect]);

  return {
    connected,
    liveSteps,
    pipelineStatus,
    totalSeconds,
    timelineEvents,
    reset,
  };
}
