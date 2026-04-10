import { useState, useEffect, useRef, useCallback } from "react";
import type { WsMessage, PipelineStepResult } from "@/types/api";
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
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();
  const pingTimer = useRef<ReturnType<typeof setInterval>>();

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
        const msg: WsMessage = JSON.parse(evt.data);
        if (msg.type === "auth_ok") return;
        if (msg.type === "pong") return;

        if (msg.type === "pipeline_step") {
          setLiveSteps((prev) => {
            const existing = prev.findIndex((s) => s.step === msg.step);
            const entry: PipelineStepResult = {
              step: msg.step,
              status: msg.status as PipelineStepResult["status"],
              duration_seconds: msg.duration ?? 0,
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
          setPipelineStatus("running");
        } else if (msg.type === "pipeline_complete") {
          setPipelineStatus("completed");
          setTotalSeconds(msg.total_seconds);
        } else if (msg.type === "pipeline_error") {
          setPipelineStatus("failed");
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
  }, []);

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
    reset,
  };
}
