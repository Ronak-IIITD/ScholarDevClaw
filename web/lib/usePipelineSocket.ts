// WebSocket hook for real-time pipeline updates

import { useEffect, useRef, useState, useCallback } from "react";
import { api, API_KEY } from "./api";
import type { PipelineRunStatus, WSMessage } from "./types";

interface UsePipelineSocketOptions {
  onMessage?: (msg: WSMessage) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (err: Event) => void;
  autoConnect?: boolean;
}

export function usePipelineSocket(options: UsePipelineSocketOptions = {}) {
  const { onMessage, onOpen, onClose, onError, autoConnect = true } = options;
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null);
  const [runStatus, setRunStatus] = useState<PipelineRunStatus | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const callbacks = useRef({ onMessage, onOpen, onClose, onError });

  useEffect(() => {
    callbacks.current = { onMessage, onOpen, onClose, onError };
  }, [onMessage, onOpen, onClose, onError]);

  const connect = useCallback(function connect() {
    if (wsRef.current) return;
    const ws = new WebSocket(api.pipeline.ws());
    wsRef.current = ws;

    ws.onopen = () => {
      if (API_KEY) {
        ws.send(JSON.stringify({ type: "auth", token: API_KEY }));
      }
      setConnected(true);
      reconnectAttempts.current = 0;
      callbacks.current.onOpen?.();
    };

    ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data);
        setLastMessage(msg);
        callbacks.current.onMessage?.(msg);

        // Handle pipeline snapshot on connect
        if (msg.type === "pipeline_snapshot" && msg.run) {
          setRunStatus(msg.run);
        }
        // Handle legacy snapshot (raw run payload)
        else if (msg.run_id && msg.status && !msg.type) {
          setRunStatus(msg as unknown as PipelineRunStatus);
        }
        // Handle step updates
        else if (msg.type === "pipeline_step" && msg.run_id && msg.step) {
          setRunStatus((prev) => {
            if (!prev || prev.run_id !== msg.run_id) return prev;
            const stepIdx = prev.steps.findIndex((s) => s.sequence === msg.sequence);
            const newStep = {
              step: msg.step!,
              status: msg.status as "running" | "completed" | "failed",
              sequence: msg.sequence || 0,
              started_at: msg.started_at || Date.now() / 1000,
              finished_at: msg.finished_at || Date.now() / 1000,
              duration_seconds: msg.duration || 0,
              data: msg.data || {},
              error: msg.error || null,
            };
            const newSteps = [...prev.steps];
            if (stepIdx >= 0) newSteps[stepIdx] = newStep;
            else newSteps.push(newStep);
            return { ...prev, steps: newSteps };
          });
        }
        // Handle completion
        else if (msg.type === "pipeline_complete" && msg.run_id) {
          setRunStatus((prev) => {
            if (!prev || prev.run_id !== msg.run_id) return prev;
            return {
              ...prev,
              status: "completed",
              finished_at: Date.now() / 1000,
              total_seconds: msg.total_seconds || 0,
            };
          });
        }
        // Handle error
        else if (msg.type === "pipeline_error" && msg.run_id) {
          setRunStatus((prev) => {
            if (!prev || prev.run_id !== msg.run_id) return prev;
            return {
              ...prev,
              status: "failed",
              finished_at: Date.now() / 1000,
              total_seconds: msg.total_seconds || 0,
            };
          });
        }
      } catch {
        // Ignore parse errors
      }
    };

    ws.onclose = () => {
      wsRef.current = null;
      setConnected(false);
      callbacks.current.onClose?.();

      // Auto-reconnect with backoff
      if (reconnectAttempts.current < maxReconnectAttempts) {
        const delay = Math.min(1000 * 2 ** reconnectAttempts.current, 30000);
        reconnectTimeoutRef.current = setTimeout(() => {
          reconnectTimeoutRef.current = null;
          reconnectAttempts.current++;
          connect();
        }, delay);
      }
    };

    ws.onerror = (err) => {
      callbacks.current.onError?.(err);
    };
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (wsRef.current) {
      const socket = wsRef.current;
      wsRef.current = null;
      socket.onopen = socket.onmessage = socket.onclose = socket.onerror = null;
      socket.close();
    }
    reconnectAttempts.current = 0;
    setConnected(false);
  }, []);

  const send = useCallback((msg: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  useEffect(() => {
    if (autoConnect) connect();
    return () => disconnect();
  }, [autoConnect, connect, disconnect]);

  return { connected, lastMessage, runStatus, send, connect, disconnect };
}
