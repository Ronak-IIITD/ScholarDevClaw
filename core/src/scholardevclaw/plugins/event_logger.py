"""
event_logger — Built-in hook plugin that logs all pipeline events.

Hooks into every hook point and writes structured log entries.
Useful for debugging, auditing, and understanding pipeline flow.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .hooks import HookEvent, HookPoint, HookRegistry

logger = logging.getLogger(__name__)

PLUGIN_METADATA = {
    "name": "event_logger",
    "version": "1.0.0",
    "description": "Logs all pipeline events for debugging and auditing",
    "author": "ScholarDevClaw",
    "plugin_type": "hook",
}


class EventLoggerPlugin:
    """Logs every pipeline hook event with timing and payload summaries.

    Configuration keys:
        log_level (str): Python log level name (default ``"INFO"``).
        log_payloads (bool): Include payload key summary (default ``True``).
        max_events (int): Max events to buffer (default ``500``).
    """

    HOOK_POINTS = [hp.value for hp in HookPoint]

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}
        self._events: list[dict[str, Any]] = []
        self._max_events = 500

    def initialize(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._max_events = int(self.config.get("max_events", 500))

    def get_name(self) -> str:
        return "event_logger"

    def register_hooks(self, registry: HookRegistry) -> None:
        for hp in HookPoint:
            registry.register(
                hp,
                self._on_event,
                plugin_name=self.get_name(),
                priority=250,  # Run last.
            )

    def teardown(self) -> None:
        self._events.clear()

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------

    def _on_event(self, event: HookEvent) -> None:
        log_level_name = self.config.get("log_level", "INFO").upper()
        log_level = getattr(logging, log_level_name, logging.INFO)
        include_payloads = self.config.get("log_payloads", True)

        entry: dict[str, Any] = {
            "hook": event.hook.value,
            "stage": event.stage,
            "timestamp": time.time(),
            "cancelled": event.cancelled,
            "error_count": len(event.errors),
        }

        if include_payloads:
            entry["payload_keys"] = list(event.payload.keys())[:20]

        if event.metadata:
            entry["metadata_keys"] = list(event.metadata.keys())[:10]

        # Buffer the event.
        self._events.append(entry)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events :]

        # Emit log.
        msg = f"[event_logger] {event.hook.value}"
        if event.stage and event.stage != event.hook.value:
            msg += f" stage={event.stage}"
        if event.errors:
            msg += f" errors={len(event.errors)}"
        if event.cancelled:
            msg += " CANCELLED"

        logger.log(log_level, msg)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def events(self) -> list[dict[str, Any]]:
        """Return buffered events."""
        return list(self._events)

    def clear(self) -> None:
        """Clear the event buffer."""
        self._events.clear()

    def get_event_counts(self) -> dict[str, int]:
        """Return per-hook-point event counts."""
        counts: dict[str, int] = {}
        for ev in self._events:
            hook = ev.get("hook", "unknown")
            counts[hook] = counts.get(hook, 0) + 1
        return counts


def get_plugin_instance() -> EventLoggerPlugin:
    return EventLoggerPlugin()
