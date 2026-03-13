"""
metrics_collector — Built-in hook plugin that collects timing and size metrics.

Hooks into pipeline start/complete and all before/after stage hooks to
build a comprehensive metrics report with per-stage timing, payload sizes,
and aggregate statistics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from .hooks import HookEvent, HookPoint, HookRegistry

logger = logging.getLogger(__name__)

PLUGIN_METADATA = {
    "name": "metrics_collector",
    "version": "1.0.0",
    "description": "Collects pipeline timing and size metrics",
    "author": "ScholarDevClaw",
    "plugin_type": "hook",
}


@dataclass
class StageMetrics:
    """Timing and size data for a single pipeline stage."""

    stage: str
    started_at: float = 0.0
    finished_at: float = 0.0
    duration_ms: float = 0.0
    payload_keys: int = 0
    payload_size_estimate: int = 0
    error: str | None = None


class MetricsCollectorPlugin:
    """Collects per-stage timing and payload size metrics during pipeline execution.

    Access collected metrics via ``.metrics`` and ``.summary()`` after the
    pipeline has finished.
    """

    HOOK_POINTS = [
        HookPoint.PIPELINE_START.value,
        HookPoint.PIPELINE_COMPLETE.value,
        HookPoint.PIPELINE_ERROR.value,
        HookPoint.BEFORE_ANALYZE.value,
        HookPoint.AFTER_ANALYZE.value,
        HookPoint.BEFORE_SUGGEST.value,
        HookPoint.AFTER_SUGGEST.value,
        HookPoint.BEFORE_SEARCH.value,
        HookPoint.AFTER_SEARCH.value,
        HookPoint.BEFORE_MAP.value,
        HookPoint.AFTER_MAP.value,
        HookPoint.BEFORE_GENERATE.value,
        HookPoint.AFTER_GENERATE.value,
        HookPoint.BEFORE_VALIDATE.value,
        HookPoint.AFTER_VALIDATE.value,
        HookPoint.BEFORE_INTEGRATE.value,
        HookPoint.AFTER_INTEGRATE.value,
    ]

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}
        self._stage_starts: dict[str, float] = {}
        self._metrics: list[StageMetrics] = []
        self._pipeline_start: float = 0.0
        self._pipeline_end: float = 0.0

    def initialize(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def get_name(self) -> str:
        return "metrics_collector"

    def register_hooks(self, registry: HookRegistry) -> None:
        # Pipeline lifecycle.
        registry.register(
            HookPoint.PIPELINE_START,
            self._on_pipeline_start,
            plugin_name=self.get_name(),
            priority=10,  # Run early.
        )
        registry.register(
            HookPoint.PIPELINE_COMPLETE,
            self._on_pipeline_complete,
            plugin_name=self.get_name(),
            priority=200,  # Run late.
        )
        registry.register(
            HookPoint.PIPELINE_ERROR,
            self._on_pipeline_error,
            plugin_name=self.get_name(),
            priority=200,
        )

        # Per-stage before/after pairs.
        stage_pairs = [
            (HookPoint.BEFORE_ANALYZE, HookPoint.AFTER_ANALYZE, "analyze"),
            (HookPoint.BEFORE_SUGGEST, HookPoint.AFTER_SUGGEST, "suggest"),
            (HookPoint.BEFORE_SEARCH, HookPoint.AFTER_SEARCH, "search"),
            (HookPoint.BEFORE_MAP, HookPoint.AFTER_MAP, "map"),
            (HookPoint.BEFORE_GENERATE, HookPoint.AFTER_GENERATE, "generate"),
            (HookPoint.BEFORE_VALIDATE, HookPoint.AFTER_VALIDATE, "validate"),
            (HookPoint.BEFORE_INTEGRATE, HookPoint.AFTER_INTEGRATE, "integrate"),
        ]

        for before_hp, after_hp, stage_name in stage_pairs:
            # Capture stage_name in closure.
            registry.register(
                before_hp,
                self._make_before_cb(stage_name),
                plugin_name=self.get_name(),
                priority=10,
            )
            registry.register(
                after_hp,
                self._make_after_cb(stage_name),
                plugin_name=self.get_name(),
                priority=200,
            )

    def teardown(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_pipeline_start(self, event: HookEvent) -> None:
        self._pipeline_start = time.monotonic()
        self._metrics.clear()
        self._stage_starts.clear()

    def _on_pipeline_complete(self, event: HookEvent) -> None:
        self._pipeline_end = time.monotonic()
        event.payload["plugin_metrics"] = self.summary()

    def _on_pipeline_error(self, event: HookEvent) -> None:
        self._pipeline_end = time.monotonic()
        event.payload["plugin_metrics"] = self.summary()

    def _make_before_cb(self, stage: str):  # noqa: ANN202
        def _cb(event: HookEvent) -> None:
            self._stage_starts[stage] = time.monotonic()

        return _cb

    def _make_after_cb(self, stage: str):  # noqa: ANN202
        def _cb(event: HookEvent) -> None:
            started = self._stage_starts.pop(stage, None)
            finished = time.monotonic()
            duration = (finished - started) * 1000 if started else 0.0

            payload_keys = len(event.payload)
            size_est = self._estimate_size(event.payload)

            sm = StageMetrics(
                stage=stage,
                started_at=started or 0.0,
                finished_at=finished,
                duration_ms=round(duration, 2),
                payload_keys=payload_keys,
                payload_size_estimate=size_est,
                error=event.errors[-1] if event.errors else None,
            )
            self._metrics.append(sm)

        return _cb

    # ------------------------------------------------------------------
    # Metrics access
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> list[StageMetrics]:
        return list(self._metrics)

    def summary(self) -> dict[str, Any]:
        """Return a structured summary of all collected metrics."""
        total_ms = (self._pipeline_end - self._pipeline_start) * 1000 if self._pipeline_start else 0
        stages = [
            {
                "stage": m.stage,
                "duration_ms": m.duration_ms,
                "payload_keys": m.payload_keys,
                "payload_size_estimate": m.payload_size_estimate,
                "error": m.error,
            }
            for m in self._metrics
        ]
        return {
            "total_pipeline_ms": round(total_ms, 2),
            "stage_count": len(self._metrics),
            "stages": stages,
            "slowest_stage": max(stages, key=lambda s: s["duration_ms"])["stage"]
            if stages
            else None,
        }

    def reset(self) -> None:
        """Clear all collected metrics."""
        self._metrics.clear()
        self._stage_starts.clear()
        self._pipeline_start = 0.0
        self._pipeline_end = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_size(payload: dict[str, Any]) -> int:
        """Rough byte-size estimate of a payload dict."""
        try:
            import json

            return len(json.dumps(payload, default=str))
        except Exception:
            return 0


def get_plugin_instance() -> MetricsCollectorPlugin:
    return MetricsCollectorPlugin()
