"""
Plugin hook system for ScholarDevClaw.

Defines pipeline hook points, the HookEvent dataclass, and the HookRegistry
that manages callback registration, ordering, and execution.

Hook points correspond to pipeline stages:
    on_before_analyze, on_after_analyze,
    on_before_suggest, on_after_suggest,
    on_before_search, on_after_search,
    on_before_map, on_after_map,
    on_before_generate, on_after_generate,
    on_patch_created,
    on_before_validate, on_after_validate,
    on_before_integrate, on_after_integrate,
    on_pipeline_start, on_pipeline_complete, on_pipeline_error.

Each callback receives a HookEvent with context data and may optionally
modify the payload (mutations propagate to subsequent hooks and the pipeline).
"""

from __future__ import annotations

import enum
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class HookPoint(str, enum.Enum):
    """Named hook points in the pipeline lifecycle."""

    # Pipeline-level
    PIPELINE_START = "on_pipeline_start"
    PIPELINE_COMPLETE = "on_pipeline_complete"
    PIPELINE_ERROR = "on_pipeline_error"

    # Analyze
    BEFORE_ANALYZE = "on_before_analyze"
    AFTER_ANALYZE = "on_after_analyze"

    # Suggest
    BEFORE_SUGGEST = "on_before_suggest"
    AFTER_SUGGEST = "on_after_suggest"

    # Search
    BEFORE_SEARCH = "on_before_search"
    AFTER_SEARCH = "on_after_search"

    # Map
    BEFORE_MAP = "on_before_map"
    AFTER_MAP = "on_after_map"

    # Generate
    BEFORE_GENERATE = "on_before_generate"
    AFTER_GENERATE = "on_after_generate"

    # Patch created (after file/transformation artifacts are ready)
    PATCH_CREATED = "on_patch_created"

    # Validate
    BEFORE_VALIDATE = "on_before_validate"
    AFTER_VALIDATE = "on_after_validate"

    # Integrate (full pipeline)
    BEFORE_INTEGRATE = "on_before_integrate"
    AFTER_INTEGRATE = "on_after_integrate"


@dataclass
class HookEvent:
    """Payload delivered to hook callbacks.

    Attributes:
        hook: The hook point that triggered this event.
        stage: Human-readable pipeline stage name.
        payload: Mutable dict of stage-specific data.  Callbacks may mutate
            this dict; mutations propagate to later hooks and back to the
            pipeline when the hook returns.
        metadata: Read-only metadata (repo_path, spec_name, timing, etc.).
        cancelled: Set to ``True`` inside a *before_** hook to skip the
            corresponding pipeline stage.  Ignored for *after_** hooks.
        errors: Accumulated error messages from hook execution.
    """

    hook: HookPoint
    stage: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False
    errors: list[str] = field(default_factory=list)


# Type alias for hook callbacks.
HookCallback = Callable[[HookEvent], None]


@dataclass
class _RegisteredHook:
    """Internal wrapper around a registered callback."""

    callback: HookCallback
    plugin_name: str
    priority: int  # lower == runs first


class HookRegistry:
    """Central registry for hook callbacks.

    Thread-safety note: this class is **not** thread-safe.  It is designed for
    single-threaded pipeline execution (the standard ScholarDevClaw model).
    """

    def __init__(self) -> None:
        self._hooks: dict[HookPoint, list[_RegisteredHook]] = {hp: [] for hp in HookPoint}
        self._execution_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        hook_point: HookPoint | str,
        callback: HookCallback,
        *,
        plugin_name: str = "<anonymous>",
        priority: int = 100,
    ) -> None:
        """Register a callback for a hook point.

        Args:
            hook_point: A ``HookPoint`` enum member or its string value.
            callback: Callable that accepts a ``HookEvent``.
            plugin_name: Name of the plugin registering this hook.
            priority: Execution order — lower values run first (default 100).
        """
        hp = self._resolve(hook_point)
        entry = _RegisteredHook(callback=callback, plugin_name=plugin_name, priority=priority)
        self._hooks[hp].append(entry)
        # Keep sorted by priority (stable sort preserves registration order
        # for equal priorities).
        self._hooks[hp].sort(key=lambda h: h.priority)

    def unregister(self, hook_point: HookPoint | str, *, plugin_name: str) -> int:
        """Remove all callbacks registered by *plugin_name* for a hook point.

        Returns the number of callbacks removed.
        """
        hp = self._resolve(hook_point)
        before = len(self._hooks[hp])
        self._hooks[hp] = [h for h in self._hooks[hp] if h.plugin_name != plugin_name]
        return before - len(self._hooks[hp])

    def unregister_all(self, *, plugin_name: str) -> int:
        """Remove all callbacks registered by *plugin_name* across all hook points.

        Returns the total number of callbacks removed.
        """
        total = 0
        for hp in HookPoint:
            total += self.unregister(hp, plugin_name=plugin_name)
        return total

    def clear(self) -> None:
        """Remove all registered hooks."""
        for hp in HookPoint:
            self._hooks[hp].clear()
        self._execution_log.clear()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def fire(
        self,
        hook_point: HookPoint | str,
        *,
        stage: str = "",
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HookEvent:
        """Fire all callbacks registered for *hook_point* and return the event.

        Callbacks are invoked in priority order (lower first).  If a callback
        raises an exception the error is logged and recorded in
        ``event.errors`` but execution continues with the next callback.

        Args:
            hook_point: The hook point to fire.
            stage: Human-readable stage label (e.g. ``"analyze"``).
            payload: Mutable data dict for the event.
            metadata: Read-only metadata dict.

        Returns:
            The ``HookEvent`` (possibly mutated by callbacks).
        """
        hp = self._resolve(hook_point)
        event = HookEvent(
            hook=hp,
            stage=stage or hp.value,
            payload=payload if payload is not None else {},
            metadata=metadata if metadata is not None else {},
        )

        callbacks = self._hooks[hp]
        if not callbacks:
            return event

        for entry in callbacks:
            t0 = time.monotonic()
            try:
                entry.callback(event)
            except Exception as exc:
                error_msg = (
                    f"Hook error [{entry.plugin_name}] on {hp.value}: {type(exc).__name__}: {exc}"
                )
                logger.warning(error_msg)
                event.errors.append(error_msg)
            elapsed = time.monotonic() - t0
            self._execution_log.append(
                {
                    "hook": hp.value,
                    "plugin": entry.plugin_name,
                    "elapsed_ms": round(elapsed * 1000, 2),
                    "error": event.errors[-1]
                    if event.errors
                    and event.errors[-1].startswith(f"Hook error [{entry.plugin_name}]")
                    else None,
                }
            )

        return event

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_hooks(self, hook_point: HookPoint | str | None = None) -> list[dict[str, Any]]:
        """Return a list of registered hooks.

        If *hook_point* is ``None``, returns hooks for all points.
        """
        points = [self._resolve(hook_point)] if hook_point is not None else list(HookPoint)
        result = []
        for hp in points:
            for entry in self._hooks[hp]:
                result.append(
                    {
                        "hook": hp.value,
                        "plugin": entry.plugin_name,
                        "priority": entry.priority,
                    }
                )
        return result

    def get_execution_log(self) -> list[dict[str, Any]]:
        """Return the execution log (list of dicts with timing info)."""
        return list(self._execution_log)

    def clear_execution_log(self) -> None:
        """Clear the execution log."""
        self._execution_log.clear()

    def has_hooks(self, hook_point: HookPoint | str) -> bool:
        """Return whether any callbacks are registered for *hook_point*."""
        hp = self._resolve(hook_point)
        return len(self._hooks[hp]) > 0

    @property
    def hook_count(self) -> int:
        """Total number of registered callbacks across all hook points."""
        return sum(len(cbs) for cbs in self._hooks.values())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(hook_point: HookPoint | str | None) -> HookPoint:
        if hook_point is None:
            raise ValueError("hook_point must not be None")
        if isinstance(hook_point, HookPoint):
            return hook_point
        # Accept either the enum name or value string.
        try:
            return HookPoint(hook_point)
        except ValueError:
            pass
        try:
            return HookPoint[hook_point.upper()]
        except KeyError:
            pass
        raise ValueError(
            f"Unknown hook point: {hook_point!r}. Valid values: {[hp.value for hp in HookPoint]}"
        )


# ------------------------------------------------------------------
# Module-level singleton (lazy)
# ------------------------------------------------------------------

_global_registry: HookRegistry | None = None


def get_hook_registry() -> HookRegistry:
    """Return the global ``HookRegistry`` singleton."""
    global _global_registry
    if _global_registry is None:
        _global_registry = HookRegistry()
    return _global_registry
