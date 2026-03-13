from __future__ import annotations

from .auto_lint import AutoLintPlugin
from .event_logger import EventLoggerPlugin
from .hooks import HookCallback, HookEvent, HookPoint, HookRegistry, get_hook_registry
from .manager import (
    AnalyzerPlugin,
    Plugin,
    PluginInterface,
    PluginManager,
    PluginMetadata,
    SpecProviderPlugin,
    ValidatorPlugin,
)
from .metrics_collector import MetricsCollectorPlugin


def get_plugin_manager(plugin_dir: str | None = None) -> PluginManager:
    return PluginManager(plugin_dir)


__all__ = [
    # Manager
    "PluginManager",
    "Plugin",
    "PluginMetadata",
    "AnalyzerPlugin",
    "SpecProviderPlugin",
    "ValidatorPlugin",
    "PluginInterface",
    "get_plugin_manager",
    # Hooks
    "HookPoint",
    "HookEvent",
    "HookRegistry",
    "HookCallback",
    "get_hook_registry",
    # Built-in hook plugins
    "AutoLintPlugin",
    "MetricsCollectorPlugin",
    "EventLoggerPlugin",
]
