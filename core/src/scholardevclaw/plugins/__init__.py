from __future__ import annotations

from .manager import PluginManager, Plugin, PluginMetadata
from .manager import AnalyzerPlugin, SpecProviderPlugin, ValidatorPlugin
from .manager import PluginInterface

from .hooks import HookPoint, HookEvent, HookRegistry, HookCallback, get_hook_registry

from .auto_lint import AutoLintPlugin
from .metrics_collector import MetricsCollectorPlugin
from .event_logger import EventLoggerPlugin


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
