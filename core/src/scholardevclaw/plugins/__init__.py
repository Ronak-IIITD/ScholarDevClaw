from __future__ import annotations

from .manager import PluginManager, Plugin, PluginMetadata
from .manager import AnalyzerPlugin, SpecProviderPlugin, ValidatorPlugin
from .manager import PluginInterface


def get_plugin_manager(plugin_dir: str | None = None) -> PluginManager:
    return PluginManager(plugin_dir)


__all__ = [
    "PluginManager",
    "Plugin",
    "PluginMetadata",
    "AnalyzerPlugin",
    "SpecProviderPlugin",
    "ValidatorPlugin",
    "PluginInterface",
    "get_plugin_manager",
]
