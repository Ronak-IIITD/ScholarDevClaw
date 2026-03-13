"""
Plugin manager for ScholarDevClaw.

Responsibilities:
  - File-based plugin discovery (``~/.scholardevclaw/plugins/*.py``)
  - ``setuptools`` entry-point discovery (group ``scholardevclaw.plugins``)
  - Built-in plugin loading from the ``plugins`` sub-package
  - Per-plugin persistent enable/disable state
  - Per-plugin configuration via ``plugins.toml``
  - Hook registration on load, unregistration on unload
  - Scaffold generation for new plugins
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .hooks import HookRegistry, get_hook_registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class PluginMetadata:
    """Metadata for a discovered or loaded plugin."""

    name: str
    version: str
    description: str
    author: str
    plugin_type: str
    entry_point: str | None = None
    source: str = "file"  # "file", "entrypoint", "builtin"
    enabled: bool = True
    hooks: list[str] = field(default_factory=list)


@dataclass
class Plugin:
    """A loaded plugin instance."""

    metadata: PluginMetadata
    module: Any
    instance: Any


# ---------------------------------------------------------------------------
# Plugin interfaces
# ---------------------------------------------------------------------------


class PluginInterface(ABC):
    """Base interface every plugin should implement."""

    @abstractmethod
    def initialize(self, config: dict[str, Any] | None = None) -> None:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    def register_hooks(self, registry: HookRegistry) -> None:
        """Override to register hook callbacks.

        Called automatically when the plugin is loaded.
        """

    def teardown(self) -> None:
        """Override for cleanup when the plugin is unloaded."""


class AnalyzerPlugin(PluginInterface):
    """Plugin that analyses a repository."""

    @abstractmethod
    def analyze(self, repo_path: str) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        pass


class SpecProviderPlugin(PluginInterface):
    """Plugin that provides research paper specifications."""

    @abstractmethod
    def get_specs(self) -> dict[str, dict[str, Any]]:
        pass

    @abstractmethod
    def search(self, query: str) -> list[dict[str, Any]]:
        pass


class ValidatorPlugin(PluginInterface):
    """Plugin that validates patches."""

    @abstractmethod
    def validate(self, repo_path: str, patch_result: dict[str, Any]) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_validation_type(self) -> str:
        pass


# ---------------------------------------------------------------------------
# Configuration persistence helpers
# ---------------------------------------------------------------------------

_CONFIG_FILENAME = "plugins.toml"
_STATE_FILENAME = "plugin_state.json"


def _config_dir() -> Path:
    """Return the ScholarDevClaw configuration directory."""
    d = Path.home() / ".scholardevclaw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_state(config_root: Path | None = None) -> dict[str, Any]:
    """Load persistent plugin state (enabled/disabled + per-plugin config)."""
    root = config_root or _config_dir()
    state_file = root / _STATE_FILENAME
    if not state_file.exists():
        return {"enabled": {}, "config": {}}
    try:
        return json.loads(state_file.read_text())
    except Exception:
        return {"enabled": {}, "config": {}}


def _save_state(state: dict[str, Any], config_root: Path | None = None) -> None:
    """Save persistent plugin state."""
    root = config_root or _config_dir()
    root.mkdir(parents=True, exist_ok=True)
    state_file = root / _STATE_FILENAME
    state_file.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Built-in plugin registry
# ---------------------------------------------------------------------------

_BUILTIN_PLUGINS = [
    "scholardevclaw.plugins.security",
    "scholardevclaw.plugins.rustlang",
    "scholardevclaw.plugins.javalang",
    "scholardevclaw.plugins.jsts",
    "scholardevclaw.plugins.auto_lint",
    "scholardevclaw.plugins.metrics_collector",
    "scholardevclaw.plugins.event_logger",
]


# ---------------------------------------------------------------------------
# PluginManager
# ---------------------------------------------------------------------------


class PluginManager:
    """Central plugin manager.

    Supports three discovery sources:
      1. Built-in plugins shipped with the package.
      2. File-based plugins in ``~/.scholardevclaw/plugins/``.
      3. ``setuptools`` entry-point plugins (group ``scholardevclaw.plugins``).
    """

    ENTRYPOINT_GROUP = "scholardevclaw.plugins"

    def __init__(
        self,
        plugin_dir: str | None = None,
        hook_registry: HookRegistry | None = None,
        config_root: Path | None = None,
    ):
        if plugin_dir:
            self.plugin_dir = Path(plugin_dir)
        else:
            self.plugin_dir = Path.home() / ".scholardevclaw" / "plugins"

        self.plugin_dir.mkdir(parents=True, exist_ok=True)

        self._config_root = config_root
        self._hook_registry = hook_registry or get_hook_registry()

        # Loaded plugins keyed by name.
        self._plugins: dict[str, Plugin] = {}
        self._analyzers: dict[str, AnalyzerPlugin] = {}
        self._spec_providers: dict[str, SpecProviderPlugin] = {}
        self._validators: dict[str, ValidatorPlugin] = {}

        # Persistent state.
        self._state = _load_state(self._config_root)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        _save_state(self._state, self._config_root)

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    def enable_plugin(self, name: str) -> None:
        """Mark a plugin as enabled (persisted across sessions)."""
        self._state.setdefault("enabled", {})[name] = True
        self._persist()
        logger.info("Plugin enabled: %s", name)

    def disable_plugin(self, name: str) -> None:
        """Mark a plugin as disabled (persisted across sessions)."""
        self._state.setdefault("enabled", {})[name] = False
        self._persist()
        # If currently loaded, unload.
        if name in self._plugins:
            self.unload_plugin(name)
        logger.info("Plugin disabled: %s", name)

    def is_enabled(self, name: str) -> bool:
        """Check whether *name* is enabled (default: ``True``)."""
        return self._state.get("enabled", {}).get(name, True)

    # ------------------------------------------------------------------
    # Per-plugin configuration
    # ------------------------------------------------------------------

    def get_plugin_config(self, name: str) -> dict[str, Any]:
        """Return per-plugin configuration dict."""
        return dict(self._state.get("config", {}).get(name, {}))

    def set_plugin_config(self, name: str, config: dict[str, Any]) -> None:
        """Persist per-plugin configuration."""
        self._state.setdefault("config", {})[name] = config
        self._persist()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_plugins(self) -> list[PluginMetadata]:
        """Discover all available plugins from all sources.

        Returns metadata for each discovered plugin.  Does NOT load them.
        """
        discovered: dict[str, PluginMetadata] = {}

        # 1. Built-in plugins.
        for module_path in _BUILTIN_PLUGINS:
            try:
                meta = self._probe_module(module_path, source="builtin")
                if meta:
                    meta.enabled = self.is_enabled(meta.name)
                    discovered[meta.name] = meta
            except Exception:
                continue

        # 2. File-based plugins.
        for file in sorted(self.plugin_dir.glob("*.py")):
            if file.name.startswith("_"):
                continue
            try:
                meta = self._probe_file(file)
                if meta and meta.name not in discovered:
                    meta.enabled = self.is_enabled(meta.name)
                    discovered[meta.name] = meta
            except Exception:
                continue

        # 3. Entrypoint-based plugins.
        for meta in self._discover_entrypoints():
            if meta.name not in discovered:
                meta.enabled = self.is_enabled(meta.name)
                discovered[meta.name] = meta

        return list(discovered.values())

    def _probe_module(self, module_path: str, *, source: str = "builtin") -> PluginMetadata | None:
        """Import a module and extract PLUGIN_METADATA without keeping it loaded."""
        try:
            mod = importlib.import_module(module_path)
        except Exception:
            return None
        raw = getattr(mod, "PLUGIN_METADATA", None)
        if not isinstance(raw, dict):
            return None

        hooks_list: list[str] = []
        inst = None
        factory = getattr(mod, "get_plugin_instance", None)
        if callable(factory):
            try:
                inst = factory()
                if hasattr(inst, "register_hooks"):
                    hooks_list = getattr(inst, "HOOK_POINTS", [])
            except Exception:
                pass

        return PluginMetadata(
            name=raw.get("name", module_path.rsplit(".", 1)[-1]),
            version=raw.get("version", "1.0.0"),
            description=raw.get("description", ""),
            author=raw.get("author", "Unknown"),
            plugin_type=raw.get("plugin_type", "custom"),
            entry_point=module_path,
            source=source,
            hooks=hooks_list,
        )

    def _probe_file(self, file: Path) -> PluginMetadata | None:
        """Load a file-based plugin and extract metadata."""
        try:
            spec = importlib.util.spec_from_file_location(file.stem, file)
            if not spec or not spec.loader:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception:
            return None

        raw = getattr(module, "PLUGIN_METADATA", None)
        if not isinstance(raw, dict):
            return None

        return PluginMetadata(
            name=raw.get("name", file.stem),
            version=raw.get("version", "1.0.0"),
            description=raw.get("description", ""),
            author=raw.get("author", "Unknown"),
            plugin_type=raw.get("plugin_type", "custom"),
            entry_point=file.stem,
            source="file",
        )

    def _discover_entrypoints(self) -> list[PluginMetadata]:
        """Discover plugins registered via setuptools entry_points."""
        results = []
        try:
            eps = importlib.metadata.entry_points()
            # Python 3.12+ returns a SelectableGroups; 3.10/3.11 returns dict.
            if isinstance(eps, dict):
                group = eps.get(self.ENTRYPOINT_GROUP, [])
            else:
                group = eps.select(group=self.ENTRYPOINT_GROUP)

            for ep in group:
                try:
                    plugin_cls_or_fn = ep.load()
                    raw: dict[str, Any] = {}
                    if hasattr(plugin_cls_or_fn, "PLUGIN_METADATA"):
                        raw = plugin_cls_or_fn.PLUGIN_METADATA
                    elif isinstance(plugin_cls_or_fn, dict):
                        raw = plugin_cls_or_fn

                    results.append(
                        PluginMetadata(
                            name=raw.get("name", ep.name),
                            version=raw.get("version", "1.0.0"),
                            description=raw.get("description", ""),
                            author=raw.get("author", "Unknown"),
                            plugin_type=raw.get("plugin_type", "custom"),
                            entry_point=f"{ep.group}:{ep.name}",
                            source="entrypoint",
                        )
                    )
                except Exception:
                    continue
        except Exception:
            pass
        return results

    # ------------------------------------------------------------------
    # Loading / unloading
    # ------------------------------------------------------------------

    def load_plugin(self, name: str) -> Plugin | None:
        """Load and activate a plugin by name.

        Searches built-in modules, file-based plugins, then entry-points.
        Returns ``None`` if the plugin cannot be found or loaded.
        """
        if name in self._plugins:
            return self._plugins[name]

        if not self.is_enabled(name):
            logger.info("Plugin '%s' is disabled — skipping load", name)
            return None

        plugin = (
            self._load_builtin(name)
            or self._load_from_file(name)
            or self._load_from_entrypoint(name)
        )
        if plugin is None:
            return None

        self._register(plugin)
        return plugin

    def load_all(self, *, include_disabled: bool = False) -> list[Plugin]:
        """Discover and load all available plugins.

        Skips disabled plugins unless *include_disabled* is ``True``.
        """
        loaded = []
        for meta in self.discover_plugins():
            if not include_disabled and not meta.enabled:
                continue
            plugin = self.load_plugin(meta.name)
            if plugin:
                loaded.append(plugin)
        return loaded

    def _load_builtin(self, name: str) -> Plugin | None:
        for module_path in _BUILTIN_PLUGINS:
            mod_name = module_path.rsplit(".", 1)[-1]
            if mod_name != name:
                continue
            try:
                mod = importlib.import_module(module_path)
                raw = getattr(mod, "PLUGIN_METADATA", None)
                if not isinstance(raw, dict):
                    return None
                instance = None
                factory = getattr(mod, "get_plugin_instance", None)
                if callable(factory):
                    instance = factory()
                config = self.get_plugin_config(name)
                if instance and hasattr(instance, "initialize"):
                    try:
                        instance.initialize(config or None)
                    except Exception:
                        pass
                return Plugin(
                    metadata=PluginMetadata(
                        name=raw.get("name", name),
                        version=raw.get("version", "1.0.0"),
                        description=raw.get("description", ""),
                        author=raw.get("author", "Unknown"),
                        plugin_type=raw.get("plugin_type", "custom"),
                        entry_point=module_path,
                        source="builtin",
                        enabled=True,
                    ),
                    module=mod,
                    instance=instance,
                )
            except Exception:
                return None
        return None

    def _load_from_file(self, name: str) -> Plugin | None:
        plugin_file = self.plugin_dir / f"{name}.py"
        if not plugin_file.exists():
            return None
        try:
            spec = importlib.util.spec_from_file_location(name, plugin_file)
            if not spec or not spec.loader:
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            raw = getattr(module, "PLUGIN_METADATA", None)
            if not isinstance(raw, dict):
                return None
            instance = None
            factory = getattr(module, "get_plugin_instance", None)
            if callable(factory):
                instance = factory()
            config = self.get_plugin_config(name)
            if instance and hasattr(instance, "initialize"):
                try:
                    instance.initialize(config or None)
                except Exception:
                    pass
            return Plugin(
                metadata=PluginMetadata(
                    name=raw.get("name", name),
                    version=raw.get("version", "1.0.0"),
                    description=raw.get("description", ""),
                    author=raw.get("author", "Unknown"),
                    plugin_type=raw.get("plugin_type", "custom"),
                    entry_point=name,
                    source="file",
                    enabled=True,
                ),
                module=module,
                instance=instance,
            )
        except Exception:
            return None

    def _load_from_entrypoint(self, name: str) -> Plugin | None:
        try:
            eps = importlib.metadata.entry_points()
            if isinstance(eps, dict):
                group = eps.get(self.ENTRYPOINT_GROUP, [])
            else:
                group = eps.select(group=self.ENTRYPOINT_GROUP)

            for ep in group:
                if ep.name != name:
                    continue
                plugin_obj = ep.load()
                instance = None
                raw: dict[str, Any] = {}
                if hasattr(plugin_obj, "PLUGIN_METADATA"):
                    raw = plugin_obj.PLUGIN_METADATA
                if hasattr(plugin_obj, "get_plugin_instance"):
                    instance = plugin_obj.get_plugin_instance()
                elif callable(plugin_obj):
                    instance = plugin_obj()
                return Plugin(
                    metadata=PluginMetadata(
                        name=raw.get("name", name),
                        version=raw.get("version", "1.0.0"),
                        description=raw.get("description", ""),
                        author=raw.get("author", "Unknown"),
                        plugin_type=raw.get("plugin_type", "custom"),
                        entry_point=f"{self.ENTRYPOINT_GROUP}:{name}",
                        source="entrypoint",
                        enabled=True,
                    ),
                    module=None,
                    instance=instance,
                )
        except Exception:
            pass
        return None

    def _register(self, plugin: Plugin) -> None:
        """Add a loaded plugin to internal registries + register hooks."""
        name = plugin.metadata.name
        ptype = plugin.metadata.plugin_type
        self._plugins[name] = plugin

        if ptype == "analyzer" and plugin.instance:
            self._analyzers[name] = plugin.instance
        elif ptype == "spec_provider" and plugin.instance:
            self._spec_providers[name] = plugin.instance
        elif ptype == "validator" and plugin.instance:
            self._validators[name] = plugin.instance

        # Register hooks if the plugin supports them.
        if plugin.instance and hasattr(plugin.instance, "register_hooks"):
            try:
                plugin.instance.register_hooks(self._hook_registry)
                logger.info("Plugin '%s' registered hooks", name)
            except Exception as exc:
                logger.warning("Plugin '%s' hook registration failed: %s", name, exc)

    def unload_plugin(self, name: str) -> None:
        """Unload a plugin, unregister its hooks, and call teardown."""
        if name not in self._plugins:
            return

        plugin = self._plugins[name]
        ptype = plugin.metadata.plugin_type

        # Teardown.
        if plugin.instance and hasattr(plugin.instance, "teardown"):
            try:
                plugin.instance.teardown()
            except Exception:
                pass

        # Unregister hooks.
        self._hook_registry.unregister_all(plugin_name=name)

        # Remove from type-specific registries.
        if ptype == "analyzer":
            self._analyzers.pop(name, None)
        elif ptype == "spec_provider":
            self._spec_providers.pop(name, None)
        elif ptype == "validator":
            self._validators.pop(name, None)

        del self._plugins[name]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_plugin(self, name: str) -> Plugin | None:
        return self._plugins.get(name)

    def list_plugins(self) -> list[PluginMetadata]:
        return [p.metadata for p in self._plugins.values()]

    def get_analyzer(self, name: str) -> AnalyzerPlugin | None:
        return self._analyzers.get(name)

    def list_analyzers(self) -> list[str]:
        return list(self._analyzers.keys())

    def get_spec_provider(self, name: str) -> SpecProviderPlugin | None:
        return self._spec_providers.get(name)

    def list_spec_providers(self) -> list[str]:
        return list(self._spec_providers.keys())

    def get_validator(self, name: str) -> ValidatorPlugin | None:
        return self._validators.get(name)

    def list_validators(self) -> list[str]:
        return list(self._validators.keys())

    @property
    def hook_registry(self) -> HookRegistry:
        return self._hook_registry

    # ------------------------------------------------------------------
    # Scaffold generation
    # ------------------------------------------------------------------

    def create_plugin_scaffold(
        self,
        name: str,
        plugin_type: str,
        output_dir: str | None = None,
    ) -> Path:
        if output_dir:
            target_dir = Path(output_dir)
        else:
            target_dir = self.plugin_dir

        target_dir.mkdir(parents=True, exist_ok=True)
        scaffold_file = target_dir / f"{name}.py"

        if plugin_type == "analyzer":
            content = _ANALYZER_SCAFFOLD.format(name=name)
        elif plugin_type == "spec_provider":
            content = _SPEC_PROVIDER_SCAFFOLD.format(name=name)
        elif plugin_type == "validator":
            content = _VALIDATOR_SCAFFOLD.format(name=name)
        elif plugin_type == "hook":
            content = _HOOK_SCAFFOLD.format(name=name)
        else:
            content = _GENERIC_SCAFFOLD.format(name=name)

        scaffold_file.write_text(content)
        return scaffold_file


# ---------------------------------------------------------------------------
# Scaffold templates
# ---------------------------------------------------------------------------

_ANALYZER_SCAFFOLD = '''"""
{name} - Custom Analyzer Plugin
"""

PLUGIN_METADATA = {{
    "name": "{name}",
    "version": "1.0.0",
    "description": "Custom analyzer for repository analysis",
    "author": "Your Name",
    "plugin_type": "analyzer",
}}


class {name}Analyzer:
    def initialize(self, config: dict | None = None) -> None:
        self.config = config or {{}}

    def get_name(self) -> str:
        return "{name}"

    def analyze(self, repo_path: str) -> dict:
        """Analyze the repository and return results"""
        return {{
            "languages": [],
            "frameworks": [],
            "patterns": {{}},
        }}

    def get_supported_languages(self) -> list[str]:
        return ["python"]

    def register_hooks(self, registry) -> None:
        """Register hook callbacks (optional)."""
        pass

    def teardown(self) -> None:
        """Cleanup on unload (optional)."""
        pass


def get_plugin_instance():
    return {name}Analyzer()
'''

_SPEC_PROVIDER_SCAFFOLD = '''"""
{name} - Custom Spec Provider Plugin
"""

PLUGIN_METADATA = {{
    "name": "{name}",
    "version": "1.0.0",
    "description": "Custom spec provider for research papers",
    "author": "Your Name",
    "plugin_type": "spec_provider",
}}


class {name}SpecProvider:
    def initialize(self, config: dict | None = None) -> None:
        self.config = config or {{}}

    def get_name(self) -> str:
        return "{name}"

    def get_specs(self) -> dict:
        """Return all available specs"""
        return {{}}

    def search(self, query: str) -> list[dict]:
        """Search for specs matching query"""
        return []

    def register_hooks(self, registry) -> None:
        """Register hook callbacks (optional)."""
        pass

    def teardown(self) -> None:
        """Cleanup on unload (optional)."""
        pass


def get_plugin_instance():
    return {name}SpecProvider()
'''

_VALIDATOR_SCAFFOLD = '''"""
{name} - Custom Validator Plugin
"""

PLUGIN_METADATA = {{
    "name": "{name}",
    "version": "1.0.0",
    "description": "Custom validator for patches",
    "author": "Your Name",
    "plugin_type": "validator",
}}


class {name}Validator:
    def initialize(self, config: dict | None = None) -> None:
        self.config = config or {{}}

    def get_name(self) -> str:
        return "{name}"

    def validate(self, repo_path: str, patch_result: dict) -> dict:
        """Validate the patch and return results"""
        return {{
            "passed": True,
            "issues": [],
        }}

    def get_validation_type(self) -> str:
        return "custom"

    def register_hooks(self, registry) -> None:
        """Register hook callbacks (optional)."""
        pass

    def teardown(self) -> None:
        """Cleanup on unload (optional)."""
        pass


def get_plugin_instance():
    return {name}Validator()
'''

_HOOK_SCAFFOLD = '''"""
{name} - Hook Plugin

This plugin uses the hook system to react to pipeline events.
"""
from scholardevclaw.plugins.hooks import HookPoint

PLUGIN_METADATA = {{
    "name": "{name}",
    "version": "1.0.0",
    "description": "Custom hook plugin",
    "author": "Your Name",
    "plugin_type": "hook",
}}


class {name}HookPlugin:
    HOOK_POINTS = [
        HookPoint.AFTER_ANALYZE.value,
        HookPoint.AFTER_GENERATE.value,
    ]

    def initialize(self, config: dict | None = None) -> None:
        self.config = config or {{}}

    def get_name(self) -> str:
        return "{name}"

    def register_hooks(self, registry) -> None:
        registry.register(
            HookPoint.AFTER_ANALYZE,
            self._on_after_analyze,
            plugin_name=self.get_name(),
        )
        registry.register(
            HookPoint.AFTER_GENERATE,
            self._on_after_generate,
            plugin_name=self.get_name(),
        )

    def _on_after_analyze(self, event) -> None:
        print(f"[{{self.get_name()}}] Analysis complete: {{event.stage}}")

    def _on_after_generate(self, event) -> None:
        print(f"[{{self.get_name()}}] Generation complete: {{event.stage}}")

    def teardown(self) -> None:
        pass


def get_plugin_instance():
    return {name}HookPlugin()
'''

_GENERIC_SCAFFOLD = '''"""
{name} - Custom Plugin
"""

PLUGIN_METADATA = {{
    "name": "{name}",
    "version": "1.0.0",
    "description": "Custom plugin",
    "author": "Your Name",
    "plugin_type": "custom",
}}


class {name}Plugin:
    def initialize(self, config: dict | None = None) -> None:
        self.config = config or {{}}

    def get_name(self) -> str:
        return "{name}"

    def register_hooks(self, registry) -> None:
        """Register hook callbacks (optional)."""
        pass

    def teardown(self) -> None:
        """Cleanup on unload (optional)."""
        pass


def get_plugin_instance():
    return {name}Plugin()
'''
