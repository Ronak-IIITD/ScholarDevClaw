from __future__ import annotations

import importlib
import importlib.util
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class PluginMetadata:
    name: str
    version: str
    description: str
    author: str
    plugin_type: str
    entry_point: str | None = None


@dataclass
class Plugin:
    metadata: PluginMetadata
    module: Any
    instance: Any


class PluginInterface(ABC):
    @abstractmethod
    def initialize(self, config: dict[str, Any] | None = None) -> None:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class AnalyzerPlugin(PluginInterface):
    @abstractmethod
    def analyze(self, repo_path: str) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        pass


class SpecProviderPlugin(PluginInterface):
    @abstractmethod
    def get_specs(self) -> dict[str, dict[str, Any]]:
        pass

    @abstractmethod
    def search(self, query: str) -> list[dict[str, Any]]:
        pass


class ValidatorPlugin(PluginInterface):
    @abstractmethod
    def validate(self, repo_path: str, patch_result: dict[str, Any]) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_validation_type(self) -> str:
        pass


class PluginManager:
    def __init__(self, plugin_dir: str | None = None):
        if plugin_dir:
            self.plugin_dir = Path(plugin_dir)
        else:
            self.plugin_dir = Path.home() / ".scholardevclaw" / "plugins"

        self.plugin_dir.mkdir(parents=True, exist_ok=True)

        self._plugins: dict[str, Plugin] = {}
        self._analyzers: dict[str, AnalyzerPlugin] = {}
        self._spec_providers: dict[str, SpecProviderPlugin] = {}
        self._validators: dict[str, ValidatorPlugin] = {}

    def discover_plugins(self) -> list[PluginMetadata]:
        discovered = []

        for file in self.plugin_dir.glob("*.py"):
            if file.name.startswith("_"):
                continue

            try:
                spec = importlib.util.spec_from_file_location(file.stem, file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[file.stem] = module
                    spec.loader.exec_module(module)

                    if hasattr(module, "PLUGIN_METADATA"):
                        meta = module.PLUGIN_METADATA
                        discovered.append(
                            PluginMetadata(
                                name=meta.get("name", file.stem),
                                version=meta.get("version", "1.0.0"),
                                description=meta.get("description", ""),
                                author=meta.get("author", "Unknown"),
                                plugin_type=meta.get("plugin_type", "custom"),
                                entry_point=file.stem,
                            )
                        )
            except Exception:
                continue

        return discovered

    def load_plugin(self, name: str) -> Plugin | None:
        if name in self._plugins:
            return self._plugins[name]

        plugin_file = self.plugin_dir / f"{name}.py"
        if not plugin_file.exists():
            return None

        try:
            spec = importlib.util.spec_from_file_location(name, plugin_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)

                if hasattr(module, "PLUGIN_METADATA"):
                    meta = module.PLUGIN_METADATA
                    plugin_type = meta.get("plugin_type", "custom")

                    instance = None
                    if hasattr(module, "get_plugin_instance"):
                        instance = module.get_plugin_instance()

                    plugin = Plugin(
                        metadata=PluginMetadata(
                            name=meta.get("name", name),
                            version=meta.get("version", "1.0.0"),
                            description=meta.get("description", ""),
                            author=meta.get("author", "Unknown"),
                            plugin_type=plugin_type,
                            entry_point=name,
                        ),
                        module=module,
                        instance=instance,
                    )

                    self._plugins[name] = plugin

                    if plugin_type == "analyzer" and instance:
                        self._analyzers[name] = instance
                    elif plugin_type == "spec_provider" and instance:
                        self._spec_providers[name] = instance
                    elif plugin_type == "validator" and instance:
                        self._validators[name] = instance

                    return plugin

        except Exception:
            pass

        return None

    def unload_plugin(self, name: str) -> None:
        if name in self._plugins:
            plugin = self._plugins[name]
            plugin_type = plugin.metadata.plugin_type

            if plugin_type == "analyzer" and name in self._analyzers:
                del self._analyzers[name]
            elif plugin_type == "spec_provider" and name in self._spec_providers:
                del self._spec_providers[name]
            elif plugin_type == "validator" and name in self._validators:
                del self._validators[name]

            del self._plugins[name]

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
        else:
            content = _GENERIC_SCAFFOLD.format(name=name)

        scaffold_file.write_text(content)
        return scaffold_file


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


def get_plugin_instance():
    return {name}Validator()
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


def get_plugin_instance():
    return {name}Plugin()
'''
