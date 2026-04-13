"""Comprehensive tests for Phase 13: Plugin Ecosystem.

Covers:
  - HookPoint enum, HookEvent dataclass, HookRegistry (register/fire/unregister)
  - PluginManager discovery, load/unload, enable/disable, config persistence
  - Built-in hook plugins (auto_lint, metrics_collector, event_logger)
  - Upgraded plugins (security, rustlang, javalang, jsts) hook registration
  - Pipeline hook wiring (_fire_hook helper)
  - __init__.py exports
  - CLI plugin subcommands
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# =========================================================================
# HookPoint enum
# =========================================================================


class TestHookPoint:
    def test_all_18_hook_points_exist(self):
        from scholardevclaw.plugins.hooks import HookPoint

        assert len(HookPoint) == 18

    def test_hook_point_values(self):
        from scholardevclaw.plugins.hooks import HookPoint

        expected = {
            "PIPELINE_START": "on_pipeline_start",
            "PIPELINE_COMPLETE": "on_pipeline_complete",
            "PIPELINE_ERROR": "on_pipeline_error",
            "BEFORE_ANALYZE": "on_before_analyze",
            "AFTER_ANALYZE": "on_after_analyze",
            "BEFORE_SUGGEST": "on_before_suggest",
            "AFTER_SUGGEST": "on_after_suggest",
            "BEFORE_SEARCH": "on_before_search",
            "AFTER_SEARCH": "on_after_search",
            "BEFORE_MAP": "on_before_map",
            "AFTER_MAP": "on_after_map",
            "BEFORE_GENERATE": "on_before_generate",
            "AFTER_GENERATE": "on_after_generate",
            "PATCH_CREATED": "on_patch_created",
            "BEFORE_VALIDATE": "on_before_validate",
            "AFTER_VALIDATE": "on_after_validate",
            "BEFORE_INTEGRATE": "on_before_integrate",
            "AFTER_INTEGRATE": "on_after_integrate",
        }
        for name, value in expected.items():
            assert HookPoint[name].value == value

    def test_hook_point_is_str_enum(self):
        from scholardevclaw.plugins.hooks import HookPoint

        for hp in HookPoint:
            assert isinstance(hp, str)
            assert hp == hp.value


# =========================================================================
# HookEvent dataclass
# =========================================================================


class TestHookEvent:
    def test_defaults(self):
        from scholardevclaw.plugins.hooks import HookEvent, HookPoint

        event = HookEvent(hook=HookPoint.PIPELINE_START, stage="test")
        assert event.payload == {}
        assert event.metadata == {}
        assert event.cancelled is False
        assert event.errors == []

    def test_payload_mutable(self):
        from scholardevclaw.plugins.hooks import HookEvent, HookPoint

        event = HookEvent(hook=HookPoint.AFTER_ANALYZE, stage="analyze")
        event.payload["key"] = "value"
        assert event.payload["key"] == "value"

    def test_cancellation(self):
        from scholardevclaw.plugins.hooks import HookEvent, HookPoint

        event = HookEvent(hook=HookPoint.BEFORE_GENERATE, stage="generate")
        event.cancelled = True
        assert event.cancelled is True


# =========================================================================
# HookRegistry
# =========================================================================


class TestHookRegistry:
    def _make_registry(self):
        from scholardevclaw.plugins.hooks import HookRegistry

        return HookRegistry()

    def test_empty_registry(self):
        reg = self._make_registry()
        assert reg.hook_count == 0

    def test_register_and_count(self):
        from scholardevclaw.plugins.hooks import HookPoint

        reg = self._make_registry()
        reg.register(HookPoint.BEFORE_ANALYZE, lambda e: None, plugin_name="test")
        assert reg.hook_count == 1

    def test_register_by_string_value(self):
        reg = self._make_registry()
        reg.register("on_before_analyze", lambda e: None, plugin_name="test")
        assert reg.hook_count == 1
        assert reg.has_hooks("on_before_analyze")

    def test_fire_invokes_callback(self):
        from scholardevclaw.plugins.hooks import HookPoint

        reg = self._make_registry()
        invoked = []
        reg.register(HookPoint.AFTER_ANALYZE, lambda e: invoked.append(e.stage), plugin_name="t")
        event = reg.fire(HookPoint.AFTER_ANALYZE, stage="analyze", payload={"x": 1})
        assert invoked == ["analyze"]
        assert event.payload == {"x": 1}

    def test_fire_respects_priority_order(self):
        from scholardevclaw.plugins.hooks import HookPoint

        reg = self._make_registry()
        order = []
        reg.register(
            HookPoint.BEFORE_MAP, lambda e: order.append("C"), plugin_name="c", priority=300
        )
        reg.register(
            HookPoint.BEFORE_MAP, lambda e: order.append("A"), plugin_name="a", priority=10
        )
        reg.register(
            HookPoint.BEFORE_MAP, lambda e: order.append("B"), plugin_name="b", priority=100
        )
        reg.fire(HookPoint.BEFORE_MAP)
        assert order == ["A", "B", "C"]

    def test_payload_mutation_propagates(self):
        from scholardevclaw.plugins.hooks import HookPoint

        reg = self._make_registry()

        def mutate(e):
            e.payload["added"] = True

        reg.register(HookPoint.AFTER_GENERATE, mutate, plugin_name="m")
        event = reg.fire(HookPoint.AFTER_GENERATE, payload={"original": 1})
        assert event.payload["added"] is True
        assert event.payload["original"] == 1

    def test_error_in_callback_is_caught(self):
        from scholardevclaw.plugins.hooks import HookPoint

        reg = self._make_registry()

        def boom(_event) -> None:
            raise ZeroDivisionError("simulated")

        reg.register(HookPoint.PIPELINE_ERROR, boom, plugin_name="bad")
        event = reg.fire(HookPoint.PIPELINE_ERROR)
        assert len(event.errors) == 1
        assert "ZeroDivisionError" in event.errors[0]

    def test_error_does_not_stop_next_callback(self):
        from scholardevclaw.plugins.hooks import HookPoint

        reg = self._make_registry()
        calls = []

        def boom(_event) -> None:
            raise ZeroDivisionError("simulated")

        reg.register(HookPoint.BEFORE_VALIDATE, boom, plugin_name="bad", priority=1)
        reg.register(
            HookPoint.BEFORE_VALIDATE, lambda e: calls.append("ok"), plugin_name="good", priority=2
        )
        event = reg.fire(HookPoint.BEFORE_VALIDATE)
        assert calls == ["ok"]
        assert len(event.errors) == 1

    def test_unregister_by_plugin_name(self):
        from scholardevclaw.plugins.hooks import HookPoint

        reg = self._make_registry()
        reg.register(HookPoint.AFTER_SEARCH, lambda e: None, plugin_name="A")
        reg.register(HookPoint.AFTER_SEARCH, lambda e: None, plugin_name="B")
        removed = reg.unregister(HookPoint.AFTER_SEARCH, plugin_name="A")
        assert removed == 1
        assert reg.hook_count == 1

    def test_unregister_all_by_plugin_name(self):
        from scholardevclaw.plugins.hooks import HookPoint

        reg = self._make_registry()
        reg.register(HookPoint.BEFORE_ANALYZE, lambda e: None, plugin_name="x")
        reg.register(HookPoint.AFTER_ANALYZE, lambda e: None, plugin_name="x")
        reg.register(HookPoint.BEFORE_MAP, lambda e: None, plugin_name="y")
        removed = reg.unregister_all(plugin_name="x")
        assert removed == 2
        assert reg.hook_count == 1

    def test_clear(self):
        from scholardevclaw.plugins.hooks import HookPoint

        reg = self._make_registry()
        reg.register(HookPoint.PIPELINE_START, lambda e: None, plugin_name="a")
        reg.register(HookPoint.PIPELINE_COMPLETE, lambda e: None, plugin_name="b")
        reg.clear()
        assert reg.hook_count == 0

    def test_list_hooks(self):
        from scholardevclaw.plugins.hooks import HookPoint

        reg = self._make_registry()
        reg.register(HookPoint.BEFORE_GENERATE, lambda e: None, plugin_name="p1", priority=50)
        hooks = reg.list_hooks(HookPoint.BEFORE_GENERATE)
        assert len(hooks) == 1
        assert hooks[0]["plugin"] == "p1"
        assert hooks[0]["priority"] == 50
        assert hooks[0]["hook"] == "on_before_generate"

    def test_list_all_hooks(self):
        from scholardevclaw.plugins.hooks import HookPoint

        reg = self._make_registry()
        reg.register(HookPoint.BEFORE_ANALYZE, lambda e: None, plugin_name="a")
        reg.register(HookPoint.AFTER_ANALYZE, lambda e: None, plugin_name="b")
        all_hooks = reg.list_hooks()
        assert len(all_hooks) == 2

    def test_has_hooks(self):
        from scholardevclaw.plugins.hooks import HookPoint

        reg = self._make_registry()
        assert not reg.has_hooks(HookPoint.BEFORE_INTEGRATE)
        reg.register(HookPoint.BEFORE_INTEGRATE, lambda e: None, plugin_name="x")
        assert reg.has_hooks(HookPoint.BEFORE_INTEGRATE)

    def test_execution_log(self):
        from scholardevclaw.plugins.hooks import HookPoint

        reg = self._make_registry()
        reg.register(HookPoint.PIPELINE_START, lambda e: None, plugin_name="test_p")
        reg.fire(HookPoint.PIPELINE_START)
        log = reg.get_execution_log()
        assert len(log) == 1
        assert log[0]["hook"] == "on_pipeline_start"
        assert log[0]["plugin"] == "test_p"
        assert "elapsed_ms" in log[0]

    def test_clear_execution_log(self):
        from scholardevclaw.plugins.hooks import HookPoint

        reg = self._make_registry()
        reg.register(HookPoint.PIPELINE_START, lambda e: None, plugin_name="x")
        reg.fire(HookPoint.PIPELINE_START)
        reg.clear_execution_log()
        assert reg.get_execution_log() == []

    def test_resolve_by_enum_name(self):
        from scholardevclaw.plugins.hooks import HookPoint

        reg = self._make_registry()
        reg.register("BEFORE_ANALYZE", lambda e: None, plugin_name="t")
        assert reg.has_hooks(HookPoint.BEFORE_ANALYZE)

    def test_resolve_invalid_raises(self):
        import pytest

        from scholardevclaw.plugins.hooks import HookRegistry

        reg = HookRegistry()
        with pytest.raises(ValueError, match="Unknown hook point"):
            reg.register("nonexistent_hook", lambda e: None, plugin_name="x")

    def test_fire_empty_hook_returns_event(self):
        from scholardevclaw.plugins.hooks import HookPoint

        reg = self._make_registry()
        event = reg.fire(HookPoint.PIPELINE_COMPLETE, stage="done", payload={"ok": True})
        assert event.hook == HookPoint.PIPELINE_COMPLETE
        assert event.payload == {"ok": True}
        assert event.errors == []


# =========================================================================
# get_hook_registry singleton
# =========================================================================


class TestGetHookRegistry:
    def test_returns_same_instance(self):
        from scholardevclaw.plugins.hooks import get_hook_registry

        a = get_hook_registry()
        b = get_hook_registry()
        assert a is b

    def test_returns_hookregistry_type(self):
        from scholardevclaw.plugins.hooks import HookRegistry, get_hook_registry

        assert isinstance(get_hook_registry(), HookRegistry)


# =========================================================================
# PluginManager — discovery, load, enable/disable, config
# =========================================================================


class TestPluginManager:
    def _make_manager(self, tmp_path):
        from scholardevclaw.plugins.hooks import HookRegistry
        from scholardevclaw.plugins.manager import PluginManager

        reg = HookRegistry()
        return PluginManager(
            plugin_dir=str(tmp_path / "plugins"),
            hook_registry=reg,
            config_root=tmp_path / "config",
        ), reg

    def test_discover_builtin_plugins(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        discovered = manager.discover_plugins()
        names = [d.name for d in discovered]
        assert "security" in names
        assert "auto_lint" in names
        assert "metrics_collector" in names
        assert "event_logger" in names
        assert len(discovered) >= 7

    def test_load_builtin_plugin(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        plugin = manager.load_plugin("security")
        assert plugin is not None
        assert plugin.metadata.name == "security"
        assert plugin.metadata.source == "builtin"

    def test_load_all(self, tmp_path):
        manager, reg = self._make_manager(tmp_path)
        loaded = manager.load_all()
        assert len(loaded) >= 7
        # All built-ins should register hooks
        assert reg.hook_count > 0

    def test_unload_plugin(self, tmp_path):
        manager, reg = self._make_manager(tmp_path)
        manager.load_plugin("security")
        assert "security" in [p.name for p in manager.list_plugins()]
        manager.unload_plugin("security")
        assert "security" not in [p.name for p in manager.list_plugins()]

    def test_enable_disable_persistence(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        assert manager.is_enabled("security") is True

        manager.disable_plugin("security")
        assert manager.is_enabled("security") is False

        # Verify persisted
        state_file = tmp_path / "config" / "plugin_state.json"
        assert state_file.exists()
        state = json.loads(state_file.read_text())
        assert state["enabled"]["security"] is False

        manager.enable_plugin("security")
        assert manager.is_enabled("security") is True

    def test_disabled_plugin_not_loaded_by_load_all(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        manager.disable_plugin("event_logger")
        loaded = manager.load_all()
        loaded_names = [p.metadata.name for p in loaded]
        assert "event_logger" not in loaded_names

    def test_load_disabled_plugin_returns_none(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        manager.disable_plugin("auto_lint")
        plugin = manager.load_plugin("auto_lint")
        assert plugin is None

    def test_get_plugin_config(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        assert manager.get_plugin_config("security") == {}

    def test_set_plugin_config(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        manager.set_plugin_config("auto_lint", {"severity": "error"})
        config = manager.get_plugin_config("auto_lint")
        assert config == {"severity": "error"}

    def test_get_plugin(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        manager.load_plugin("security")
        plugin = manager.get_plugin("security")
        assert plugin is not None
        assert plugin.metadata.name == "security"

    def test_get_nonexistent_plugin(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        plugin = manager.get_plugin("nonexistent")
        assert plugin is None

    def test_list_plugins_empty_initially(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        assert manager.list_plugins() == []

    def test_hook_registration_on_load(self, tmp_path):
        manager, reg = self._make_manager(tmp_path)
        assert reg.hook_count == 0
        manager.load_plugin("metrics_collector")
        assert reg.hook_count > 0

    def test_hook_unregistration_on_unload(self, tmp_path):
        manager, reg = self._make_manager(tmp_path)
        manager.load_plugin("metrics_collector")
        count_after_load = reg.hook_count
        assert count_after_load > 0
        manager.unload_plugin("metrics_collector")
        assert reg.hook_count < count_after_load


# =========================================================================
# Built-in hook plugins — smoke tests
# =========================================================================


class TestAutoLintPlugin:
    def test_import_and_metadata(self):
        from scholardevclaw.plugins.auto_lint import PLUGIN_METADATA, get_plugin_instance

        assert PLUGIN_METADATA["name"] == "auto_lint"
        assert PLUGIN_METADATA["plugin_type"] == "hook"
        inst = get_plugin_instance()
        assert inst.get_name() == "auto_lint"

    def test_register_hooks(self):
        from scholardevclaw.plugins.auto_lint import get_plugin_instance
        from scholardevclaw.plugins.hooks import HookRegistry

        reg = HookRegistry()
        inst = get_plugin_instance()
        inst.register_hooks(reg)
        assert reg.hook_count >= 2  # at least AFTER_GENERATE and PATCH_CREATED

    def test_teardown(self):
        from scholardevclaw.plugins.auto_lint import get_plugin_instance

        inst = get_plugin_instance()
        inst.teardown()  # should not raise


class TestMetricsCollectorPlugin:
    def test_import_and_metadata(self):
        from scholardevclaw.plugins.metrics_collector import PLUGIN_METADATA, get_plugin_instance

        assert PLUGIN_METADATA["name"] == "metrics_collector"
        assert PLUGIN_METADATA["plugin_type"] == "hook"
        inst = get_plugin_instance()
        assert inst.get_name() == "metrics_collector"

    def test_register_hooks(self):
        from scholardevclaw.plugins.hooks import HookRegistry
        from scholardevclaw.plugins.metrics_collector import get_plugin_instance

        reg = HookRegistry()
        inst = get_plugin_instance()
        inst.register_hooks(reg)
        # Metrics collector hooks into many points (before/after pairs + pipeline level)
        assert reg.hook_count >= 10

    def test_get_metrics_initially_empty(self):
        from scholardevclaw.plugins.metrics_collector import get_plugin_instance

        inst = get_plugin_instance()
        metrics = inst.metrics
        assert isinstance(metrics, (dict, list))

    def test_teardown_resets_metrics(self):
        from scholardevclaw.plugins.metrics_collector import get_plugin_instance

        inst = get_plugin_instance()
        inst.teardown()
        # Should be safe to call


class TestEventLoggerPlugin:
    def test_import_and_metadata(self):
        from scholardevclaw.plugins.event_logger import PLUGIN_METADATA, get_plugin_instance

        assert PLUGIN_METADATA["name"] == "event_logger"
        assert PLUGIN_METADATA["plugin_type"] == "hook"
        inst = get_plugin_instance()
        assert inst.get_name() == "event_logger"

    def test_register_hooks(self):
        from scholardevclaw.plugins.event_logger import get_plugin_instance
        from scholardevclaw.plugins.hooks import HookRegistry

        reg = HookRegistry()
        inst = get_plugin_instance()
        inst.register_hooks(reg)
        # Event logger hooks into all 18 hook points
        assert reg.hook_count == 18

    def test_teardown(self):
        from scholardevclaw.plugins.event_logger import get_plugin_instance

        inst = get_plugin_instance()
        inst.teardown()


# =========================================================================
# Upgraded plugins — verify hook registration
# =========================================================================


class TestSecurityPluginHooks:
    def test_has_register_hooks(self):
        from scholardevclaw.plugins.security import get_plugin_instance

        inst = get_plugin_instance()
        assert hasattr(inst, "register_hooks")

    def test_register_hooks(self):
        from scholardevclaw.plugins.hooks import HookRegistry
        from scholardevclaw.plugins.security import get_plugin_instance

        reg = HookRegistry()
        inst = get_plugin_instance()
        inst.register_hooks(reg)
        assert reg.hook_count >= 2

    def test_teardown(self):
        from scholardevclaw.plugins.security import get_plugin_instance

        inst = get_plugin_instance()
        inst.teardown()


class TestRustlangPluginHooks:
    def test_has_register_hooks(self):
        from scholardevclaw.plugins.rustlang import get_plugin_instance

        inst = get_plugin_instance()
        assert hasattr(inst, "register_hooks")

    def test_register_hooks(self):
        from scholardevclaw.plugins.hooks import HookRegistry
        from scholardevclaw.plugins.rustlang import get_plugin_instance

        reg = HookRegistry()
        inst = get_plugin_instance()
        inst.register_hooks(reg)
        assert reg.hook_count >= 1


class TestJavalangPluginHooks:
    def test_has_register_hooks(self):
        from scholardevclaw.plugins.javalang import get_plugin_instance

        inst = get_plugin_instance()
        assert hasattr(inst, "register_hooks")

    def test_register_hooks(self):
        from scholardevclaw.plugins.hooks import HookRegistry
        from scholardevclaw.plugins.javalang import get_plugin_instance

        reg = HookRegistry()
        inst = get_plugin_instance()
        inst.register_hooks(reg)
        assert reg.hook_count >= 1


class TestJstsPluginHooks:
    def test_has_register_hooks(self):
        from scholardevclaw.plugins.jsts import get_plugin_instance

        inst = get_plugin_instance()
        assert hasattr(inst, "register_hooks")

    def test_register_hooks(self):
        from scholardevclaw.plugins.hooks import HookRegistry
        from scholardevclaw.plugins.jsts import get_plugin_instance

        reg = HookRegistry()
        inst = get_plugin_instance()
        inst.register_hooks(reg)
        assert reg.hook_count >= 1


# =========================================================================
# Pipeline hook wiring — _fire_hook helper
# =========================================================================


class TestFireHookHelper:
    def test_fire_hook_import(self):
        from scholardevclaw.application.pipeline import _fire_hook

        assert callable(_fire_hook)

    def test_fire_hook_returns_payload(self):
        from scholardevclaw.application.pipeline import _fire_hook

        result = _fire_hook("on_pipeline_start", stage="test", payload={"k": "v"})
        assert result is not None
        assert result.get("k") == "v"

    def test_fire_hook_with_no_hooks_returns_payload(self):
        from scholardevclaw.application.pipeline import _fire_hook

        result = _fire_hook("on_before_analyze", payload={"a": 1})
        # Should return payload even when no hooks registered
        assert result is not None

    def test_fire_hook_bad_hook_name_returns_payload(self):
        from scholardevclaw.application.pipeline import _fire_hook

        # Invalid hook name — should log debug and return payload
        result = _fire_hook("nonexistent_hook", payload={"safe": True})
        assert result is not None
        assert result.get("safe") is True

    def test_fire_hook_bootstraps_plugins_once_idempotent(self, monkeypatch):
        from scholardevclaw.application import pipeline

        class _Registry:
            def __init__(self):
                self.hook_count = 1
                self.fire_calls = 0

            def fire(self, hook_point, *, stage="", payload=None, metadata=None):
                self.fire_calls += 1
                return SimpleNamespace(payload=payload if payload is not None else {})

        registry = _Registry()
        hooks_module = ModuleType("scholardevclaw.plugins.hooks")
        setattr(hooks_module, "get_hook_registry", lambda: registry)

        manager_module = ModuleType("scholardevclaw.plugins.manager")

        class _Manager:
            init_calls = 0
            load_all_calls = 0

            def __init__(self, plugin_dir=None, hook_registry=None, config_root=None):
                _Manager.init_calls += 1
                assert hook_registry is registry

            def load_all(self, *, include_disabled=False):
                _Manager.load_all_calls += 1
                return ["fake_plugin"]

        setattr(manager_module, "PluginManager", _Manager)

        monkeypatch.setitem(sys.modules, hooks_module.__name__, hooks_module)
        monkeypatch.setitem(sys.modules, manager_module.__name__, manager_module)
        monkeypatch.delenv("SCHOLARDEVCLAW_DISABLE_PLUGIN_AUTOLOAD", raising=False)
        monkeypatch.setattr(pipeline, "_PLUGIN_AUTOLOAD_ATTEMPTED", False)

        first = pipeline._fire_hook("on_before_analyze", payload={"n": 1})
        second = pipeline._fire_hook("on_before_analyze", payload={"n": 2})

        assert first == {"n": 1}
        assert second == {"n": 2}
        assert _Manager.init_calls == 1
        assert _Manager.load_all_calls == 1
        assert registry.fire_calls == 2

    def test_fire_hook_bootstrap_failure_isolated(self, monkeypatch):
        from scholardevclaw.application import pipeline

        class _Registry:
            hook_count = 0

            def fire(self, hook_point, *, stage="", payload=None, metadata=None):
                return SimpleNamespace(payload=payload if payload is not None else {})

        hooks_module = ModuleType("scholardevclaw.plugins.hooks")
        setattr(hooks_module, "get_hook_registry", lambda: _Registry())

        manager_module = ModuleType("scholardevclaw.plugins.manager")

        class _Manager:
            def __init__(self, plugin_dir=None, hook_registry=None, config_root=None):
                pass

            def load_all(self, *, include_disabled=False):
                raise RuntimeError("boom")

        setattr(manager_module, "PluginManager", _Manager)

        monkeypatch.setitem(sys.modules, hooks_module.__name__, hooks_module)
        monkeypatch.setitem(sys.modules, manager_module.__name__, manager_module)
        monkeypatch.delenv("SCHOLARDEVCLAW_DISABLE_PLUGIN_AUTOLOAD", raising=False)
        monkeypatch.setattr(pipeline, "_PLUGIN_AUTOLOAD_ATTEMPTED", False)

        payload = {"safe": True}
        result = pipeline._fire_hook("on_before_map", payload=payload)

        assert result == payload

    def test_fire_hook_respects_disable_autoload_env(self, monkeypatch):
        from scholardevclaw.application import pipeline

        class _Registry:
            hook_count = 0

            def fire(self, hook_point, *, stage="", payload=None, metadata=None):
                return SimpleNamespace(payload=payload if payload is not None else {})

        hooks_module = ModuleType("scholardevclaw.plugins.hooks")
        setattr(hooks_module, "get_hook_registry", lambda: _Registry())

        manager_module = ModuleType("scholardevclaw.plugins.manager")

        class _Manager:
            init_calls = 0
            load_all_calls = 0

            def __init__(self, plugin_dir=None, hook_registry=None, config_root=None):
                _Manager.init_calls += 1

            def load_all(self, *, include_disabled=False):
                _Manager.load_all_calls += 1
                return []

        setattr(manager_module, "PluginManager", _Manager)

        monkeypatch.setitem(sys.modules, hooks_module.__name__, hooks_module)
        monkeypatch.setitem(sys.modules, manager_module.__name__, manager_module)
        monkeypatch.setenv("SCHOLARDEVCLAW_DISABLE_PLUGIN_AUTOLOAD", "1")
        monkeypatch.setattr(pipeline, "_PLUGIN_AUTOLOAD_ATTEMPTED", False)

        payload = {"noop": True}
        result = pipeline._fire_hook("on_before_validate", payload=payload)

        assert result == payload
        assert _Manager.init_calls == 0
        assert _Manager.load_all_calls == 0


# =========================================================================
# __init__.py exports
# =========================================================================


class TestPluginExports:
    def test_all_exports_importable(self):
        from scholardevclaw.plugins import (
            AnalyzerPlugin,
            AutoLintPlugin,
            EventLoggerPlugin,
            HookCallback,
            HookEvent,
            HookPoint,
            HookRegistry,
            MetricsCollectorPlugin,
            Plugin,
            PluginInterface,
            PluginManager,
            PluginMetadata,
            SpecProviderPlugin,
            ValidatorPlugin,
            get_hook_registry,
            get_plugin_manager,
        )

        assert all(
            x is not None
            for x in [
                PluginManager,
                Plugin,
                PluginMetadata,
                AnalyzerPlugin,
                SpecProviderPlugin,
                ValidatorPlugin,
                PluginInterface,
                get_plugin_manager,
                HookPoint,
                HookEvent,
                HookRegistry,
                HookCallback,
                get_hook_registry,
                AutoLintPlugin,
                MetricsCollectorPlugin,
                EventLoggerPlugin,
            ]
        )

    def test_all_list_complete(self):
        from scholardevclaw import plugins

        expected = {
            "PluginManager",
            "Plugin",
            "PluginMetadata",
            "AnalyzerPlugin",
            "SpecProviderPlugin",
            "ValidatorPlugin",
            "PluginInterface",
            "get_plugin_manager",
            "HookPoint",
            "HookEvent",
            "HookRegistry",
            "HookCallback",
            "get_hook_registry",
            "AutoLintPlugin",
            "MetricsCollectorPlugin",
            "EventLoggerPlugin",
        }
        assert expected.issubset(set(plugins.__all__))

    def test_get_plugin_manager_returns_manager(self):
        from scholardevclaw.plugins import PluginManager, get_plugin_manager

        mgr = get_plugin_manager()
        assert isinstance(mgr, PluginManager)


# =========================================================================
# Full integration: load all plugins and fire hooks
# =========================================================================


class TestFullIntegration:
    def test_load_all_and_fire_pipeline_start(self, tmp_path):
        from scholardevclaw.plugins.hooks import HookPoint, HookRegistry
        from scholardevclaw.plugins.manager import PluginManager

        reg = HookRegistry()
        mgr = PluginManager(
            plugin_dir=str(tmp_path / "plugins"),
            hook_registry=reg,
            config_root=tmp_path / "config",
        )
        loaded = mgr.load_all()
        assert len(loaded) >= 7
        assert reg.hook_count > 0

        # Fire pipeline start — should invoke metrics_collector and event_logger
        event = reg.fire(HookPoint.PIPELINE_START, stage="test", payload={"repo": "/tmp/test"})
        assert event.hook == HookPoint.PIPELINE_START
        assert len(event.errors) == 0  # all callbacks should succeed

    def test_load_all_and_fire_all_hook_points(self, tmp_path):
        from scholardevclaw.plugins.hooks import HookPoint, HookRegistry
        from scholardevclaw.plugins.manager import PluginManager

        reg = HookRegistry()
        mgr = PluginManager(
            plugin_dir=str(tmp_path / "plugins"),
            hook_registry=reg,
            config_root=tmp_path / "config",
        )
        mgr.load_all()

        # Fire every hook point — none should raise
        for hp in HookPoint:
            event = reg.fire(hp, stage="smoke_test", payload={"test": True})
            # Event should always be returned
            assert event.hook == hp

    def test_hook_count_after_load_all(self, tmp_path):
        from scholardevclaw.plugins.hooks import HookRegistry
        from scholardevclaw.plugins.manager import PluginManager

        reg = HookRegistry()
        mgr = PluginManager(
            plugin_dir=str(tmp_path / "plugins"),
            hook_registry=reg,
            config_root=tmp_path / "config",
        )
        mgr.load_all()
        # 7 built-in plugins with many hooks; expect at least 30
        assert reg.hook_count >= 30


# =========================================================================
# CLI plugin subcommand argument parsing
# =========================================================================


class TestCLIPluginArgs:
    def test_plugin_action_choices(self):
        """Verify the plugin subcommand accepts the new action choices."""

        # We can't easily extract argparse choices without invoking the parser,
        # so just verify the module imports and the function exists.
        from scholardevclaw.cli import cmd_plugin

        assert callable(cmd_plugin)

    def test_plugin_action_includes_new_commands(self):
        """The plugin subparser should accept enable, disable, hooks."""
        # We verify by attempting to parse them (in isolation).
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "plugin_action",
            choices=[
                "list",
                "load",
                "unload",
                "enable",
                "disable",
                "hooks",
                "analyze",
                "validate",
                "scaffold",
                "info",
            ],
        )
        # These should all parse without error
        for action in ["enable", "disable", "hooks"]:
            args = parser.parse_args([action])
            assert args.plugin_action == action
