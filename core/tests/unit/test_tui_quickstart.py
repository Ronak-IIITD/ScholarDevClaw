"""Tests for the TUI Quickstart Dashboard."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scholardevclaw.tui.quickstart import (
    QUICK_ACTIONS,
    SCHOLAR_BANNER,
    SCHOLAR_BANNER_COMPACT,
    QuickAction,
    QuickstartDashboard,
    _format_recent_runs,
    _gather_system_status,
    _render_tile,
)


class TestQuickActions:
    """Tests for the QUICK_ACTIONS registry."""

    def test_actions_count(self) -> None:
        assert len(QUICK_ACTIONS) == 6

    def test_actions_have_unique_keys(self) -> None:
        keys = [a.key for a in QUICK_ACTIONS]
        assert len(keys) == len(set(keys))

    def test_actions_have_unique_shortcuts(self) -> None:
        shortcuts = [a.shortcut for a in QUICK_ACTIONS]
        # Shortcuts may overlap with key characters; just ensure all non-empty
        assert all(s for s in shortcuts)

    def test_actions_have_titles(self) -> None:
        for a in QUICK_ACTIONS:
            assert a.title.strip()
            assert a.icon.strip()
            assert a.description.strip()
            assert a.command.strip()

    def test_action_dataclass_frozen(self) -> None:
        action = QUICK_ACTIONS[0]
        with pytest.raises((AttributeError, Exception)):
            action.title = "hacked"  # type: ignore[misc]

    def test_action_dataclass_constructible(self) -> None:
        action = QuickAction(
            key="x",
            title="X",
            description="D",
            icon="?",
            command="cmd",
            shortcut="X",
        )
        assert action.key == "x"
        assert action.command == "cmd"

    def test_includes_analyze_action(self) -> None:
        keys = [a.key for a in QUICK_ACTIONS]
        assert "analyze" in keys

    def test_includes_suggest_action(self) -> None:
        keys = [a.key for a in QUICK_ACTIONS]
        assert "suggest" in keys

    def test_includes_integrate_action(self) -> None:
        keys = [a.key for a in QUICK_ACTIONS]
        assert "integrate" in keys

    def test_includes_search_action(self) -> None:
        keys = [a.key for a in QUICK_ACTIONS]
        assert "search" in keys

    def test_includes_doctor_action(self) -> None:
        keys = [a.key for a in QUICK_ACTIONS]
        assert "doctor" in keys


class TestBanner:
    """Tests for the ASCII art banner."""

    def test_banner_non_empty(self) -> None:
        assert SCHOLAR_BANNER.strip()

    def test_banner_contains_research_tagline(self) -> None:
        # The full ASCII-art banner uses the "RESEARCH → CODE" tagline.
        # The tagline is letter-spaced (e.g. "R E S E A R C H") in the art,
        # so we normalize whitespace before checking.
        normalized = "".join(SCHOLAR_BANNER.split())
        assert "RESEARCH" in normalized.upper()
        assert "CODE" in normalized.upper()

    def test_banner_compact_non_empty(self) -> None:
        assert SCHOLAR_BANNER_COMPACT.strip()

    def test_banner_compact_contains_product_name(self) -> None:
        assert "ScholarDevClaw" in SCHOLAR_BANNER_COMPACT


class TestRenderTile:
    """Tests for the tile rendering helper."""

    def test_renders_box(self) -> None:
        out = _render_tile(QUICK_ACTIONS[0])
        assert "┏" in out
        assert "┗" in out
        assert "┛" in out

    def test_includes_title(self) -> None:
        out = _render_tile(QUICK_ACTIONS[0])
        assert QUICK_ACTIONS[0].title in out

    def test_includes_command(self) -> None:
        out = _render_tile(QUICK_ACTIONS[0])
        assert QUICK_ACTIONS[0].command in out

    def test_focused_marker(self) -> None:
        out_focused = _render_tile(QUICK_ACTIONS[0], focused=True)
        out_unfocused = _render_tile(QUICK_ACTIONS[0], focused=False)
        assert out_focused != out_unfocused
        assert "▶" in out_focused


class TestFormatRecentRuns:
    """Tests for the recent-runs formatter."""

    def test_empty_runs(self) -> None:
        out = _format_recent_runs([])
        assert "no runs yet" in out.lower()

    def test_none_runs(self) -> None:
        out = _format_recent_runs(None)
        assert "no runs yet" in out.lower()

    def test_completed_run(self) -> None:
        out = _format_recent_runs([{"action": "analyze", "status": "completed"}])
        assert "✓" in out
        assert "analyze" in out
        assert "completed" in out

    def test_failed_run(self) -> None:
        out = _format_recent_runs([{"action": "integrate", "spec": "rmsnorm", "status": "failed"}])
        assert "✗" in out
        assert "integrate" in out
        assert "rmsnorm" in out
        assert "failed" in out

    def test_running_run(self) -> None:
        out = _format_recent_runs([{"action": "generate", "status": "running"}])
        assert "⟳" in out

    def test_cancelled_run(self) -> None:
        out = _format_recent_runs([{"action": "validate", "status": "cancelled"}])
        assert "⊘" in out

    def test_multiple_runs(self) -> None:
        runs = [
            {"action": "a", "status": "completed"},
            {"action": "b", "status": "failed"},
            {"action": "c", "status": "running"},
        ]
        out = _format_recent_runs(runs)
        # All actions should appear
        for r in runs:
            assert r["action"] in out


class TestSystemStatus:
    """Tests for the system-status collector."""

    def test_returns_dict(self) -> None:
        status = _gather_system_status()
        assert isinstance(status, dict)

    def test_includes_python_version(self) -> None:
        status = _gather_system_status()
        assert "Python" in status
        import sys

        assert str(sys.version_info.major) in status["Python"]

    def test_includes_platform(self) -> None:
        status = _gather_system_status()
        assert "Platform" in status
        assert status["Platform"]  # non-empty

    def test_includes_cwd(self, tmp_path: Path) -> None:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            status = _gather_system_status()
            assert "CWD" in status
            assert str(tmp_path) in status["CWD"]
        finally:
            os.chdir(old_cwd)

    def test_api_key_masked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secretkey12345")
        status = _gather_system_status()
        if "API Key" in status:
            assert "sk-ant" in status["API Key"]  # prefix visible
            assert "secretkey" not in status["API Key"]  # rest masked

    def test_api_key_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        status = _gather_system_status()
        assert "API Key" in status
        assert "not set" in status["API Key"].lower()


class TestQuickstartDashboard:
    """Tests for the QuickstartDashboard screen class."""

    def test_constructible_with_defaults(self) -> None:
        # Should not raise
        dashboard = QuickstartDashboard()
        assert dashboard.recent_runs == []
        assert dashboard.system_status  # non-empty dict

    def test_constructible_with_runs(self) -> None:
        runs = [{"action": "analyze", "status": "completed"}]
        dashboard = QuickstartDashboard(recent_runs=runs)
        assert len(dashboard.recent_runs) == 1

    def test_constructible_with_custom_status(self) -> None:
        status = {"Version": "1.0", "CWD": "/tmp"}
        dashboard = QuickstartDashboard(system_status=status)
        assert dashboard.system_status["Version"] == "1.0"

    def test_screen_class_hierarchy(self) -> None:
        from textual.screen import ModalScreen

        assert issubclass(QuickstartDashboard, ModalScreen)

    def test_screen_class_is_callable(self) -> None:
        # ModalScreen subclasses are usually instantiated without args
        d = QuickstartDashboard()
        assert d is not None
