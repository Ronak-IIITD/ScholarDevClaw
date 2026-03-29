from __future__ import annotations

import os

from scholardevclaw.tui.app import ScholarDevClawApp


def _minimal_app_for_unit() -> ScholarDevClawApp:
    app = ScholarDevClawApp()
    app._validate_repo_path = lambda path: (
        bool(path),
        "Repository path is required" if not path else "",
    )
    app._validate_spec = lambda spec: (False, "Unknown spec") if spec == "bad" else (True, "")
    app._check_git_status = lambda path: (True, "Repository has uncommitted changes")
    return app


def test_parse_natural_command_extracts_action_repo_and_spec():
    app = _minimal_app_for_unit()

    action, ctx = app._parse_natural_command("please integrate rmsnorm in ./repo")

    assert action == "integrate"
    assert ctx.get("repo_path") == "./repo"
    assert ctx.get("spec") == "rmsnorm"


def test_validate_request_inputs_errors_and_warnings():
    app = _minimal_app_for_unit()

    ok, errors, warnings = app._validate_request_inputs(
        {
            "action": "search",
            "repo_path": "",
            "query": "",
            "include_arxiv": False,
            "include_web": False,
            "spec": "",
        }
    )

    assert ok is False
    assert any("Search query is required" in e for e in errors)
    assert any("Only local spec index" in w for w in warnings)


def test_validate_request_inputs_integrate_spec_validation_and_git_warning():
    app = _minimal_app_for_unit()

    ok, errors, warnings = app._validate_request_inputs(
        {
            "action": "integrate",
            "repo_path": "/tmp/repo",
            "query": "",
            "include_arxiv": False,
            "include_web": False,
            "spec": "bad",
            "integrate_require_clean": True,
        }
    )

    assert ok is False
    assert any("Unknown spec" in e for e in errors)
    assert any("uncommitted changes" in w for w in warnings)


def test_apply_and_restore_provider_env_roundtrip(monkeypatch):
    app = _minimal_app_for_unit()

    monkeypatch.setenv("SCHOLARDEVCLAW_API_PROVIDER", "old-provider")
    monkeypatch.setenv("SCHOLARDEVCLAW_API_MODEL", "old-model")

    app._resolve_model_provider = lambda: ("openai", "gpt-test")
    prev = app._apply_provider_env()

    assert os.environ.get("SCHOLARDEVCLAW_API_PROVIDER") == "openai"
    assert os.environ.get("SCHOLARDEVCLAW_API_MODEL") == "gpt-test"

    app._restore_provider_env(prev)

    assert os.environ.get("SCHOLARDEVCLAW_API_PROVIDER") == "old-provider"
    assert os.environ.get("SCHOLARDEVCLAW_API_MODEL") == "old-model"


def test_handle_escape_double_press_triggers_stop(monkeypatch):
    app = _minimal_app_for_unit()
    events: list[str] = []

    app._set_status = lambda message, level="info": events.append(f"{level}:{message}")
    app.on_stop = lambda: events.append("stopped")
    app._escape_pressed_count = 0
    app._escape_warning_shown = False
    app._last_escape_time = 0.0

    times = iter([1000.0, 1000.5])
    monkeypatch.setattr("scholardevclaw.tui.app.time.time", lambda: next(times))

    app.action_handle_escape()
    app.action_handle_escape()

    assert any("warning:Press ESC again" in e for e in events)
    assert "stopped" in events
