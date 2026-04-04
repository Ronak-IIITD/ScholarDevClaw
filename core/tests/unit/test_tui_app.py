from __future__ import annotations

import os
import inspect

from scholardevclaw.tui.app import ScholarDevClawApp


def _minimal_app_for_unit() -> ScholarDevClawApp:
    app = ScholarDevClawApp()
    app._provider = "setup"
    app._model = ""
    app._directory = "."
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


def test_parse_natural_command_uses_chat_for_plain_greeting():
    app = _minimal_app_for_unit()

    action, ctx = app._parse_natural_command("hi there")

    assert action == "chat"
    assert ctx["prompt"] == "hi there"


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


def test_build_request_supports_mode_shorthand():
    app = _minimal_app_for_unit()

    action, req = app._build_request(":search")

    assert action == "set_mode"
    assert req == {"mode": "search"}


def test_build_request_supports_setup_and_provider_commands():
    app = _minimal_app_for_unit()

    action, req = app._build_request("set provider openrouter")
    assert action == "set_provider"
    assert req == {"provider": "openrouter"}

    action, req = app._build_request("setup")
    assert action == "setup"
    assert req == {}


def test_build_request_routes_plain_text_to_chat():
    app = _minimal_app_for_unit()

    action, req = app._build_request("hello model")

    assert action == "chat"
    assert req["prompt"] == "hello model"


def test_compute_suggestions_prioritizes_best_match():
    app = _minimal_app_for_unit()

    suggestions = app._compute_suggestions("ana")

    assert suggestions
    assert suggestions[0] == "analyze ./repo"


def test_compute_suggestions_supports_fuzzy_matching():
    app = _minimal_app_for_unit()

    suggestions = app._compute_suggestions("gnrt")

    assert "generate ./repo rmsnorm" in suggestions


def test_suggest_next_commands_are_action_specific():
    app = _minimal_app_for_unit()

    suggestions = app._suggest_next_commands(
        "generate",
        {"branch_name": "feature/rmsnorm"},
        {"repo_path": "./repo", "spec": "rmsnorm"},
    )

    assert suggestions == [
        "validate ./repo",
        "integrate ./repo rmsnorm",
        ":analyze",
    ]


def test_resolve_model_provider_prefers_selected_provider():
    app = _minimal_app_for_unit()
    app._provider = "openrouter"
    app._model = "anthropic/claude-sonnet-4"

    provider, model = app._resolve_model_provider()

    assert provider == "openrouter"
    assert model == "anthropic/claude-sonnet-4"


def test_action_cancel_task_exits_when_idle():
    app = _minimal_app_for_unit()
    events: list[str] = []
    app.exit = lambda *args, **kwargs: events.append("exit")

    app.action_cancel_task()

    assert events == ["exit"]


def test_on_mount_does_not_use_zero_delay_timer():
    source = inspect.getsource(ScholarDevClawApp.on_mount)

    assert "set_timer(0," not in source
    assert "set_timer(" in source


def test_save_provider_setup_openrouter_rejects_malformed_key(monkeypatch, tmp_path):
    app = _minimal_app_for_unit()
    app._save_runtime_state = lambda: None
    monkeypatch.setenv("SCHOLARDEVCLAW_AUTH_DIR", str(tmp_path))

    ok, message = app._save_provider_setup("openrouter", "anthropic/claude-sonnet-4", "bad-key")

    assert ok is False
    assert "Invalid key format for openrouter" in message


def test_save_provider_setup_openrouter_accepts_valid_key(monkeypatch, tmp_path):
    app = _minimal_app_for_unit()
    app._save_runtime_state = lambda: None
    monkeypatch.setenv("SCHOLARDEVCLAW_AUTH_DIR", str(tmp_path))

    key = "sk-or-1234567890"
    ok, message = app._save_provider_setup("openrouter", "anthropic/claude-sonnet-4", key)

    assert ok is True
    assert message == "OK"
    assert app._provider == "openrouter"
    assert os.environ.get("OPENROUTER_API_KEY") == key
