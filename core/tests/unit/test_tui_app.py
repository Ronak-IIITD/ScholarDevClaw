from __future__ import annotations

import inspect
import os
import time

import pytest

pytest.importorskip("textual")

from scholardevclaw.llm.client import LLMAPIError
from scholardevclaw.tui.app import ScholarDevClawApp, TaskCompleted
from scholardevclaw.tui.widgets import HistoryPane, PhaseTracker


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


def test_validate_repo_path_rejects_outside_allowed_roots(monkeypatch, tmp_path):
    app = ScholarDevClawApp()
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()

    monkeypatch.setenv("SCHOLARDEVCLAW_ALLOWED_REPO_DIRS", str(allowed))

    ok, message = app._validate_repo_path(str(outside))

    assert ok is False
    assert "outside allowed roots" in message.lower()


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


def test_bindings_include_escape_and_command_palette_shortcut():
    source = inspect.getsource(ScholarDevClawApp)

    assert '("escape", "handle_escape", "ESC")' in source
    assert '("ctrl+j", "open_command_palette", "Palette")' in source


def test_compose_includes_phase_tracker_and_history_pane():
    source = inspect.getsource(ScholarDevClawApp.compose)

    assert "PhaseTracker" in source
    assert "HistoryPane" in source


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


def test_action_cancel_task_sets_cancel_event_when_running():
    app = _minimal_app_for_unit()
    app._running_action = "analyze"
    app._active_token = 7

    import threading

    event = threading.Event()
    app._cancel_events = {7: event}

    class _DummyStatus:
        def stop_timer(self):
            return None

    app.query_one = lambda *_a, **_k: _DummyStatus()  # type: ignore[assignment]
    app._clear_progress = lambda: None
    app._append_output = lambda *_a, **_k: None
    app._set_status = lambda *_a, **_k: None

    app.action_cancel_task()

    assert event.is_set() is True


def test_is_task_cancelled_reflects_event_state():
    app = _minimal_app_for_unit()
    import threading

    event = threading.Event()
    app._cancel_events = {5: event}
    assert app._is_task_cancelled(5) is False

    event.set()
    assert app._is_task_cancelled(5) is True


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


def test_saved_key_selection_ignores_empty_values(monkeypatch, tmp_path):
    app = _minimal_app_for_unit()
    app._save_runtime_state = lambda: None
    monkeypatch.setenv("SCHOLARDEVCLAW_AUTH_DIR", str(tmp_path))

    # Simulate historical bad state (empty key entry first)
    app._save_provider_setup("openrouter", "anthropic/claude-sonnet-4", "sk-or-aaaaaaaa")
    app._save_provider_setup("openrouter", "anthropic/claude-sonnet-4", "sk-or-bbbbbbbb")

    from scholardevclaw.auth.store import AuthStore
    from scholardevclaw.auth.types import AuthProvider

    store = AuthStore(enable_audit=False, enable_rate_limit=False)
    config = store.get_config()
    # Force an empty key to be default to emulate previously broken state
    for key_obj in config.api_keys:
        if key_obj.provider == AuthProvider.OPENROUTER:
            key_obj.key = ""
            config.default_key_id = key_obj.id
            break
    store._save_config(config)

    resolved = app._get_saved_key_for_provider(AuthProvider.OPENROUTER)

    # Should not return empty/whitespace key
    assert resolved in {None, "sk-or-aaaaaaaa", "sk-or-bbbbbbbb"}
    if resolved is not None:
        assert resolved.strip()


def test_provider_has_credentials_false_when_only_empty_openrouter_key(monkeypatch, tmp_path):
    app = _minimal_app_for_unit()
    app._provider = "openrouter"
    app._model = "qwen/qwen3.6-plus:free"
    app._save_runtime_state = lambda: None
    monkeypatch.setenv("SCHOLARDEVCLAW_AUTH_DIR", str(tmp_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    from scholardevclaw.auth.store import AuthStore
    from scholardevclaw.auth.types import AuthProvider

    store = AuthStore(enable_audit=False, enable_rate_limit=False)
    store.add_api_key(
        "sk-or-cccccccc",
        "temp",
        AuthProvider.OPENROUTER,
        set_default=True,
        validate=True,
        metadata={"source": "test"},
    )
    config = store.get_config()
    for key_obj in config.api_keys:
        if key_obj.provider == AuthProvider.OPENROUTER:
            key_obj.key = ""
            config.default_key_id = key_obj.id
            break
    store._save_config(config)

    assert app._provider_has_credentials("openrouter") is False


def test_save_provider_setup_openrouter_requires_key_even_free_model(monkeypatch, tmp_path):
    app = _minimal_app_for_unit()
    app._save_runtime_state = lambda: None
    monkeypatch.setenv("SCHOLARDEVCLAW_AUTH_DIR", str(tmp_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    ok, message = app._save_provider_setup("openrouter", "qwen/qwen3.6-plus:free", "")

    assert ok is False
    assert "requires an API key" in message


def test_build_request_analyze_current_repo_uses_active_directory():
    app = _minimal_app_for_unit()
    app._directory = "/tmp/repo"

    action, req = app._build_request("analyze this current repo")

    assert action == "analyze"
    assert req["repo_path"] == "/tmp/repo"


def test_build_request_generate_this_repo_uses_active_directory_and_spec():
    app = _minimal_app_for_unit()
    app._directory = "/tmp/repo"

    action, req = app._build_request("generate this repo rmsnorm")

    assert action == "generate"
    assert req["repo_path"] == "/tmp/repo"
    assert req["spec"] == "rmsnorm"


def test_build_request_generate_with_explicit_repo_and_spec_parses_correctly():
    app = _minimal_app_for_unit()

    action, req = app._build_request("generate ./repo rmsnorm")

    assert action == "generate"
    assert req["repo_path"] == "./repo"
    assert req["spec"] == "rmsnorm"


def test_build_request_map_with_explicit_repo_and_spec_parses_correctly():
    app = _minimal_app_for_unit()

    action, req = app._build_request("map ./repo flashattention")

    assert action == "map"
    assert req["repo_path"] == "./repo"
    assert req["spec"] == "flashattention"


def test_build_request_integrate_with_explicit_repo_and_spec_parses_correctly():
    app = _minimal_app_for_unit()

    action, req = app._build_request("integrate ./repo rmsnorm")

    assert action == "integrate"
    assert req["repo_path"] == "./repo"
    assert req["spec"] == "rmsnorm"


def test_format_chat_error_429_is_concise_and_actionable():
    app = _minimal_app_for_unit()

    message = app._format_chat_error(LLMAPIError("openrouter", 429, "large blob"))

    assert "Rate limit" in message
    assert "set provider ollama" in message
    assert "large blob" not in message


def test_update_command_meta_uses_palette_accent_not_hardcoded_hex():
    source = inspect.getsource(ScholarDevClawApp._update_command_meta)

    assert "#7dd3fc" not in source
    assert "$accent" in source


def test_tui_styles_resolve_from_theme_palette():
    from scholardevclaw.tui.theme import COLORS

    assert ScholarDevClawApp.STYLES["background"] == COLORS["background"]
    assert ScholarDevClawApp.STYLES["surface"] == COLORS["surface"]
    assert ScholarDevClawApp.STYLES["text"] == COLORS["text"]
    assert ScholarDevClawApp.STYLES["text-muted"] == COLORS["text-muted"]
    assert ScholarDevClawApp.STYLES["accent"] == COLORS["accent"]


def test_remember_model_for_provider_tracks_supported_provider_only():
    app = _minimal_app_for_unit()

    app._remember_model_for_provider("openrouter", "openai/gpt-4.1-mini")
    app._remember_model_for_provider("unknown", "model-x")

    assert app._models_by_provider.get("openrouter") == "openai/gpt-4.1-mini"
    assert "unknown" not in app._models_by_provider


def test_model_for_provider_uses_saved_then_default():
    app = _minimal_app_for_unit()
    app._models_by_provider = {"openrouter": "openai/gpt-4.1-mini"}

    assert app._model_for_provider("openrouter") == "openai/gpt-4.1-mini"
    assert app._model_for_provider("ollama")


def test_startup_preflight_recovers_missing_directory(tmp_path):
    app = _minimal_app_for_unit()
    app._directory = str(tmp_path / "missing-dir")
    logs: list[str] = []
    app._append_output = lambda line, level="auto": logs.append(f"{level}:{line}")
    app._set_status = lambda *_a, **_k: None
    app._save_runtime_state = lambda: None

    app._startup_preflight()

    assert os.path.isdir(app._directory)
    assert any("previous directory missing" in line for line in logs)


def test_set_provider_uses_provider_specific_model_memory(monkeypatch):
    app = _minimal_app_for_unit()
    app._provider = "openrouter"
    app._model = "openai/gpt-4.1-mini"
    app._models_by_provider = {
        "openrouter": "openai/gpt-4.1-mini",
        "ollama": "llama3.1",
    }
    app._save_runtime_state = lambda: None
    app._sync_status_bar = lambda: None
    app._set_status = lambda *_a, **_k: None
    app._update_command_meta = lambda: None
    app._append_output = lambda *_a, **_k: None

    # No setup popup needed for ollama
    app._provider_has_credentials = lambda provider=None: True

    action, req = app._build_request("set provider ollama")
    assert action == "set_provider"

    # Execute equivalent branch directly.
    provider = str(req.get("provider", "") or "").strip().lower()
    app._provider = provider
    app._model = app._model_for_provider(provider)
    app._remember_model_for_provider(provider, app._model)

    assert app._provider == "ollama"
    assert app._model == "llama3.1"


def test_format_chat_error_for_bad_model_is_actionable():
    app = _minimal_app_for_unit()

    message = app._format_chat_error(LLMAPIError("openrouter", 404, "model not found"))

    assert "Model unavailable" in message


def test_build_chat_system_prompt_contains_natural_greeting_guidance():
    app = _minimal_app_for_unit()
    app._directory = "/tmp"

    prompt = app._build_chat_system_prompt()

    assert "For short greetings, reply naturally" in prompt


def test_parse_natural_command_repeated_prefixes_still_resolve_action():
    app = _minimal_app_for_unit()

    action, ctx = app._parse_natural_command("please can you analyze this repo")

    assert action == "analyze"


def test_parse_natural_command_supports_analyse_alias():
    app = _minimal_app_for_unit()

    action, _ctx = app._parse_natural_command("analyse ./repo")

    assert action == "analyze"


def test_extract_spec_from_tokens_skips_filler_words():
    app = _minimal_app_for_unit()

    spec = app._extract_spec_from_tokens(["this", "repo", "rmsnorm"])

    assert spec == "rmsnorm"


def test_on_task_completed_creates_run_history_entry_for_success():
    app = _minimal_app_for_unit()
    app._active_token = 41
    app._running_action = "analyze"
    app._run_started_at = {41: time.perf_counter() - 1.0}

    entries: list[dict[str, object]] = []

    class _DummyStatus:
        def stop_timer(self):
            return None

    class _DummyPrompt:
        def focus(self):
            return None

    class _DummyHistory:
        def add_entry(self, **kwargs):
            entries.append(kwargs)

    def _query_one(selector, *_args, **_kwargs):
        if selector == "#status-bar":
            return _DummyStatus()
        if selector == "#prompt-input":
            return _DummyPrompt()
        if selector == "#history-pane":
            return _DummyHistory()
        raise AssertionError(f"unexpected selector: {selector}")

    app.query_one = _query_one  # type: ignore[assignment]
    app._set_progress = lambda *_a, **_k: None
    app._clear_progress = lambda: None
    app._append_output = lambda *_a, **_k: None
    app._set_status = lambda *_a, **_k: None
    app._summarize_result = lambda *_a, **_k: []
    app._suggest_next_commands = lambda *_a, **_k: []
    app._update_command_meta = lambda: None
    app._set_phase = lambda *_a, **_k: None

    result = type("Result", (), {"ok": True, "payload": {}, "error": "", "logs": []})()
    request = {
        "action": "analyze",
        "repo_path": "./repo",
        "spec": "rmsnorm",
        "_original_command": "analyze ./repo",
    }
    app.on_task_completed(TaskCompleted(41, "analyze", result, request))

    assert len(entries) == 1
    assert entries[0]["run_id"] == 41
    assert entries[0]["action"] == "analyze"
    assert entries[0]["status"] == "Success"
    assert entries[0]["repo"] == "./repo"
    assert app._run_replay_map[41]["command"] == "analyze ./repo"


def test_save_runtime_state_failure_is_handled_without_raising():
    app = _minimal_app_for_unit()
    observed: list[tuple[str, str]] = []

    class _FailingPath:
        def write_text(self, *_args, **_kwargs):
            raise OSError("disk full")

    app._runtime_state_path = lambda: _FailingPath()  # type: ignore[assignment]
    app._append_output = lambda line, level="auto": observed.append((level, line))
    app._set_status = lambda message, level="info": observed.append((level, message))

    app._save_runtime_state()

    assert any("failed to save TUI runtime state" in value for _, value in observed)
    assert ("warning", "Runtime State Warning") in observed
