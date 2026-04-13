from __future__ import annotations

import inspect
import json
import os
import time

import pytest

pytest.importorskip("textual")

from scholardevclaw.llm.client import LLMAPIError
from scholardevclaw.tui.app import (
    RunArtifact,
    RunEvent,
    RunLifecycleState,
    ScholarDevClawApp,
    TaskCompleted,
)
from scholardevclaw.tui.widgets import RunInspector


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
    assert '("ctrl+i", "focus_inspector", "Inspector")' in source


def test_compose_includes_split_workspace_and_side_inspector_history_pane():
    source = inspect.getsource(ScholarDevClawApp.compose)

    assert 'Horizontal(id="workspace")' in source
    assert 'Vertical(id="main-pane")' in source
    assert 'Vertical(id="side-pane")' in source
    assert "PhaseTracker" in source
    assert "LogView" in source
    assert "HistoryPane" in source
    assert "RunInspector" in source


def test_action_focus_inspector_focuses_widget_and_sets_status():
    app = _minimal_app_for_unit()
    called: list[str] = []
    statuses: list[tuple[str, str]] = []

    class _DummyInspector:
        def focus(self):
            called.append("focus")

    app.query_one = lambda *_a, **_k: _DummyInspector()  # type: ignore[assignment]
    app._set_status = lambda message, level="info": statuses.append((level, message))

    app.action_focus_inspector()

    assert called == ["focus"]
    assert statuses[-1][0] == "accent"
    assert "Inspector focused" in statuses[-1][1]


def test_on_inspector_action_routes_to_show_events_and_rerun():
    app = _minimal_app_for_unit()
    calls: list[tuple[str, dict[str, int], str | None]] = []

    class _DummyPrompt:
        def focus(self):
            return None

    def _query_one(selector, *_args, **_kwargs):
        if selector == "#prompt-input":
            return _DummyPrompt()
        raise AssertionError(f"unexpected selector: {selector}")

    app.query_one = _query_one  # type: ignore[assignment]

    def _exec(action, request, *, command=None):
        calls.append((action, request, command))

    app._execute_action_request = _exec  # type: ignore[assignment]

    app.on_inspector_action(RunInspector.InspectorAction("show", 8, 3))
    app.on_inspector_action(RunInspector.InspectorAction("events", 8, 3))
    app.on_inspector_action(RunInspector.InspectorAction("rerun", 8, 3))

    assert calls[0] == ("run_show", {"run_id": 8}, "run show 8")
    assert calls[1] == ("run_events", {"run_id": 8}, "run events 8")
    assert calls[2] == ("run_rerun", {"run_id": 8}, "run rerun 8")


def test_on_inspector_action_warns_when_no_run_selected():
    app = _minimal_app_for_unit()
    output: list[tuple[str, str]] = []
    statuses: list[tuple[str, str]] = []

    class _DummyPrompt:
        def focus(self):
            return None

    def _query_one(selector, *_args, **_kwargs):
        if selector == "#prompt-input":
            return _DummyPrompt()
        raise AssertionError(f"unexpected selector: {selector}")

    app.query_one = _query_one  # type: ignore[assignment]
    app._append_output = lambda line, level="auto": output.append((level, line))
    app._set_status = lambda message, level="info": statuses.append((level, message))

    app.on_inspector_action(RunInspector.InspectorAction("show", None, None))

    assert output[-1][0] == "warning"
    assert "no run selected" in output[-1][1].lower()
    assert statuses[-1][0] == "warning"


def test_refresh_run_inspector_prefers_active_run_then_latest_artifact():
    app = _minimal_app_for_unit()
    app._active_token = 22
    app._running_action = "validate"
    app._run_state = RunLifecycleState.RUNNING
    app._active_request = {"repo_path": "./active-repo", "spec": "rmsnorm", "query": ""}
    app._run_started_at = {22: time.perf_counter() - 0.2}
    app._recent_run_artifacts = [
        RunArtifact(
            run_id=10,
            action="analyze",
            status="Success",
            repo_path="./old",
            spec="",
            duration_seconds=1.1,
            terminal_state=RunLifecycleState.COMPLETED.value,
            summary_lines=["old summary"],
        )
    ]

    active_snapshot = app._build_run_inspector_snapshot()
    assert active_snapshot is not None
    assert active_snapshot["run_id"] == 22
    assert active_snapshot["action"] == "validate"
    assert active_snapshot["repo"] == "./active-repo"

    app._running_action = None
    app._active_request = None
    latest_snapshot = app._build_run_inspector_snapshot()
    assert latest_snapshot is not None
    assert latest_snapshot["run_id"] == 10
    assert latest_snapshot["action"] == "analyze"
    assert latest_snapshot["repo"] == "./old"


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


def test_build_request_supports_ask_namespace():
    app = _minimal_app_for_unit()

    action, req = app._build_request("/ask what does this repo do?")

    assert action == "chat"
    assert req["prompt"] == "what does this repo do?"


def test_build_request_supports_run_namespace_analyze():
    app = _minimal_app_for_unit()

    action, req = app._build_request("/run analyze ./repo")

    assert action == "analyze"
    assert req["action"] == "analyze"
    assert req["repo_path"] == "./repo"


def test_build_request_supports_run_namespace_generate():
    app = _minimal_app_for_unit()

    action, req = app._build_request("/run generate ./repo rmsnorm")

    assert action == "generate"
    assert req["action"] == "generate"
    assert req["repo_path"] == "./repo"
    assert req["spec"] == "rmsnorm"


def test_build_request_routes_plain_text_to_chat():
    app = _minimal_app_for_unit()

    action, req = app._build_request("hello model")

    assert action == "chat"
    assert req["prompt"] == "hello model"


def test_build_request_natural_action_routing_disabled_by_default():
    app = _minimal_app_for_unit()

    action, req = app._build_request("please analyze ./repo")

    assert action == "chat"
    assert req["prompt"] == "please analyze ./repo"


def test_build_request_natural_action_routing_enabled_by_env(monkeypatch):
    app = _minimal_app_for_unit()
    monkeypatch.setenv("SCHOLARDEVCLAW_TUI_ENABLE_NATURAL_ACTION_ROUTING", "true")

    action, req = app._build_request("please analyze ./repo")

    assert action == "analyze"
    assert req["action"] == "analyze"
    assert req["repo_path"] == "./repo"


def test_parse_approval_value_accepts_common_truthy_and_falsey_values():
    app = _minimal_app_for_unit()

    assert app._parse_approval_value("yes") is True
    assert app._parse_approval_value("Approve") is True
    assert app._parse_approval_value("no") is False
    assert app._parse_approval_value("reject") is False
    assert app._parse_approval_value("maybe") is None


def test_build_integrate_approval_callback_env_enabled_uses_stage_override(monkeypatch):
    app = _minimal_app_for_unit()
    monkeypatch.setenv("SCHOLARDEVCLAW_TUI_APPROVAL_GATES", "true")
    monkeypatch.setenv("SCHOLARDEVCLAW_TUI_APPROVAL_PATCH_APPLICATION", "true")

    emitted: list[str] = []
    app.call_from_thread = lambda fn, *args, **kwargs: fn(*args, **kwargs)  # type: ignore[assignment]
    app.post_message = lambda msg: emitted.append(getattr(msg, "line", ""))  # type: ignore[assignment]

    callback = app._build_integrate_approval_callback(5, input_reader=lambda _prompt: "n")
    assert callback is not None

    decision = callback("patch_application", {})

    assert decision is True
    assert any("Approval required [patch_application]" in line for line in emitted)
    assert any("Approval [patch_application] from env: approved" in line for line in emitted)


def test_build_integrate_approval_callback_noninteractive_auto_approves(monkeypatch):
    app = _minimal_app_for_unit()
    monkeypatch.setenv("SCHOLARDEVCLAW_TUI_APPROVAL_GATES", "true")
    monkeypatch.delenv("SCHOLARDEVCLAW_TUI_APPROVAL_DEFAULT", raising=False)

    emitted: list[str] = []
    app.call_from_thread = lambda fn, *args, **kwargs: fn(*args, **kwargs)  # type: ignore[assignment]
    app.post_message = lambda msg: emitted.append(getattr(msg, "line", ""))  # type: ignore[assignment]

    monkeypatch.setattr("scholardevclaw.tui.app.sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("scholardevclaw.tui.app.sys.stdout.isatty", lambda: False)

    callback = app._build_integrate_approval_callback(
        8,
        input_reader=lambda _prompt: (_ for _ in ()).throw(
            AssertionError("input should not be used")
        ),
    )
    assert callback is not None

    decision = callback("impact_acceptance", {})

    assert decision is True
    assert any("auto-approved (non-interactive terminal)" in line for line in emitted)


def test_build_integrate_approval_callback_patch_review_waits_for_submit(monkeypatch):
    app = _minimal_app_for_unit()
    monkeypatch.setenv("SCHOLARDEVCLAW_TUI_APPROVAL_GATES", "true")
    monkeypatch.delenv("SCHOLARDEVCLAW_TUI_APPROVAL_DEFAULT", raising=False)
    monkeypatch.delenv("SCHOLARDEVCLAW_TUI_APPROVAL_PATCH_APPLICATION", raising=False)

    emitted: list[str] = []
    app.call_from_thread = lambda fn, *args, **kwargs: fn(*args, **kwargs)  # type: ignore[assignment]
    app.post_message = lambda msg: emitted.append(getattr(msg, "line", ""))  # type: ignore[assignment]
    app._refresh_run_inspector = lambda: ["review"]
    app._set_status = lambda *_a, **_k: None

    monkeypatch.setattr("scholardevclaw.tui.app.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("scholardevclaw.tui.app.sys.stdout.isatty", lambda: True)

    callback = app._build_integrate_approval_callback(44, input_reader=lambda _prompt: "n")
    assert callback is not None

    import threading

    result_box: dict[str, object] = {}

    def _invoke():
        result_box["value"] = callback(
            "patch_application",
            {
                "hunks": [
                    {"id": "h1", "file": "a.py", "header": "@@"},
                    {"id": "h2", "file": "b.py", "header": "@@"},
                ]
            },
        )

    thread = threading.Thread(target=_invoke)
    thread.start()

    for _ in range(50):
        review = app._get_pending_integrate_review(44, "patch_application")
        if review is not None:
            break
        time.sleep(0.01)
    else:
        pytest.fail("pending review was not registered")

    submit_ok = app._submit_pending_review(
        token=44,
        stage="patch_application",
        approved=True,
        hunk_decisions={"h1": "accept", "h2": "reject"},
    )
    assert submit_ok is True

    thread.join(timeout=2.0)
    assert not thread.is_alive()

    decision = result_box.get("value")
    assert isinstance(decision, dict)
    assert decision["approved"] is True
    assert decision["hunk_decisions"]["h1"] == "accept"
    assert decision["hunk_decisions"]["h2"] == "reject"
    assert any("waiting for inspector review" in line for line in emitted)


def test_build_integrate_approval_callback_patch_review_noninteractive_returns_hunk_payload(
    monkeypatch,
):
    app = _minimal_app_for_unit()
    monkeypatch.setenv("SCHOLARDEVCLAW_TUI_APPROVAL_GATES", "true")
    monkeypatch.delenv("SCHOLARDEVCLAW_TUI_APPROVAL_PATCH_APPLICATION", raising=False)
    monkeypatch.setattr("scholardevclaw.tui.app.sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("scholardevclaw.tui.app.sys.stdout.isatty", lambda: False)

    app.call_from_thread = lambda fn, *args, **kwargs: fn(*args, **kwargs)  # type: ignore[assignment]
    app.post_message = lambda _msg: True  # type: ignore[assignment]
    app._refresh_run_inspector = lambda: ["review"]
    app._set_status = lambda *_a, **_k: None

    callback = app._build_integrate_approval_callback(55)
    assert callback is not None

    decision = callback(
        "patch_application",
        {
            "hunks": [
                {"id": "h1", "file": "a.py", "header": "@@"},
                {"id": "h2", "file": "b.py", "header": "@@"},
            ]
        },
    )

    assert isinstance(decision, dict)
    assert decision["approved"] is True
    assert decision["hunk_decisions"]["h1"] == "accept"
    assert decision["hunk_decisions"]["h2"] == "accept"


def test_on_inspector_action_review_submit_routes_and_updates_status():
    app = _minimal_app_for_unit()
    app._set_pending_integrate_review(
        token=71,
        stage="patch_application",
        hunks=[
            {"key": "h1", "id": "h1", "file": "a.py", "header": "@@"},
            {"key": "h2", "id": "h2", "file": "b.py", "header": "@@"},
        ],
    )

    output: list[tuple[str, str]] = []
    statuses: list[tuple[str, str]] = []

    class _DummyPrompt:
        def focus(self):
            return None

    def _query_one(selector, *_args, **_kwargs):
        if selector == "#prompt-input":
            return _DummyPrompt()
        raise AssertionError(f"unexpected selector: {selector}")

    app.query_one = _query_one  # type: ignore[assignment]
    app._append_output = lambda line, level="auto": output.append((level, line))
    app._set_status = lambda message, level="info": statuses.append((level, message))
    app._refresh_run_inspector = lambda: ["review"]

    app.on_inspector_action(
        RunInspector.InspectorAction(
            "review_submit",
            71,
            None,
            payload={
                "token": 71,
                "stage": "patch_application",
                "approved": True,
                "hunk_decisions": {"h1": "accept", "h2": "regenerate"},
            },
        )
    )

    review = app._get_pending_integrate_review(71, "patch_application")
    assert review is not None
    submitted = dict(review.get("submitted") or {})
    assert submitted.get("approved") is True
    assert submitted.get("hunk_decisions", {}).get("h1") == "accept"
    assert submitted.get("hunk_decisions", {}).get("h2") == "regenerate"
    assert output
    assert "Review submitted" in output[-1][1]
    assert statuses
    assert "Review [patch_application] submitted" in statuses[-1][1]


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


def test_chat_result_from_text_empty_response_is_failure():
    app = _minimal_app_for_unit()

    result = app._chat_result_from_text("hello", "   \n")

    assert result.ok is False
    assert "empty response" in result.error.lower()


def test_chat_prompt_includes_recent_run_context_after_completed_run():
    app = _minimal_app_for_unit()
    app._active_token = 51
    app._running_action = "analyze"
    app._run_started_at = {51: time.perf_counter() - 1.0}

    class _DummyStatus:
        def stop_timer(self):
            return None

    class _DummyPrompt:
        def focus(self):
            return None

    class _DummyHistory:
        def add_entry(self, **kwargs):
            return None

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
    app._summarize_result = lambda *_a, **_k: ["Languages: python", "Entry points: 2"]
    app._suggest_next_commands = lambda *_a, **_k: []
    app._update_command_meta = lambda: None
    app._set_phase = lambda *_a, **_k: None

    result = type("Result", (), {"ok": True, "payload": {}, "error": "", "logs": []})()
    request = {
        "action": "analyze",
        "repo_path": "./repo",
        "spec": "",
        "_original_command": "analyze ./repo",
    }
    app.on_task_completed(TaskCompleted(51, "analyze", result, request))

    prompt = app._build_chat_system_prompt()
    assert "Recent run context" in prompt
    assert "run #51: analyze [Success]" in prompt
    assert "repo=./repo" in prompt
    assert "Languages: python" in prompt


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


def test_transition_run_state_updates_phase_and_status_message():
    app = _minimal_app_for_unit()
    phases: list[str] = []
    statuses: list[tuple[str, str]] = []

    app._set_phase = lambda phase: phases.append(phase)
    app._set_status = lambda message, level="info": statuses.append((level, message))

    app._transition_run_state(RunLifecycleState.QUEUED, action="generate", detail="dispatch")

    assert app._run_state == RunLifecycleState.QUEUED
    assert phases[-1] == "validating"
    assert statuses[-1][0] == "accent"
    assert "Run state: QUEUED | generate | dispatch" in statuses[-1][1]


def test_append_run_event_sequence_is_monotonic_per_run():
    app = _minimal_app_for_unit()

    first = app._append_run_event(17, "run.accepted", message="accepted")
    second = app._append_run_event(17, "run.queued", message="queued")
    other = app._append_run_event(23, "run.accepted", message="accepted")

    assert first.seq == 1
    assert second.seq == 2
    assert other.seq == 1
    assert app._run_event_seq[17] == 2


def test_start_task_emits_core_lifecycle_events(monkeypatch):
    app = _minimal_app_for_unit()
    app._validate_request_inputs = lambda req: (True, [], [])
    app._set_status = lambda *_a, **_k: None
    app._emit_progress = lambda *_a, **_k: None
    app._update_command_meta = lambda: None

    class _DummyStatus:
        def start_timer(self):
            return None

    app.query_one = lambda *_a, **_k: _DummyStatus()  # type: ignore[assignment]

    class _NoopThread:
        def __init__(self, *args, **kwargs):
            return None

        def start(self):
            return None

    monkeypatch.setattr("scholardevclaw.tui.app.threading.Thread", _NoopThread)

    app._start_task("analyze", {"action": "analyze", "repo_path": "./repo"}, command="analyze")

    run_id = app._active_token
    event_types = [item.type for item in app._run_events[run_id][:3]]
    assert event_types == ["run.accepted", "run.queued", "run.running"]


def test_on_chat_delta_records_sampled_event():
    app = _minimal_app_for_unit()
    app._active_token = 31
    app._running_action = "chat"

    class _DummyStatus:
        def update_timer(self):
            return None

    app.query_one = lambda *_a, **_k: _DummyStatus()  # type: ignore[assignment]
    app._set_live_text = lambda *_a, **_k: None

    from scholardevclaw.tui.app import ChatDelta

    app.on_chat_delta(ChatDelta(31, "hello there"))

    assert any(event.type == "chat.delta" for event in app._run_events[31])


def test_run_events_command_renders_known_unknown_and_limit():
    app = _minimal_app_for_unit()
    app._run_events = {
        44: [
            RunEvent(run_id=44, seq=1, timestamp=1.0, type="run.accepted", message="accepted"),
            RunEvent(run_id=44, seq=2, timestamp=2.0, type="run.queued", message="queued"),
            RunEvent(run_id=44, seq=3, timestamp=3.0, type="run.running", message="running"),
            RunEvent(run_id=44, seq=4, timestamp=4.0, type="run.completed", message="done"),
        ]
    }
    output: list[str] = []
    statuses: list[tuple[str, str]] = []
    app._append_output = lambda line, level="auto": output.append(line)
    app._set_status = lambda message, level="info": statuses.append((level, message))

    app._execute_action_request("run_events", {"run_id": 44}, command="run events 44")

    assert output
    assert output[0].startswith("001 run.accepted")
    assert statuses[-1][1] == "Run #44 events"

    output.clear()
    app._execute_action_request("run_events", {"run_id": 44, "limit": 2}, command="run events 44 2")
    assert len(output) == 2
    assert output[0].startswith("003 run.running")
    assert output[1].startswith("004 run.completed")
    assert statuses[-1][1] == "Run #44 events (last 2)"

    output.clear()
    app._execute_action_request("run_events", {"run_id": 999}, command="run events 999")
    assert output[0] == "Run #999 has no recorded events"


def test_run_events_command_warns_when_run_id_missing():
    app = _minimal_app_for_unit()
    output: list[tuple[str, str]] = []
    statuses: list[tuple[str, str]] = []
    app._append_output = lambda line, level="auto": output.append((level, line))
    app._set_status = lambda message, level="info": statuses.append((level, message))

    app._execute_action_request("run_events", {}, command="run events")

    assert output[0][0] == "warning"
    assert "run id required" in output[0][1].lower()
    assert statuses[-1][1] == "Run events requires id"


def test_build_request_supports_run_events_commands_and_limit():
    app = _minimal_app_for_unit()

    action, req = app._build_request("run events")
    assert action == "run_events"
    assert req == {}

    action, req = app._build_request("run events 7")
    assert action == "run_events"
    assert req == {"run_id": 7}

    action, req = app._build_request("run events 7 3")
    assert action == "run_events"
    assert req == {"run_id": 7, "limit": 3}


def test_on_task_completed_emits_terminal_completion_event():
    app = _minimal_app_for_unit()
    app._active_token = 88
    app._running_action = "analyze"
    app._run_started_at = {88: time.perf_counter() - 0.3}

    class _DummyStatus:
        def stop_timer(self):
            return None

    class _DummyPrompt:
        def focus(self):
            return None

    class _DummyHistory:
        def add_entry(self, **kwargs):
            return None

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
    app._save_runtime_state = lambda: None

    result = type("Result", (), {"ok": True, "payload": {}, "error": "", "logs": []})()
    request = {
        "action": "analyze",
        "repo_path": "./repo",
        "spec": "",
        "_original_command": "analyze ./repo",
    }

    app.on_task_completed(TaskCompleted(88, "analyze", result, request))

    assert any(event.type == "run.completed" for event in app._run_events[88])


def test_on_task_completed_failure_sets_failure_code_and_inspector_lines():
    app = _minimal_app_for_unit()
    app._active_token = 77
    app._running_action = "search"
    app._active_request = {"action": "search", "query": "flash attention"}
    app._run_started_at = {77: time.perf_counter() - 0.2}

    class _DummyStatus:
        def stop_timer(self):
            return None

    class _DummyPrompt:
        def focus(self):
            return None

    class _DummyHistory:
        def add_entry(self, **kwargs):
            return None

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
    app._summarize_result = lambda *_a, **_k: ["search failed"]
    app._update_command_meta = lambda: None
    app._save_runtime_state = lambda: None

    result = type(
        "Result",
        (),
        {
            "ok": False,
            "payload": {},
            "error": "Rate limit reached (429) while querying model",
            "logs": [],
        },
    )()
    request = {"action": "search", "query": "flash attention", "_original_command": "search"}

    app.on_task_completed(TaskCompleted(77, "search", result, request))

    artifact = app._recent_run_artifacts[-1]
    assert artifact.failure_code == "E_LLM_RATE_LIMIT"
    assert "Rate limit" in artifact.error
    assert any("failure=E_LLM_RATE_LIMIT" in line for line in app._inspector_lines)
    assert any("Error:" in line for line in app._inspector_lines)


def test_inspect_command_outputs_current_inspector_snapshot_lines():
    app = _minimal_app_for_unit()
    app._recent_run_artifacts = [
        RunArtifact(
            run_id=5,
            action="generate",
            status="Failed",
            repo_path="./repo",
            spec="rmsnorm",
            duration_seconds=3.1,
            terminal_state=RunLifecycleState.FAILED.value,
            failure_code="E_RUNTIME_EXCEPTION",
            error="boom",
            summary_lines=["failed summary"],
        )
    ]
    app._run_events = {
        5: [RunEvent(run_id=5, seq=1, timestamp=1.0, type="run.failed", message="boom")]
    }
    output: list[str] = []
    statuses: list[tuple[str, str]] = []
    app._append_output = lambda line, level="auto": output.append(line)
    app._set_status = lambda message, level="info": statuses.append((level, message))

    expected = app._refresh_run_inspector()
    output.clear()
    app._execute_action_request("inspect", {}, command="inspect")

    assert output == expected
    assert statuses[-1][1] == "Run inspector snapshot"


def test_runs_command_outputs_compact_recent_runs():
    app = _minimal_app_for_unit()
    app._recent_run_artifacts = [
        RunArtifact(
            run_id=1,
            action="analyze",
            status="Success",
            repo_path="./repo",
            spec="",
            duration_seconds=1.4,
            terminal_state=RunLifecycleState.COMPLETED.value,
            summary_lines=["Languages: python"],
        ),
        RunArtifact(
            run_id=2,
            action="generate",
            status="Failed",
            repo_path="./repo",
            spec="rmsnorm",
            duration_seconds=3.2,
            terminal_state=RunLifecycleState.FAILED.value,
            summary_lines=["Branch: feature/rmsnorm"],
        ),
    ]
    output: list[str] = []
    statuses: list[str] = []
    app._append_output = lambda line, level="auto": output.append(line)
    app._set_status = lambda message, level="info": statuses.append(message)

    app._execute_action_request("runs", {}, command="runs")

    assert output
    assert output[0].startswith("#2 generate")
    assert "Failed" in output[0]
    assert output[1].startswith("#1 analyze")
    assert "Recent runs listed" in statuses[-1]


def test_run_show_command_renders_known_and_unknown_ids():
    app = _minimal_app_for_unit()
    app._recent_run_artifacts = [
        RunArtifact(
            run_id=7,
            action="map",
            status="Success",
            repo_path="./repo",
            spec="rmsnorm",
            duration_seconds=2.0,
            terminal_state=RunLifecycleState.COMPLETED.value,
            summary_lines=["Targets: 3", "Confidence: 90%"],
        )
    ]
    app._run_replay_map = {
        7: {
            "command": "map ./repo rmsnorm",
            "action": "map",
            "request": {
                "action": "map",
                "repo_path": "./repo",
                "spec": "rmsnorm",
            },
            "terminal_state": RunLifecycleState.COMPLETED.value,
            "status": "Success",
            "duration_seconds": 2.0,
            "summary_lines": ["Targets: 3", "Confidence: 90%"],
        }
    }
    output: list[str] = []
    statuses: list[tuple[str, str]] = []
    app._append_output = lambda line, level="auto": output.append(line)
    app._set_status = lambda message, level="info": statuses.append((level, message))

    app._execute_action_request("run_show", {"run_id": 7}, command="run show 7")

    assert any(line == "Run #7" for line in output)
    assert any("Action: map" in line for line in output)
    assert any("Summary:" in line for line in output)
    assert statuses[-1][1] == "Run #7 details"

    output.clear()
    app._execute_action_request("run_show", {"run_id": 404}, command="run show 404")
    assert output[0] == "Run #404 not found"


def test_run_rerun_command_routes_through_replay_path():
    app = _minimal_app_for_unit()
    called: list[int] = []
    app._rerun_history_item = lambda run_id: called.append(run_id)

    app._execute_action_request("run_rerun", {"run_id": 12}, command="run rerun 12")

    assert called == [12]


def test_runtime_state_persists_recent_run_artifacts_replay_map_and_events(monkeypatch, tmp_path):
    monkeypatch.setenv("SCHOLARDEVCLAW_AUTH_DIR", str(tmp_path))
    app = _minimal_app_for_unit()
    app._provider = "openrouter"
    app._model = "anthropic/claude-sonnet-4"
    app._directory = "/tmp/repo"
    app._recent_run_artifacts = [
        RunArtifact(
            run_id=9,
            action="validate",
            status="Cancelled",
            repo_path="./repo",
            spec="rmsnorm",
            query="",
            duration_seconds=4.2,
            terminal_state=RunLifecycleState.CANCELLED.value,
            failure_code="E_CANCELLED_BY_USER",
            error="Task cancelled",
            summary_lines=["Stage: generated"],
        )
    ]
    app._run_replay_map = {
        9: {
            "command": "validate ./repo",
            "action": "validate",
            "request": {
                "action": "validate",
                "repo_path": "./repo",
                "spec": "rmsnorm",
                "prompt": "secret should not persist",
            },
            "status": "Cancelled",
            "terminal_state": RunLifecycleState.CANCELLED.value,
            "duration_seconds": 4.2,
            "failure_code": "E_CANCELLED_BY_USER",
            "error": "Task cancelled",
            "summary_lines": ["Stage: generated"],
        }
    }
    long_chat = "chunk-" * 80
    app._run_events = {
        9: [
            RunEvent(
                run_id=9,
                seq=1,
                timestamp=1.0,
                type="run.accepted",
                message="accepted",
            ),
            RunEvent(
                run_id=9,
                seq=2,
                timestamp=2.0,
                type="chat.delta",
                message=long_chat,
                payload={"chars": len(long_chat), "chunk": long_chat},
            ),
        ]
    }
    app._run_event_seq = {9: 2}

    app._save_runtime_state()

    state_path = app._runtime_state_path()
    saved = json.loads(state_path.read_text())
    assert saved["recent_run_artifacts"][0]["run_id"] == 9
    assert saved["recent_run_artifacts"][0]["failure_code"] == "E_CANCELLED_BY_USER"
    assert "Task cancelled" in saved["recent_run_artifacts"][0]["error"]
    assert saved["replay_map"]["9"]["failure_code"] == "E_CANCELLED_BY_USER"
    assert "prompt" not in saved["replay_map"]["9"]["request"]
    assert "run_events" in saved
    assert saved["run_events"]["9"][1]["type"] == "chat.delta"
    assert len(saved["run_events"]["9"][1]["message"]) <= 220
    assert len(saved["run_events"]["9"][1]["payload"]["chunk"]) <= 180

    # Also include malformed entries; loader should ignore safely.
    saved["recent_run_artifacts"].append({"run_id": None, "action": 1})
    saved["replay_map"]["not-a-run"] = {"action": "analyze"}
    saved["run_events"]["bad"] = [{"seq": "x"}]
    saved["run_events"]["9"].append({"seq": "bad", "timestamp": "nan", "type": 1})
    state_path.write_text(json.dumps(saved))

    reloaded = ScholarDevClawApp()
    assert any(item.run_id == 9 for item in reloaded._recent_run_artifacts)
    assert 9 in reloaded._run_replay_map
    assert 9 in reloaded._run_events
    assert any(event.type == "chat.delta" for event in reloaded._run_events[9])
    assert "prompt" not in reloaded._run_replay_map[9]["request"]
    assert reloaded._run_replay_map[9]["failure_code"] == "E_CANCELLED_BY_USER"
    restored = next(item for item in reloaded._recent_run_artifacts if item.run_id == 9)
    assert restored.failure_code == "E_CANCELLED_BY_USER"


def test_build_request_supports_runs_show_and_rerun_commands():
    app = _minimal_app_for_unit()

    action, req = app._build_request("runs")
    assert action == "runs"

    action, req = app._build_request("inspect")
    assert action == "inspect"
    assert req == {}

    action, req = app._build_request("run show 7")
    assert action == "run_show"
    assert req == {"run_id": 7}

    action, req = app._build_request("run rerun 7")
    assert action == "run_rerun"
    assert req == {"run_id": 7}
