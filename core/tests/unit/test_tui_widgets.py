from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("textual")

from scholardevclaw.tui.widgets import (
    HistoryPane,
    LogView,
    PhaseTracker,
    PromptInput,
    RunInspector,
    StatusBar,
)


def test_logview_detect_level_classification():
    assert LogView._detect_level("Error: boom") == "error"
    assert LogView._detect_level("warning: careful") == "warning"
    assert LogView._detect_level("Task complete") == "success"
    assert LogView._detect_level("=== section ===") == "accent"
    assert LogView._detect_level("--- step ---") == "system"
    assert LogView._detect_level("hello") == "info"


def test_logview_progress_line_is_reused():
    pytest.importorskip("textual")
    log = LogView()
    mounted: list[object] = []

    def _mount(*widgets, **_kwargs):
        mounted.extend(widgets)

    object.__setattr__(log, "mount", _mount)

    log.set_progress("Scanning repository...")
    first = log._progress_line
    log.set_progress("Scanning repository... [████░░░░░░] 40%")

    assert first is not None
    assert log._progress_line is first
    assert len(mounted) == 1


def test_historypane_keeps_only_latest_20_entries():
    pane = HistoryPane()
    pane._render_entries = lambda: None
    for i in range(1, 26):
        pane.add_entry(i, "analyze", "Done", duration=0.1 * i)

    assert len(pane._entries) == 20
    assert pane._entries[0]["id"] == 25
    assert pane._entries[-1]["id"] == 6


def test_historypane_keyboard_activate_posts_selected_run():
    pytest.importorskip("textual")
    pane = HistoryPane()
    pane._render_entries = lambda: None
    pane.add_entry(41, "analyze", "Done", duration=1.0)

    posted: list[int] = []

    def _post_message(message):
        posted.append(int(getattr(message, "run_id", 0)))
        return True

    object.__setattr__(pane, "post_message", _post_message)
    pane._selected_index = 0

    pane.on_key(SimpleNamespace(key="enter", stop=lambda: None))  # type: ignore[arg-type]

    assert posted == [41]


def test_phasetracker_set_phase_updates_state_without_crashing():
    pytest.importorskip("textual")
    tracker = PhaseTracker()

    def _query_one(*_a, **_k):
        return SimpleNamespace(
            update=lambda *_x, **_y: None,
            styles=SimpleNamespace(background=None, width=None),
        )

    object.__setattr__(tracker, "query_one", _query_one)

    tracker.set_phase("complete")

    assert tracker.current_phase == "complete"


def test_statusbar_refresh_does_not_shadow_textual_render():
    status = StatusBar()

    status.set_context(
        mode="search", provider="openrouter", model="anthropic/claude-sonnet-4", directory="./repo"
    )
    status.set_usage(session_tokens=1536, last_tokens=320)
    status.set_status("Running", "accent")
    status.set_step(1, 3)
    rendered = status.render()

    assert callable(getattr(status, "_render", None))
    assert "MODE: search" in str(rendered)
    assert "PROVIDER: openrouter" in str(rendered)
    assert "TOKENS: 1.5k" in str(rendered)


def test_statusbar_shows_fallback_model_and_truncated_directory():
    status = StatusBar()

    status.set_context(
        mode="analyze",
        provider="openrouter",
        model="",
        directory="/very/long/path/that/should/be/truncated/in/status/bar/for/readability",
    )
    rendered = str(status.render())

    assert "MODEL: unset" in rendered
    assert "DIR: …" in rendered


def test_statusbar_supports_multiline_render_for_long_context():
    status = StatusBar()

    status.set_context(
        mode="analyze",
        provider="openrouter",
        model="very-long-model-name-that-exceeds-common-terminal-width",
        directory="/this/is/a/very/long/path/that/would/otherwise/push/dir/offscreen",
    )

    rendered = str(status.render())
    assert "DIR:" in rendered


def test_promptinput_delegates_key_handling_to_app_level():
    """PromptInput no longer overrides _on_key; app handles special keys."""
    # PromptInput should not have its own on_key override
    assert not hasattr(PromptInput, "on_key")
    # App-level on_key intercepts up/down/tab/escape before Input consumes them.


def test_run_inspector_render_snapshot_lines_truncates_error_and_limits_events():
    long_error = "x" * 300
    lines = RunInspector.render_snapshot_lines(
        {
            "run_id": 9,
            "action": "generate",
            "status": "Failed",
            "duration": 3.2,
            "terminal_state": "failed",
            "failure_code": "E_RUNTIME_EXCEPTION",
            "error": long_error,
            "repo": "./repo",
            "spec": "rmsnorm",
            "query": "",
            "summary_lines": ["step-a", "step-b", "step-c"],
            "event_lines": [
                "001 run.accepted idle accepted",
                "002 run.running running running",
                "003 log.line running phase 1",
                "004 run.failed failed boom",
            ],
        },
        max_events=2,
        max_error_chars=80,
    )

    assert any(line.startswith("Run #9 | generate | Failed") for line in lines)
    assert any(line.startswith("Error: ") for line in lines)
    error_line = next(line for line in lines if line.startswith("Error: "))
    assert len(error_line) <= 88
    assert error_line.endswith("…")
    assert "Events:" in lines
    assert "003 log.line running phase 1" in lines
    assert "004 run.failed failed boom" in lines
    assert "001 run.accepted idle accepted" not in lines


def test_run_inspector_keyboard_navigation_and_action_messages():
    inspector = RunInspector()
    inspector.set_lines(
        [
            "Run #9 | generate | Failed | 3.2s",
            "Events:",
            "001 run.accepted idle accepted",
            "002 run.running running running",
            "003 run.failed failed boom",
        ],
        run_id=9,
    )

    posted: list[tuple[str, int | None, int | None]] = []

    def _post_message(message):
        posted.append((message.action, message.run_id, message.seq))
        return True

    object.__setattr__(inspector, "post_message", _post_message)

    inspector.on_key(SimpleNamespace(key="down", stop=lambda: None))  # type: ignore[arg-type]
    assert inspector._selected_event_index == 1

    inspector.on_key(SimpleNamespace(key="enter", stop=lambda: None))  # type: ignore[arg-type]
    inspector.on_key(SimpleNamespace(key="r", stop=lambda: None))  # type: ignore[arg-type]
    inspector.on_key(SimpleNamespace(key="s", stop=lambda: None))  # type: ignore[arg-type]
    inspector.on_key(SimpleNamespace(key="e", stop=lambda: None))  # type: ignore[arg-type]

    assert posted[0] == ("events", 9, 2)
    assert posted[1] == ("rerun", 9, 2)
    assert posted[2] == ("show", 9, 2)
    assert posted[3] == ("events", 9, 2)


def test_run_inspector_review_mode_renders_compact_hunk_lines_and_counts():
    inspector = RunInspector()
    inspector.set_review(
        token=12,
        stage="patch_application",
        hunks=[
            {"id": "h1", "file": "src/a.py", "header": "@@ -1,4 +1,5 @@"},
            {"id": "h2", "file": "src/b.py", "header": "@@ -10,3 +11,7 @@"},
        ],
        decisions={"h1": "accept", "h2": "pending"},
        run_id=12,
    )

    rendered = str(inspector.render())
    assert "Review pending (patch_application)" in rendered
    assert "A:1 X:0 G:0 P:1" in rendered
    assert "src/a.py #h1" in rendered
    assert "src/b.py #h2" in rendered


def test_run_inspector_review_mode_keyboard_updates_and_submit_payload():
    inspector = RunInspector()
    inspector.set_review(
        token=22,
        stage="patch_application",
        hunks=[
            {"id": "1", "file": "src/a.py", "header": "@@"},
            {"id": "2", "file": "src/b.py", "header": "@@"},
        ],
        decisions={"1": "pending", "2": "pending"},
        run_id=22,
    )

    posted: list[tuple[str, dict[str, Any]]] = []

    def _post_message(message):
        posted.append((message.action, dict(getattr(message, "payload", {}) or {})))
        return True

    object.__setattr__(inspector, "post_message", _post_message)

    inspector.on_key(SimpleNamespace(key="a", stop=lambda: None))  # type: ignore[arg-type]
    inspector.on_key(SimpleNamespace(key="down", stop=lambda: None))  # type: ignore[arg-type]
    inspector.on_key(SimpleNamespace(key="g", stop=lambda: None))  # type: ignore[arg-type]
    inspector.on_key(SimpleNamespace(key="enter", stop=lambda: None))  # type: ignore[arg-type]

    assert posted[0][0] == "review_update"
    assert posted[0][1]["hunk_decisions"]["1"] == "accept"
    assert posted[1][0] == "review_update"
    assert posted[1][1]["hunk_decisions"]["2"] == "regenerate"
    assert posted[2][0] == "review_submit"
    assert posted[2][1]["approved"] is True
    assert posted[2][1]["hunk_decisions"]["1"] == "accept"
    assert posted[2][1]["hunk_decisions"]["2"] == "regenerate"
