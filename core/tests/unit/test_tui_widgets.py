from __future__ import annotations

import inspect
from types import SimpleNamespace

from scholardevclaw.tui.widgets import HistoryPane, LogView, PhaseTracker, PromptInput, StatusBar


def test_logview_detect_level_classification():
    assert LogView._detect_level("Error: boom") == "error"
    assert LogView._detect_level("warning: careful") == "warning"
    assert LogView._detect_level("Task complete") == "success"
    assert LogView._detect_level("=== section ===") == "accent"
    assert LogView._detect_level("--- step ---") == "system"
    assert LogView._detect_level("hello") == "info"


def test_logview_progress_line_is_reused():
    log = LogView()
    mounted: list[object] = []
    log.mount = lambda widget: mounted.append(widget)

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
    pane = HistoryPane()
    pane._render_entries = lambda: None
    pane.add_entry(41, "analyze", "Done", duration=1.0)

    posted: list[int] = []
    pane.post_message = lambda msg: posted.append(msg.run_id)
    pane._selected_index = 0

    pane.on_key(SimpleNamespace(key="enter", stop=lambda: None))

    assert posted == [41]


def test_phasetracker_set_phase_updates_state_without_crashing():
    tracker = PhaseTracker()
    tracker.query_one = lambda *_a, **_k: SimpleNamespace(
        update=lambda *_x, **_y: None,
        styles=SimpleNamespace(background=None, width=None),
    )

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


def test_promptinput_delegates_key_handling_to_app_level():
    """PromptInput no longer overrides _on_key; app handles special keys."""
    # PromptInput should not have its own on_key override
    assert not hasattr(PromptInput, "on_key")
    # App-level on_key intercepts up/down/tab/escape before Input consumes them.
