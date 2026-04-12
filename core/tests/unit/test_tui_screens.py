from __future__ import annotations

import inspect

import pytest

pytest.importorskip("textual")

from scholardevclaw.tui.screens import (
    DEFAULT_OPENROUTER_MODEL,
    HELP_TEXT,
    CommandPalette,
    ProviderSetupScreen,
)


def test_provider_setup_submitted_handlers_are_field_scoped():
    source = inspect.getsource(ProviderSetupScreen)

    assert '@on(Input.Submitted, "#setup-provider")' in source
    assert '@on(Input.Submitted, "#setup-model")' in source
    assert '@on(Input.Submitted, "#setup-key")' in source
    assert "@on(Input.Submitted)\n    def on_input_submitted" not in source


def test_help_text_uses_current_openrouter_default_model():
    assert DEFAULT_OPENROUTER_MODEL in HELP_TEXT


def test_no_hardcoded_hex_palette_in_help_text():
    assert "#" not in HELP_TEXT


def test_help_text_mentions_ask_and_run_namespaces():
    assert "/ask <question>" in HELP_TEXT
    assert "/run <action> [args...]" in HELP_TEXT


def test_help_and_palette_include_run_events_command():
    assert "run events <id> [limit]" in HELP_TEXT
    assert "run events 1" in CommandPalette.PALETTE_COMMANDS


def test_help_and_palette_include_inspect_command():
    assert "inspect" in HELP_TEXT
    assert "inspect" in CommandPalette.PALETTE_COMMANDS
