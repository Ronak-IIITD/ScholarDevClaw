from __future__ import annotations

import inspect

import pytest

pytest.importorskip("textual")

from scholardevclaw.tui.screens import ProviderSetupScreen


def test_provider_setup_submitted_handlers_are_field_scoped():
    source = inspect.getsource(ProviderSetupScreen)

    assert '@on(Input.Submitted, "#setup-provider")' in source
    assert '@on(Input.Submitted, "#setup-model")' in source
    assert '@on(Input.Submitted, "#setup-key")' in source
    assert "@on(Input.Submitted)\n    def on_input_submitted" not in source


def test_help_text_uses_current_openrouter_default_model():
    from scholardevclaw.tui.screens import DEFAULT_OPENROUTER_MODEL, HELP_TEXT

    assert DEFAULT_OPENROUTER_MODEL in HELP_TEXT


def test_no_hardcoded_hex_palette_in_help_text():
    from scholardevclaw.tui.screens import HELP_TEXT

    assert "#" not in HELP_TEXT


def test_help_text_mentions_ask_and_run_namespaces():
    from scholardevclaw.tui.screens import HELP_TEXT

    assert "/ask <question>" in HELP_TEXT
    assert "/run <action> [args...]" in HELP_TEXT
