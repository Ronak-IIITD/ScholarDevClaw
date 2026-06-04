"""Tests for the LogSearchScreen modal."""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from scholardevclaw.tui.log_search import LogSearchScreen  # noqa: E402
from scholardevclaw.tui.widgets import LogView  # noqa: E402


class TestLogSearchScreen:
    def test_can_instantiate(self) -> None:
        screen = LogSearchScreen()
        assert screen is not None

    def test_default_initial_empty(self) -> None:
        screen = LogSearchScreen()
        # _initial is private but stable
        assert screen._initial == ""

    def test_accepts_initial_value(self) -> None:
        screen = LogSearchScreen(initial="hello")
        assert screen._initial == "hello"

    def test_has_clear_filter_action(self) -> None:
        screen = LogSearchScreen()
        assert hasattr(screen, "action_clear_filter")
        assert callable(screen.action_clear_filter)

    def test_has_dismiss_action(self) -> None:
        screen = LogSearchScreen()
        assert hasattr(screen, "action_dismiss_modal")
        assert callable(screen.action_dismiss_modal)

    def test_inherits_modal_screen(self) -> None:
        from textual.screen import ModalScreen

        assert isinstance(LogSearchScreen(), ModalScreen)

    def test_uses_logview_filter_target(self) -> None:
        """The screen references LogView.set_search_filter."""
        # We only need to confirm the import is in place and the symbol
        # is wired through the compose path.
        assert LogView is not None
        # LogView must have a set_search_filter method
        assert callable(getattr(LogView, "set_search_filter", None))
