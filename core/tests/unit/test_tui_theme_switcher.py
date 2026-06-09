"""Tests for the ThemeSwitcherScreen modal."""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from scholardevclaw.tui.theme_switcher import ThemeSwitcherScreen  # noqa: E402


class TestThemeSwitcherScreenInstantiation:
    def test_can_instantiate(self) -> None:
        screen = ThemeSwitcherScreen()
        assert screen is not None

    def test_default_theme_is_opencode(self) -> None:
        screen = ThemeSwitcherScreen()
        assert screen._current_theme == "opencode"
        assert screen._selected_index == 0

    def test_accepts_current_theme(self) -> None:
        screen = ThemeSwitcherScreen(current_theme="minimal")
        assert screen._current_theme == "minimal"
        assert screen._selected_index == 2

    def test_inherits_modal_screen(self) -> None:
        from textual.screen import ModalScreen

        assert isinstance(ThemeSwitcherScreen(), ModalScreen)


class TestThemeSwitcherThemes:
    def test_has_all_themes(self) -> None:
        screen = ThemeSwitcherScreen()
        assert "opencode" in screen._themes
        assert "claude" in screen._themes
        assert "minimal" in screen._themes
        assert "high_contrast" in screen._themes

    def test_theme_order(self) -> None:
        screen = ThemeSwitcherScreen()
        assert screen._themes == ["opencode", "claude", "minimal", "high_contrast"]


class TestThemeSwitcherActions:
    def test_has_dismiss_action(self) -> None:
        screen = ThemeSwitcherScreen()
        assert callable(getattr(screen, "action_dismiss_modal", None))

    def test_has_select_action(self) -> None:
        screen = ThemeSwitcherScreen()
        assert callable(getattr(screen, "action_select_theme", None))


class TestThemeSwitcherPreview:
    def test_preview_updates_on_selection(self) -> None:
        screen = ThemeSwitcherScreen()
        # Simulate selection change
        screen._selected_index = 1
        # The preview update method should not raise
        screen._update_preview()
