"""Theme switcher modal for the TUI.

Allows users to preview and select from available themes.
Updated to use the new theme system with 4 polished themes.
"""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import ListItem, ListView, Static

from .theme import THEMES, get_theme_colors

HINT_TEXT = "[dim]↑/↓ navigate · Enter select · Esc cancel[/dim]"


class ThemeSwitcherScreen(ModalScreen[str | None]):
    """Browse and select a theme."""

    BINDINGS = [
        ("escape", "dismiss_modal", "Cancel"),
        ("enter", "select_theme", "Select"),
    ]

    DEFAULT_CSS = """
    ThemeSwitcherScreen {
        align: center middle;
        background: $background 70%;
        layer: overlay;
    }

    ThemeSwitcherScreen > Vertical {
        width: 70%;
        height: auto;
        max-height: 80%;
        max-width: 90;
        padding: 1 2;
        border: tall $accent;
        background: $surface;
    }

    ThemeSwitcherScreen #theme-switcher-title {
        width: 100%;
        height: 1;
        margin-bottom: 1;
        color: $accent;
        text-style: bold;
    }

    ThemeSwitcherScreen #theme-switcher-list {
        width: 100%;
        height: auto;
        max-height: 15;
        background: $background;
        border: round $border;
    }

    ThemeSwitcherScreen #theme-switcher-preview {
        width: 100%;
        height: auto;
        margin-top: 1;
        padding: 1;
        background: $background;
        border: round $border;
    }

    ThemeSwitcherScreen #theme-switcher-hint {
        width: 100%;
        height: auto;
        margin-top: 1;
        color: $text-muted;
    }
    """

    def __init__(self, current_theme: str = "opencode", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._current_theme = current_theme
        self._themes = list(THEMES.keys())
        self._selected_index = (
            self._themes.index(current_theme) if current_theme in self._themes else 0
        )

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Theme Switcher", id="theme-switcher-title")
            yield ListView(id="theme-switcher-list")
            yield Static("", id="theme-switcher-preview")
            yield Static(HINT_TEXT, id="theme-switcher-hint")

    def on_mount(self) -> None:
        self._refresh_list()
        self._update_preview()
        try:
            list_view = self.query_one("#theme-switcher-list", ListView)
            list_view.index = self._selected_index
        except Exception:
            pass

    def _refresh_list(self) -> None:
        list_view = self.query_one("#theme-switcher-list", ListView)
        list_view.clear()
        for theme_key in self._themes:
            theme = THEMES[theme_key]
            marker = " ● " if theme_key == self._current_theme else "   "
            label = f"{marker}{theme['name']}  [dim]{theme['description']}[/dim]"
            list_view.append(ListItem(Static(label)))

    def _update_preview(self) -> None:
        theme_key = self._themes[self._selected_index]
        colors = get_theme_colors(theme_key)
        preview = (
            f"[b]{THEMES[theme_key]['name']}[/b]  {THEMES[theme_key]['description']}\n"
            f"  background: {colors['background']}\n"
            f"  surface:    {colors['surface']}\n"
            f"  text:       {colors['text']}\n"
            f"  accent:     {colors['accent']}  {'█' * 8}\n"
            f"  success:    {colors['success']}  {'█' * 8}\n"
            f"  warning:    {colors['warning']}  {'█' * 8}\n"
            f"  error:      {colors['error']}  {'█' * 8}"
        )
        try:
            self.query_one("#theme-switcher-preview", Static).update(preview)
        except Exception:
            pass

    @on(ListView.Selected, "#theme-switcher-list")
    def on_list_selected(self, event: ListView.Selected) -> None:
        if event.item is not None and hasattr(event.item, "index"):
            self._selected_index = event.item.index
        self._update_preview()

    @on(ListView.Highlighted, "#theme-switcher-list")
    def on_list_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is not None and hasattr(event.item, "index"):
            self._selected_index = event.item.index
        self._update_preview()

    def action_select_theme(self) -> None:
        self.dismiss(self._themes[self._selected_index])

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)
