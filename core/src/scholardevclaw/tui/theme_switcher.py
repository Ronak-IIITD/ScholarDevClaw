"""Theme switcher modal for the TUI.

Allows users to preview and select from available themes.
"""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import ListItem, ListView, Static

from .theme import get_theme

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
        width: 60%;
        height: auto;
        max-height: 80%;
        max-width: 80;
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

    def __init__(self, current_theme: str = "default", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._current_theme = current_theme
        self._themes = ["default", "minimal", "high_contrast"]
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
        for theme_name in self._themes:
            marker = " ● " if theme_name == self._current_theme else "   "
            label = f"{marker}{theme_name}"
            list_view.append(ListItem(Static(label)))

    def _update_preview(self) -> None:
        theme_name = self._themes[self._selected_index]
        theme = get_theme(theme_name)
        colors = theme["colors"]
        preview = (
            f"[b]{theme_name}[/b]\n"
            f"  background: {colors['background']}\n"
            f"  surface:    {colors['surface']}\n"
            f"  text:       {colors['text']}\n"
            f"  accent:     {colors['accent']}\n"
            f"  success:    {colors['success']}\n"
            f"  warning:    {colors['warning']}\n"
            f"  error:      {colors['error']}"
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
