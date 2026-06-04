"""Reverse-i-search modal for command history.

Mirrors bash's ``Ctrl+R`` behaviour: a small modal that filters the
``_command_history`` list by substring as the user types, shows the
current best match in real time, and lets the user navigate to the
next earlier / later match. ``Enter`` accepts the current match and
dismisses; ``Escape`` cancels and dismisses.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

HINT_TEXT = (
    "[dim]Type to filter history (case-insensitive). Enter accepts, "
    "F2 / Shift+Tab go back, Shift+F2 go forward, Esc cancels.[/dim]"
)


class ReverseSearchScreen(ModalScreen[str | None]):
    """Live-filter the command history; returns the chosen command (or None)."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("enter", "accept", "Accept"),
        ("f2", "find_prev", "Prev"),
        ("shift+tab", "find_prev", "Prev"),
        ("shift+f2", "find_next", "Next"),
    ]

    DEFAULT_CSS = """
    ReverseSearchScreen {
        align: center top;
        background: $background 60%;
        layer: overlay;
    }

    ReverseSearchScreen > Vertical {
        width: 80%;
        height: auto;
        max-width: 110;
        margin-top: 1;
        padding: 1 2;
        border: tall $accent;
        background: $surface;
    }

    ReverseSearchScreen Input {
        width: 100%;
        height: 3;
        margin-bottom: 1;
    }

    ReverseSearchScreen #rev-search-label {
        width: 100%;
        height: auto;
        margin-bottom: 1;
        color: $text-muted;
    }

    ReverseSearchScreen #rev-search-current {
        width: 100%;
        height: auto;
        margin-bottom: 1;
        color: $accent;
        text-style: bold;
    }

    ReverseSearchScreen #rev-search-empty {
        width: 100%;
        height: auto;
        color: $text-muted;
    }

    ReverseSearchScreen #rev-search-hint {
        width: 100%;
        height: auto;
        margin-top: 1;
        color: $text-muted;
    }
    """

    def __init__(self, history: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Most-recent-first matches the user's mental model
        self._history: list[str] = list(reversed(history or []))
        self._query: str = ""
        self._matches: list[str] = []
        self._match_index: int = 0  # index into self._matches
        self._cancelled: bool = False

    # ------------------------------------------------------------------
    # composition
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("(reverse-i-search):", id="rev-search-label")
            yield Input(placeholder="type to filter…", id="rev-search-input")
            yield Static("", id="rev-search-current")
            yield Static("", id="rev-search-empty")
            yield Static(HINT_TEXT, id="rev-search-hint")

    def on_mount(self) -> None:
        try:
            self.query_one("#rev-search-input", Input).focus()
        except Exception:
            pass
        self._update_view()

    # ------------------------------------------------------------------
    # event handlers
    # ------------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "rev-search-input":
            return
        self._query = event.value or ""
        self._match_index = 0
        self._refresh_matches()
        self._update_view()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _refresh_matches(self) -> None:
        """Rebuild self._matches from self._history and self._query."""
        if not self._query:
            self._matches = []
            return
        q = self._query.lower()
        self._matches = [cmd for cmd in self._history if q in cmd.lower()]

    def _update_view(self) -> None:
        # Update label
        try:
            label = self.query_one("#rev-search-label", Static)
            label.update(f"(reverse-i-search) '{self._query}':")
        except Exception:
            pass
        # Update current match display
        current = self._matches[self._match_index] if self._matches else ""
        try:
            current_widget = self.query_one("#rev-search-current", Static)
            if current:
                # Truncate long matches for display
                shown = current if len(current) <= 80 else current[:77] + "…"
                current_widget.update(f"  → {shown}")
            else:
                current_widget.update("")
        except Exception:
            pass
        # Update empty placeholder
        try:
            empty_widget = self.query_one("#rev-search-empty", Static)
            if not self._query:
                empty_widget.update("[dim]start typing to search history…[/dim]")
            elif not self._matches:
                empty_widget.update("[dim](no matching history)[/dim]")
            else:
                pos = self._match_index + 1
                total = len(self._matches)
                empty_widget.update(f"[dim]match {pos}/{total}[/dim]")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------

    def action_find_prev(self) -> None:
        """Find next earlier match (older entry in history)."""
        if not self._matches:
            return
        self._match_index = (self._match_index + 1) % len(self._matches)
        self._update_view()

    def action_find_next(self) -> None:
        """Find next later match (newer entry in history)."""
        if not self._matches:
            return
        self._match_index = (self._match_index - 1) % len(self._matches)
        self._update_view()

    def action_accept(self) -> None:
        """Accept the current match and dismiss."""
        if self._matches:
            self.dismiss(self._matches[self._match_index])
        else:
            # No match — accept the raw query as a new command
            self.dismiss(self._query)

    def action_cancel(self) -> None:
        """Cancel and return None."""
        self.dismiss(None)
