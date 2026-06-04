"""Log search modal: live filter the LogView by substring.

Triggered from the main app with ``/``. As the user types, the underlying
``LogView``'s search filter is updated in real time. Pressing ``Enter``
or ``Escape`` dismisses the modal; ``Ctrl+U`` clears the filter.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from .widgets import LogView

HINT_TEXT = (
    "[dim]Type to filter the log view in real time. Enter applies, Esc closes, Ctrl+U clears.[/dim]"
)


class LogSearchScreen(ModalScreen[None]):
    """Live-filter the main LogView by substring."""

    BINDINGS = [
        ("escape", "dismiss_modal", "Close"),
        ("enter", "dismiss_modal", "Apply"),
        ("ctrl+u", "clear_filter", "Clear"),
    ]

    DEFAULT_CSS = """
    LogSearchScreen {
        align: center top;
        background: $background 60%;
        layer: overlay;
    }

    LogSearchScreen > Vertical {
        width: 70%;
        height: auto;
        max-width: 96;
        margin-top: 2;
        padding: 1 2;
        border: tall $accent;
        background: $surface;
    }

    LogSearchScreen Input {
        width: 100%;
        height: 3;
        margin-bottom: 1;
    }

    LogSearchScreen #log-search-hint {
        width: 100%;
        height: auto;
        color: $text-muted;
    }
    """

    def __init__(self, initial: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._initial = initial or ""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Filter log view", classes="log-search-title")
            yield Input(
                value=self._initial,
                placeholder="search substring…",
                id="log-search-input",
            )
            yield Static(HINT_TEXT, id="log-search-hint")

    def on_mount(self) -> None:
        # Focus the input automatically
        try:
            self.query_one("#log-search-input", Input).focus()
        except Exception:
            pass

    def on_input_changed(self, event: Input.Changed) -> None:
        # Live-update the LogView's search filter
        if event.input.id != "log-search-input":
            return
        try:
            log_view = self.app.query_one("#main-output", LogView)
        except Exception:
            return
        try:
            log_view.set_search_filter(event.value or "")
        except Exception:
            pass

    def action_clear_filter(self) -> None:
        """Clear the input and the LogView's search filter."""
        try:
            input_widget = self.query_one("#log-search-input", Input)
            input_widget.value = ""
        except Exception:
            pass
        try:
            log_view = self.app.query_one("#main-output", LogView)
            log_view.set_search_filter("")
        except Exception:
            pass

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)
