"""Minimal modal screens for the command-first TUI."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static


WELCOME_TEXT = (
    "ScholarDevClaw\n\n"
    "Keyboard-first research-to-code shell.\n"
    "Type commands like:\n"
    "  analyze ./repo\n"
    "  set mode search\n"
    "  :edit\n\n"
    "Press Enter or Esc to continue."
)


HELP_TEXT = (
    "Keys\n"
    "Tab autocomplete\n"
    "Up/Down history\n"
    "Ctrl+C cancel task\n"
    "Ctrl+K clear output\n"
    "Enter execute\n"
    "Esc dismiss suggestions\n\n"
    "Modes\n"
    ":analyze\n"
    ":search\n"
    ":edit"
)


class WelcomeScreen(ModalScreen[None]):
    BINDINGS = [("escape", "dismiss", "Dismiss"), ("enter", "dismiss", "Dismiss")]

    DEFAULT_CSS = """
    WelcomeScreen {
        align: left top;
        background: $background 80%;
    }

    WelcomeScreen > Vertical {
        width: 100%;
        height: auto;
        padding: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(WELCOME_TEXT)


class HelpOverlay(ModalScreen[None]):
    BINDINGS = [("escape", "dismiss", "Dismiss"), ("enter", "dismiss", "Dismiss")]

    DEFAULT_CSS = """
    HelpOverlay {
        align: left top;
        background: $background 80%;
    }

    HelpOverlay > Vertical {
        width: 100%;
        height: auto;
        padding: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(HELP_TEXT)


class CommandPalette(ModalScreen[str | None]):
    """Thin command chooser for keyboard fallback."""

    BINDINGS = [
        ("escape", "dismiss_none", "Dismiss"),
        ("enter", "run_selected", "Run"),
        ("down", "select_next", "Next"),
        ("up", "select_prev", "Prev"),
    ]

    PALETTE_COMMANDS = [
        "analyze ./repo",
        "suggest ./repo",
        "search layer normalization",
        "map ./repo rmsnorm",
        "generate ./repo rmsnorm",
        "validate ./repo",
        "set mode analyze",
        "set model auto",
        "set dir ./repo",
        ":analyze",
        ":search",
        ":edit",
    ]

    DEFAULT_CSS = """
    CommandPalette {
        align: left top;
        background: $background 80%;
    }

    CommandPalette > Vertical {
        width: 100%;
        height: auto;
        padding: 1 2;
    }

    CommandPalette Input {
        width: 100%;
        height: 1;
        border: none;
        padding: 0;
    }

    CommandPalette .palette-line {
        width: 100%;
        height: 1;
        color: $text-muted;
    }

    CommandPalette .palette-line.-selected {
        color: $accent;
        text-style: bold;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._selected_index = 0
        self._filtered_commands = list(self.PALETTE_COMMANDS)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Commands")
            yield Input(placeholder="Filter commands", id="palette-input")
            for command in self.PALETTE_COMMANDS[:3]:
                yield Static(command, classes="palette-line")

    def on_mount(self) -> None:
        self.query_one("#palette-input", Input).focus()
        self._refresh()

    def action_dismiss_none(self) -> None:
        self.dismiss(None)

    def action_select_next(self) -> None:
        if self._filtered_commands:
            self._selected_index = (self._selected_index + 1) % len(self._filtered_commands)
            self._refresh()

    def action_select_prev(self) -> None:
        if self._filtered_commands:
            self._selected_index = (self._selected_index - 1) % len(self._filtered_commands)
            self._refresh()

    def action_run_selected(self) -> None:
        if not self._filtered_commands:
            self.dismiss(None)
            return
        self.dismiss(self._filtered_commands[self._selected_index])

    @on(Input.Changed, "#palette-input")
    def on_input_changed(self, event: Input.Changed) -> None:
        query = event.value.strip().lower()
        if not query:
            self._filtered_commands = list(self.PALETTE_COMMANDS)
        else:
            self._filtered_commands = [cmd for cmd in self.PALETTE_COMMANDS if query in cmd.lower()]
        self._selected_index = 0
        self._refresh()

    def _refresh(self) -> None:
        lines = list(self.query(".palette-line"))
        for line in lines:
            line.remove()
        if not self._filtered_commands:
            self.mount(Static("No matches", classes="palette-line"))
            return
        for index, command in enumerate(self._filtered_commands[:3]):
            classes = "palette-line"
            if index == self._selected_index:
                classes += " -selected"
            self.mount(Static(command, classes=classes))
