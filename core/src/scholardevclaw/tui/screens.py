"""Minimal modal screens for the command-first TUI."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from scholardevclaw.auth.types import AuthProvider
from scholardevclaw.llm.client import DEFAULT_MODELS


DEFAULT_OPENROUTER_MODEL = DEFAULT_MODELS[AuthProvider.OPENROUTER]


WELCOME_TEXT = (
    "ScholarDevClaw\n\n"
    "Keyboard-first research-to-code shell.\n"
    "Type commands like:\n"
    "  setup\n"
    "  analyze ./repo\n"
    "  chat explain this repository\n"
    "  set mode search\n"
    "  :edit\n\n"
    "Press Enter or Esc to continue."
)


HELP_TEXT = (
    "Keys\n"
    "Tab autocomplete\n"
    "Up/Down history\n"
    "Ctrl+C cancel task or exit\n"
    "Ctrl+K clear output\n"
    "Enter execute\n"
    "Esc dismiss suggestions\n\n"
    "Setup\n"
    "setup\n"
    "set provider openrouter\n"
    "set provider ollama\n"
    f"set model {DEFAULT_OPENROUTER_MODEL}\n\n"
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


class ProviderSetupScreen(ModalScreen[dict[str, str] | None]):
    """Keyboard-first provider onboarding for the TUI shell."""

    BINDINGS = [
        ("ctrl+s", "submit_setup", "Save"),
        ("escape", "dismiss_skip", "Skip"),
    ]

    DEFAULT_CSS = """
    ProviderSetupScreen {
        align: center middle;
        background: $background 80%;
    }

    ProviderSetupScreen > Vertical {
        width: 60;
        height: 14;
        padding: 1 2;
        border: round $accent;
        background: $surface;
        overflow-y: auto;
    }

    ProviderSetupScreen Input {
        width: 100%;
        height: 3;
        border: solid $border;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    ProviderSetupScreen Input:focus {
        border: solid $accent;
    }

    #setup-hint {
        width: 100%;
        height: auto;
        color: $text-muted;
        margin: 0 0 1 0;
    }

    #setup-error {
        width: 100%;
        height: auto;
        color: $error;
    }
    """

    def __init__(
        self,
        *,
        provider: str = "openrouter",
        model: str = "",
        has_saved_key: bool = False,
    ) -> None:
        super().__init__()
        self._provider = provider or "openrouter"
        self._model = model
        self._has_saved_key = has_saved_key

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("LLM Setup")
            yield Static("", id="setup-hint")
            yield Input(
                value=self._provider,
                placeholder="Provider: openrouter or ollama",
                id="setup-provider",
            )
            yield Input(value=self._model, placeholder="Model ID", id="setup-model")
            yield Input(password=True, placeholder="OpenRouter API key", id="setup-key")
            yield Static("", id="setup-error")

    def on_mount(self) -> None:
        self._refresh_hint()
        self.query_one("#setup-provider", Input).focus()

    def action_dismiss_skip(self) -> None:
        self.dismiss(None)

    def action_submit_setup(self) -> None:
        provider = self.query_one("#setup-provider", Input).value.strip().lower()
        model = self.query_one("#setup-model", Input).value.strip()
        api_key = self.query_one("#setup-key", Input).value.strip()
        error = self.query_one("#setup-error", Static)

        if provider not in {"openrouter", "ollama"}:
            error.update("Error: provider must be openrouter or ollama")
            return
        if not model:
            error.update("Error: model is required")
            return
        if provider == "openrouter" and not api_key and not self._has_saved_key:
            error.update("Error: OpenRouter requires an API key")
            return

        self.dismiss(
            {
                "provider": provider,
                "model": model,
                "api_key": api_key,
            }
        )

    @on(Input.Changed, "#setup-provider")
    def on_provider_changed(self, event: Input.Changed) -> None:
        self._provider = event.value.strip().lower()
        self._refresh_hint()

    @on(Input.Submitted, "#setup-provider")
    def on_provider_submitted(self) -> None:
        self.query_one("#setup-model", Input).focus()

    @on(Input.Submitted, "#setup-model")
    def on_model_submitted(self) -> None:
        provider = self.query_one("#setup-provider", Input).value.strip().lower()
        if provider == "ollama":
            self.action_submit_setup()
            return
        self.query_one("#setup-key", Input).focus()

    @on(Input.Submitted, "#setup-key")
    def on_key_submitted(self) -> None:
        self.action_submit_setup()

    def _refresh_hint(self) -> None:
        hint = self.query_one("#setup-hint", Static)
        if self._provider == "ollama":
            hint.update(
                "Provider -> Ollama\n"
                "Model -> any local Ollama tag, for example `llama3.1`\n"
                "Key -> not required\n"
                "Save -> Ctrl+S or Enter"
            )
            return

        reuse = (
            "leave key blank to reuse saved key"
            if self._has_saved_key
            else "paste your OpenRouter key"
        )
        hint.update(
            "Provider -> OpenRouter\n"
            f"Model -> full OpenRouter model id, for example `{DEFAULT_OPENROUTER_MODEL}`\n"
            f"Key -> {reuse}\n"
            "Save -> Ctrl+S or Enter"
        )


class CommandPalette(ModalScreen[str | None]):
    """Thin command chooser for keyboard fallback."""

    BINDINGS = [
        ("escape", "dismiss_none", "Dismiss"),
        ("enter", "run_selected", "Run"),
        ("down", "select_next", "Next"),
        ("up", "select_prev", "Prev"),
    ]

    PALETTE_COMMANDS = [
        "setup",
        "analyze ./repo",
        "suggest ./repo",
        "chat hello",
        "search layer normalization",
        "map ./repo rmsnorm",
        "generate ./repo rmsnorm",
        "validate ./repo",
        "set provider openrouter",
        "set provider ollama",
        f"set model {DEFAULT_OPENROUTER_MODEL}",
        "set mode analyze",
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
