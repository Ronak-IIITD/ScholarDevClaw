"""TUI screens: Welcome, Help Overlay, Command Palette."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Markdown

WELCOME_MD = """
# ScholarDevClaw

**Research-to-Code AI Agent**

ScholarDevClaw analyzes your repository, discovers research improvements,
maps them onto your codebase, generates validated patches, and reports outcomes.

---

## Quick Start

| Key | Action |
|-----|--------|
| `Ctrl+R` | Run selected workflow |
| `Ctrl+K` | Command palette |
| `Ctrl+H` | Keyboard shortcuts |
| `Ctrl+L` | Clear logs |
| `Ctrl+A` | Quick Analyze |
| `Ctrl+S` | Quick Suggest |
| `Ctrl+I` | Quick Integrate |
| `Esc` | Stop agent / close overlay |

## Workflows

- **Analyze** — Scan repository structure, languages, frameworks
- **Suggest** — Discover research papers relevant to your codebase
- **Search** — Find papers on arXiv or the web
- **Map** — Match research patterns to code locations
- **Generate** — Create code patches using concrete-syntax-tree transforms
- **Validate** — Run tests and benchmarks to verify patches
- **Integrate** — End-to-end: analyze → suggest → map → generate → validate

---

Press **Enter** or **Esc** to dismiss.
"""


HELP_MD = """
# Keyboard Shortcuts

## Global
| Key | Action |
|-----|--------|
| `Ctrl+C` | Quit |
| `Ctrl+R` | Run selected workflow |
| `Ctrl+K` | Open command palette |
| `Ctrl+H` | Show this help |
| `Esc` | Stop agent / close overlay |
| `Esc` x2 | Stop running agent |

## Quick Actions
| Key | Action |
|-----|--------|
| `Ctrl+A` | Quick Analyze |
| `Ctrl+S` | Quick Suggest |
| `Ctrl+I` | Quick Integrate |

## Logs
| Key | Action |
|-----|--------|
| `Ctrl+L` | Clear logs |

## Navigation
| Key | Action |
|-----|--------|
| `Tab` | Cycle focus forward |
| `Shift+Tab` | Cycle focus backward |
| `Up/Down` | Navigate sidebar items |
| `Enter` | Activate focused item |

## Prompt Bar
| Key | Action |
|-----|--------|
| `Enter` | Submit prompt |
| `Up` | Previous command in history |
| `Down` | Next command in history |
"""


class WelcomeScreen(ModalScreen[None]):
    """Welcome overlay shown on first launch."""

    BINDINGS = [
        ("escape", "dismiss", "Dismiss"),
        ("enter", "dismiss", "Dismiss"),
    ]

    CSS = """
    WelcomeScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }

    WelcomeScreen > Container {
        width: 70;
        max-width: 80;
        height: auto;
        max-height: 85%;
        background: $panel;
        border: thick $accent;
        padding: 2 3;
    }

    WelcomeScreen Markdown {
        width: 100%;
        height: auto;
    }

    WelcomeScreen .dismiss-hint {
        width: 100%;
        text-align: center;
        color: $text-muted;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Container():
            yield Markdown(WELCOME_MD)
            yield Label("[dim]Press Enter or Esc to start[/]", classes="dismiss-hint")


class HelpOverlay(ModalScreen[None]):
    """Keyboard shortcuts overlay."""

    BINDINGS = [
        ("escape", "dismiss", "Dismiss"),
        ("question_mark", "dismiss", "Dismiss"),
    ]

    CSS = """
    HelpOverlay {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }

    HelpOverlay > Container {
        width: 60;
        max-width: 75;
        height: auto;
        max-height: 80%;
        background: $panel;
        border: thick $accent;
        padding: 2 3;
    }

    HelpOverlay Markdown {
        width: 100%;
        height: auto;
    }
    """

    def compose(self) -> ComposeResult:
        with Container():
            yield Markdown(HELP_MD)


class CommandPalette(ModalScreen[str | None]):
    """Command palette — fuzzy search workflows and actions."""

    BINDINGS = [
        ("escape", "dismiss_none", "Close"),
    ]

    COMMANDS = [
        ("analyze", "Scan repository structure", "analyze"),
        ("suggest", "Discover research improvements", "suggest"),
        ("search", "Search arXiv / web for papers", "search"),
        ("specs", "List available paper specs", "specs"),
        ("map", "Map research patterns to code", "map"),
        ("generate", "Generate code patches", "generate"),
        ("validate", "Run tests & benchmarks", "validate"),
        ("integrate", "Full end-to-end pipeline", "integrate"),
        ("clear", "Clear logs and results", "clear"),
        ("quit", "Exit ScholarDevClaw", "quit"),
    ]

    CSS = """
    CommandPalette {
        align: center top;
        background: rgba(0, 0, 0, 0.5);
    }

    CommandPalette > Vertical {
        width: 55;
        max-width: 70;
        height: auto;
        max-height: 60%;
        background: $panel;
        border: thick $accent;
        padding: 1;
        margin-top: 8;
    }

    CommandPalette Input {
        width: 100%;
        margin-bottom: 1;
        background: $surface-dark;
        border: solid $border;
    }

    CommandPalette .command-list {
        width: 100%;
        height: auto;
        max-height: 30;
        overflow-y: auto;
    }

    CommandPalette .command-item {
        width: 100%;
        padding: 0 1;
        height: auto;
    }

    CommandPalette .command-item:hover {
        background: $accent 20%;
    }

    CommandPalette .command-item.selected {
        background: $accent 30%;
        text-style: bold;
    }

    CommandPalette .cmd-name {
        color: $accent;
        text-style: bold;
    }

    CommandPalette .cmd-desc {
        color: $text-muted;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._selected_index = 0
        self._filtered_commands = list(self.COMMANDS)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Input(placeholder="Type a command...", id="palette-input")
            with Vertical(classes="command-list", id="command-list"):
                for name, desc, _ in self.COMMANDS:
                    with Horizontal(classes="command-item", id=f"cmd-{name}"):
                        yield Label(f"  {name}", classes="cmd-name")
                        yield Label(f"  {desc}", classes="cmd-desc")

    def on_mount(self) -> None:
        self.query_one("#palette-input", Input).focus()

    def action_dismiss_none(self) -> None:
        self.dismiss(None)

    @on(Input.Changed, "#palette-input")
    def on_input_changed(self, event: Input.Changed) -> None:
        query = event.value.strip().lower()
        container = self.query_one("#command-list")
        container.remove_children()

        if not query:
            self._filtered_commands = list(self.COMMANDS)
        else:
            self._filtered_commands = [
                (name, desc, action)
                for name, desc, action in self.COMMANDS
                if query in name or query in desc.lower()
            ]

        for name, desc, _ in self._filtered_commands:
            with container:
                item = Horizontal(classes="command-item", id=f"cmd-{name}")
                item.mount(Label(f"  {name}", classes="cmd-name"))
                item.mount(Label(f"  {desc}", classes="cmd-desc"))

        self._selected_index = 0

    @on(Input.Submitted, "#palette-input")
    def on_submit(self) -> None:
        if self._filtered_commands:
            self.dismiss(self._filtered_commands[0][2])
        else:
            self.dismiss(None)

    @on(Button.Pressed)
    def on_command_click(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id.startswith("cmd-"):
            name = btn_id[4:]
            for cmd_name, _, action in self.COMMANDS:
                if cmd_name == name:
                    self.dismiss(action)
                    return
        self.dismiss(None)
