"""TUI screens: Welcome, Help Overlay, Command Palette."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Markdown

WELCOME_MD = """
# ScholarDevClaw

**Research-to-Code Control Surface**

ScholarDevClaw analyzes your repository, discovers research improvements,
maps them onto your codebase, generates validated patches, and reports outcomes.

---

## Start in 10 seconds

| Key | Action |
|-----|--------|
| `ctrl+r` | run selected workflow |
| `ctrl+k` | command palette |
| `ctrl+h` | keyboard shortcuts |
| `ctrl+p` / `ctrl+g` | jump prompt / history |
| `ctrl+o` | toggle config bar |
| `ctrl+shift+r` | rerun latest |
| `Esc` | stop agent / close overlay |

## Workflows

- **Analyze** — Scan repository structure, languages, frameworks
- **Suggest** — Discover research papers relevant to your codebase
- **Search** — Find papers on arXiv or the web
- **Map** — Match research patterns to code locations
- **Generate** — Create code patches using concrete-syntax-tree transforms
- **Validate** — Run tests and benchmarks to verify patches
- **Integrate** — End-to-end: analyze → suggest → map → generate → validate

You can also type natural language in the prompt bar:
`analyze ./my-project` or `apply rmsnorm to /path/to/repo`

---

Press **Enter** or **Esc** to continue.
"""


HELP_MD = """
# Keyboard Shortcuts

## Global
| Key | Action |
|-----|--------|
| `ctrl+c` | quit |
| `ctrl+r` | run selected workflow |
| `ctrl+shift+r` | rerun latest workflow |
| `ctrl+k` | open command palette |
| `ctrl+h` | show this help |
| `ctrl+o` | toggle config panel |
| `ctrl+l` | clear logs |
| `ctrl+n` | new session |
| `ctrl+e` | export logs |
| `tab` / `shift+tab` | cycle focus |
| `ctrl+p` | focus prompt |
| `ctrl+g` | focus run history |
| `esc` | stop agent / close overlay |
| `esc` x2 | stop running agent |

## History Pane
| Key | Action |
|-----|--------|
| `up` / `k` | previous run |
| `down` / `j` | next run |
| `enter` / `space` | rerun selected |

## Prompt Bar
| Key | Action |
|-----|--------|
| `enter` | submit prompt |
| `up` | previous command in history |
| `down` | next command in history |

## Slash Commands
| Cmd | Action |
|-----|--------|
| `/commands` | open command palette |
| `/new` | start new session |
| `/export` | export current chat log |
| `/clear` | clear logs |
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
        background: #11111b 82%;
    }

    WelcomeScreen > Container {
        width: 82;
        max-width: 94;
        height: auto;
        max-height: 85%;
        background: #1e1e2e;
        border: round #45475a;
        border-top: thick #74c7ec;
        padding: 2 4;
    }

    WelcomeScreen Markdown {
        width: 100%;
        height: auto;
    }

    WelcomeScreen .dismiss-hint {
        width: 100%;
        text-align: center;
        color: #a6adc8;
        margin-top: 2;
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
        background: #11111b 82%;
    }

    HelpOverlay > Container {
        width: 74;
        max-width: 90;
        height: auto;
        max-height: 80%;
        background: #1e1e2e;
        border: round #45475a;
        border-top: thick #74c7ec;
        padding: 2 4;
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
        ("down", "select_next", "next"),
        ("up", "select_prev", "prev"),
        ("enter", "run_selected", "run"),
    ]

    PALETTE_COMMANDS = [
        ("analyze", "Scan repository structure", "analyze"),
        ("suggest", "Discover research improvements", "suggest"),
        ("search", "Search arXiv / web for papers", "search"),
        ("specs", "List available paper specs", "specs"),
        ("map", "Map research patterns to code", "map"),
        ("generate", "Generate code patches", "generate"),
        ("validate", "Run tests & benchmarks", "validate"),
        ("integrate", "Full end-to-end pipeline", "integrate"),
        ("clear", "Clear logs and results", "clear"),
        ("new-session", "Start new clean session", "new_session"),
        ("export-log", "Export current logs to file", "export_log"),
        ("quit", "Exit ScholarDevClaw", "quit"),
    ]

    CSS = """
    CommandPalette {
        align: center top;
        background: #11111b 76%;
    }

    CommandPalette > Vertical {
        width: 66;
        max-width: 82;
        height: auto;
        max-height: 68%;
        background: #1e1e2e;
        border: round #45475a;
        border-top: thick #74c7ec;
        padding: 1 1 2 1;
        margin-top: 5;
    }

    CommandPalette .palette-title {
        color: #cdd6f4;
        text-style: bold;
        padding: 0 1;
        margin-bottom: 0;
    }

    CommandPalette .palette-subtitle {
        color: #7f849c;
        padding: 0 1;
        margin-bottom: 1;
    }

    CommandPalette Input {
        width: 100%;
        margin-bottom: 1;
        background: #181825;
        border: solid #45475a;
        height: 3;
    }

    CommandPalette .command-list {
        width: 100%;
        height: auto;
        max-height: 30;
        overflow-y: auto;
    }

    CommandPalette .command-item {
        width: 100%;
        height: 3;
        margin-bottom: 1;
        background: #181825;
        border: solid #313244;
        border-left: thick #313244;
        color: #cdd6f4;
        text-align: left;
    }

    CommandPalette .command-item:hover {
        background: #313244 80%;
        border-left: thick #89b4fa;
    }

    CommandPalette .command-item.selected {
        background: #313244;
        border-left: thick #89b4fa;
        text-style: bold;
    }

    CommandPalette .palette-empty {
        color: #6c7086;
        padding: 1 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._selected_index = 0
        self._filtered_commands = list(self.PALETTE_COMMANDS)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Command palette · type to filter · Enter to run", classes="palette-title")
            yield Label(
                "Jump to workflows and operator actions without leaving the keyboard.",
                classes="palette-subtitle",
            )
            yield Input(placeholder="Type a command…", id="palette-input")
            with Vertical(classes="command-list", id="command-list"):
                for name, desc, _ in self.PALETTE_COMMANDS:
                    yield Button(f"{name:<11}  {desc}", id=f"cmd-{name}", classes="command-item")

    def on_mount(self) -> None:
        self.query_one("#palette-input", Input).focus()
        self._refresh_selection()

    def action_dismiss_none(self) -> None:
        self.dismiss(None)

    @on(Input.Changed, "#palette-input")
    def on_input_changed(self, event: Input.Changed) -> None:
        query = event.value.strip().lower()
        container = self.query_one("#command-list", Vertical)
        container.remove_children()

        if not query:
            self._filtered_commands = list(self.PALETTE_COMMANDS)
        else:
            self._filtered_commands = [
                (name, desc, action)
                for name, desc, action in self.PALETTE_COMMANDS
                if query in name or query in desc.lower()
            ]

        if not self._filtered_commands:
            container.mount(Label("No matching commands", classes="palette-empty"))
        else:
            for name, desc, _ in self._filtered_commands:
                container.mount(
                    Button(f"{name:<11}  {desc}", id=f"cmd-{name}", classes="command-item")
                )

        self._selected_index = 0
        self._refresh_selection()

    @on(Input.Submitted, "#palette-input")
    def on_submit(self) -> None:
        self.action_run_selected()

    def _refresh_selection(self) -> None:
        buttons = list(self.query(Button))
        for idx, button in enumerate(buttons):
            if idx == self._selected_index:
                button.add_class("selected")
            else:
                button.remove_class("selected")

    def action_select_next(self) -> None:
        if not self._filtered_commands:
            return
        self._selected_index = (self._selected_index + 1) % len(self._filtered_commands)
        self._refresh_selection()

    def action_select_prev(self) -> None:
        if not self._filtered_commands:
            return
        self._selected_index = (self._selected_index - 1) % len(self._filtered_commands)
        self._refresh_selection()

    def action_run_selected(self) -> None:
        if not self._filtered_commands:
            self.dismiss(None)
            return
        idx = max(0, min(self._selected_index, len(self._filtered_commands) - 1))
        self.dismiss(self._filtered_commands[idx][2])

    @on(Button.Pressed)
    def on_command_click(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id.startswith("cmd-"):
            name = btn_id[4:]
            for cmd_name, _, action in self.PALETTE_COMMANDS:
                if cmd_name == name:
                    self.dismiss(action)
                    return
        self.dismiss(None)
