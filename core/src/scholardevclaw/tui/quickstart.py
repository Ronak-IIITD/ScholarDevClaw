"""Quickstart dashboard for ScholarDevClaw TUI.

A beautiful, keyboard-navigable first-run experience that showcases
the product with:
- ASCII art banner
- System status panel (provider/model/dir/version)
- 6 clickable quick-action tiles (analyze, suggest, integrate, etc.)
- Recent runs (last 5) with status icons
- Helpful tips and shortcuts
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SCHOLAR_BANNER = r"""
   _____ _    _  ___   ___   _____ _      _____          _      _
  / ____| |  | |/ _ \ / _ \ / ____| |    |_   _|__   ___ | | ___| |_
 | (___ | |__| | | | | | | | |    | |__    | |/ _ \ / _ \| |/ _ \ __|
  \___ \|  __  | | | | | | | |    |  __|   | | (_) | (_) | |  __/ |_
  ____) | |  | | |_| | |_| | |____| |      | |\___/ \___/|_|\___|\__|
 |_____/|_|  |_|\___/ \___/ \_____|_|      |_|

   R E S E A R C H  →  C O D E
"""

SCHOLAR_BANNER_COMPACT = r"""
 ╔══════════════════════════════════════╗
 ║  ScholarDevClaw                      ║
 ║  Research → Code                     ║
 ╚══════════════════════════════════════╝
"""


@dataclass(frozen=True)
class QuickAction:
    """A quick-action tile in the dashboard."""

    key: str
    title: str
    description: str
    icon: str
    command: str
    shortcut: str = ""


QUICK_ACTIONS: tuple[QuickAction, ...] = (
    QuickAction(
        key="analyze",
        title="Analyze Repo",
        description="Detect languages, frameworks, entry points",
        icon="◉",
        command="analyze .",
        shortcut="A",
    ),
    QuickAction(
        key="suggest",
        title="Suggest Improvements",
        description="AI-powered research-to-code recommendations",
        icon="◆",
        command="suggest .",
        shortcut="S",
    ),
    QuickAction(
        key="integrate",
        title="Integrate Patch",
        description="Apply a paper spec to your repo end-to-end",
        icon="▶",
        command="integrate . rmsnorm",
        shortcut="I",
    ),
    QuickAction(
        key="search",
        title="Search Papers",
        description="Find arXiv papers and implementations",
        icon="⌕",
        command="search transformer",
        shortcut="/",
    ),
    QuickAction(
        key="map",
        title="Map Code",
        description="Find code locations for a paper's algorithm",
        icon="⇄",
        command="map . rmsnorm",
        shortcut="M",
    ),
    QuickAction(
        key="doctor",
        title="Health Check",
        description="Verify setup, deps, and API keys",
        icon="✚",
        command="doctor",
        shortcut="?",
    ),
)


def _status_icon(ok: bool | None) -> str:
    if ok is True:
        return "●"
    if ok is False:
        return "○"
    return "·"


def _gather_system_status() -> dict[str, str]:
    """Collect system information for the status panel."""
    info: dict[str, str] = {
        "Version": _get_version(),
        "Python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "Platform": platform.system(),
        "CWD": str(Path.cwd()),
    }
    # API key env hints
    for env_name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        if os.environ.get(env_name):
            masked = os.environ[env_name][:6] + "…" + os.environ[env_name][-4:]
            info["API Key"] = f"{env_name} ({masked})"
            break
    else:
        info["API Key"] = "(not set)"
    return info


def _get_version() -> str:
    try:
        from scholardevclaw import __version__

        return str(__version__)
    except (ImportError, AttributeError):
        return "2.0"


def _format_recent_runs(runs: list[dict[str, Any]] | None) -> str:
    """Format the last few runs for display."""
    if not runs:
        return "  (no runs yet — try one of the actions above)"
    lines: list[str] = []
    for run in runs[-5:][::-1]:
        status = str(run.get("status", "unknown"))
        icon = {
            "completed": "✓",
            "running": "⟳",
            "failed": "✗",
            "cancelled": "⊘",
            "queued": "⋯",
        }.get(status, "?")
        action = run.get("action", "?")
        spec = run.get("spec", "")
        spec_str = f" → {spec}" if spec else ""
        lines.append(f"  {icon} {action}{spec_str}  [{status}]")
    return "\n".join(lines)


def _render_tile(action: QuickAction, *, focused: bool = False) -> str:
    """Render a single quick-action tile as ASCII box."""
    if focused:
        border = "┃"
        prefix = "▶"
    else:
        border = "│"
        prefix = " "
    icon_line = f"{border}  {prefix} {action.icon}  {action.title:<24}{border}"
    desc_line = f"{border}     {action.description:<28}{border}"
    cmd_line = f"{border}     $ {action.command:<28}{border}"
    shortcut_line = f"{border}     [{action.shortcut}]{'':<26}{border}"
    top = f"┏{'━' * 38}┓"
    mid = f"┣{'━' * 38}┫"
    bottom = f"┗{'━' * 38}┛"
    return "\n".join([top, icon_line, desc_line, cmd_line, shortcut_line, mid, bottom])


# -----------------------------------------------------------------------------
# Widgets
# -----------------------------------------------------------------------------


class QuickActionTile(Button):
    """A clickable quick-action tile."""

    def __init__(self, action: QuickAction, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # `Button` has its own `action: str | None` attribute, so we
        # use a distinct name to avoid clashing with the parent class.
        self.quick_action = action
        self.can_focus = True
        # Render the tile text directly
        self.label = f"{action.icon}  {action.title}\n  {action.description}\n  $ {action.command}\n  [{action.shortcut}]"

    DEFAULT_CSS = """
    QuickActionTile {
        width: 1fr;
        height: 5;
        min-width: 32;
        margin: 0 1;
        padding: 0 1;
        border: solid $accent;
        background: $surface;
        color: $text;
        content-align: left top;
    }
    QuickActionTile:hover {
        background: $accent;
        color: $text-bright;
    }
    QuickActionTile:focus {
        border: double $success;
        background: $accent;
        text-style: bold;
    }
    """


class QuickstartDashboard(ModalScreen[str | None]):
    """Beautiful first-run quickstart dashboard.

    Dismisses with a string command (the user picked an action),
    or None if the user dismissed the dashboard.
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "dismiss_dashboard", "Dismiss"),
        Binding("q", "dismiss_dashboard", "Dismiss"),
        Binding("enter", "activate_focused", "Run"),
    ]

    DEFAULT_CSS = """
    QuickstartDashboard {
        align: center middle;
        background: $background 92%;
    }
    QuickstartDashboard > Vertical {
        width: 95%;
        max-width: 120;
        height: auto;
        max-height: 95%;
        padding: 1 2;
        background: $surface;
        border: thick $accent;
    }
    .dashboard-banner {
        height: auto;
        color: $accent;
        text-style: bold;
        padding: 1 0;
    }
    .dashboard-section-title {
        height: 1;
        color: $text-muted;
        text-style: bold;
        padding: 1 0 0 0;
    }
    .dashboard-status {
        height: auto;
        padding: 0 1;
        color: $text;
    }
    .dashboard-actions-row {
        height: auto;
        padding: 0 0;
    }
    .dashboard-tips {
        height: auto;
        padding: 0 1;
        color: $text-muted;
    }
    .dashboard-runs {
        height: auto;
        padding: 0 1;
        color: $text;
    }
    """

    def __init__(
        self,
        *,
        recent_runs: list[dict[str, Any]] | None = None,
        system_status: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.recent_runs = recent_runs or []
        self.system_status = system_status or _gather_system_status()
        self._focus_index = 0

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self._render_banner(), classes="dashboard-banner")

            yield Static("SYSTEM STATUS", classes="dashboard-section-title")
            yield Static(self._render_status(), classes="dashboard-status")

            yield Static("QUICK ACTIONS", classes="dashboard-section-title")
            with Horizontal(classes="dashboard-actions-row"):
                # Render the first row of actions
                for action in QUICK_ACTIONS[:3]:
                    yield QuickActionTile(action, id=f"tile-{action.key}")
            with Horizontal(classes="dashboard-actions-row"):
                for action in QUICK_ACTIONS[3:]:
                    yield QuickActionTile(action, id=f"tile-{action.key}")

            yield Static("RECENT RUNS", classes="dashboard-section-title")
            yield Static(_format_recent_runs(self.recent_runs), classes="dashboard-runs")

            yield Static("TIPS", classes="dashboard-section-title")
            yield Static(self._render_tips(), classes="dashboard-tips")

    def _render_banner(self) -> str:
        """Render the ASCII art banner."""
        # Pick a compact version if the terminal is narrow
        if os.get_terminal_size().columns < 80:
            return SCHOLAR_BANNER_COMPACT
        return SCHOLAR_BANNER

    def _render_status(self) -> str:
        """Render the system status panel."""
        lines: list[str] = []
        for key, value in self.system_status.items():
            # Truncate long values
            display = value if len(value) <= 60 else value[:57] + "…"
            lines.append(f"  {key:<12} {display}")
        return "\n".join(lines)

    def _render_tips(self) -> str:
        """Render helpful tips."""
        return (
            "  • Press Tab to cycle through actions  • Enter to run  • Esc to dismiss\n"
            "  • In the main TUI: Ctrl+H for help  • Ctrl+P for paper workflow\n"
            "  • /ask <question>  • /run <action>  • runs (list)  • inspect"
        )

    def on_mount(self) -> None:
        """Focus the first action tile on mount."""
        first_tile = self.query_one("#tile-analyze", QuickActionTile)
        first_tile.focus()

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle a tile click."""
        tile = event.button
        if isinstance(tile, QuickActionTile):
            self._dispatch_action(tile.quick_action)

    def action_activate_focused(self) -> None:
        """Activate whichever tile currently has focus."""
        focused = self.focused
        if isinstance(focused, QuickActionTile):
            self._dispatch_action(focused.quick_action)

    def _dispatch_action(self, action: QuickAction) -> None:
        """Dismiss the dashboard and emit the chosen command."""
        self.dismiss(action.command)

    def action_dismiss_dashboard(self) -> None:
        """Dismiss without taking an action."""
        self.dismiss(None)
