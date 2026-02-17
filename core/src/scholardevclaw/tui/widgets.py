from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Button, Footer, Header


class StatusBar(Static):
    """Status bar showing current operation and system info."""

    def __init__(self, status: str = "Ready", **kwargs):
        super().__init__(**kwargs)
        self._status = status

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Static(f"Status: {self._status}", id="status-text"),
            Static("", id="memory-usage"),
            Static("", id="time-elapsed"),
        )

    def update_status(self, status: str) -> None:
        """Update status text."""
        self._status = status
        self.query_one("#status-text", Static).update(f"Status: {status}")

    def update_memory(self, memory_mb: float) -> None:
        """Update memory usage display."""
        self.query_one("#memory-usage", Static).update(f"Memory: {memory_mb:.1f}MB")


class CommandPalette(Vertical):
    """Command palette for quick actions."""

    BINDINGS = [
        ("ctrl+k", "focus_command_input", "Command"),
        ("ctrl+a", "start_analyze", "Analyze"),
        ("ctrl+g", "start_generate", "Generate"),
        ("ctrl+v", "start_validate", "Validate"),
        ("ctrl+p", "toggle_plugins", "Plugins"),
    ]

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Static("Quick Actions:", id="palette-header"),
            Button("Analyze", id="btn-analyze", variant="primary"),
            Button("Generate", id="btn-generate", variant="success"),
            Button("Validate", id="btn-validate", variant="warning"),
            Button("Planner", id="btn-planner", variant="default"),
        )


class KeyboardShortcuts(Footer):
    """Enhanced footer with keyboard shortcuts."""

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Static(
                "Ctrl+K: Command | Ctrl+A: Analyze | Ctrl+G: Generate | Ctrl+V: Validate | Ctrl+P: Planner | Q: Quit"
            ),
        )


class ProgressIndicator(Static):
    """Progress indicator for long-running operations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._progress = 0
        self._total = 100

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Static("", id="progress-bar"),
            Static("0%", id="progress-text"),
        )

    def set_progress(self, current: int, total: int) -> None:
        """Set progress values."""
        self._current = current
        self._total = total
        if total > 0:
            percent = int(current / total * 100)
            bar = "█" * (percent // 5) + "░" * (20 - percent // 5)
            self.query_one("#progress-bar", Static).update(f"[{bar}]")
            self.query_one("#progress-text", Static).update(f"{percent}%")


class TabbedContent(Vertical):
    """Tabbed content container for multiple views."""

    def __init__(self, tabs: list[str], **kwargs):
        super().__init__(**kwargs)
        self._tabs = tabs
        self._active_tab = 0

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Static(" | ".join(self._tabs), id="tab-bar"),
        )


def create_wizard_layout() -> dict:
    """Create enhanced wizard layout configuration."""
    return {
        "hero": {
            "dock": "top",
            "height": 1,
        },
        "main": {
            "layout": "horizontal",
            "fraction": 1,
        },
        "wizard": {
            "width": "45%",
            "border": "tall #1d4ed8",
        },
        "output": {
            "width": "55%",
            "border": "tall #2563eb",
        },
        "status_bar": {
            "dock": "bottom",
            "height": 1,
        },
    }
