"""Enhanced TUI widgets: Sidebar, PhaseTracker, LogView, StatusBar."""

from __future__ import annotations

import time
from typing import Any

from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Input, Label, Static

# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------


class SidebarItem(Static):
    """A single sidebar navigation item with icon and label."""

    can_focus = True

    BINDINGS = [
        ("enter", "activate", "activate"),
        ("space", "activate", "activate"),
    ]

    def __init__(self, name: str, icon: str, action: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.item_name = name
        self.item_icon = icon
        self.item_action = action

    class Selected(Message):
        """Posted when sidebar item is selected."""

        def __init__(self, action: str) -> None:
            super().__init__()
            self.action = action

    def on_click(self) -> None:
        self.post_message(self.Selected(self.item_action))

    def action_activate(self) -> None:
        self.post_message(self.Selected(self.item_action))

    def on_focus(self) -> None:
        self.add_class("focused")

    def on_blur(self) -> None:
        self.remove_class("focused")

    def on_key(self, event: events.Key) -> None:
        key = event.key
        if key in ("enter", "space"):
            self.action_activate()
            event.stop()
            return

        siblings = list(self.parent.query(SidebarItem)) if self.parent is not None else []
        if not siblings:
            return

        try:
            index = siblings.index(self)
        except ValueError:
            return

        if key in ("up", "k"):
            siblings[max(0, index - 1)].focus()
            event.stop()
        elif key in ("down", "j"):
            siblings[min(len(siblings) - 1, index + 1)].focus()
            event.stop()

    def compose(self) -> ComposeResult:
        yield Label(f" {self.item_icon} {self.item_name}", classes="sidebar-label")


class Sidebar(Vertical):
    """Left sidebar with workflow steps, quick actions, and navigation."""

    class ActionSelected(Message):
        """Posted when a sidebar action is selected."""

        def __init__(self, action: str) -> None:
            super().__init__()
            self.action = action

    WORKFLOW_ITEMS = [
        ("analyze", " ", "Analyze"),
        ("suggest", " ", "Suggest"),
        ("search", " ", "Search"),
        ("specs", " ", "Specs"),
        ("map", " ", "Map"),
        ("generate", " ", "Generate"),
        ("validate", " ", "Validate"),
        ("integrate", " ", "Integrate"),
    ]

    QUICK_ACTIONS = [
        ("qa-analyze", "Quick Analyze"),
        ("qa-suggest", "Quick Suggest"),
        ("qa-integrate", "Quick Integrate"),
    ]

    CSS = """
    Sidebar {
        width: 22;
        min-width: 18;
        max-width: 28;
        background: $panel;
        border-right: tall $border;
        padding: 1 0;
    }

    Sidebar .sidebar-header {
        width: 100%;
        height: 2;
        padding: 0 1;
        background: $surface-dark;
        border-bottom: solid $border;
        content-align: center middle;
        margin-bottom: 1;
    }

    Sidebar .sidebar-section-title {
        width: 100%;
        height: 1;
        padding: 0 2;
        color: $text-muted;
        text-style: bold;
        margin-top: 1;
        margin-bottom: 1;
    }

    SidebarItem {
        width: 100%;
        height: 1;
        padding: 0 2;
        color: $text-muted;
        margin-bottom: 1;
    }

    SidebarItem:hover {
        background: $accent 15%;
        color: $text;
    }

    SidebarItem.selected {
        background: $accent 25%;
        color: $accent;
        text-style: bold;
    }

    SidebarItem.focused {
        border-left: thick $accent;
        color: $text;
    }

    Sidebar .sidebar-label {
        width: 100%;
    }

    Sidebar .quick-actions {
        width: 100%;
        padding: 0 1;
        margin-top: 1;
    }

    Sidebar .quick-btn {
        width: 100%;
        margin-bottom: 1;
        background: transparent;
        color: $text-muted;
        border: none;
        text-align: left;
        padding: 0 2;
        height: 1;
    }

    Sidebar .quick-btn:hover {
        background: $accent 15%;
        color: $text;
    }
    """

    def compose(self) -> ComposeResult:
        with Static(classes="sidebar-header"):
            yield Label("[bold]ScholarDevClaw[/]")

        yield Label("  Workflows", classes="sidebar-section-title")
        for action, icon, name in self.WORKFLOW_ITEMS:
            yield SidebarItem(name, icon, action, id=f"sidebar-{action}")

        yield Static("")

        yield Label("  Quick Actions", classes="sidebar-section-title")
        for btn_id, label in self.QUICK_ACTIONS:
            yield Button(label, id=btn_id, classes="quick-btn")

    def set_selected(self, action: str) -> None:
        """Highlight the selected workflow item."""
        for item in self.query(SidebarItem):
            if item.item_action == action:
                item.add_class("selected")
            else:
                item.remove_class("selected")

    @on(SidebarItem.Selected)
    def on_item_selected(self, msg: SidebarItem.Selected) -> None:
        self.set_selected(msg.action)
        self.post_message(self.ActionSelected(msg.action))

    @on(Button.Pressed, ".quick-btn")
    def on_quick_btn(self, msg: Button.Pressed) -> None:
        button_id = msg.button.id or ""
        if button_id == "qa-analyze":
            self.post_message(self.ActionSelected("quick-analyze"))
        elif button_id == "qa-suggest":
            self.post_message(self.ActionSelected("quick-suggest"))
        elif button_id == "qa-integrate":
            self.post_message(self.ActionSelected("quick-integrate"))


# ---------------------------------------------------------------------------
# Phase Tracker -- multi-step progress bar
# ---------------------------------------------------------------------------


class PhaseTracker(Static):
    """Multi-step progress indicator with named phases."""

    PHASES = [
        ("idle", "Ready"),
        ("validating", "Validate"),
        ("analyzing", "Analyze"),
        ("research", "Research"),
        ("mapping", "Map"),
        ("generating", "Generate"),
        ("validating_patches", "Verify"),
        ("complete", "Done"),
    ]

    current_phase: reactive[str] = reactive("idle")

    CSS = """
    PhaseTracker {
        width: 100%;
        height: 3;
        background: $surface-dark;
        border-bottom: solid $border;
        padding: 0 1;
    }

    PhaseTracker .phase-bar {
        width: 100%;
        height: 3;
        margin-top: 0;
    }

    PhaseTracker .progress-fill {
        height: 100%;
        background: $accent;
        transition: width 0.4s ease;
    }

    PhaseTracker .progress-track {
        width: 100%;
        height: 1;
        background: $border;
        margin-top: 0;
    }

    PhaseTracker .phase-label {
        width: 100%;
        height: 1;
        text-align: center;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(classes="phase-bar"):
            with Vertical():
                yield Static("", classes="phase-label", id="phase-label-text")
                with Static(classes="progress-track"):
                    yield Static("", classes="progress-fill", id="progress-fill")

    def watch_current_phase(self, old: str, new: str) -> None:
        self._update_phase(new)

    def set_phase(self, phase: str) -> None:
        self.current_phase = phase
        self._update_phase(phase)

    def _update_phase(self, phase: str) -> None:
        try:
            label = self.query_one("#phase-label-text", Static)
            fill = self.query_one("#progress-fill", Static)

            phase_label = "Ready"
            progress = 0
            for name, lbl in self.PHASES:
                if name == phase:
                    phase_label = lbl
                    break
                progress += 1

            total = len(self.PHASES) - 1
            pct = int((progress / total) * 100) if total > 0 else 0

            if phase == "complete":
                label.update(f"[bold $success] {phase_label}[/]")
                fill.styles.background = "$success"
            elif phase == "idle":
                label.update(f"[dim]{phase_label}[/]")
                fill.styles.background = "$accent"
            else:
                label.update(f"[bold $accent] {phase_label}...[/]")
                fill.styles.background = "$accent"

            fill.styles.width = f"{pct}%"
        except Exception:
            pass


# ---------------------------------------------------------------------------
# LogView -- styled log stream with syntax awareness
# ---------------------------------------------------------------------------


class LogEntry(Static):
    """A single log entry with optional styling."""

    CSS = """
    LogEntry {
        width: 100%;
        height: auto;
        padding: 0 1;
        color: $text;
    }

    LogEntry.info { color: $text; }
    LogEntry.success { color: $success; }
    LogEntry.error { color: $error; }
    LogEntry.warning { color: $warning; }
    LogEntry.dim { color: $text-muted; }
    LogEntry.accent { color: $accent; }
    LogEntry.system {
        color: $text-muted;
        text-style: italic;
    }
    """

    def __init__(self, text: str, level: str = "info", **kwargs: Any) -> None:
        super().__init__(text, **kwargs)
        self._level = level
        self.add_class(level)


class LogView(VerticalScroll):
    """Scrollable log view with styled entries."""

    can_focus = True

    CSS = """
    LogView {
        width: 100%;
        height: 1fr;
        background: $surface;
        border: solid $border;
        padding: 0 1;
        scrollbar-size: 1 1;
        scrollbar-gutter: stable;
    }

    LogView LogEntry {
        margin: 0;
    }
    """

    _entry_count: int = 0
    _max_entries: int = 500

    def add_log(self, text: str, level: str = "auto") -> None:
        """Add a log entry with auto-detected or specified level."""
        if level == "auto":
            level = self._detect_level(text)

        entry = LogEntry(text, level)
        self.mount(entry)
        self._entry_count += 1

        if self._entry_count > self._max_entries:
            children = list(self.children)
            if len(children) > self._max_entries:
                to_remove = children[: len(children) - self._max_entries]
                for child in to_remove:
                    child.remove()
                self._entry_count = self._max_entries

        self.scroll_end(animate=False)

    def add_logs(self, lines: list[str], level: str = "auto") -> None:
        """Add multiple log entries."""
        for line in lines:
            self.add_log(line, level)

    def clear_logs(self) -> None:
        """Remove all log entries."""
        self.remove_children()
        self._entry_count = 0

    @staticmethod
    def _detect_level(text: str) -> str:
        """Auto-detect log level from text content."""
        lower = text.lower().strip()
        if lower.startswith("error") or "failed" in lower or "error:" in lower:
            return "error"
        if lower.startswith("warn") or "warning" in lower:
            return "warning"
        if "complete" in lower or "done" in lower or "success" in lower:
            return "success"
        if lower.startswith("analyz") or lower.startswith("scan"):
            return "accent"
        if lower.startswith("[") or lower.startswith("---"):
            return "system"
        return "info"


# ---------------------------------------------------------------------------
# StatusBar -- bottom status bar with context info
# ---------------------------------------------------------------------------


class StatusBar(Static):
    """Status bar showing current operation, git info, and timing."""

    CSS = """
    StatusBar {
        width: 100%;
        height: 1;
        background: $surface-dark;
        border-top: solid $border;
        padding: 0 1;
        layout: horizontal;
    }

    StatusBar .status-left {
        width: 1fr;
        height: 1;
        color: $text-muted;
    }

    StatusBar .status-center {
        width: 1fr;
        height: 1;
        text-align: center;
        color: $text-muted;
    }

    StatusBar .status-right {
        width: 1fr;
        height: 1;
        text-align: right;
        color: $text-muted;
    }

    StatusBar .status-error { color: $error; }
    StatusBar .status-success { color: $success; }
    StatusBar .status-warning { color: $warning; }
    StatusBar .status-accent { color: $accent; }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._start_time: float = 0.0

    def compose(self) -> ComposeResult:
        yield Static("Ready", classes="status-left", id="status-left")
        yield Static("", classes="status-center", id="status-center")
        yield Static("", classes="status-right", id="status-right")

    def set_status(self, message: str, level: str = "info") -> None:
        """Update the left status message."""
        try:
            widget = self.query_one("#status-left", Static)
            widget.remove_class("status-error", "status-success", "status-warning", "status-accent")
            if level != "info":
                widget.add_class(f"status-{level}")
            widget.update(message)
        except Exception:
            pass

    def set_center(self, message: str) -> None:
        """Update the center status text."""
        try:
            self.query_one("#status-center", Static).update(message)
        except Exception:
            pass

    def set_right(self, message: str) -> None:
        """Update the right status text."""
        try:
            self.query_one("#status-right", Static).update(message)
        except Exception:
            pass

    def start_timer(self) -> None:
        """Start the elapsed time tracker."""
        self._start_time = time.perf_counter()

    def update_timer(self) -> None:
        """Update the elapsed time display."""
        if self._start_time > 0:
            elapsed = time.perf_counter() - self._start_time
            try:
                self.query_one("#status-right", Static).update(f" {elapsed:.1f}s")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# ResultCard -- styled result display
# ---------------------------------------------------------------------------


class ResultCard(Static):
    """A styled card for displaying pipeline results."""

    CSS = """
    ResultCard {
        width: 100%;
        height: auto;
        background: $surface-dark;
        border: solid $border;
        padding: 1;
        margin-bottom: 1;
    }

    ResultCard .card-title {
        color: $accent;
        text-style: bold;
        margin-bottom: 0;
    }

    ResultCard .card-body {
        color: $text;
        width: 100%;
        height: auto;
    }

    ResultCard .card-error {
        color: $error;
        text-style: bold;
    }

    ResultCard .card-success {
        color: $success;
    }

    ResultCard .card-meta {
        color: $text-muted;
        margin-top: 0;
    }
    """

    def __init__(self, title: str, content: str, status: str = "info", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._content = content
        self._status = status

    def compose(self) -> ComposeResult:
        yield Label(self._title, classes="card-title")
        yield Static(self._content, classes="card-body")
        if self._status == "error":
            yield Label(" Failed", classes="card-error")
        elif self._status == "success":
            yield Label(" Complete", classes="card-success")


# ---------------------------------------------------------------------------
# HistoryPane -- run history with selection
# ---------------------------------------------------------------------------


class HistoryPane(VerticalScroll):
    """Sidebar-style history of past runs."""

    class EntrySelected(Message):
        """Posted when a history entry is clicked."""

        def __init__(self, run_id: int) -> None:
            super().__init__()
            self.run_id = run_id

    CSS = """
    HistoryPane {
        width: 100%;
        height: auto;
        max-height: 15;
        background: $surface;
        border: solid $border;
        padding: 0 1;
        scrollbar-size: 1 1;
        margin-bottom: 1;
    }

    HistoryPane .history-entry {
        width: 100%;
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }

    HistoryPane .history-entry:hover {
        background: $accent 15%;
        color: $text;
    }

    HistoryPane .history-entry.success { border-left: thick $success; }
    HistoryPane .history-entry.failed { border-left: thick $error; }
    HistoryPane .history-entry.running { border-left: thick $accent; }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._entries: list[dict[str, Any]] = []

    def add_entry(self, run_id: int, action: str, status: str, duration: float) -> None:
        """Add a history entry."""
        entry = {"id": run_id, "action": action, "status": status, "duration": duration}
        self._entries.insert(0, entry)
        self._entries = self._entries[:20]
        self._render_entries()

    def _render_entries(self) -> None:
        self.remove_children()
        for entry in self._entries:
            status_class = (
                "success"
                if entry["status"] == "Done"
                else "failed"
                if entry["status"] == "Failed"
                else ""
            )
            icon = " " if entry["status"] == "Done" else " " if entry["status"] == "Failed" else " "
            line = f" {icon} #{entry['id']} {entry['action'][:8]:8} {entry['duration']:.1f}s"
            lbl = Label(line, classes=f"history-entry {status_class}")
            self.mount(lbl)

    def clear_history(self) -> None:
        self._entries.clear()
        self.remove_children()


# ---------------------------------------------------------------------------
# AgentStatus -- compact agent status indicator
# ---------------------------------------------------------------------------


class AgentStatus(Static):
    """Compact agent status with dot indicator."""

    CSS = """
    AgentStatus {
        width: auto;
        height: 1;
        padding: 0 1;
    }

    AgentStatus .dot-offline { color: $text-muted; }
    AgentStatus .dot-online { color: $success; }
    AgentStatus .dot-error { color: $error; }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._status = "Offline"

    def compose(self) -> ComposeResult:
        yield Static(" Offline", classes="dot-offline", id="agent-dot")

    def set_status(self, status: str) -> None:
        self._status = status
        try:
            dot = self.query_one("#agent-dot", Static)
            dot.remove_class("dot-offline", "dot-online", "dot-error")
            if status == "Online":
                dot.update(" Online")
                dot.add_class("dot-online")
            elif status == "Error":
                dot.update(" Error")
                dot.add_class("dot-error")
            else:
                dot.update(" Offline")
                dot.add_class("dot-offline")
        except Exception:
            pass


class PromptInput(Input):
    """Input with keyboard history navigation events."""

    class HistoryPrev(Message):
        def __init__(self) -> None:
            super().__init__()

    class HistoryNext(Message):
        def __init__(self) -> None:
            super().__init__()

    def on_key(self, event: events.Key) -> None:
        if event.key == "up":
            self.post_message(self.HistoryPrev())
            event.stop()
        elif event.key == "down":
            self.post_message(self.HistoryNext())
            event.stop()
