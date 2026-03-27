"""TUI widgets: LogView, StatusBar, PhaseTracker, ChatLog, PromptInput."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Input, Label, Markdown, Static

# ---------------------------------------------------------------------------
# Phase Tracker -- thin progress bar
# ---------------------------------------------------------------------------


class PhaseTracker(Static):
    """Single-line progress indicator with named phase."""

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
        height: 1;
        background: $surface-dark;
        padding: 0 1;
    }

    PhaseTracker .phase-bar {
        width: 100%;
        height: 1;
    }

    PhaseTracker .progress-track {
        width: 100%;
        height: 1;
        background: $border;
    }

    PhaseTracker .progress-fill {
        height: 100%;
        background: $accent;
        transition: width 0.3s linear;
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
                label.update(f"[bold $success]{phase_label}[/]")
                fill.styles.background = "$success"
            elif phase == "idle":
                label.update(f"[dim]{phase_label}[/]")
                fill.styles.background = "$accent"
            else:
                label.update(f"[bold $accent]{phase_label}...[/]")
                fill.styles.background = "$accent"

            fill.styles.width = f"{pct}%"
        except Exception:
            pass


# ---------------------------------------------------------------------------
# LogView -- styled log stream
# ---------------------------------------------------------------------------


class LogEntry(Static):
    """A single log entry with auto-detected level."""

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
        background: $panel;
        scrollbar-size: 1 1;
        scrollbar-gutter: stable;
        padding: 0 1;
    }

    LogView LogEntry {
        margin: 0;
    }
    """

    _entry_count: int = 0
    _max_entries: int = 500

    def add_log(self, text: str, level: str = "auto") -> None:
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
        for line in lines:
            self.add_log(line, level)

    def clear_logs(self) -> None:
        self.remove_children()
        self._entry_count = 0

    @staticmethod
    def _detect_level(text: str) -> str:
        lower = text.lower().strip()
        if lower.startswith("error") or "failed" in lower or "error:" in lower:
            return "error"
        if lower.startswith("warn") or "warning" in lower:
            return "warning"
        if "complete" in lower or "done" in lower or "success" in lower:
            return "success"
        if lower.startswith("==="):
            return "accent"
        if lower.startswith("[") or lower.startswith("---"):
            return "system"
        return "info"


# ---------------------------------------------------------------------------
# StatusBar -- single-line status with timer
# ---------------------------------------------------------------------------


class StatusBar(Static):
    """Status bar showing current operation and timing."""

    CSS = """
    StatusBar {
        width: 100%;
        height: 1;
        background: $panel;
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
        self._step_text: str = ""

    def compose(self) -> ComposeResult:
        yield Static("Ready", classes="status-left", id="status-left")
        yield Static("", classes="status-center", id="status-center")
        yield Static("", classes="status-right", id="status-right")

    def set_status(self, message: str, level: str = "info") -> None:
        try:
            widget = self.query_one("#status-left", Static)
            widget.remove_class("status-error", "status-success", "status-warning", "status-accent")
            if level != "info":
                widget.add_class(f"status-{level}")
            widget.update(message)
        except Exception:
            pass

    def set_center(self, message: str) -> None:
        try:
            self.query_one("#status-center", Static).update(message)
        except Exception:
            pass

    def set_step(self, current: int, total: int) -> None:
        if total > 0:
            self._step_text = f"step {current}/{total}"
        else:
            self._step_text = ""
        self._render_right()

    def start_timer(self) -> None:
        self._start_time = time.perf_counter()
        self._render_right()

    def stop_timer(self) -> None:
        self._render_right()

    def update_timer(self) -> None:
        self._render_right()

    def _render_right(self) -> None:
        parts: list[str] = []
        if self._step_text:
            parts.append(self._step_text)
        if self._start_time > 0:
            elapsed = time.perf_counter() - self._start_time
            parts.append(f"{elapsed:.1f}s")
        try:
            self.query_one("#status-right", Static).update(" | ".join(parts))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# HistoryPane -- compact run history
# ---------------------------------------------------------------------------


class HistoryPane(VerticalScroll):
    """Scrollable history of past runs."""

    can_focus = True

    class RunSelected(Message):
        """Posted when a specific run entry is activated."""

        def __init__(self, run_id: int):
            super().__init__()
            self.run_id = run_id

    CSS = """
    HistoryPane {
        width: 100%;
        height: auto;
        max-height: 10;
        background: $panel;
        scrollbar-size: 1 1;
        margin-top: 1;
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
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._entries: list[dict[str, Any]] = []

    def add_entry(
        self,
        run_id: int,
        action: str,
        status: str,
        duration: float,
        *,
        repo: str = "",
        spec: str = "",
    ) -> None:
        entry = {
            "id": run_id,
            "action": action,
            "status": status,
            "duration": duration,
            "repo": repo,
            "spec": spec,
        }
        self._entries.insert(0, entry)
        self._entries = self._entries[:20]
        self._render_entries()

    def _render_entries(self) -> None:
        self.remove_children()
        if not self._entries:
            self.mount(
                Label("No runs yet. Execute a workflow to build history.", classes="history-entry")
            )
            return

        for entry in self._entries:
            status_class = (
                "success"
                if entry["status"] == "Done"
                else "failed"
                if entry["status"] == "Failed"
                else ""
            )
            icon = "✓" if entry["status"] == "Done" else "✗" if entry["status"] == "Failed" else "•"

            repo_name = entry.get("repo", "") or "-"
            repo_name = repo_name.replace("\\", "/").rstrip("/").split("/")[-1] or "-"
            spec_name = entry.get("spec", "") or "-"
            line = (
                f"#{entry['id']:02d} {entry['action'][:9]:9} {icon} {entry['duration']:>5.1f}s "
                f"· {repo_name} · {spec_name}"
            )
            button = Button(
                line,
                id=f"history-run-{entry['id']}",
                classes=f"history-entry {status_class}",
            )
            self.mount(button)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if not button_id.startswith("history-run-"):
            return
        try:
            run_id = int(button_id.removeprefix("history-run-"))
        except ValueError:
            return
        self.post_message(self.RunSelected(run_id))

    def clear_history(self) -> None:
        self._entries.clear()
        self._render_entries()


# ---------------------------------------------------------------------------
# AgentStatus -- compact status indicator
# ---------------------------------------------------------------------------


class AgentStatus(Static):
    """Agent status with dot indicator."""

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


# ---------------------------------------------------------------------------
# PromptInput -- input with history navigation
# ---------------------------------------------------------------------------


class PromptInput(Input):
    """Input with keyboard history navigation events."""

    class HistoryPrev(Message):
        pass

    class HistoryNext(Message):
        pass

    def on_key(self, event: events.Key) -> None:
        if event.key == "up":
            self.post_message(self.HistoryPrev())
            event.stop()
        elif event.key == "down":
            self.post_message(self.HistoryNext())
            event.stop()


# ---------------------------------------------------------------------------
# ChatLog -- markdown chat timeline
# ---------------------------------------------------------------------------


class ChatLog(VerticalScroll):
    """Scrollable markdown chat/log timeline."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._entries: list[str] = []

    CSS = """
    ChatLog {
        width: 100%;
        height: 1fr;
        background: $panel;
        scrollbar-size: 1 1;
        padding: 0 1;
    }

    ChatLog .chat-entry {
        width: 100%;
        margin-bottom: 1;
        padding: 0 1;
        border-left: thick $border;
        background: $surface-dark;
    }

    ChatLog .chat-entry.user {
        border-left: thick $accent;
    }

    ChatLog .chat-entry.agent {
        border-left: thick $success;
    }

    ChatLog .chat-entry.system {
        border-left: thick $warning;
    }

    ChatLog Markdown {
        width: 100%;
        height: auto;
    }
    """

    def add_entry(self, role: str, content: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        safe = content.replace("\r", "").strip()
        block = f"**{role}**  `{ts}`\n\n{safe if safe else '_empty_'}"
        self._entries.append(block)
        md = Markdown(block, classes=f"chat-entry {role.lower()}")
        self.mount(md)
        self.scroll_end(animate=False)

    def clear_entries(self) -> None:
        self.remove_children()
        self._entries.clear()

    def export_markdown(self) -> str:
        if not self._entries:
            return "No log entries"
        return "\n\n---\n\n".join(self._entries)
