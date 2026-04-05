"""Keyboard-first TUI widgets for the command shell."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from textual import events
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import Input, Static


class PhaseTracker(Static):
    """Thin single-line phase indicator."""

    PHASES = [
        ("idle", "Ready"),
        ("validating", "Preflight"),
        ("analyzing", "Analyze"),
        ("research", "Search"),
        ("mapping", "Map"),
        ("generating", "Generate"),
        ("validating_patches", "Validate"),
        ("complete", "Done"),
    ]

    DEFAULT_CSS = """
    PhaseTracker {
        width: 100%;
        height: 1;
        color: $text-muted;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self.current_phase = "idle"

    def set_phase(self, phase: str) -> None:
        self.current_phase = phase
        self._update_phase(phase)

    def _update_phase(self, phase: str) -> None:
        labels = [label for _, label in self.PHASES]
        try:
            index = next(idx for idx, (name, _) in enumerate(self.PHASES) if name == phase)
        except StopIteration:
            index = 0
        width = max(1, min(10, len(labels) + 2))
        filled = min(width, int(((index + 1) / max(1, len(self.PHASES))) * width))
        bar = f"[{'█' * filled}{'░' * (width - filled)}]"
        try:
            self.update(f"{bar} {labels[index]}")
        except Exception:
            pass


class LogView(VerticalScroll):
    """Plain streaming text area."""

    can_focus = True

    DEFAULT_CSS = """
    LogView {
        width: 100%;
        height: 1fr;
        padding: 0;
        background: $background;
    }

    LogView .log-line {
        width: 100%;
        height: auto;
        padding: 0;
        margin: 0;
    }

    LogView .log-line.info { color: $text; }
    LogView .log-line.success { color: $success; }
    LogView .log-line.error { color: $error; }
    LogView .log-line.warning { color: $warning; }
    LogView .log-line.accent {
        color: $accent;
        text-style: bold;
    }
    LogView .log-line.system { color: $text-muted; }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._entry_count = 0
        self._max_entries = 800
        self._progress_line: Static | None = None

    def add_log(self, text: str, level: str = "auto") -> None:
        if level == "auto":
            level = self._detect_level(text)
        line = Static(text, classes=f"log-line {level}")
        self.mount(line)
        self._entry_count += 1
        if self._entry_count > self._max_entries:
            children = list(self.children)
            if len(children) > self._max_entries:
                for child in children[: len(children) - self._max_entries]:
                    child.remove()
                self._entry_count = self._max_entries
        self.scroll_end(animate=False)

    def add_logs(self, lines: list[str], level: str = "auto") -> None:
        for line in lines:
            self.add_log(line, level)

    def set_progress(self, text: str, level: str = "system") -> None:
        if self._progress_line is None:
            self._progress_line = Static(text, classes=f"log-line {level}")
            self.mount(self._progress_line)
            self._entry_count += 1
        else:
            self._progress_line.update(text)
            self._progress_line.set_classes(f"log-line {level}")
        self.scroll_end(animate=False)

    def clear_progress(self) -> None:
        if self._progress_line is None:
            return
        self._progress_line.remove()
        self._progress_line = None
        self._entry_count = max(0, self._entry_count - 1)

    def clear_logs(self) -> None:
        self.remove_children()
        self._entry_count = 0
        self._progress_line = None

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


class StatusBar(Static):
    """Inline mode/model/directory status."""

    DEFAULT_CSS = """
    StatusBar {
        width: 100%;
        height: auto;
        min-height: 1;
        color: $text-muted;
    }

    StatusBar.-error { color: $error; }
    StatusBar.-success { color: $success; }
    StatusBar.-warning { color: $warning; }
    StatusBar.-accent { color: $accent; }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._mode = "analyze"
        self._provider = "setup"
        self._model = "auto"
        self._directory = "."
        self._session_tokens = 0
        self._last_tokens = 0
        self._message = "Ready"
        self._step_text = ""
        self._start_time = 0.0
        self._level = "info"
        self._refresh_display()

    def set_context(
        self,
        *,
        mode: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        directory: str | None = None,
    ) -> None:
        if mode is not None:
            self._mode = mode
        if provider is not None:
            self._provider = provider
        if model is not None:
            self._model = model
        if directory is not None:
            self._directory = directory
        self._refresh_display()

    def set_usage(
        self, *, session_tokens: int | None = None, last_tokens: int | None = None
    ) -> None:
        if session_tokens is not None:
            self._session_tokens = session_tokens
        if last_tokens is not None:
            self._last_tokens = last_tokens
        self._refresh_display()

    def set_status(self, message: str, level: str = "info") -> None:
        self._message = message
        self._level = level
        self.remove_class("-error", "-success", "-warning", "-accent")
        if level != "info":
            self.add_class(f"-{level}")
        self._refresh_display()

    def set_center(self, message: str) -> None:
        self._message = message
        self._refresh_display()

    def set_step(self, current: int, total: int) -> None:
        self._step_text = f"{current}/{total}" if total > 0 else ""
        self._refresh_display()

    def start_timer(self) -> None:
        self._start_time = time.perf_counter()
        self._refresh_display()

    def stop_timer(self) -> None:
        self._refresh_display()

    def update_timer(self) -> None:
        self._refresh_display()

    def _refresh_display(self) -> None:
        model_value = self._model or "unset"
        directory_value = self._directory or "."
        if len(directory_value) > 40:
            directory_value = f"…{directory_value[-39:]}"

        parts = [
            f"MODE: {self._mode}",
            f"PROVIDER: {self._provider}",
            f"MODEL: {model_value}",
            f"TOKENS: {self._format_tokens(self._session_tokens)}",
            f"DIR: {directory_value}",
        ]
        tail: list[str] = []
        if self._message:
            tail.append(self._message)
        if self._step_text:
            tail.append(self._step_text)
        if self._last_tokens:
            tail.append(f"last {self._format_tokens(self._last_tokens)}")
        if self._start_time:
            tail.append(f"{time.perf_counter() - self._start_time:.1f}s")
        suffix = f"   {' | '.join(tail)}" if tail else ""
        self.update("   ".join(parts) + suffix)

    @staticmethod
    def _format_tokens(value: int) -> str:
        if value < 1000:
            return str(value)
        return f"{value / 1000:.1f}k"


class HistoryPane(VerticalScroll):
    """Keyboard-only command history / run history."""

    can_focus = True

    class RunSelected(Message):
        def __init__(self, run_id: int):
            super().__init__()
            self.run_id = run_id

    DEFAULT_CSS = """
    HistoryPane {
        width: 100%;
        height: auto;
        max-height: 6;
        color: $text-muted;
    }

    HistoryPane .history-line {
        width: 100%;
        height: 1;
        color: $text-muted;
    }

    HistoryPane .history-line.-selected {
        color: $accent;
        text-style: bold;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._entries: list[dict[str, Any]] = []
        self._selected_index = 0

    def add_entry(
        self,
        run_id: int,
        action: str,
        status: str,
        duration: float,
        *,
        repo: str = "",
        spec: str = "",
        finished_at: str = "--:--:--",
    ) -> None:
        self._entries.insert(
            0,
            {
                "id": run_id,
                "action": action,
                "status": status,
                "duration": duration,
                "repo": repo,
                "spec": spec,
                "finished_at": finished_at,
            },
        )
        self._entries = self._entries[:20]
        self._selected_index = 0
        self._render_entries()

    def _render_entries(self) -> None:
        self.remove_children()
        if not self._entries:
            self.mount(Static("No recent runs", classes="history-line"))
            return
        self._selected_index = max(0, min(self._selected_index, len(self._entries) - 1))
        for idx, entry in enumerate(self._entries):
            prefix = ">" if idx == self._selected_index else " "
            line = (
                f"{prefix} #{entry['id']:02d} {entry['action']} "
                f"{entry['status']} {entry['duration']:.1f}s"
            )
            classes = "history-line"
            if idx == self._selected_index:
                classes += " -selected"
            self.mount(Static(line, classes=classes))

    def _move_selection(self, delta: int) -> None:
        if not self._entries:
            return
        self._selected_index = (self._selected_index + delta) % len(self._entries)
        self._render_entries()

    def _activate_selected(self) -> None:
        if not self._entries:
            return
        self.post_message(self.RunSelected(int(self._entries[self._selected_index]["id"])))

    def on_key(self, event: events.Key) -> None:
        if event.key in {"up", "k"}:
            self._move_selection(-1)
            event.stop()
        elif event.key in {"down", "j"}:
            self._move_selection(1)
            event.stop()
        elif event.key in {"enter", "space"}:
            self._activate_selected()
            event.stop()

    def clear_history(self) -> None:
        self._entries.clear()
        self._selected_index = 0
        self._render_entries()


class AgentStatus(Static):
    """Minimal inline agent status."""

    DEFAULT_CSS = """
    AgentStatus {
        width: auto;
        height: 1;
        color: $text-muted;
    }
    """

    def set_status(self, status: str) -> None:
        self.update(status)


class PromptInput(Input):
    """Command input with shell-style key events."""

    class HistoryPrev(Message):
        pass

    class HistoryNext(Message):
        pass

    class AutoComplete(Message):
        pass

    class SuggestionDismiss(Message):
        pass

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "up":
            self.post_message(self.HistoryPrev())
            event.stop()
            return
        elif event.key == "down":
            self.post_message(self.HistoryNext())
            event.stop()
            return
        elif event.key == "tab":
            self.post_message(self.AutoComplete())
            event.stop()
            return
        elif event.key == "escape":
            self.post_message(self.SuggestionDismiss())
            event.stop()
            return
        # All other keys fall through to Input's default handler.
        await super()._on_key(event)


class ChatLog(VerticalScroll):
    """Minimal transcript log."""

    DEFAULT_CSS = """
    ChatLog {
        width: 100%;
        height: 1fr;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._entries: list[str] = []

    def add_entry(self, role: str, content: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"{role} {timestamp} {content}"
        self._entries.append(entry)
        self.mount(Static(entry))
        self.scroll_end(animate=False)

    def clear_entries(self) -> None:
        self.remove_children()
        self._entries.clear()

    def export_markdown(self) -> str:
        return "\n".join(self._entries) if self._entries else "No log entries"
