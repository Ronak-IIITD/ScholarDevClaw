"""Keyboard-first TUI widgets for the command shell."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from textual import events
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
        ("paper_ingest", "Paper: Ingest"),
        ("paper_understand", "Paper: Understand"),
        ("paper_plan", "Paper: Plan"),
        ("paper_generate", "Paper: Generate"),
        ("paper_execute", "Paper: Test"),
        ("paper_score", "Paper: Score"),
        ("paper_product", "Paper: Product"),
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
    LogView .log-line.debug { color: $text-muted; }
    """

    LEVEL_ICONS = {
        "info": "ℹ",
        "success": "✓",
        "warning": "⚠",
        "error": "✗",
        "accent": "▶",
        "system": "▸",
        "debug": "·",
    }

    def __init__(self, show_timestamps: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._entry_count = 0
        self._max_entries = 800
        self._progress_line: Static | None = None
        self._show_timestamps = show_timestamps

    def add_log(self, text: str, level: str = "auto") -> None:
        if level == "auto":
            level = self._detect_level(text)

        # Add icon prefix based on level
        icon = self.LEVEL_ICONS.get(level, "▸")

        # Add timestamp if enabled
        timestamp = ""
        if self._show_timestamps:
            from datetime import datetime

            timestamp = f"[{datetime.now().strftime('%H:%M:%S')}] "

        line = Static(f"{timestamp}{icon} {text}", classes=f"log-line {level}")
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

        # Status message with icon based on level
        status_icon = {
            "info": "●",
            "success": "✓",
            "warning": "⚠",
            "error": "✗",
            "accent": "▶",
        }.get(self._level, "●")

        tail: list[str] = [f"{status_icon} {self._message}"]
        if self._step_text:
            tail.append(f"[{self._step_text}]")
        if self._last_tokens:
            tail.append(f"+{self._format_tokens(self._last_tokens)}")
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


class RunInspector(Static):
    """Compact, always-visible run inspector pane."""

    can_focus = True

    class InspectorAction(Message):
        def __init__(
            self,
            action: str,
            run_id: int | None,
            seq: int | None = None,
            payload: dict[str, Any] | None = None,
        ):
            super().__init__()
            self.action = action
            self.run_id = run_id
            self.seq = seq
            self.payload = payload or {}

    DEFAULT_CSS = """
    RunInspector {
        width: 100%;
        height: auto;
        max-height: 10;
        color: $text-muted;
    }

    RunInspector:focus {
        color: $text;
        text-style: bold;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._run_id: int | None = None
        self._lines: list[str] = []
        self._event_line_indices: list[int] = []
        self._selected_event_index = 0
        self._review_mode = False
        self._review_token: int | None = None
        self._review_stage = ""
        self._review_hunks: list[dict[str, str]] = []
        self._review_decisions: dict[str, str] = {}
        self._selected_hunk_index = 0
        self.set_placeholder("Run Inspector: no runs yet")

    @staticmethod
    def _decision_icon(decision: str) -> str:
        return {
            "accept": "✓",
            "reject": "✗",
            "regenerate": "↻",
            "pending": "·",
        }.get(str(decision or "pending"), "·")

    @staticmethod
    def _normalize_decision(value: str) -> str:
        decision = str(value or "").strip().lower()
        if decision in {"accept", "reject", "regenerate"}:
            return decision
        return "pending"

    @classmethod
    def _normalize_hunk_summary(cls, hunk: dict[str, Any], index: int) -> dict[str, str]:
        hunk_id = str(
            hunk.get("id")
            or hunk.get("hunk_id")
            or hunk.get("key")
            or hunk.get("index")
            or (index + 1)
        ).strip()
        if not hunk_id:
            hunk_id = str(index + 1)
        key = hunk_id
        file_path = cls._trim(
            str(hunk.get("file") or hunk.get("file_path") or hunk.get("path") or "?"),
            36,
        )
        header = cls._trim(str(hunk.get("header") or hunk.get("summary") or ""), 56)
        return {
            "key": key,
            "id": hunk_id,
            "file": file_path,
            "header": header,
        }

    @classmethod
    def render_review_lines(
        cls,
        *,
        stage: str,
        hunks: list[dict[str, str]],
        decisions: dict[str, str],
        selected_index: int = 0,
        max_lines: int = 10,
    ) -> list[str]:
        clean_hunks = [
            cls._normalize_hunk_summary(item, index)
            for index, item in enumerate(hunks)
            if isinstance(item, dict)
        ]
        if not clean_hunks:
            return [f"Review pending ({stage})", "No hunks available for review"]

        accepted = sum(1 for value in decisions.values() if value == "accept")
        rejected = sum(1 for value in decisions.values() if value == "reject")
        regenerated = sum(1 for value in decisions.values() if value == "regenerate")
        pending = max(0, len(clean_hunks) - accepted - rejected - regenerated)

        lines: list[str] = [
            f"Review pending ({stage}) | {len(clean_hunks)} hunks",
            f"A:{accepted} X:{rejected} G:{regenerated} P:{pending}",
            "Keys: j/k move a/x/g set A/X/G all Enter submit",
        ]

        visible_slots = max(1, max_lines - len(lines))
        selected = max(0, min(selected_index, len(clean_hunks) - 1))
        start = max(0, selected - visible_slots + 1)
        end = min(len(clean_hunks), start + visible_slots)
        start = max(0, end - visible_slots)

        for index in range(start, end):
            hunk = clean_hunks[index]
            key = hunk.get("key", str(index + 1))
            decision = cls._normalize_decision(decisions.get(key, "pending"))
            icon = cls._decision_icon(decision)
            marker = "▶" if index == selected else " "
            header = hunk.get("header", "")
            suffix = f" | {header}" if header else ""
            lines.append(
                f"{marker} [{icon}] {hunk.get('file', '?')} #{hunk.get('id', key)}{suffix}"
            )

        return lines[:max_lines]

    @staticmethod
    def _trim(text: str, limit: int) -> str:
        value = str(text or "").replace("\n", " ").strip()
        if len(value) <= limit:
            return value
        return value[: max(0, limit - 1)].rstrip() + "…"

    @classmethod
    def render_snapshot_lines(
        cls,
        snapshot: dict[str, Any],
        *,
        max_events: int = 3,
        max_summary: int = 2,
        max_error_chars: int = 160,
        max_lines: int = 10,
    ) -> list[str]:
        run_id = int(snapshot.get("run_id", 0) or 0)
        action = str(snapshot.get("action", "") or "unknown")
        status = str(snapshot.get("status", "") or "Unknown")
        duration = float(snapshot.get("duration", 0.0) or 0.0)
        terminal_state = str(snapshot.get("terminal_state", "") or "")
        failure_code = str(snapshot.get("failure_code", "") or "")
        repo = str(snapshot.get("repo", "") or "")
        spec = str(snapshot.get("spec", "") or "")
        query = str(snapshot.get("query", "") or "")
        error = cls._trim(str(snapshot.get("error", "") or ""), max_error_chars)
        summary_lines = [
            cls._trim(str(line), 140)
            for line in list(snapshot.get("summary_lines") or [])
            if str(line).strip()
        ]
        event_lines = [
            cls._trim(str(line), 180)
            for line in list(snapshot.get("event_lines") or [])
            if str(line).strip()
        ]

        header = f"Run #{run_id} | {action} | {status} | {max(0.0, duration):.1f}s"
        lines: list[str] = [header]

        state_line_parts: list[str] = []
        if terminal_state:
            state_line_parts.append(f"state={terminal_state}")
        if failure_code:
            state_line_parts.append(f"failure={failure_code}")
        if state_line_parts:
            lines.append(" | ".join(state_line_parts))

        context_parts: list[str] = []
        if repo:
            context_parts.append(f"repo={cls._trim(repo, 44)}")
        if spec:
            context_parts.append(f"spec={cls._trim(spec, 24)}")
        if query:
            context_parts.append(f"query={cls._trim(query, 36)}")
        if context_parts:
            lines.append(" | ".join(context_parts))

        if error:
            lines.append(f"Error: {error}")

        if summary_lines:
            lines.append("Summary:")
            for line in summary_lines[:max_summary]:
                lines.append(f"- {line}")

        if event_lines:
            lines.append("Events:")
            for line in event_lines[-max_events:]:
                lines.append(line)

        return lines[:max_lines]

    def set_placeholder(self, message: str) -> None:
        self._run_id = None
        self._review_mode = False
        self._review_token = None
        self._review_stage = ""
        self._review_hunks = []
        self._review_decisions = {}
        self._selected_hunk_index = 0
        self._lines = [message]
        self._event_line_indices = []
        self._selected_event_index = 0
        self.update(message)

    def _extract_event_seq(self, line: str) -> int | None:
        token = str(line or "").strip().split(" ", 1)[0]
        if not token.isdigit():
            return None
        return int(token)

    def _selected_event_seq(self) -> int | None:
        if not self._event_line_indices:
            return None
        line_index = self._event_line_indices[self._selected_event_index]
        return self._extract_event_seq(self._lines[line_index])

    def _selected_hunk_key(self) -> str | None:
        if not self._review_hunks:
            return None
        hunk = self._review_hunks[self._selected_hunk_index]
        return str(hunk.get("key") or "") or None

    def _set_selected_hunk_decision(self, decision: str) -> None:
        key = self._selected_hunk_key()
        if key is None:
            return
        self._review_decisions[key] = self._normalize_decision(decision)
        self._render_with_selection()
        self._emit_action(
            "review_update",
            payload={
                "token": self._review_token,
                "stage": self._review_stage,
                "hunk_decisions": dict(self._review_decisions),
            },
        )

    def _set_all_hunk_decisions(self, decision: str) -> None:
        normalized = self._normalize_decision(decision)
        for hunk in self._review_hunks:
            key = str(hunk.get("key") or "").strip()
            if key:
                self._review_decisions[key] = normalized
        self._render_with_selection()
        self._emit_action(
            "review_update",
            payload={
                "token": self._review_token,
                "stage": self._review_stage,
                "hunk_decisions": dict(self._review_decisions),
            },
        )

    def _render_with_selection(self) -> None:
        if self._review_mode:
            self._selected_hunk_index = max(
                0,
                min(self._selected_hunk_index, max(0, len(self._review_hunks) - 1)),
            )
            self.update(
                "\n".join(
                    self.render_review_lines(
                        stage=self._review_stage,
                        hunks=self._review_hunks,
                        decisions=self._review_decisions,
                        selected_index=self._selected_hunk_index,
                    )
                )
            )
            return
        if not self._lines:
            self.update("Run Inspector: no runs yet")
            return
        lines = list(self._lines)
        if self._event_line_indices:
            selected_line = self._event_line_indices[self._selected_event_index]
            lines[selected_line] = f"▶ {lines[selected_line]}"
        self.update("\n".join(lines))

    def _move_selection(self, delta: int) -> None:
        if self._review_mode:
            if not self._review_hunks:
                return
            self._selected_hunk_index = (self._selected_hunk_index + delta) % len(
                self._review_hunks
            )
            self._render_with_selection()
            return
        if not self._event_line_indices:
            return
        self._selected_event_index = (self._selected_event_index + delta) % len(
            self._event_line_indices
        )
        self._render_with_selection()

    def _emit_action(self, action: str, payload: dict[str, Any] | None = None) -> None:
        self.post_message(
            self.InspectorAction(
                action=action,
                run_id=self._run_id,
                seq=None if self._review_mode else self._selected_event_seq(),
                payload=payload,
            )
        )

    def on_key(self, event: events.Key) -> None:
        if self._review_mode:
            if event.key in {"up", "k"}:
                self._move_selection(-1)
                event.stop()
                return
            if event.key in {"down", "j"}:
                self._move_selection(1)
                event.stop()
                return
            if event.key in {"a"}:
                self._set_selected_hunk_decision("accept")
                event.stop()
                return
            if event.key in {"x"}:
                self._set_selected_hunk_decision("reject")
                event.stop()
                return
            if event.key in {"g"}:
                self._set_selected_hunk_decision("regenerate")
                event.stop()
                return
            if event.key in {"A"}:
                self._set_all_hunk_decisions("accept")
                event.stop()
                return
            if event.key in {"X"}:
                self._set_all_hunk_decisions("reject")
                event.stop()
                return
            if event.key in {"G"}:
                self._set_all_hunk_decisions("regenerate")
                event.stop()
                return
            if event.key in {"enter", "space"}:
                self._emit_action(
                    "review_submit",
                    payload={
                        "token": self._review_token,
                        "stage": self._review_stage,
                        "approved": True,
                        "hunk_decisions": dict(self._review_decisions),
                    },
                )
                event.stop()
                return

        if event.key in {"up", "k"}:
            self._move_selection(-1)
            event.stop()
            return
        if event.key in {"down", "j"}:
            self._move_selection(1)
            event.stop()
            return
        if event.key in {"enter", "space"}:
            self._emit_action("events")
            event.stop()
            return
        if event.key == "r":
            self._emit_action("rerun")
            event.stop()
            return
        if event.key == "s":
            self._emit_action("show")
            event.stop()
            return
        if event.key == "e":
            self._emit_action("events")
            event.stop()

    def set_lines(self, lines: list[str], *, run_id: int | None = None) -> None:
        clean = [str(line).rstrip() for line in lines if str(line).strip()]
        if not clean:
            self.set_placeholder("Run Inspector: no runs yet")
            return

        self._review_mode = False
        self._review_token = None
        self._review_stage = ""
        self._review_hunks = []
        self._review_decisions = {}
        self._selected_hunk_index = 0

        selected_seq = self._selected_event_seq()
        self._run_id = run_id
        self._lines = clean
        self._event_line_indices = [
            index for index, line in enumerate(clean) if self._extract_event_seq(line) is not None
        ]
        if not self._event_line_indices:
            self._selected_event_index = 0
            self.update("\n".join(clean))
            return

        if selected_seq is not None:
            for idx, line_index in enumerate(self._event_line_indices):
                if self._extract_event_seq(self._lines[line_index]) == selected_seq:
                    self._selected_event_index = idx
                    break
            else:
                self._selected_event_index = 0
        else:
            self._selected_event_index = min(
                self._selected_event_index, len(self._event_line_indices) - 1
            )

        self._render_with_selection()

    def set_review(
        self,
        *,
        token: int,
        stage: str,
        hunks: list[dict[str, Any]],
        decisions: dict[str, str] | None = None,
        run_id: int | None = None,
    ) -> None:
        selected_key = self._selected_hunk_key() if self._review_mode else None
        clean_hunks = [
            self._normalize_hunk_summary(item, index)
            for index, item in enumerate(hunks)
            if isinstance(item, dict)
        ]
        if not clean_hunks:
            self.set_lines(
                [f"Review pending [{stage}]", "No hunks available for review"], run_id=run_id
            )
            return

        merged_decisions: dict[str, str] = {}
        provided = decisions or {}
        for hunk in clean_hunks:
            key = str(hunk.get("key") or "").strip()
            if not key:
                continue
            merged_decisions[key] = self._normalize_decision(provided.get(key, "pending"))

        self._review_mode = True
        self._review_token = token
        self._review_stage = stage
        self._review_hunks = clean_hunks
        self._review_decisions = merged_decisions
        self._run_id = run_id if run_id is not None else token
        self._event_line_indices = []
        self._selected_event_index = 0

        if selected_key is not None:
            for idx, hunk in enumerate(self._review_hunks):
                if hunk.get("key") == selected_key:
                    self._selected_hunk_index = idx
                    break
            else:
                self._selected_hunk_index = min(
                    self._selected_hunk_index,
                    max(0, len(self._review_hunks) - 1),
                )
        else:
            self._selected_hunk_index = min(
                self._selected_hunk_index,
                max(0, len(self._review_hunks) - 1),
            )

        self._lines = self.render_review_lines(
            stage=stage,
            hunks=self._review_hunks,
            decisions=self._review_decisions,
            selected_index=self._selected_hunk_index,
        )
        self._render_with_selection()

    def set_snapshot(
        self,
        *,
        run_id: int,
        action: str,
        status: str,
        terminal_state: str,
        duration: float,
        repo: str,
        spec: str,
        query: str,
        failure_code: str,
        error: str,
        summary_lines: list[str],
        event_lines: list[str],
    ) -> None:
        self.set_lines(
            self.render_snapshot_lines(
                {
                    "run_id": run_id,
                    "action": action,
                    "status": status,
                    "terminal_state": terminal_state,
                    "duration": duration,
                    "repo": repo,
                    "spec": spec,
                    "query": query,
                    "failure_code": failure_code,
                    "error": error,
                    "summary_lines": summary_lines,
                    "event_lines": event_lines,
                }
            ),
            run_id=run_id,
        )


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
