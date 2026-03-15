from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Checkbox, Footer, Header, Input, Label, Pretty, Select, TextArea

from scholardevclaw.application.pipeline import (
    run_analyze,
    run_generate,
    run_integrate,
    run_map,
    run_search,
    run_specs,
    run_suggest,
    run_validate,
)


class TaskCompleted(Message):
    def __init__(self, title: str, result: dict[str, Any], logs: list[str], error: str | None):
        super().__init__()
        self.title = title
        self.result = result
        self.logs = logs
        self.error = error


class TaskLog(Message):
    def __init__(self, line: str):
        super().__init__()
        self.line = line


class AgentLog(Message):
    def __init__(self, line: str):
        super().__init__()
        self.line = line


class ScholarDevClawApp(App[None]):
    TITLE = "ScholarDevClaw"
    SUB_TITLE = "Research-to-Code Assistant"

    PHASES = [
        ("idle", "Ready", 0),
        ("analyzing", "Analyzing repository...", 0.15),
        ("research", "Fetching research...", 0.3),
        ("mapping", "Mapping patterns...", 0.5),
        ("generating", "Generating patches...", 0.7),
        ("validating", "Validating...", 0.85),
        ("complete", "Complete", 1.0),
    ]

    AVAILABLE_SPECS = [
        "rmsnorm",
        "flashattention",
        "swiglu",
        "geglu",
        "gqa",
        "rope",
        "preln",
        "alibi",
        "qknorm",
    ]

    CSS = """
    /* Professional Dark Theme */

    Screen {
        layout: vertical;
        background: $surface;
        color: $text;
    }

    Header {
        background: $header;
        color: $text;
        dock: top;
        height: 3;
    }

    Footer {
        background: $header;
        color: $text-muted;
        dock: bottom;
        height: 1;
    }

    #main-container {
        height: 100%;
        padding: 1 2;
        background: $surface;
    }

    .panel {
        background: $panel;
        border: solid $border;
        border-radius: 8px;
        padding: 1;
    }

    .section-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    #wizard-panel {
        width: 30%;
        background: $panel;
        border: solid $border;
        border-radius: 8px;
        padding: 1 1;
        margin-right: 1;
    }

    #output-panel {
        width: 70%;
        background: $panel;
        border: solid $border;
        border-radius: 8px;
        padding: 1 1;
    }

    #agent-panel {
        height: 28%;
        dock: bottom;
        background: $panel;
        border: solid $border;
        border-radius: 8px 8px 0 0;
        margin: 0 2;
        padding: 1;
    }

    #phase-container {
        dock: top;
        height: 2;
        background: $surface-dark;
        border-radius: 4px;
        margin-bottom: 1;
    }

    #phase-progress {
        width: 0%;
        height: 100%;
        background: $accent;
        border-radius: 4px;
    }

    #phase-label {
        dock: top;
        height: 1;
        color: $text-muted;
    }

    #result {
        height: 10;
        background: $surface-dark;
        border-radius: 6px;
        padding: 0 1;
    }

    #logs {
        height: 7;
        background: $surface-dark;
        border-radius: 6px;
        margin-top: 1;
    }

    #history {
        height: 4;
        background: $surface-dark;
        border-radius: 6px;
        margin-top: 1;
    }

    #agent-logs {
        height: 1fr;
        background: $surface-dark;
        border-radius: 6px;
    }

    #prompt-bar {
        dock: bottom;
        height: 3;
        background: $panel;
        border-top: 1px solid $border;
        padding: 0 1;
    }

    #prompt-input {
        width: 100%;
        background: transparent;
        color: $accent;
    }

    #prompt-input::placeholder {
        color: $text-muted;
    }

    #quick-actions {
        dock: top;
        height: 3;
        background: transparent;
        padding: 0 1;
    }

    Input {
        background: $surface-dark;
        color: $text;
        border: solid $border;
        border-radius: 6px;
        padding: 0 1;
    }

    Input:focus {
        border: $accent;
    }

    Input::placeholder {
        color: $text-muted;
    }

    Select {
        background: $surface-dark;
        color: $text;
        border: solid $border;
        border-radius: 6px;
    }

    Select:focus {
        border: $accent;
    }

    Checkbox {
        color: $text-muted;
    }

    Checkbox:focus {
        color: $accent;
    }

    Button {
        border: none;
        background: $button;
        color: $text;
        border-radius: 6px;
        padding: 0 1;
        min-width: 12;
    }

    Button:hover {
        background: $button-hover;
    }

    Button.-primary {
        background: $accent;
        color: $text-inverse;
        text-style: bold;
    }

    Button.-primary:hover {
        background: $accent-hover;
    }

    Button.-success {
        background: $success;
        color: $text-inverse;
        text-style: bold;
    }

    Button.-error {
        background: $error;
        color: $text-inverse;
        text-style: bold;
    }

    Button:disabled {
        opacity: 0.4;
    }

    .quick-btn {
        background: transparent;
        color: $text-muted;
        border: 1px solid $border;
        border-radius: 6px;
        padding: 0 1;
        min-width: 10;
    }

    .quick-btn:hover {
        background: $button-hover;
        color: $text;
    }

    Label {
        color: $text-muted;
    }

    #run-status {
        dock: top;
        height: 1;
        background: transparent;
        color: $text-muted;
        padding: 0 1;
    }

    .spaced { margin-top: 1; }
    .spaced-small { margin-top: 0; }

    Pretty {
        background: transparent;
        color: $text;
    }

    TextArea {
        background: $surface-dark;
        color: $text-muted;
        border: none;
        border-radius: 6px;
    }

    TextArea:focus {
        border: 1px solid $border;
    }

    .agent-status {
        dock: top;
        height: 1;
        color: $text-muted;
    }

    .agent-status.online {
        color: $success;
    }

    .agent-status.offline {
        color: $text-muted;
    }

    Horizontal { height: auto; }
    Vertical { height: auto; }

    $surface: #0d1117;
    $surface-dark: #161b22;
    $panel: #21262d;
    $border: #30363d;
    $text: #c9d1d9;
    $text-muted: #8b949e;
    $accent: #58a6ff;
    $accent-hover: #79c0ff;
    $button: #21262d;
    $button-hover: #30363d;
    $success: #238636;
    $error: #da3633;
    $text-inverse: #ffffff;
    $header: #161b22;
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+r", "run_selected", "Run"),
        ("escape", "handle_escape", "Stop"),
        ("ctrl+l", "clear_logs", "Clear"),
        ("ctrl+a", "quick_action_analyze", "Analyze"),
        ("ctrl+s", "quick_action_suggest", "Suggest"),
        ("ctrl+i", "quick_action_integrate", "Integrate"),
    ]

    action_mode_options = [
        ("Analyze", "analyze"),
        ("Suggest", "suggest"),
        ("Search", "search"),
        ("Specs", "specs"),
        ("Map", "map"),
        ("Generate", "generate"),
        ("Validate", "validate"),
        ("Integrate", "integrate"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._agent_process: subprocess.Popen[str] | None = None
        self._live_logs_enabled = False
        self._run_history: list[dict[str, Any]] = []
        self._history_limit = 20
        self._next_run_id = 1
        self._active_run_request: dict[str, Any] | None = None
        self._active_run_started_at = 0.0
        self._escape_pressed_count = 0
        self._escape_warning_shown = False
        self._last_escape_time = 0.0
        self._agent_stdin: Any = None
        self._agent_running = False
        self._current_phase = "idle"
        self._command_history: list[str] = []
        self._history_index = -1
        self._context_file = Path.home() / ".scholardevclaw" / "tui_context.json"
        self._saved_context: dict[str, Any] = {}
        self._load_context()

    def _load_context(self) -> None:
        try:
            if self._context_file.exists():
                self._saved_context = json.loads(self._context_file.read_text())
        except Exception:
            self._saved_context = {}

    def _save_context(self) -> None:
        try:
            self._context_file.parent.mkdir(parents=True, exist_ok=True)
            self._context_file.write_text(json.dumps(self._saved_context, indent=2))
        except Exception:
            pass

    def _update_saved_context(self, key: str, value: Any) -> None:
        self._saved_context[key] = value
        self._save_context()

    def _parse_natural_command(self, prompt: str) -> tuple[str, dict[str, Any]]:
        prompt_lower = prompt.strip().lower()
        ctx: dict[str, Any] = {}
        command = "help"

        path_match = re.search(r"(?:to|on|in|at|for)\s+([/\w~.][^\s]+)", prompt)
        if path_match:
            ctx["repo_path"] = path_match.group(1)

        for spec in self.AVAILABLE_SPECS:
            if spec in prompt_lower:
                ctx["spec"] = spec
                break

        if any(kw in prompt_lower for kw in ["analyze", "scan", "inspect", "examine"]):
            command = "analyze"
        elif any(kw in prompt_lower for kw in ["suggest", "recommend", "improvement", "ideas"]):
            command = "suggest"
        elif any(kw in prompt_lower for kw in ["integrate", "apply", "implement", "add to"]):
            command = "integrate"
        elif any(kw in prompt_lower for kw in ["search", "find", "look for"]):
            command = "search"
        elif any(kw in prompt_lower for kw in ["map", "connect"]):
            command = "map"
        elif any(kw in prompt_lower for kw in ["generate", "create patch"]):
            command = "generate"
        elif any(kw in prompt_lower for kw in ["validate", "test"]):
            command = "validate"
        elif any(kw in prompt_lower for kw in ["specs", "list available"]):
            command = "specs"

        return command, ctx

    def _set_phase(self, phase: str) -> None:
        self._current_phase = phase
        for phase_name, label, progress in self.PHASES:
            if phase_name == phase:
                try:
                    bar = self.query_one("#phase-progress")
                    bar.styles.width = f"{int(progress * 100)}%"
                    label_widget = self.query_one("#phase-label")
                    label_widget.update(label)
                except Exception:
                    pass
                break

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        yield Label("Ready", id="phase-label")
        with Horizontal(id="phase-container"):
            yield Label("", id="phase-progress")

        with Horizontal(id="main-container"):
            with Vertical(id="wizard-panel", classes="panel"):
                yield Label("Workflow", classes="section-title")
                yield Select(
                    self.action_mode_options,
                    value=self._saved_context.get("last_action", "analyze"),
                    id="action",
                )
                yield Input(
                    value=self._saved_context.get("last_repo", str(Path.cwd())),
                    placeholder="/path/to/repository",
                    id="repo-path",
                )
                yield Input(
                    value=self._saved_context.get("last_query", "layer normalization"),
                    placeholder="Search query",
                    id="query",
                )
                with Horizontal(classes="spaced-small"):
                    yield Checkbox("arXiv", value=False, id="search-arxiv")
                    yield Checkbox("Web", value=False, id="search-web")
                yield Input(
                    value=self._saved_context.get("last_language", "python"),
                    placeholder="Language",
                    id="search-language",
                )
                yield Input(value="10", placeholder="Max", id="search-max-results")
                yield Input(
                    value=self._saved_context.get("last_spec", "rmsnorm"),
                    placeholder="Spec name",
                    id="spec",
                )
                yield Input(value="", placeholder="Output dir (optional)", id="output-dir")
                with Horizontal(classes="spaced-small"):
                    yield Checkbox("Dry-run", value=False, id="integrate-dry-run")
                    yield Checkbox("Clean git", value=False, id="integrate-require-clean")
                yield Label("Ready", id="run-status")
                with Horizontal(classes="spaced"):
                    yield Button("Run", id="run", variant="primary")
                    yield Button("Clear", id="clear")

            with Vertical(id="output-panel", classes="panel"):
                yield Label("Results", classes="section-title")
                yield Pretty({}, id="result")
                yield Label("Logs", classes="section-title")
                yield TextArea("", id="logs", read_only=True)
                yield Label("History", classes="section-title")
                yield TextArea("No runs yet.", id="history", read_only=True)
                with Horizontal(classes="spaced-small"):
                    yield Input(value="", placeholder="#", id="history-id")
                    yield Button("Rerun", id="rerun-history")
                    yield Button("View", id="view-history")

        with Vertical(id="agent-panel", classes="panel"):
            yield Label("Agent Mode", classes="section-title")
            with Horizontal(id="quick-actions"):
                yield Button("Launch", id="launch-agent", variant="primary")
                yield Button("Stop", id="stop-agent", variant="error")
                yield Button("Analyze", id="quick-analyze", classes="quick-btn")
                yield Button("Suggest", id="quick-suggest", classes="quick-btn")
                yield Button("Integrate", id="quick-integrate", classes="quick-btn")
                yield Label("Offline", id="agent-status", classes="agent-status offline")
            yield TextArea("", id="agent-logs", read_only=True)

        with Horizontal(id="prompt-bar"):
            yield Input(
                value="",
                placeholder="> Type request... (e.g., 'apply rmsnorm to /path')",
                id="prompt-input",
            )

        yield Footer()

    def _append_logs(self, widget_id: str, lines: list[str]) -> None:
        area = self.query_one(f"#{widget_id}", TextArea)
        current = area.text
        merged = (current + "\n" if current else "") + "\n".join(lines)
        area.load_text(merged)

    def _capture_run_request(self) -> dict[str, Any]:
        return {
            "action": self.query_one("#action", Select).value,
            "repo_path": self.query_one("#repo-path", Input).value.strip(),
            "query": self.query_one("#query", Input).value.strip(),
            "include_arxiv": self.query_one("#search-arxiv", Checkbox).value,
            "include_web": self.query_one("#search-web", Checkbox).value,
            "search_language": self.query_one("#search-language", Input).value.strip() or "python",
            "max_results_raw": self.query_one("#search-max-results", Input).value.strip() or "10",
            "spec": self.query_one("#spec", Input).value.strip(),
            "output_dir": self.query_one("#output-dir", Input).value.strip() or None,
            "integrate_dry_run": self.query_one("#integrate-dry-run", Checkbox).value,
            "integrate_require_clean": self.query_one("#integrate-require-clean", Checkbox).value,
        }

    def _apply_run_request(self, request: dict[str, Any]) -> None:
        self.query_one("#action", Select).value = request.get("action", "analyze")
        self.query_one("#repo-path", Input).value = request.get("repo_path", "")
        self.query_one("#query", Input).value = request.get("query", "")
        self.query_one("#search-arxiv", Checkbox).value = bool(request.get("include_arxiv", False))
        self.query_one("#search-web", Checkbox).value = bool(request.get("include_web", False))
        self.query_one("#search-language", Input).value = request.get("search_language", "python")
        self.query_one("#search-max-results", Input).value = request.get("max_results_raw", "10")
        self.query_one("#spec", Input).value = request.get("spec", "")
        self.query_one("#output-dir", Input).value = request.get("output_dir") or ""
        self.query_one("#integrate-dry-run", Checkbox).value = bool(
            request.get("integrate_dry_run", False)
        )
        self.query_one("#integrate-require-clean", Checkbox).value = bool(
            request.get("integrate_require_clean", False)
        )
        self._refresh_action_input_state()

    def _render_history(self) -> None:
        if not self._run_history:
            self.query_one("#history", TextArea).load_text("No runs yet.")
            return
        lines = [
            f"#{item['id']} | {item['action'][:4]} | {item['status']} | {item['duration_s']:.1f}s"
            for item in self._run_history[:10]
        ]
        self.query_one("#history", TextArea).load_text("\n".join(lines))

    def _resolve_history_record(self, history_id_raw: str) -> dict[str, Any] | None:
        if not self._run_history:
            return None
        if not history_id_raw:
            return self._run_history[0]
        try:
            history_id = int(history_id_raw)
        except ValueError:
            return None
        return next((entry for entry in self._run_history if entry["id"] == history_id), None)

    def _append_history(
        self,
        action: str,
        status: str,
        duration_s: float,
        request: dict[str, Any],
        *,
        title: str,
        result: dict[str, Any],
        error: str | None,
    ) -> None:
        record = {
            "id": self._next_run_id,
            "action": action,
            "status": status,
            "duration_s": duration_s,
            "request": request,
            "title": title,
            "result": result,
            "error": error,
        }
        self._next_run_id += 1
        self._run_history.insert(0, record)
        self._run_history = self._run_history[: self._history_limit]
        self.query_one("#history-id", Input).value = str(record["id"])
        self._render_history()
        self._update_saved_context("last_action", action)
        self._update_saved_context("last_repo", request.get("repo_path", ""))
        self._update_saved_context("last_spec", request.get("spec", ""))

    def _refresh_action_input_state(self) -> None:
        action = self.query_one("#action", Select).value
        is_search = action == "search"
        needs_spec = action in {"map", "generate", "integrate"}
        supports_output_dir = action == "generate"
        is_integrate = action == "integrate"
        self.query_one("#query", Input).disabled = not is_search
        self.query_one("#search-arxiv", Checkbox).disabled = not is_search
        self.query_one("#search-web", Checkbox).disabled = not is_search
        self.query_one("#search-language", Input).disabled = not is_search
        self.query_one("#search-max-results", Input).disabled = not is_search
        self.query_one("#spec", Input).disabled = not needs_spec
        self.query_one("#output-dir", Input).disabled = not supports_output_dir
        self.query_one("#integrate-dry-run", Checkbox).disabled = not is_integrate
        self.query_one("#integrate-require-clean", Checkbox).disabled = not is_integrate

    def _update_agent_status(self, online: bool) -> None:
        status = self.query_one("#agent-status", Label)
        if online:
            status.update("Online")
            status.add_class("online")
            status.remove_class("offline")
        else:
            status.update("Offline")
            status.add_class("offline")
            status.remove_class("online")

    def _execute_quick_action(self, action: str) -> None:
        repo_path = self.query_one("#repo-path", Input).value.strip()

        if not repo_path:
            self._append_logs("agent-logs", ["Error: Set repository path first"])
            return

        self.query_one("#action", Select).value = action
        self._set_phase("analyzing")
        self._run_selected_workflow()

    def action_quick_action_analyze(self) -> None:
        self._execute_quick_action("analyze")

    def action_quick_action_suggest(self) -> None:
        self._execute_quick_action("suggest")

    def action_quick_action_integrate(self) -> None:
        self._execute_quick_action("integrate")

    def on_mount(self) -> None:
        self._refresh_action_input_state()
        self._update_agent_status(False)
        self._set_phase("idle")

    def action_run_selected(self) -> None:
        self._run_selected_workflow()

    def action_clear_logs(self) -> None:
        self.query_one("#result", Pretty).update({})
        self.query_one("#logs", TextArea).load_text("")

    @on(Button.Pressed, "#run")
    def on_run_button(self) -> None:
        self._run_selected_workflow()

    @on(Button.Pressed, "#clear")
    def on_clear_button(self) -> None:
        self.action_clear_logs()

    @on(Select.Changed, "#action")
    def on_action_changed(self) -> None:
        self._refresh_action_input_state()

    @on(Button.Pressed, "#rerun-history")
    def on_rerun_history(self) -> None:
        history_id_raw = self.query_one("#history-id", Input).value.strip()
        if not self._run_history:
            return
        record = self._resolve_history_record(history_id_raw)
        if record:
            self._apply_run_request(record["request"])
            self._run_selected_workflow(override_request=record["request"])

    @on(Button.Pressed, "#view-history")
    def on_view_history(self) -> None:
        history_id_raw = self.query_one("#history-id", Input).value.strip()
        record = self._resolve_history_record(history_id_raw)
        if record:
            self._append_logs("logs", [f"Viewing #{record['id']}: {record['action']}"])

    @on(Button.Pressed, "#quick-analyze")
    def on_quick_analyze(self) -> None:
        self._execute_quick_action("analyze")

    @on(Button.Pressed, "#quick-suggest")
    def on_quick_suggest(self) -> None:
        self._execute_quick_action("suggest")

    @on(Button.Pressed, "#quick-integrate")
    def on_quick_integrate(self) -> None:
        self._execute_quick_action("integrate")

    def _run_selected_workflow(self, override_request: dict[str, Any] | None = None) -> None:
        request = override_request or self._capture_run_request()
        action = request.get("action", "analyze")
        repo_path = request.get("repo_path", "")
        query = request.get("query", "")
        include_arxiv = bool(request.get("include_arxiv", False))
        include_web = bool(request.get("include_web", False))
        search_language = request.get("search_language", "python")
        max_results_raw = request.get("max_results_raw", "10")
        spec = request.get("spec", "")
        output_dir = request.get("output_dir")
        integrate_dry_run = bool(request.get("integrate_dry_run", False))
        integrate_require_clean = bool(request.get("integrate_require_clean", False))

        try:
            max_results = max(1, int(max_results_raw))
        except ValueError:
            max_results = 10

        run_button = self.query_one("#run", Button)
        if run_button.disabled:
            return
        run_button.disabled = True

        for btn in [
            "rerun-history",
            "view-history",
            "quick-analyze",
            "quick-suggest",
            "quick-integrate",
        ]:
            try:
                self.query_one(f"#{btn}", Button).disabled = True
            except Exception:
                pass

        self.query_one("#run-status", Label).update(f"Running '{action}'...")
        self._live_logs_enabled = True
        self._active_run_request = request
        self._active_run_started_at = time.perf_counter()

        phase_map = {
            "analyze": "analyzing",
            "suggest": "research",
            "search": "research",
            "map": "mapping",
            "generate": "generating",
            "validate": "validating",
            "integrate": "analyzing",
            "specs": "idle",
        }
        self._set_phase(phase_map.get(action, "analyzing"))

        def _runner() -> None:
            def _emit(line: str) -> None:
                self.post_message(TaskLog(line))

            if action == "analyze":
                result = run_analyze(repo_path, log_callback=_emit)
            elif action == "suggest":
                result = run_suggest(repo_path, log_callback=_emit)
            elif action == "search":
                result = run_search(
                    query or "layer normalization",
                    include_arxiv=include_arxiv,
                    include_web=include_web,
                    language=search_language,
                    max_results=max_results,
                    log_callback=_emit,
                )
            elif action == "map":
                result = run_map(repo_path, spec or "rmsnorm", log_callback=_emit)
            elif action == "generate":
                result = run_generate(
                    repo_path, spec or "rmsnorm", output_dir=output_dir, log_callback=_emit
                )
            elif action == "validate":
                result = run_validate(repo_path, log_callback=_emit)
            elif action == "integrate":
                result = run_integrate(
                    repo_path,
                    spec or None,
                    dry_run=integrate_dry_run,
                    require_clean=integrate_require_clean,
                    output_dir=output_dir,
                    log_callback=_emit,
                )
            else:
                result = run_specs(detailed=True, log_callback=_emit)

            self.post_message(
                TaskCompleted(result.title, result.payload, result.logs, result.error)
            )

        threading.Thread(target=_runner, daemon=True).start()
        self._append_logs("logs", [f"Started: {action} on {repo_path or 'default'}"])

    @on(TaskCompleted)
    def on_task_completed(self, message: TaskCompleted) -> None:
        self._set_phase("complete")

        payload = {"title": message.title, "error": message.error, "result": message.result}
        self.query_one("#result", Pretty).update(payload)

        self._live_logs_enabled = False

        run_button = self.query_one("#run", Button)
        run_button.disabled = False
        for btn in [
            "rerun-history",
            "view-history",
            "quick-analyze",
            "quick-suggest",
            "quick-integrate",
        ]:
            try:
                self.query_one(f"#{btn}", Button).disabled = False
            except Exception:
                pass

        status = "Done" if message.error is None else "Failed"
        self.query_one("#run-status", Label).update(f"{status} ({message.title})")

        action = (self._active_run_request or {}).get("action", "unknown")
        duration_s = max(0.0, time.perf_counter() - self._active_run_started_at)

        if self._active_run_request is not None:
            self._append_history(
                action,
                status,
                duration_s,
                self._active_run_request,
                title=message.title,
                result=message.result,
                error=message.error,
            )

        self._active_run_request = None
        self._active_run_started_at = 0.0

    @on(TaskLog)
    def on_task_log(self, message: TaskLog) -> None:
        self._append_logs("logs", [message.line])

    @on(Button.Pressed, "#launch-agent")
    def on_launch_agent(self) -> None:
        if self._agent_process and self._agent_process.poll() is None:
            self._append_logs("agent-logs", ["Agent already running"])
            return

        project_root = Path(__file__).resolve().parents[4]
        agent_dir = project_root / "agent"

        if not agent_dir.exists():
            self._append_logs("agent-logs", [f"Agent directory not found: {agent_dir}"])
            return

        try:
            self._agent_process = subprocess.Popen(
                ["bun", "run", "start", "--repl"],
                cwd=agent_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._agent_stdin = self._agent_process.stdin
            self._agent_running = True
            self._update_agent_status(True)
        except Exception as exc:
            self._append_logs("agent-logs", [f"Failed to launch: {exc}"])
            return

        self._append_logs(
            "agent-logs",
            [
                "Agent launched in REPL mode",
                "Commands: analyze, suggest, integrate, search...",
                "Type 'help' for more, 'exit' to quit",
            ],
        )

        def _read_logs() -> None:
            if not self._agent_process or not self._agent_process.stdout:
                return
            for line in self._agent_process.stdout:
                if line.strip():
                    self.post_message(AgentLog(line.rstrip()))

        threading.Thread(target=_read_logs, daemon=True).start()

    @on(AgentLog)
    def on_agent_log(self, message: AgentLog) -> None:
        self._append_logs("agent-logs", [message.line])
        line_lower = message.line.lower()
        if "analyzing" in line_lower:
            self._set_phase("analyzing")
        elif "research" in line_lower or "searching" in line_lower:
            self._set_phase("research")
        elif "mapping" in line_lower:
            self._set_phase("mapping")
        elif "generating" in line_lower or "creating" in line_lower:
            self._set_phase("generating")
        elif "validating" in line_lower or "testing" in line_lower:
            self._set_phase("validating")
        elif "complete" in line_lower or "done" in line_lower or "finished" in line_lower:
            self._set_phase("complete")

    @on(Button.Pressed, "#stop-agent")
    def on_stop_agent(self) -> None:
        if not self._agent_process or self._agent_process.poll() is not None:
            self._append_logs("agent-logs", ["No running agent"])
            self._update_agent_status(False)
            return
        self._agent_running = False
        self._agent_process.terminate()
        self._update_agent_status(False)
        self._append_logs("agent-logs", ["Agent stopped"])

    @on(Input.Submitted, "#prompt-input")
    def on_prompt_submit(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return

        self._command_history.append(prompt)
        self._history_index = len(self._command_history)
        event.input.value = ""

        if prompt.lower() in ("help", "?"):
            self._append_logs(
                "agent-logs",
                [
                    "Commands: analyze, suggest, integrate, search, map, generate, validate",
                    "Natural language: 'apply rmsnorm to /path'",
                    "Type 'exit' to quit",
                ],
            )
            return

        if not self._agent_running or not self._agent_stdin:
            command, ctx = self._parse_natural_command(prompt)
            self._append_logs("agent-logs", [f"User: {prompt}"])

            if "repo_path" in ctx:
                self.query_one("#repo-path", Input).value = ctx["repo_path"]
            if "spec" in ctx:
                self.query_one("#spec", Input).value = ctx["spec"]
            if "query" in ctx:
                self.query_one("#query", Input).value = ctx["query"]

            self.query_one("#action", Select).value = command
            self._set_phase("analyzing")
            self._run_selected_workflow()
            return

        if prompt.lower() in ("exit", "quit"):
            self.on_stop_agent()
            return

        self._append_logs("agent-logs", [f"User: {prompt}"])

        try:
            self._agent_stdin.write(prompt + "\n")
            self._agent_stdin.flush()
        except Exception as exc:
            self._append_logs("agent-logs", [f"Error: {exc}"])

    def on_unmount(self) -> None:
        if self._agent_process and self._agent_process.poll() is None:
            self._agent_process.terminate()

    def action_handle_escape(self) -> None:
        current_time = time.time()
        if current_time - self._last_escape_time > 2.0:
            self._escape_pressed_count = 0
            self._escape_warning_shown = False
        self._last_escape_time = current_time
        self._escape_pressed_count += 1

        if self._escape_pressed_count == 1:
            self._show_warning_bar("Press ESC again to stop agent...")
            self._escape_warning_shown = True
        elif self._escape_pressed_count >= 2 and self._escape_warning_shown:
            self._hide_warning_bar()
            self.on_stop_agent()
            self._escape_pressed_count = 0
            self._escape_warning_shown = False

    def _show_warning_bar(self, message: str) -> None:
        try:
            warning_bar = self.query_one("#warning-bar", Label)
            warning_bar.update(message)
            warning_bar.add_class("visible")
        except Exception:
            pass

    def _hide_warning_bar(self) -> None:
        try:
            warning_bar = self.query_one("#warning-bar", Label)
            warning_bar.update("")
            warning_bar.remove_class("visible")
        except Exception:
            pass


def run_tui() -> None:
    app = ScholarDevClawApp()
    app.run()


if __name__ == "__main__":
    run_tui()
