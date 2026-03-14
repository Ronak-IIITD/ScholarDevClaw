from __future__ import annotations

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
from scholardevclaw.application.schema_contract import evaluate_payload_compatibility


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


class AgentResponse(Message):
    def __init__(self, response: str):
        super().__init__()
        self.response = response


class ScholarDevClawApp(App[None]):
    TITLE = "ScholarDevClaw"
    SUB_TITLE = "Research → Code Assistant"

    CSS = """
    /* ========================================
       PREMIUM MODERN DARK THEME
       Glassmorphism + Gradient Accents
       ======================================== */

    Screen {
        layout: vertical;
        background: #0a0a0f;
        color: #e2e8f0;
    }

    /* Header with gradient border */
    Header {
        background: linear-gradient(90deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f1f5f9;
        dock: top;
        height: 3;
    }

    Header > .header--clock {
        color: #22d3ee;
    }

    /* Footer */
    Footer {
        background: #0f172a;
        color: #64748b;
        dock: bottom;
        height: 1;
    }

    /* Main container with subtle border */
    #main-container {
        height: 100%;
        padding: 1 2;
        background: #0a0a0f;
    }

    /* Glass panel base style */
    .glass-panel {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.15);
        backdrop-blur: 12px;
        border-radius: 12px;
        padding: 1;
    }

    /* Section titles */
    .section-title {
        text-style: bold;
        color: #22d3ee;
        text-shadow: 0 0 10px rgba(34, 211, 238, 0.3);
        margin-bottom: 1;
        dock: top;
        height: auto;
    }

    .section-subtitle {
        color: #94a3b8;
        margin-bottom: 1;
        dock: top;
        height: auto;
    }

    /* Left panel - Workflow Wizard */
    #wizard-panel {
        width: 38%;
        border: solid rgba(59, 130, 246, 0.4);
        background: rgba(15, 23, 42, 0.6);
        border-radius: 12px;
        padding: 1 1;
        margin-right: 1;
    }

    /* Right panel - Output */
    #output-panel {
        width: 62%;
        border: solid rgba(139, 92, 246, 0.4);
        background: rgba(15, 23, 42, 0.6);
        border-radius: 12px;
        padding: 1 1;
    }

    /* Agent Interactive Panel */
    #agent-panel {
        height: 35%;
        dock: bottom;
        border: solid rgba(34, 211, 238, 0.5);
        background: rgba(8, 47, 73, 0.5);
        border-radius: 12px 12px 0 0;
        margin: 0 2;
        padding: 1;
    }

    /* Result display */
    #result {
        height: 12;
        border: none;
        background: rgba(2, 6, 23, 0.8);
        border-radius: 8px;
        padding: 0 1;
    }

    /* Logs area */
    #logs {
        height: 10;
        border: none;
        background: rgba(2, 6, 23, 0.8);
        border-radius: 8px;
        margin-top: 1;
    }

    /* Run history */
    #history {
        height: 6;
        border: none;
        background: rgba(2, 6, 23, 0.8);
        border-radius: 8px;
        margin-top: 1;
    }

    /* Run details */
    #run-details {
        height: 10;
        border: none;
        background: rgba(2, 6, 23, 0.8);
        border-radius: 8px;
        margin-top: 1;
    }

    /* Agent logs */
    #agent-logs {
        height: 1fr;
        border: none;
        background: rgba(2, 6, 23, 0.9);
        border-radius: 8px;
    }

    /* Prompt bar */
    #prompt-bar {
        dock: bottom;
        height: 3;
        background: rgba(15, 23, 42, 0.9);
        border-top: 1px solid rgba(34, 211, 238, 0.3);
        padding: 0 1;
        margin-bottom: 0;
    }

    #prompt-input {
        width: 100%;
        border: none;
        background: transparent;
        color: #22d3ee;
        text-style: bold;
    }

    #prompt-input::placeholder {
        color: #475569;
    }

    /* Input styling */
    Input {
        background: rgba(2, 6, 23, 0.8);
        color: #e2e8f0;
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        padding: 0 1;
    }

    Input:focus {
        border: 1px solid #22d3ee;
        box-shadow: 0 0 15px rgba(34, 211, 238, 0.2);
    }

    Input::placeholder {
        color: #475569;
    }

    /* Select widget */
    Select {
        background: rgba(2, 6, 23, 0.8);
        color: #e2e8f0;
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
    }

    Select:focus {
        border: 1px solid #22d3ee;
    }

    /* Checkbox */
    Checkbox {
        color: #94a3b8;
    }

    Checkbox:focus {
        color: #22d3ee;
    }

    /* Buttons */
    Button {
        border: none;
        background: rgba(59, 130, 246, 0.2);
        color: #93c5fd;
        border-radius: 8px;
        padding: 0 2;
        min-width: 16;
    }

    Button:hover {
        background: rgba(59, 130, 246, 0.4);
        border: 1px solid #3b82f6;
    }

    Button.-primary {
        background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%);
        color: #ffffff;
        text-style: bold;
        border: none;
    }

    Button.-primary:hover {
        background: linear-gradient(135deg, #38bdf8 0%, #60a5fa 100%);
        box-shadow: 0 0 20px rgba(14, 165, 233, 0.4);
    }

    Button.-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff;
        text-style: bold;
    }

    Button.-success:hover {
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
    }

    Button.-error {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: #ffffff;
        text-style: bold;
    }

    Button.-error:hover {
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
    }

    Button:disabled {
        opacity: 0.4;
    }

    /* Labels */
    Label {
        color: #94a3b8;
    }

    /* Status bar */
    #run-status {
        dock: top;
        height: 1;
        background: rgba(15, 23, 42, 0.9);
        color: #22d3ee;
        text-style: bold;
        padding: 0 1;
        border-radius: 4px;
    }

    /* Warning bar */
    #warning-bar {
        dock: bottom;
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        padding: 0 1;
        height: 1;
        text-style: bold;
        display: none;
    }

    #warning-bar.visible {
        display: block;
    }

    /* Spacing */
    .spaced {
        margin-top: 1;
    }

    .spaced-small {
        margin-top: 0;
    }

    /* Horizontal containers */
    Horizontal {
        height: auto;
    }

    /* Vertical containers */
    Vertical {
        height: auto;
    }

    /* Pretty widget */
    Pretty {
        background: transparent;
        color: #e2e8f0;
    }

    /* TextArea */
    TextArea {
        background: rgba(2, 6, 23, 0.8);
        color: #94a3b8;
        border: none;
        border-radius: 8px;
    }

    TextArea:focus {
        border: 1px solid rgba(34, 211, 238, 0.3);
    }

    /* Agent status indicator */
    .agent-status {
        dock: top;
        height: 1;
        color: #64748b;
    }

    .agent-status.online {
        color: #10b981;
        text-style: bold;
    }

    .agent-status.offline {
        color: #ef4444;
    }

    /* Quick action buttons */
    .quick-actions {
        height: 3;
        dock: top;
        background: rgba(15, 23, 42, 0.5);
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+r", "run_selected", "Run"),
        ("escape", "handle_escape", "Stop Agent"),
    ]

    action_mode_options = [
        ("Analyze repository", "analyze"),
        ("Suggest improvements", "suggest"),
        ("Search research", "search"),
        ("List specs", "specs"),
        ("Map spec to repository", "map"),
        ("Generate patch artifacts", "generate"),
        ("Validate repository", "validate"),
        ("Integrate workflow", "integrate"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._agent_process: subprocess.Popen[str] | None = None
        self._live_logs_enabled = False
        self._run_history: list[dict[str, Any]] = []
        self._history_limit = 15
        self._next_run_id = 1
        self._active_run_request: dict[str, Any] | None = None
        self._active_run_started_at = 0.0
        self._escape_pressed_count = 0
        self._escape_warning_shown = False
        self._last_escape_time = 0.0
        self._agent_stdin: Any = None
        self._agent_running = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="main-container"):
            # Left Panel - Workflow Wizard
            with Vertical(id="wizard-panel", classes="glass-panel"):
                yield Label("⚡ Workflow Wizard", classes="section-title")
                yield Label("Configure your research-to-code pipeline", classes="section-subtitle")
                yield Select(self.action_mode_options, value="analyze", id="action")
                yield Input(
                    value=str(Path.cwd()), placeholder="/path/to/repository", id="repo-path"
                )
                yield Input(value="layer normalization", placeholder="Search query...", id="query")
                with Horizontal(classes="spaced-small"):
                    yield Checkbox("Include arXiv", value=False, id="search-arxiv")
                    yield Checkbox("Include web", value=False, id="search-web")
                yield Input(value="python", placeholder="Language filter", id="search-language")
                yield Input(value="10", placeholder="Max results", id="search-max-results")
                yield Input(value="rmsnorm", placeholder="Spec name", id="spec")
                yield Input(value="", placeholder="Output directory (optional)", id="output-dir")
                with Horizontal(classes="spaced-small"):
                    yield Checkbox("Dry-run", value=False, id="integrate-dry-run")
                    yield Checkbox("Require clean git", value=False, id="integrate-require-clean")
                yield Label("Status: Idle", id="run-status")
                with Horizontal(classes="spaced"):
                    yield Button("▶ Run", id="run", variant="success")
                    yield Button("Clear", id="clear", variant="default")

            # Right Panel - Output
            with Vertical(id="output-panel", classes="glass-panel"):
                yield Label("📊 Results", classes="section-title")
                yield Pretty({}, id="result")
                yield Label("Execution Logs", classes="section-title")
                yield TextArea("", id="logs", read_only=True)
                yield Label("Run History", classes="section-title")
                yield TextArea("No runs yet.", id="history", read_only=True)
                with Horizontal(classes="spaced-small"):
                    yield Input(value="", placeholder="Run ID", id="history-id")
                    yield Button("Rerun", id="rerun-history", variant="primary")
                    yield Button("View", id="view-history", variant="default")

        # Agent Interactive Panel
        with Vertical(id="agent-panel", classes="glass-panel"):
            yield Label("🤖 Agent Interactive Mode", classes="section-title")
            with Horizontal(classes="quick-actions"):
                yield Button("🚀 Launch Agent", id="launch-agent", variant="primary")
                yield Button("⏹ Stop Agent", id="stop-agent", variant="error")
                yield Label("", id="agent-status", classes="agent-status offline")
            yield TextArea("", id="agent-logs", read_only=True)

        # Prompt Bar
        with Horizontal(id="prompt-bar"):
            yield Input(
                value="",
                placeholder="> Type your request to the agent... (e.g., 'Analyze /path/to/repo')",
                id="prompt-input",
            )

        yield Label("", id="warning-bar")
        yield Footer()

    def _append_logs(self, widget_id: str, lines: list[str]) -> None:
        area = self.query_one(f"#{widget_id}", TextArea)
        current = area.text
        merged = (current + "\n" if current else "") + "\n".join(lines)
        area.load_text(merged)

    def _payload_compat_messages(
        self, payload: dict[str, Any], *, expected_types: set[str] | None = None
    ) -> list[str]:
        report = evaluate_payload_compatibility(payload, expected_types=expected_types)
        lines: list[str] = []
        lines.extend([f"⚠️ {item}" for item in report.issues])
        lines.extend([f"📝 {item}" for item in report.warnings])
        lines.extend([f"ℹ️ {item}" for item in report.notes])
        return lines

    def _extract_artifacts(self, record: dict[str, Any]) -> list[dict[str, str]]:
        result = record.get("result", {})
        if not isinstance(result, dict):
            return []

        generation = result.get("generation")
        if not isinstance(generation, dict):
            if any(key in result for key in ("new_files", "transformations", "written_files")):
                generation = result
            else:
                generation = {}

        artifacts: list[dict[str, str]] = []

        for item in generation.get("new_files", []) or []:
            path = item.get("path") if isinstance(item, dict) else None
            content = item.get("content") if isinstance(item, dict) else None
            if not path:
                continue
            artifacts.append(
                {"label": f"📄 {path}", "content": content or f"No content for {path}"}
            )

        for file_path in generation.get("written_files", []) or []:
            artifacts.append(
                {"label": f"💾 {file_path}", "content": f"Written to disk:\n{file_path}"}
            )

        for item in generation.get("transformations", []) or []:
            if not isinstance(item, dict):
                continue
            file_name = item.get("file") or "unknown"
            artifacts.append(
                {
                    "label": f"✏️ {file_name}",
                    "content": f"Original:\n{item.get('original', '')[:500]}\n\nModified:\n{item.get('modified', '')[:500]}",
                }
            )

        return artifacts

    def _render_artifact_preview(
        self, record: dict[str, Any] | None, artifact_id_raw: str = ""
    ) -> None:
        if record is None:
            return

        artifacts = self._extract_artifacts(record)
        if not artifacts:
            return

        lines = ["=== Artifacts ==="]
        for i, art in enumerate(artifacts, 1):
            lines.append(f"\n[{i}] {art['label']}")
            lines.append(art["content"][:300])

        self._append_logs("agent-logs", lines)

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
            f"#{item['id']} | {item['action']} | {item['status']} | {item['duration_s']:.2f}s"
            for item in self._run_history
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

    def _render_run_details(self, record: dict[str, Any] | None) -> None:
        details = self.query_one("#run-details", TextArea)
        if record is None:
            details.load_text("No run details.")
            return
        request = record.get("request", {})
        lines = [
            f"Run #{record.get('id')} | {record.get('action')} | {record.get('status')}",
            f"Duration: {record.get('duration_s', 0):.2f}s | Error: {record.get('error') or 'none'}",
            f"Repo: {request.get('repo_path', 'n/a')}",
        ]
        details.load_text("\n".join(lines))

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
        self._render_run_details(record)
        self._render_artifact_preview(record)

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
            status.update("● Online")
            status.add_class("online")
            status.remove_class("offline")
        else:
            status.update("○ Offline")
            status.add_class("offline")
            status.remove_class("online")

    def on_mount(self) -> None:
        self._refresh_action_input_state()
        self._update_agent_status(False)

    def action_run_selected(self) -> None:
        self._run_selected_workflow()

    @on(Button.Pressed, "#run")
    def on_run_button(self) -> None:
        self._run_selected_workflow()

    @on(Button.Pressed, "#clear")
    def on_clear_button(self) -> None:
        self.query_one("#result", Pretty).update({})
        self.query_one("#logs", TextArea).load_text("")
        self.query_one("#run-details", TextArea).load_text("")

    @on(Select.Changed, "#action")
    def on_action_changed(self) -> None:
        self._refresh_action_input_state()

    @on(Button.Pressed, "#rerun-history")
    def on_rerun_history(self) -> None:
        history_id_raw = self.query_one("#history-id", Input).value.strip()
        if not self._run_history:
            self._append_logs("logs", ["No history to rerun."])
            return
        record = self._resolve_history_record(history_id_raw)
        if record is None:
            return
        request = record["request"]
        self._apply_run_request(request)
        self._append_logs("logs", [f"Rerunning run #{record['id']}"])
        self._run_selected_workflow(override_request=request)

    @on(Button.Pressed, "#view-history")
    def on_view_history(self) -> None:
        history_id_raw = self.query_one("#history-id", Input).value.strip()
        record = self._resolve_history_record(history_id_raw)
        if record:
            self._render_run_details(record)
            self._append_logs("logs", [f"Viewing run #{record['id']}"])

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
        self.query_one("#rerun-history", Button).disabled = True
        self.query_one("#view-history", Button).disabled = True
        self.query_one("#run-status", Label).update(f"⚡ Running '{action}'...")
        self._live_logs_enabled = True
        self._active_run_request = request
        self._active_run_started_at = time.perf_counter()

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
        self._append_logs("logs", [f"🚀 Started: {action}"])

    @on(TaskCompleted)
    def on_task_completed(self, message: TaskCompleted) -> None:
        payload = {"title": message.title, "error": message.error, "result": message.result}
        self.query_one("#result", Pretty).update(payload)
        compat_messages = self._payload_compat_messages(
            message.result,
            expected_types={"integration", "validation"}
            if message.title in ("Integration", "Validation")
            else None,
        )
        if compat_messages:
            self._append_logs("logs", compat_messages)
        if not self._live_logs_enabled:
            self._append_logs("logs", message.logs)
        self._live_logs_enabled = False
        self.query_one("#run", Button).disabled = False
        self.query_one("#rerun-history", Button).disabled = False
        self.query_one("#view-history", Button).disabled = False
        status = "✅ Done" if message.error is None else "❌ Failed"
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
            self._append_logs("agent-logs", ["🤖 Agent already running."])
            return

        project_root = Path(__file__).resolve().parents[4]
        agent_dir = project_root / "agent"

        if not agent_dir.exists():
            self._append_logs("agent-logs", [f"❌ Agent folder not found: {agent_dir}"])
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
            self._append_logs("agent-logs", [f"❌ Failed to launch: {exc}"])
            return

        self._append_logs(
            "agent-logs",
            [
                "🚀 Agent launched in REPL mode",
                f"📁 Working dir: {agent_dir}",
                "💡 Type 'help' for commands, 'exit' to quit",
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

    @on(Button.Pressed, "#stop-agent")
    def on_stop_agent(self) -> None:
        if not self._agent_process or self._agent_process.poll() is not None:
            self._append_logs("agent-logs", ["🤖 No running agent."])
            self._update_agent_status(False)
            return
        self._agent_running = False
        self._agent_process.terminate()
        self._update_agent_status(False)
        self._append_logs("agent-logs", ["🛑 Agent stopped."])

    @on(Input.Submitted, "#prompt-input")
    def on_prompt_submit(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return

        event.input.value = ""

        if not self._agent_running or not self._agent_stdin:
            self._append_logs("agent-logs", ["❌ Agent not running. Click 'Launch Agent' first."])
            return

        if prompt.lower() in ("exit", "quit"):
            self.on_stop_agent()
            return

        if prompt.lower() == "help":
            self._append_logs(
                "agent-logs",
                [
                    "📖 Available commands:",
                    "  help  - Show this help",
                    "  exit  - Stop agent",
                    "  Or type your request naturally...",
                ],
            )
            return

        self._append_logs("agent-logs", [f"\n👤 You: {prompt}"])

        try:
            self._agent_stdin.write(prompt + "\n")
            self._agent_stdin.flush()
        except Exception as exc:
            self._append_logs("agent-logs", [f"❌ Error sending to agent: {exc}"])

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
            self._show_warning_bar("⚠️ Press ESC again to stop agent...")
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
