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


class AgentPhase(Message):
    def __init__(self, phase: str, progress: float):
        super().__init__()
        self.phase = phase
        self.progress = progress


class ScholarDevClawApp(App[None]):
    TITLE = "ScholarDevClaw"
    SUB_TITLE = "Research → Code Assistant"

    PHASES = [
        ("idle", "Idle", 0),
        ("analyzing", "🔍 Analyzing...", 0.1),
        ("research", "📚 Researching...", 0.3),
        ("mapping", "🗺 Mapping...", 0.5),
        ("generating", "⚡ Generating...", 0.7),
        ("validating", "✅ Validating...", 0.9),
        ("complete", "✨ Complete!", 1.0),
    ]

    CSS = """
    /* ========================================
       PREMIUM MODERN DARK THEME V2
       Enhanced with Phase Progress + Glows
       ======================================== */

    Screen {
        layout: vertical;
        background: #06090d;
        color: #e2e8f0;
    }

    /* Header */
    Header {
        background: linear-gradient(90deg, #0c1222 0%, #1e1b4b 50%, #0c1222 100%);
        color: #f1f5f9;
        dock: top;
        height: 3;
    }

    Header > .header--clock {
        color: #22d3ee;
    }

    /* Footer */
    Footer {
        background: #0c1222;
        color: #475569;
        dock: bottom;
        height: 1;
    }

    /* Main container */
    #main-container {
        height: 100%;
        padding: 1 2;
        background: #06090d;
    }

    /* Glass panels */
    .glass-panel {
        background: rgba(12, 18, 34, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.12);
        backdrop-blur: 16px;
        border-radius: 16px;
        padding: 1;
    }

    /* Section titles */
    .section-title {
        text-style: bold;
        color: #22d3ee;
        text-shadow: 0 0 20px rgba(34, 211, 238, 0.4);
        margin-bottom: 1;
    }

    .section-subtitle {
        color: #64748b;
        margin-bottom: 1;
    }

    /* Panels */
    #wizard-panel {
        width: 35%;
        border: solid rgba(59, 130, 246, 0.3);
        background: rgba(12, 18, 34, 0.6);
        border-radius: 16px;
        padding: 1 1;
        margin-right: 1;
    }

    #output-panel {
        width: 65%;
        border: solid rgba(139, 92, 246, 0.3);
        background: rgba(12, 18, 34, 0.6);
        border-radius: 16px;
        padding: 1 1;
    }

    /* Agent panel */
    #agent-panel {
        height: 32%;
        dock: bottom;
        border: solid rgba(34, 211, 238, 0.4);
        background: rgba(8, 47, 73, 0.4);
        border-radius: 16px 16px 0 0;
        margin: 0 2;
        padding: 1;
    }

    /* Phase progress bar */
    #phase-bar {
        dock: top;
        height: 2;
        background: rgba(30, 41, 59, 0.8);
        margin-bottom: 1;
    }

    #phase-progress {
        width: 0%;
        height: 100%;
        background: linear-gradient(90deg, #06b6d4, #3b82f6, #8b5cf6);
        transition: width 0.5s ease;
    }

    #phase-label {
        dock: top;
        height: 1;
        color: #94a3b8;
        text-style: bold;
    }

    /* Result */
    #result {
        height: 10;
        border: none;
        background: rgba(2, 10, 18, 0.9);
        border-radius: 12px;
        padding: 0 1;
    }

    /* Logs */
    #logs {
        height: 8;
        border: none;
        background: rgba(2, 10, 18, 0.9);
        border-radius: 12px;
        margin-top: 1;
    }

    /* History */
    #history {
        height: 5;
        border: none;
        background: rgba(2, 10, 18, 0.9);
        border-radius: 12px;
        margin-top: 1;
    }

    /* Agent logs */
    #agent-logs {
        height: 1fr;
        border: none;
        background: rgba(2, 10, 18, 0.95);
        border-radius: 12px;
    }

    /* Prompt bar */
    #prompt-bar {
        dock: bottom;
        height: 3;
        background: rgba(12, 18, 34, 0.95);
        border-top: 1px solid rgba(34, 211, 238, 0.2);
        padding: 0 1;
    }

    #prompt-input {
        width: 100%;
        border: none;
        background: transparent;
        color: #22d3ee;
        text-style: bold;
    }

    #prompt-input::placeholder {
        color: #334155;
    }

    /* Quick actions */
    #quick-actions {
        dock: top;
        height: 3;
        background: rgba(12, 18, 34, 0.5);
        padding: 0 1;
    }

    /* Inputs */
    Input {
        background: rgba(2, 10, 18, 0.8);
        color: #e2e8f0;
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 10px;
        padding: 0 1;
    }

    Input:focus {
        border: 1px solid #22d3ee;
        box-shadow: 0 0 20px rgba(34, 211, 238, 0.25);
    }

    Input::placeholder {
        color: #334155;
    }

    Select {
        background: rgba(2, 10, 18, 0.8);
        color: #e2e8f0;
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 10px;
    }

    Select:focus {
        border: 1px solid #22d3ee;
    }

    Checkbox {
        color: #64748b;
    }

    Checkbox:focus {
        color: #22d3ee;
    }

    /* Buttons */
    Button {
        border: none;
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border-radius: 10px;
        padding: 0 1;
        min-width: 14;
    }

    Button:hover {
        background: rgba(59, 130, 246, 0.3);
        border: 1px solid #3b82f6;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.3);
    }

    Button.-primary {
        background: linear-gradient(135deg, #0891b2 0%, #3b82f6 100%);
        color: #ffffff;
        text-style: bold;
        border: none;
    }

    Button.-primary:hover {
        box-shadow: 0 0 25px rgba(14, 165, 233, 0.5);
    }

    Button.-success {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: #ffffff;
        text-style: bold;
    }

    Button.-success:hover {
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.5);
    }

    Button.-error {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: #ffffff;
        text-style: bold;
    }

    Button.-error:hover {
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.5);
    }

    Button:disabled {
        opacity: 0.3;
    }

    /* Quick action buttons */
    .quick-btn {
        background: rgba(139, 92, 246, 0.15);
        color: #a78bfa;
        border-radius: 8px;
        padding: 0 1;
        min-width: 12;
    }

    .quick-btn:hover {
        background: rgba(139, 92, 246, 0.35);
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.4);
    }

    Label {
        color: #64748b;
    }

    #run-status {
        dock: top;
        height: 1;
        background: rgba(12, 18, 34, 0.9);
        color: #22d3ee;
        text-style: bold;
        padding: 0 1;
        border-radius: 6px;
    }

    #warning-bar {
        dock: bottom;
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        padding: 0 1;
        height: 1;
        text-style: bold;
        display: none;
    }

    #warning-bar.visible {
        display: block;
    }

    .spaced { margin-top: 1; }
    .spaced-small { margin-top: 0; }

    Pretty {
        background: transparent;
        color: #e2e8f0;
    }

    TextArea {
        background: rgba(2, 10, 18, 0.85);
        color: #94a3b8;
        border: none;
        border-radius: 12px;
    }

    TextArea:focus {
        border: 1px solid rgba(34, 211, 238, 0.2);
    }

    .agent-status {
        dock: top;
        height: 1;
        color: #475569;
    }

    .agent-status.online {
        color: #10b981;
        text-style: bold;
    }

    .agent-status.offline {
        color: #ef4444;
    }

    Horizontal { height: auto; }
    Vertical { height: auto; }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+r", "run_selected", "Run"),
        ("escape", "handle_escape", "Stop"),
        ("ctrl+l", "clear_logs", "Clear"),
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
        self._current_phase = "idle"
        self._command_history: list[str] = []
        self._history_index = -1

    def _parse_natural_command(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """Parse natural language into structured command."""
        prompt = prompt.strip().lower()
        ctx: dict[str, Any] = {}
        command = "help"

        # Check for repo path patterns
        repo_match = re.search(r"(?:to|on|in|at)\s+([/\w~.-]+)", prompt)
        if repo_match:
            ctx["repo_path"] = repo_match.group(1)

        # Check for spec names
        spec_match = re.search(
            r"(?:apply|use|with)\s+(rmsnorm|flashattention|swiglu|geglu|gqa|rope|preln|ali|bi|qknorm)",
            prompt,
        )
        if spec_match:
            ctx["spec"] = spec_match.group(1).lower()

        # Detect command intent
        if any(kw in prompt for kw in ["analyze", "scan", "inspect", "examine"]):
            command = "analyze"
        elif any(kw in prompt for kw in ["suggest", "recommend", "improve", "ideas"]):
            command = "suggest"
        elif any(kw in prompt for kw in ["integrate", "apply", "implement", "add"]):
            command = "integrate"
        elif any(kw in prompt for kw in ["search", "find", "look"]):
            command = "search"
            # Extract query
            query_match = re.search(r'search(?:ing)?\s+(?:for\s+)?["\']?([^"\']+)["\']?', prompt)
            if query_match:
                ctx["query"] = query_match.group(1).strip()
            elif "query" not in ctx:
                ctx["query"] = (
                    prompt.replace("search", "").replace("find", "").strip() or "machine learning"
                )
        elif any(kw in prompt for kw in ["map", "connect", "link"]):
            command = "map"
        elif any(kw in prompt for kw in ["generate", "create", "make"]):
            command = "generate"
        elif any(kw in prompt for kw in ["validate", "test", "check"]):
            command = "validate"
        elif any(kw in prompt for kw in ["specs", "list", "show"]):
            command = "specs"

        return command, ctx

    def _set_phase(self, phase: str) -> None:
        """Update the current phase and progress bar."""
        self._current_phase = phase
        for phase_name, label, progress in self.PHASES:
            if phase_name == phase:
                try:
                    progress_bar = self.query_one("#phase-progress")
                    progress_bar.styles.width = f"{int(progress * 100)}%"
                    phase_label = self.query_one("#phase-label")
                    phase_label.update(label)
                except Exception:
                    pass
                break

    def _format_json_output(self, data: dict[str, Any]) -> str:
        """Format JSON with syntax highlighting."""
        try:
            formatted = json.dumps(data, indent=2)
            # Add some basic formatting
            lines = formatted.split("\n")
            result_lines = []
            for line in lines:
                # Highlight keys
                line = re.sub(r'("[\w]+":)', r"§\1§", line)
                # Highlight values
                line = re.sub(r':\s*(".*?"|\d+\.?\d*|true|false|null)', r": §\1§", line)
                result_lines.append(line)
            return "\n".join(result_lines)
        except Exception:
            return str(data)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="main-container"):
            # Left Panel
            with Vertical(id="wizard-panel", classes="glass-panel"):
                yield Label("⚡ Workflow Wizard", classes="section-title")
                yield Label("Configure your pipeline", classes="section-subtitle")
                yield Select(self.action_mode_options, value="analyze", id="action")
                yield Input(
                    value=str(Path.cwd()), placeholder="/path/to/repository", id="repo-path"
                )
                yield Input(value="layer normalization", placeholder="Search query...", id="query")
                with Horizontal(classes="spaced-small"):
                    yield Checkbox("arXiv", value=False, id="search-arxiv")
                    yield Checkbox("Web", value=False, id="search-web")
                yield Input(value="python", placeholder="Language", id="search-language")
                yield Input(value="10", placeholder="Max results", id="search-max-results")
                yield Input(value="rmsnorm", placeholder="Spec name", id="spec")
                yield Input(value="", placeholder="Output dir (optional)", id="output-dir")
                with Horizontal(classes="spaced-small"):
                    yield Checkbox("Dry-run", value=False, id="integrate-dry-run")
                    yield Checkbox("Clean git", value=False, id="integrate-require-clean")
                yield Label("Status: Idle", id="run-status")
                with Horizontal(classes="spaced"):
                    yield Button("▶ Run", id="run", variant="success")
                    yield Button("Clear", id="clear", variant="default")

            # Right Panel
            with Vertical(id="output-panel", classes="glass-panel"):
                # Phase progress
                yield Label("✨ Ready", id="phase-label")
                with Horizontal(id="phase-bar"):
                    yield Label("", id="phase-progress")
                yield Label("📊 Results", classes="section-title")
                yield Pretty({}, id="result")
                yield Label("Execution Logs", classes="section-title")
                yield TextArea("", id="logs", read_only=True)
                yield Label("History", classes="section-title")
                yield TextArea("No runs yet.", id="history", read_only=True)
                with Horizontal(classes="spaced-small"):
                    yield Input(value="", placeholder="Run ID", id="history-id")
                    yield Button("↻", id="rerun-history", variant="primary", tooltip="Rerun")
                    yield Button("👁", id="view-history", variant="default", tooltip="View")

        # Agent Panel
        with Vertical(id="agent-panel", classes="glass-panel"):
            yield Label("🤖 Agent Interactive", classes="section-title")
            with Horizontal(id="quick-actions"):
                yield Button("🚀 Launch", id="launch-agent", variant="primary")
                yield Button("⏹ Stop", id="stop-agent", variant="error")
                yield Button("📋 Analyze", id="quick-analyze", classes="quick-btn")
                yield Button("💡 Suggest", id="quick-suggest", classes="quick-btn")
                yield Button("🔗 Integrate", id="quick-integrate", classes="quick-btn")
                yield Label("", id="agent-status", classes="agent-status offline")
            yield TextArea("", id="agent-logs", read_only=True)

        # Prompt Bar
        with Horizontal(id="prompt-bar"):
            yield Input(
                value="",
                placeholder="> Type naturally... (e.g., 'Apply rmsnorm to /path/to/repo')",
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
            if path:
                artifacts.append({"label": f"📄 {path}", "content": content or "No content"})
        for file_path in generation.get("written_files", []) or []:
            artifacts.append({"label": f"💾 {file_path}", "content": f"Written: {file_path}"})
        for item in generation.get("transformations", []) or []:
            if isinstance(item, dict):
                file_name = item.get("file") or "unknown"
                artifacts.append(
                    {
                        "label": f"✏️ {file_name}",
                        "content": f"Original:\n{item.get('original', '')[:300]}\n\nModified:\n{item.get('modified', '')[:300]}",
                    }
                )
        return artifacts

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
            f"#{item['id']} | {item['action']} | {item['status']} | {item['duration_s']:.1f}s"
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

    def _execute_quick_action(self, action: str) -> None:
        """Execute a quick action button."""
        repo_path = self.query_one("#repo-path", Input).value.strip()
        spec = self.query_one("#spec", Input).value.strip() or "rmsnorm"

        if not repo_path:
            self._append_logs("agent-logs", ["⚠️ Please set repository path first"])
            return

        self.query_one("#action", Select).value = action
        if action == "integrate":
            self.query_one("#spec", Input).value = spec

        self._set_phase("analyzing")
        self._run_selected_workflow()

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
            self._append_logs("logs", [f"Viewing run #{record['id']}: {record['action']}"])

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
        self.query_one("#rerun-history", Button).disabled = True
        self.query_one("#view-history", Button).disabled = True
        self.query_one("#run-status", Label).update(f"⚡ Running '{action}'...")
        self._live_logs_enabled = True
        self._active_run_request = request
        self._active_run_started_at = time.perf_counter()

        # Set phase based on action
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
        self._append_logs("logs", [f"🚀 Started: {action}"])

    @on(TaskCompleted)
    def on_task_completed(self, message: TaskCompleted) -> None:
        self._set_phase("complete")
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
            self._append_logs("agent-logs", ["🤖 Agent already running"])
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
                "💡 Type naturally or use commands: analyze, suggest, integrate...",
                "📝 Try: 'apply rmsnorm to /path/to/repo'",
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
        # Update phase based on agent output
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
            self._append_logs("agent-logs", ["🤖 No running agent"])
            self._update_agent_status(False)
            return
        self._agent_running = False
        self._agent_process.terminate()
        self._update_agent_status(False)
        self._append_logs("agent-logs", ["🛑 Agent stopped"])

    @on(Input.Submitted, "#prompt-input")
    def on_prompt_submit(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return

        # Add to history
        self._command_history.append(prompt)
        self._history_index = len(self._command_history)

        event.input.value = ""

        if not self._agent_running or not self._agent_stdin:
            # Parse as natural command and run locally
            command, ctx = self._parse_natural_command(prompt)
            self._append_logs("agent-logs", [f"👤 {prompt}"])
            self._append_logs("agent-logs", [f"🔧 Parsed: {command} with {ctx}"])

            # Apply to UI and run
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

        if prompt.lower() == "help":
            self._append_logs(
                "agent-logs",
                [
                    "📖 Commands: analyze, suggest, integrate, search, map, generate, validate",
                    "💡 Or type naturally: 'apply rmsnorm to /path/to/repo'",
                    "🔧 Quick: 'set repo /path', 'set spec rmsnorm'",
                ],
            )
            return

        if prompt.lower().startswith("set "):
            self._append_logs("agent-logs", [f"⚙️ {prompt}"])
            return

        self._append_logs("agent-logs", [f"\n👤 {prompt}"])

        try:
            self._agent_stdin.write(prompt + "\n")
            self._agent_stdin.flush()
        except Exception as exc:
            self._append_logs("agent-logs", [f"❌ Error: {exc}"])

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
            self._show_warning_bar("⚠️ Press ESC again to stop...")
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
