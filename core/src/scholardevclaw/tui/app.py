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

from scholardevclaw.application.schema_contract import evaluate_payload_compatibility

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
    SUB_TITLE = "Research-driven programming assistant"

    CSS = """
    Screen {
        layout: vertical;
        background: #020617;
        color: #dbeafe;
    }

    Header {
        background: #0b1b3b;
        color: #dbeafe;
    }

    Footer {
        background: #081124;
        color: #93c5fd;
    }

    #hero {
        dock: top;
        background: #0b1220;
        color: #93c5fd;
        border-bottom: solid #1d4ed8;
        padding: 0 1;
        height: 1;
        text-style: bold;
    }

    #main-row {
        height: 1fr;
        padding: 0 1;
    }

    #wizard {
        width: 45%;
        border: tall #1d4ed8;
        background: #0b1220;
        padding: 1;
        margin-right: 1;
    }

    #output {
        width: 55%;
        border: tall #2563eb;
        background: #0b1220;
        padding: 1;
    }

    #agent-row {
        height: 16;
        border: tall #0ea5e9;
        background: #030712;
        padding: 1;
        margin: 0 1;
    }

    #agent-logs {
        height: 1fr;
        border: round #1e40af;
    }

    #result {
        height: 14;
        border: round #1d4ed8;
        background: #020617;
        padding: 0 1;
    }

    #logs {
        height: 10;
        border: round #1e40af;
    }

    #history {
        height: 8;
        border: round #1e3a8a;
    }

    #run-details {
        height: 12;
        border: round #0f766e;
        background: #03151a;
    }

    #artifact-view {
        height: 12;
        border: round #155e75;
        background: #041018;
    }

    #run-status {
        margin-top: 1;
        background: #082f49;
        color: #bfdbfe;
        border: round #0ea5e9;
        padding: 0 1;
        text-style: bold;
    }

    Input,
    Select,
    TextArea,
    Pretty {
        background: #020617;
        color: #dbeafe;
    }

    Input,
    Select {
        border: round #1d4ed8;
    }

    Input:focus,
    Select:focus,
    TextArea:focus {
        border: round #38bdf8;
    }

    Checkbox {
        color: #bfdbfe;
    }

    Button {
        border: round #1e40af;
    }

    Button.-primary {
        background: #1d4ed8;
        color: #eff6ff;
    }

    Button.-success {
        background: #0369a1;
        color: #eff6ff;
    }

    Button.-error {
        background: #1e3a8a;
        color: #eff6ff;
    }

    Button:hover {
        border: round #38bdf8;
    }

    .section-title {
        text-style: bold;
        margin-bottom: 1;
        color: #93c5fd;
    }

    .spaced {
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+r", "run_selected", "Run"),
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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("ScholarDevClaw TUI · Research → Code", id="hero")
        with Horizontal(id="main-row"):
            with Vertical(id="wizard"):
                yield Label("Workflow Wizard", classes="section-title")
                yield Select(self.action_mode_options, value="analyze", id="action")
                yield Label("Repository path")
                yield Input(value=str(Path.cwd()), placeholder="/path/to/repository", id="repo-path")
                yield Label("Search query")
                yield Input(value="layer normalization", id="query")
                with Horizontal(classes="spaced"):
                    yield Checkbox("Include arXiv", value=False, id="search-arxiv")
                    yield Checkbox("Include web", value=False, id="search-web")
                yield Label("Search language filter")
                yield Input(value="python", id="search-language")
                yield Label("Search max results")
                yield Input(value="10", id="search-max-results")
                yield Label("Spec name (for map/generate/integrate)")
                yield Input(value="rmsnorm", id="spec")
                yield Label("Output directory (generate only, optional)")
                yield Input(value="", placeholder="/tmp/sdclaw-patch", id="output-dir")
                with Horizontal(classes="spaced"):
                    yield Checkbox("Integrate dry-run", value=False, id="integrate-dry-run")
                    yield Checkbox("Require clean git", value=False, id="integrate-require-clean")
                yield Label("Status: Idle", id="run-status")
                with Horizontal(classes="spaced"):
                    yield Button("Run", id="run", variant="success")
                    yield Button("Clear output", id="clear", variant="default")
            with Vertical(id="output"):
                yield Label("Result", classes="section-title")
                yield Pretty({}, id="result")
                yield Label("Execution logs", classes="section-title")
                yield TextArea("", id="logs", read_only=True)
                yield Label("Run History", classes="section-title")
                yield TextArea("No runs yet.", id="history", read_only=True)
                with Horizontal(classes="spaced"):
                    yield Input(value="", placeholder="Run ID (blank = last)", id="history-id")
                    yield Button("Rerun", id="rerun-history", variant="primary")
                    yield Button("View run", id="view-history", variant="default")
                yield Label("Run Details", classes="section-title")
                yield TextArea("No run details yet.", id="run-details", read_only=True)
                with Horizontal(classes="spaced"):
                    yield Input(value="", placeholder="Artifact # (blank = 1)", id="artifact-id")
                    yield Button("View artifact", id="view-artifact", variant="default")
                yield Label("Artifact Viewer", classes="section-title")
                yield TextArea("No artifacts yet.", id="artifact-view", read_only=True)

        with Vertical(id="agent-row"):
            yield Label("Agent Mode (Launcher)", classes="section-title")
            with Horizontal():
                yield Button("Launch agent", id="launch-agent", variant="primary")
                yield Button("Stop agent", id="stop-agent", variant="error")
            yield TextArea("", id="agent-logs", read_only=True)

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
        lines.extend([f"Compatibility issue: {item}" for item in report.issues])
        lines.extend([f"Compatibility warning: {item}" for item in report.warnings])
        lines.extend([f"Compatibility note: {item}" for item in report.notes])
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
                {
                    "label": f"new_file:{path}",
                    "content": content or f"No in-memory content available for {path}.",
                }
            )

        for file_path in generation.get("written_files", []) or []:
            artifacts.append(
                {
                    "label": f"written_file:{file_path}",
                    "content": f"Artifact written to disk at:\n{file_path}",
                }
            )

        for item in generation.get("transformations", []) or []:
            if not isinstance(item, dict):
                continue
            file_name = item.get("file") or "unknown"
            changes = item.get("changes") or []
            summary_lines = [
                f"Transformation target: {file_name}",
                f"Declared changes: {len(changes)}",
                "",
                "Original snippet:",
                str(item.get("original", ""))[:1200] or "(none)",
                "",
                "Modified snippet:",
                str(item.get("modified", ""))[:1200] or "(none)",
            ]
            artifacts.append(
                {
                    "label": f"transformation:{file_name}",
                    "content": "\n".join(summary_lines),
                }
            )

        return artifacts

    def _render_artifact_preview(self, record: dict[str, Any] | None, artifact_id_raw: str = "") -> None:
        artifact_view = self.query_one("#artifact-view", TextArea)
        artifact_input = self.query_one("#artifact-id", Input)

        if record is None:
            artifact_view.load_text("No artifacts yet.")
            artifact_input.value = ""
            return

        artifacts = self._extract_artifacts(record)
        if not artifacts:
            artifact_view.load_text("No artifacts available for this run.")
            artifact_input.value = ""
            return

        if artifact_id_raw:
            try:
                artifact_index = int(artifact_id_raw)
            except ValueError:
                self._append_logs("logs", [f"Invalid artifact id: {artifact_id_raw}"])
                artifact_index = 1
        else:
            artifact_index = 1

        artifact_index = max(1, min(artifact_index, len(artifacts)))
        artifact_input.value = str(artifact_index)

        artifact = artifacts[artifact_index - 1]
        body = [
            f"Run #{record.get('id')} artifact {artifact_index}/{len(artifacts)}",
            f"Label: {artifact.get('label', 'unknown')}",
            "",
            artifact.get("content", ""),
        ]
        artifact_view.load_text("\n".join(body))

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

        lines: list[str] = []
        for item in self._run_history:
            lines.append(
                f"#{item['id']} | {item['action']} | {item['status']} | {item['duration_s']:.2f}s"
            )
        self.query_one("#history", TextArea).load_text("\n".join(lines))

    def _resolve_history_record(self, history_id_raw: str) -> dict[str, Any] | None:
        if not self._run_history:
            return None

        if not history_id_raw:
            return self._run_history[0]

        try:
            history_id = int(history_id_raw)
        except ValueError:
            self._append_logs("logs", [f"Invalid run id: {history_id_raw}"])
            return None

        record = next((entry for entry in self._run_history if entry["id"] == history_id), None)
        if record is None:
            self._append_logs("logs", [f"Run id not found: {history_id}"])
            return None
        return record

    def _render_run_details(self, record: dict[str, Any] | None) -> None:
        details = self.query_one("#run-details", TextArea)
        if record is None:
            details.load_text("No run details yet.")
            return

        request = record.get("request", {})
        result = record.get("result", {})

        lines = [
            f"Run #{record.get('id')}",
            f"Action: {record.get('action')}",
            f"Status: {record.get('status')}",
            f"Duration: {record.get('duration_s', 0.0):.2f}s",
            f"Title: {record.get('title') or 'n/a'}",
            f"Error: {record.get('error') or 'none'}",
            "",
            "Inputs:",
            f"  repo_path: {request.get('repo_path', '')}",
            f"  spec: {request.get('spec', '') or 'auto'}",
        ]

        meta = result.get("_meta") if isinstance(result, dict) else None
        if isinstance(meta, dict):
            lines.extend(
                [
                    f"  schema_version: {meta.get('schema_version')}",
                    f"  payload_type: {meta.get('payload_type')}",
                ]
            )

        if request.get("action") == "search":
            lines.extend(
                [
                    f"  query: {request.get('query', '')}",
                    f"  include_arxiv: {bool(request.get('include_arxiv', False))}",
                    f"  include_web: {bool(request.get('include_web', False))}",
                    f"  max_results: {request.get('max_results_raw', '10')}",
                ]
            )
        if request.get("action") == "integrate":
            lines.extend(
                [
                    f"  dry_run: {bool(request.get('integrate_dry_run', False))}",
                    f"  require_clean: {bool(request.get('integrate_require_clean', False))}",
                ]
            )

        lines.append("")
        lines.append("Outputs:")
        if isinstance(result, dict):
            if result.get("step"):
                lines.append(f"  failed_step: {result.get('step')}")
            if "spec" in result:
                lines.append(f"  selected_spec: {result.get('spec')}")
            mapping = result.get("mapping")
            if isinstance(mapping, dict):
                lines.append(f"  mapping_targets: {len(mapping.get('targets', []))}")
            validation = result.get("validation")
            if isinstance(validation, dict):
                lines.append(f"  validation_stage: {validation.get('stage')}")
                lines.append(f"  validation_passed: {validation.get('passed')}")
                scorecard = validation.get("scorecard")
                if isinstance(scorecard, dict):
                    lines.append(f"  validation_summary: {scorecard.get('summary')}")
                    highlights = scorecard.get("highlights") or []
                    if highlights:
                        lines.append("  validation_highlights:")
                        lines.extend([f"    - {item}" for item in highlights[:4]])
            generation = result.get("generation")
            if isinstance(generation, dict):
                written_files = generation.get("written_files") or []
                if written_files:
                    lines.append("  written_files:")
                    lines.extend([f"    - {item}" for item in written_files[:8]])
                else:
                    new_files = generation.get("new_files") or []
                    if new_files:
                        lines.append("  generated_files:")
                        lines.extend([f"    - {item.get('path')}" for item in new_files[:8]])
            guidance = result.get("guidance")
            if isinstance(guidance, list) and guidance:
                lines.append("  guidance:")
                lines.extend([f"    - {item}" for item in guidance[:8]])

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

    def on_mount(self) -> None:
        self._refresh_action_input_state()

    def action_run_selected(self) -> None:
        self._run_selected_workflow()

    @on(Button.Pressed, "#run")
    def on_run_button(self) -> None:
        self._run_selected_workflow()

    @on(Button.Pressed, "#clear")
    def on_clear_button(self) -> None:
        self.query_one("#result", Pretty).update({})
        self.query_one("#logs", TextArea).load_text("")
        self.query_one("#run-details", TextArea).load_text("No run details yet.")
        self.query_one("#artifact-view", TextArea).load_text("No artifacts yet.")
        self.query_one("#artifact-id", Input).value = ""

    @on(Select.Changed, "#action")
    def on_action_changed(self) -> None:
        self._refresh_action_input_state()

    @on(Button.Pressed, "#rerun-history")
    def on_rerun_history(self) -> None:
        history_id_raw = self.query_one("#history-id", Input).value.strip()

        if not self._run_history:
            self._append_logs("logs", ["No history available to rerun."])
            return

        record = self._resolve_history_record(history_id_raw)
        if record is None:
            return

        request = record["request"]
        self._apply_run_request(request)
        self._render_run_details(record)
        self._render_artifact_preview(record)
        self._append_logs("logs", [f"Rerunning from history #{record['id']} ({record['action']})"])
        self._run_selected_workflow(override_request=request)

    @on(Button.Pressed, "#view-history")
    def on_view_history(self) -> None:
        history_id_raw = self.query_one("#history-id", Input).value.strip()
        record = self._resolve_history_record(history_id_raw)
        if record is None:
            if not self._run_history:
                self._append_logs("logs", ["No history available to inspect."])
            return

        self._render_run_details(record)
        self._render_artifact_preview(record)
        self._append_logs("logs", [f"Showing details for run #{record['id']} ({record['action']})"])

    @on(Button.Pressed, "#view-artifact")
    def on_view_artifact(self) -> None:
        history_id_raw = self.query_one("#history-id", Input).value.strip()
        artifact_id_raw = self.query_one("#artifact-id", Input).value.strip()
        record = self._resolve_history_record(history_id_raw)
        if record is None:
            if not self._run_history:
                self._append_logs("logs", ["No history available to inspect artifacts."])
            return

        self._render_artifact_preview(record, artifact_id_raw=artifact_id_raw)
        self._append_logs("logs", [f"Showing artifact for run #{record['id']}"])

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
        self.query_one("#view-artifact", Button).disabled = True
        self.query_one("#run-status", Label).update(f"Status: Running '{action}'...")
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
                    repo_path,
                    spec or "rmsnorm",
                    output_dir=output_dir,
                    log_callback=_emit,
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
                TaskCompleted(
                    result.title,
                    result.payload,
                    result.logs,
                    result.error,
                )
            )

        threading.Thread(target=_runner, daemon=True).start()
        self._append_logs("logs", [f"Started workflow: {action}"])

    @on(TaskCompleted)
    def on_task_completed(self, message: TaskCompleted) -> None:
        payload = {
            "title": message.title,
            "error": message.error,
            "result": message.result,
        }
        self.query_one("#result", Pretty).update(payload)
        expected_types = {"integration"} if message.title == "Integration" else {"validation"} if message.title == "Validation" else None
        compat_messages = self._payload_compat_messages(message.result, expected_types=expected_types)
        if compat_messages:
            self._append_logs("logs", compat_messages)
        if not self._live_logs_enabled:
            self._append_logs("logs", message.logs)
        self._live_logs_enabled = False
        self.query_one("#run", Button).disabled = False
        self.query_one("#rerun-history", Button).disabled = False
        self.query_one("#view-history", Button).disabled = False
        self.query_one("#view-artifact", Button).disabled = False
        status = "Done" if message.error is None else "Failed"
        self.query_one("#run-status", Label).update(f"Status: {status} ({message.title})")
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
            self._append_logs("agent-logs", ["Agent already running."])
            return

        project_root = Path(__file__).resolve().parents[4]
        agent_dir = project_root / "agent"

        if not agent_dir.exists():
            self._append_logs("agent-logs", [f"Agent folder not found: {agent_dir}"])
            return

        try:
            self._agent_process = subprocess.Popen(
                ["bun", "run", "start"],
                cwd=agent_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self._append_logs("agent-logs", [f"Failed to launch agent: {exc}"])
            return

        self._append_logs("agent-logs", [f"Launched agent in: {agent_dir}"])

        def _read_logs() -> None:
            if not self._agent_process or not self._agent_process.stdout:
                return
            for line in self._agent_process.stdout:
                self.post_message(AgentLog(line.rstrip()))

        threading.Thread(target=_read_logs, daemon=True).start()

    @on(AgentLog)
    def on_agent_log(self, message: AgentLog) -> None:
        self._append_logs("agent-logs", [message.line])

    @on(Button.Pressed, "#stop-agent")
    def on_stop_agent(self) -> None:
        if not self._agent_process or self._agent_process.poll() is not None:
            self._append_logs("agent-logs", ["No running agent process."])
            return
        self._agent_process.terminate()
        self._append_logs("agent-logs", ["Sent terminate signal to agent."])

    def on_unmount(self) -> None:
        if self._agent_process and self._agent_process.poll() is None:
            self._agent_process.terminate()


def run_tui() -> None:
    app = ScholarDevClawApp()
    app.run()


if __name__ == "__main__":
    run_tui()
