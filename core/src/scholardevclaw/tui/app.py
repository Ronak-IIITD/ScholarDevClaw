from __future__ import annotations

import json
import logging
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

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "default_repo": "",
    "default_spec": "rmsnorm",
    "default_language": "python",
    "max_results": 10,
    "auto_validate": True,
    "require_clean_git": False,
    "theme": "dark",
}


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


class ValidationError(Exception):
    pass


class ScholarDevClawApp(App[None]):
    TITLE = "ScholarDevClaw"
    SUB_TITLE = "Research-to-Code Assistant"

    PHASES = [
        ("idle", "Ready", 0),
        ("validating", "Validating input...", 0.05),
        ("analyzing", "Analyzing repository...", 0.2),
        ("research", "Fetching research...", 0.4),
        ("mapping", "Mapping patterns...", 0.55),
        ("generating", "Generating patches...", 0.7),
        ("validating_patches", "Validating patches...", 0.85),
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
        transition: width 0.3s ease;
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

    Input.invalid {
        border: $error;
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

    .status-error { color: $error; }
    .status-success { color: $success; }
    .status-warning { color: $warning; }

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

    .agent-status.online { color: $success; }
    .agent-status.offline { color: $text-muted; }
    .agent-status.error { color: $error; }

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
    $warning: #9e6a03;
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

        # Config
        self._config_dir = Path.home() / ".scholardevclaw"
        self._config_file = self._config_dir / "config.json"
        self._context_file = self._config_dir / "tui_context.json"
        self._config = self._load_config()
        self._saved_context: dict[str, Any] = self._load_context()

    def _load_config(self) -> dict[str, Any]:
        try:
            if self._config_file.exists():
                with open(self._config_file) as f:
                    return {**DEFAULT_CONFIG, **json.load(f)}
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
        return DEFAULT_CONFIG.copy()

    def _save_config(self) -> None:
        try:
            self._config_dir.mkdir(parents=True, exist_ok=True)
            with open(self._config_file, "w") as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save config: {e}")

    def _load_context(self) -> dict[str, Any]:
        try:
            if self._context_file.exists():
                return json.loads(self._context_file.read_text())
        except Exception:
            pass
        return {}

    def _save_context(self) -> None:
        try:
            self._context_file.parent.mkdir(parents=True, exist_ok=True)
            self._context_file.write_text(json.dumps(self._saved_context, indent=2))
        except Exception:
            pass

    def _validate_repo_path(self, path: str) -> tuple[bool, str]:
        """Validate repository path."""
        if not path:
            return False, "Repository path is required"

        p = Path(path).expanduser()
        if not p.exists():
            return False, f"Path does not exist: {path}"
        if not p.is_dir():
            return False, f"Path is not a directory: {path}"
        return True, ""

    def _validate_spec(self, spec: str) -> tuple[bool, str]:
        """Validate spec name."""
        if not spec:
            return True, ""  # Optional
        if spec.lower() not in self.AVAILABLE_SPECS:
            return False, f"Unknown spec: {spec}. Valid: {', '.join(self.AVAILABLE_SPECS)}"
        return True, ""

    def _check_git_status(self, path: str) -> tuple[bool, str]:
        """Check if repo has uncommitted changes."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return True, "Repository has uncommitted changes"
            return False, ""
        except Exception:
            return False, ""  # Not a git repo or git not available

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

    def _set_status(self, message: str, level: str = "info") -> None:
        try:
            status = self.query_one("#run-status", Label)
            status.update(message)
            status.remove_class("status-error", "status-success", "status-warning")
            if level == "error":
                status.add_class("status-error")
            elif level == "success":
                status.add_class("status-success")
            elif level == "warning":
                status.add_class("status-warning")
        except Exception:
            pass

    def _update_agent_status(self, status: str) -> None:
        try:
            label = self.query_one("#agent-status", Label)
            label.update(status)
            label.remove_class("online", "offline", "error")
            if status == "Online":
                label.add_class("online")
            elif status == "Error":
                label.add_class("error")
            else:
                label.add_class("offline")
        except Exception:
            pass

    def _log(self, widget_id: str, lines: list[str]) -> None:
        area = self.query_one(f"#{widget_id}", TextArea)
        current = area.text
        merged = (current + "\n" if current else "") + "\n".join(lines)
        area.load_text(merged)

    def _parse_natural_command(self, prompt: str) -> tuple[str, dict[str, Any]]:
        prompt_lower = prompt.strip().lower()
        ctx: dict[str, Any] = {}
        command = "help"

        # Extract path
        path_match = re.search(r"(?:to|on|in|at|for)\s+([/\w~.][^\s]+)", prompt)
        if path_match:
            ctx["repo_path"] = path_match.group(1)

        # Extract spec
        for spec in self.AVAILABLE_SPECS:
            if spec in prompt_lower:
                ctx["spec"] = spec
                break

        # Detect intent
        if any(kw in prompt_lower for kw in ["analyze", "scan", "inspect"]):
            command = "analyze"
        elif any(kw in prompt_lower for kw in ["suggest", "recommend", "improvement"]):
            command = "suggest"
        elif any(kw in prompt_lower for kw in ["integrate", "apply", "implement"]):
            command = "integrate"
        elif any(kw in prompt_lower for kw in ["search", "find", "look"]):
            command = "search"
        elif any(kw in prompt_lower for kw in ["map", "connect"]):
            command = "map"
        elif any(kw in prompt_lower for kw in ["generate", "create patch"]):
            command = "generate"
        elif any(kw in prompt_lower for kw in ["validate", "test"]):
            command = "validate"

        return command, ctx

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
                    value=self._saved_context.get(
                        "last_action", self._config.get("default_action", "analyze")
                    ),
                    id="action",
                )
                yield Input(
                    value=self._saved_context.get(
                        "last_repo", self._config.get("default_repo", str(Path.cwd()))
                    ),
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
                    value=self._saved_context.get(
                        "last_language", self._config.get("default_language", "python")
                    ),
                    placeholder="Language",
                    id="search-language",
                )
                yield Input(
                    value=str(self._config.get("max_results", 10)),
                    placeholder="Max",
                    id="search-max-results",
                )
                yield Input(
                    value=self._saved_context.get(
                        "last_spec", self._config.get("default_spec", "rmsnorm")
                    ),
                    placeholder="Spec name",
                    id="spec",
                )
                yield Input(value="", placeholder="Output dir (optional)", id="output-dir")
                with Horizontal(classes="spaced-small"):
                    yield Checkbox("Dry-run", value=False, id="integrate-dry-run")
                    yield Checkbox(
                        "Clean git",
                        value=self._config.get("require_clean_git", False),
                        id="integrate-require-clean",
                    )
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

    def _capture_request(self) -> dict[str, Any]:
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

    def _apply_request(self, request: dict[str, Any]) -> None:
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
        self._refresh_action_state()

    def _refresh_action_state(self) -> None:
        action = self.query_one("#action", Select).value
        is_search = action == "search"
        needs_spec = action in {"map", "generate", "integrate"}
        supports_output_dir = action == "generate"
        is_integrate = action == "integrate"

        for id, widget in [
            ("query", Input),
            ("search-arxiv", Checkbox),
            ("search-web", Checkbox),
            ("search-language", Input),
            ("search-max-results", Input),
        ]:
            try:
                self.query_one(f"#{id}", widget).disabled = not is_search
            except Exception:
                pass

        for id, widget in [("spec", Input), ("output-dir", Input)]:
            try:
                self.query_one(f"#{id}", widget).disabled = False
            except Exception:
                pass

        try:
            self.query_one("#spec", Input).disabled = not needs_spec
            self.query_one("#output-dir", Input).disabled = not supports_output_dir
            self.query_one("#integrate-dry-run", Checkbox).disabled = not is_integrate
            self.query_one("#integrate-require-clean", Checkbox).disabled = not is_integrate
        except Exception:
            pass

    def _render_history(self) -> None:
        if not self._run_history:
            self.query_one("#history", TextArea).load_text("No runs yet.")
            return
        lines = [
            f"#{r['id']} | {r['action'][:4]} | {r['status']} | {r['duration_s']:.1f}s"
            for r in self._run_history[:10]
        ]
        self.query_one("#history", TextArea).load_text("\n".join(lines))

    def _resolve_history(self, raw: str) -> dict[str, Any] | None:
        if not self._run_history:
            return None
        if not raw:
            return self._run_history[0]
        try:
            return next((r for r in self._run_history if r["id"] == int(raw)), None)
        except ValueError:
            return None

    def _append_history(
        self,
        action: str,
        status: str,
        duration: float,
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
            "duration_s": duration,
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
        self._saved_context["last_action"] = action
        self._saved_context["last_repo"] = request.get("repo_path", "")
        self._saved_context["last_spec"] = request.get("spec", "")
        self._save_context()

    def _execute_quick(self, action: str) -> None:
        repo = self.query_one("#repo-path", Input).value.strip()

        if not repo:
            self._set_status("Error: Set repository path first", "error")
            self._log("agent-logs", ["Error: Repository path is required"])
            return

        valid, err = self._validate_repo_path(repo)
        if not valid:
            self._set_status(f"Error: {err}", "error")
            self._log("agent-logs", [f"Error: {err}"])
            return

        self.query_one("#action", Select).value = action
        self._set_phase("validating")
        self._run_workflow()

    def _run_workflow(self, override: dict[str, Any] | None = None) -> None:
        req = override or self._capture_request()
        action = req.get("action", "analyze")
        repo = req.get("repo_path", "")
        spec = req.get("spec", "")

        # Validate inputs
        valid, err = self._validate_repo_path(repo)
        if not valid:
            self._set_status(f"Error: {err}", "error")
            self._log("logs", [f"Error: {err}"])
            return

        if spec:
            valid, err = self._validate_spec(spec)
            if not valid:
                self._set_status(f"Error: {err}", "error")
                self._log("logs", [f"Error: {err}"])
                return

        # Check git if required
        if req.get("integrate_require_clean"):
            dirty, err = self._check_git_status(repo)
            if dirty:
                self._set_status("Warning: Uncommitted changes", "warning")
                self._log("logs", [f"Warning: {err}"])

        # Disable buttons
        for btn_id in [
            "run",
            "rerun-history",
            "view-history",
            "quick-analyze",
            "quick-suggest",
            "quick-integrate",
        ]:
            try:
                self.query_one(f"#{btn_id}", Button).disabled = True
            except Exception:
                pass

        self._set_status(f"Running '{action}'...", "info")
        self._live_logs_enabled = True
        self._active_run_request = req
        self._active_run_started_at = time.perf_counter()

        phase_map = {
            "analyze": "analyzing",
            "suggest": "research",
            "search": "research",
            "map": "mapping",
            "generate": "generating",
            "validate": "validating_patches",
            "integrate": "analyzing",
            "specs": "idle",
        }
        self._set_phase(phase_map.get(action, "analyzing"))

        def _run():
            def _emit(line: str):
                self.post_message(TaskLog(line))

            try:
                if action == "analyze":
                    result = run_analyze(repo, log_callback=_emit)
                elif action == "suggest":
                    result = run_suggest(repo, log_callback=_emit)
                elif action == "search":
                    result = run_search(
                        req.get("query") or "layer normalization",
                        include_arxiv=req.get("include_arxiv", False),
                        include_web=req.get("include_web", False),
                        language=req.get("search_language", "python"),
                        max_results=max(1, int(req.get("max_results_raw", 10))),
                        log_callback=_emit,
                    )
                elif action == "map":
                    result = run_map(repo, spec or "rmsnorm", log_callback=_emit)
                elif action == "generate":
                    result = run_generate(
                        repo,
                        spec or "rmsnorm",
                        output_dir=req.get("output_dir"),
                        log_callback=_emit,
                    )
                elif action == "validate":
                    result = run_validate(repo, log_callback=_emit)
                elif action == "integrate":
                    result = run_integrate(
                        repo,
                        spec or None,
                        dry_run=req.get("integrate_dry_run", False),
                        require_clean=req.get("integrate_require_clean", False),
                        output_dir=req.get("output_dir"),
                        log_callback=_emit,
                    )
                else:
                    result = run_specs(detailed=True, log_callback=_emit)

                self.post_message(
                    TaskCompleted(result.title, result.payload, result.logs, result.error)
                )
            except Exception as e:
                logger.exception("Workflow failed")
                self.post_message(TaskCompleted(action, {}, [], str(e)))

        threading.Thread(target=_run, daemon=True).start()
        self._log("logs", [f"Started: {action} on {repo}"])

    def action_quick_action_analyze(self) -> None:
        self._execute_quick("analyze")

    def action_quick_action_suggest(self) -> None:
        self._execute_quick("suggest")

    def action_quick_action_integrate(self) -> None:
        self._execute_quick("integrate")

    def on_mount(self) -> None:
        self._refresh_action_state()
        self._update_agent_status("Offline")
        self._set_phase("idle")

    def action_run_selected(self) -> None:
        self._run_workflow()

    def action_clear_logs(self) -> None:
        self.query_one("#result", Pretty).update({})
        self.query_one("#logs", TextArea).load_text("")

    @on(Button.Pressed, "#run")
    def on_run(self) -> None:
        self._run_workflow()

    @on(Button.Pressed, "#clear")
    def on_clear(self) -> None:
        self.action_clear_logs()

    @on(Select.Changed, "#action")
    def on_action_change(self) -> None:
        self._refresh_action_state()

    @on(Button.Pressed, "#rerun-history")
    def on_rerun(self) -> None:
        raw = self.query_one("#history-id", Input).value.strip()
        record = self._resolve_history(raw)
        if record:
            self._apply_request(record["request"])
            self._run_workflow(override=record["request"])

    @on(Button.Pressed, "#view-history")
    def on_view(self) -> None:
        raw = self.query_one("#history-id", Input).value.strip()
        record = self._resolve_history(raw)
        if record:
            self._log("logs", [f"Run #{record['id']}: {record['action']} - {record['status']}"])

    @on(Button.Pressed, "#quick-analyze")
    def on_quick_analyze(self) -> None:
        self._execute_quick("analyze")

    @on(Button.Pressed, "#quick-suggest")
    def on_quick_suggest(self) -> None:
        self._execute_quick("suggest")

    @on(Button.Pressed, "#quick-integrate")
    def on_quick_integrate(self) -> None:
        self._execute_quick("integrate")

    @on(TaskCompleted)
    def on_task_done(self, msg: TaskCompleted) -> None:
        self._set_phase("complete")

        payload = {"title": msg.title, "error": msg.error, "result": msg.result}
        self.query_one("#result", Pretty).update(payload)

        if not self._live_logs_enabled:
            self._log("logs", msg.logs)
        self._live_logs_enabled = False

        # Re-enable buttons
        for btn_id in [
            "run",
            "rerun-history",
            "view-history",
            "quick-analyze",
            "quick-suggest",
            "quick-integrate",
        ]:
            try:
                self.query_one(f"#{btn_id}", Button).disabled = False
            except Exception:
                pass

        status = "Done" if msg.error is None else "Failed"
        level = "success" if msg.error is None else "error"
        self._set_status(f"{status} ({msg.title})", level)

        duration = max(0.0, time.perf_counter() - self._active_run_started_at)
        if self._active_run_request:
            self._append_history(
                self._active_run_request.get("action", "unknown"),
                status,
                duration,
                self._active_run_request,
                title=msg.title,
                result=msg.result,
                error=msg.error,
            )

        self._active_run_request = None
        self._active_run_started_at = 0.0

    @on(TaskLog)
    def on_log(self, msg: TaskLog) -> None:
        self._log("logs", [msg.line])

    @on(Button.Pressed, "#launch-agent")
    def on_launch(self) -> None:
        if self._agent_process and self._agent_process.poll() is None:
            self._log("agent-logs", ["Agent already running"])
            return

        project_root = Path(__file__).resolve().parents[4]
        agent_dir = project_root / "agent"

        if not agent_dir.exists():
            self._update_agent_status("Error")
            self._log("agent-logs", [f"Agent directory not found: {agent_dir}"])
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
            self._update_agent_status("Online")
        except Exception as exc:
            self._update_agent_status("Error")
            self._log("agent-logs", [f"Failed to launch: {exc}"])
            return

        self._log(
            "agent-logs",
            [
                "Agent launched in REPL mode",
                "Commands: analyze, suggest, integrate, search...",
                "Type 'help' for more, 'exit' to quit",
            ],
        )

        def _read():
            if self._agent_process and self._agent_process.stdout:
                for line in self._agent_process.stdout:
                    if line.strip():
                        self.post_message(AgentLog(line.rstrip()))

        threading.Thread(target=_read, daemon=True).start()

    @on(AgentLog)
    def on_agent_log(self, msg: AgentLog) -> None:
        self._log("agent-logs", [msg.line])
        line = msg.line.lower()
        if "analyzing" in line:
            self._set_phase("analyzing")
        elif "research" in line or "searching" in line:
            self._set_phase("research")
        elif "mapping" in line:
            self._set_phase("mapping")
        elif "generating" in line:
            self._set_phase("generating")
        elif "validating" in line:
            self._set_phase("validating_patches")
        elif "complete" in line or "done" in line:
            self._set_phase("complete")

    @on(Button.Pressed, "#stop-agent")
    def on_stop(self) -> None:
        if not self._agent_process or self._agent_process.poll() is not None:
            self._log("agent-logs", ["No running agent"])
            self._update_agent_status("Offline")
            return
        self._agent_running = False
        self._agent_process.terminate()
        self._update_agent_status("Offline")
        self._log("agent-logs", ["Agent stopped"])

    @on(Input.Submitted, "#prompt-input")
    def on_prompt(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return

        self._command_history.append(prompt)
        self._history_index = len(self._command_history)
        event.input.value = ""

        if prompt.lower() in ("help", "?"):
            self._log(
                "agent-logs",
                [
                    "Commands: analyze, suggest, integrate, search, map, generate, validate",
                    "Natural language: 'apply rmsnorm to /path'",
                    "Type 'exit' to quit",
                ],
            )
            return

        if not self._agent_running or not self._agent_stdin:
            cmd, ctx = self._parse_natural_command(prompt)
            self._log("agent-logs", [f"User: {prompt}"])

            if ctx.get("repo_path"):
                self.query_one("#repo-path", Input).value = ctx["repo_path"]
            if ctx.get("spec"):
                self.query_one("#spec", Input).value = ctx["spec"]

            self.query_one("#action", Select).value = cmd
            self._set_phase("validating")
            self._run_workflow()
            return

        if prompt.lower() in ("exit", "quit"):
            self.on_stop()
            return

        self._log("agent-logs", [f"User: {prompt}"])

        try:
            self._agent_stdin.write(prompt + "\n")
            self._agent_stdin.flush()
        except Exception as exc:
            self._log("agent-logs", [f"Error: {exc}"])

    def on_unmount(self) -> None:
        if self._agent_process and self._agent_process.poll() is None:
            self._agent_process.terminate()

    def action_handle_escape(self) -> None:
        t = time.time()
        if t - self._last_escape_time > 2.0:
            self._escape_pressed_count = 0
            self._escape_warning_shown = False
        self._last_escape_time = t
        self._escape_pressed_count += 1

        if self._escape_pressed_count == 1:
            self._show_warning("Press ESC again to stop agent...")
            self._escape_warning_shown = True
        elif self._escape_pressed_count >= 2 and self._escape_warning_shown:
            self._hide_warning()
            self.on_stop()
            self._escape_pressed_count = 0
            self._escape_warning_shown = False

    def _show_warning(self, msg: str) -> None:
        try:
            bar = self.query_one("#warning-bar", Label)
            bar.update(msg)
            bar.add_class("visible")
        except Exception:
            pass

    def _hide_warning(self) -> None:
        try:
            bar = self.query_one("#warning-bar", Label)
            bar.update("")
            bar.remove_class("visible")
        except Exception:
            pass


def run_tui() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app = ScholarDevClawApp()
    app.run()


if __name__ == "__main__":
    run_tui()
