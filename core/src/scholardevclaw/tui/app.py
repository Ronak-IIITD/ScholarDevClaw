"""ScholarDevClaw TUI — enhanced terminal interface.

Modern terminal UI inspired by Claude Code and OpenCode, built on Textual.
Features: sidebar navigation, chat-style logs, command palette, phase
progress tracker, help overlay, run history, and agent integration.
"""

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
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    TextArea,
)

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

from .screens import CommandPalette, HelpOverlay, WelcomeScreen
from .widgets import (
    AgentStatus,
    HistoryPane,
    LogView,
    PhaseTracker,
    Sidebar,
    StatusBar,
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
    "show_welcome": True,
}


# ---------------------------------------------------------------------------
# Internal messages
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


class ScholarDevClawApp(App[None]):
    TITLE = "ScholarDevClaw"
    SUB_TITLE = "Research-to-Code Agent"

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

    # -----------------------------------------------------------------------
    # CSS Theme — Modern Dark with gradient accents
    # -----------------------------------------------------------------------

    CSS = """
    /* ---- Color tokens ---- */

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
    $success: #3fb950;
    $error: #f85149;
    $warning: #d29922;
    $text-inverse: #ffffff;
    $header: #161b22;

    /* ---- Base ---- */
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
        text-align: center;
    }

    Footer {
        background: $header;
        color: $text-muted;
        dock: bottom;
        height: 1;
    }

    /* ---- Top-level layout ---- */

    #app-body {
        width: 100%;
        height: 1fr;
        layout: horizontal;
    }

    #content-area {
        width: 1fr;
        height: 100%;
        layout: vertical;
    }

    /* ---- Main content split ---- */

    #main-split {
        width: 100%;
        height: 1fr;
        layout: horizontal;
    }

    /* ---- Left: configuration panel ---- */

    #config-panel {
        width: 32;
        min-width: 28;
        max-width: 38;
        height: 100%;
        background: $panel;
        border-right: tall $border;
        padding: 1;
        overflow-y: auto;
    }

    #config-panel .panel-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
        width: 100%;
        text-align: center;
        height: 1;
    }

    #config-panel .field-label {
        color: $text-muted;
        margin-top: 1;
        height: 1;
    }

    #config-panel .spacer {
        height: 1;
    }

    /* ---- Right: output panel ---- */

    #output-panel {
        width: 1fr;
        height: 100%;
        background: $surface;
        layout: vertical;
    }

    #output-panel .output-header {
        width: 100%;
        height: 1;
        padding: 0 1;
        background: $surface-dark;
        border-bottom: solid $border;
        color: $accent;
        text-style: bold;
    }

    /* ---- Bottom: agent / prompt bar ---- */

    #agent-section {
        width: 100%;
        height: 25%;
        min-height: 8;
        max-height: 35%;
        dock: bottom;
        background: $panel;
        border-top: tall $border;
        layout: vertical;
    }

    #agent-header {
        width: 100%;
        height: 1;
        padding: 0 1;
        background: $surface-dark;
        border-bottom: solid $border;
        layout: horizontal;
    }

    #agent-header .agent-title {
        width: 1fr;
        color: $accent;
        text-style: bold;
    }

    #agent-controls {
        width: auto;
        layout: horizontal;
        height: 1;
    }

    #agent-logs {
        width: 100%;
        height: 1fr;
        background: $surface;
    }

    #prompt-bar {
        width: 100%;
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

    /* ---- Shared widgets ---- */

    .panel-section-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
        margin-top: 1;
    }

    Input {
        background: $surface-dark;
        color: $text;
        border: solid $border;
        border-radius: 6;
        padding: 0 1;
    }

    Input:focus {
        border: solid $accent;
    }

    Input::placeholder {
        color: $text-muted;
    }

    Input.invalid {
        border: solid $error;
    }

    Select {
        background: $surface-dark;
        color: $text;
        border: solid $border;
        border-radius: 6;
    }

    Select:focus {
        border: solid $accent;
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
        border-radius: 6;
        padding: 0 1;
        min-width: 10;
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

    Label {
        color: $text-muted;
    }

    Pretty {
        background: transparent;
        color: $text;
    }

    TextArea {
        background: $surface-dark;
        color: $text-muted;
        border: none;
        border-radius: 6;
    }

    TextArea:focus {
        border: 1px solid $border;
    }

    /* ---- Status / result classes ---- */

    .status-error { color: $error; }
    .status-success { color: $success; }
    .status-warning { color: $warning; }
    .status-info { color: $text-muted; }
    """

    # -----------------------------------------------------------------------
    # Key bindings
    # -----------------------------------------------------------------------

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+r", "run_selected", "Run"),
        ("ctrl+k", "command_palette", "Commands"),
        ("ctrl+question_mark", "help", "Help"),
        ("escape", "handle_escape", "Stop/Back"),
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

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Config / context persistence
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------------

    def _validate_repo_path(self, path: str) -> tuple[bool, str]:
        if not path:
            return False, "Repository path is required"
        p = Path(path).expanduser()
        if not p.exists():
            return False, f"Path does not exist: {path}"
        if not p.is_dir():
            return False, f"Path is not a directory: {path}"
        return True, ""

    def _validate_spec(self, spec: str) -> tuple[bool, str]:
        if not spec:
            return True, ""
        if spec.lower() not in self.AVAILABLE_SPECS:
            return False, f"Unknown spec: {spec}. Valid: {', '.join(self.AVAILABLE_SPECS)}"
        return True, ""

    def _check_git_status(self, path: str) -> tuple[bool, str]:
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
            return False, ""

    # -----------------------------------------------------------------------
    # UI state helpers
    # -----------------------------------------------------------------------

    def _set_phase(self, phase: str) -> None:
        self._current_phase = phase
        try:
            tracker = self.query_one(PhaseTracker)
            tracker.set_phase(phase)
        except Exception:
            pass

    def _set_status(self, message: str, level: str = "info") -> None:
        try:
            status_bar = self.query_one(StatusBar)
            status_bar.set_status(message, level)
        except Exception:
            pass

    def _update_agent_status(self, status: str) -> None:
        try:
            agent_dot = self.query_one(AgentStatus)
            agent_dot.set_status(status)
        except Exception:
            pass

    def _log_to_view(self, lines: list[str]) -> None:
        """Append lines to the LogView widget."""
        try:
            log_view = self.query_one(LogView)
            log_view.add_logs(lines)
        except Exception:
            pass

    def _log_to_legacy(self, widget_id: str, lines: list[str]) -> None:
        """Append lines to a TextArea widget (agent-logs)."""
        try:
            area = self.query_one(f"#{widget_id}", TextArea)
            current = area.text
            merged = (current + "\n" if current else "") + "\n".join(lines)
            area.load_text(merged)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Natural language command parsing
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Compose — layout definition
    # -----------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="app-body"):
            # Left sidebar
            yield Sidebar(id="sidebar")

            # Center content
            with Vertical(id="content-area"):
                # Phase tracker
                yield PhaseTracker(id="phase-tracker")

                # Main split: config + output
                with Horizontal(id="main-split"):
                    # Left: configuration panel
                    with VerticalScroll(id="config-panel"):
                        yield Label("Workflow", classes="panel-title")
                        yield Select(
                            self.action_mode_options,
                            value=self._saved_context.get(
                                "last_action",
                                self._config.get("default_action", "analyze"),
                            ),
                            id="action",
                        )

                        yield Label("Repository Path", classes="field-label")
                        yield Input(
                            value=self._saved_context.get(
                                "last_repo",
                                self._config.get("default_repo", str(Path.cwd())),
                            ),
                            placeholder="/path/to/repository",
                            id="repo-path",
                        )

                        yield Label("Search Query", classes="field-label")
                        yield Input(
                            value=self._saved_context.get("last_query", "layer normalization"),
                            placeholder="Search query",
                            id="query",
                        )

                        with Horizontal(classes="spacer"):
                            yield Checkbox("arXiv", value=False, id="search-arxiv")
                            yield Checkbox("Web", value=False, id="search-web")

                        yield Label("Language", classes="field-label")
                        yield Input(
                            value=self._saved_context.get(
                                "last_language",
                                self._config.get("default_language", "python"),
                            ),
                            placeholder="Language",
                            id="search-language",
                        )

                        yield Label("Max Results", classes="field-label")
                        yield Input(
                            value=str(self._config.get("max_results", 10)),
                            placeholder="Max",
                            id="search-max-results",
                        )

                        yield Label("Spec Name", classes="field-label")
                        yield Input(
                            value=self._saved_context.get(
                                "last_spec",
                                self._config.get("default_spec", "rmsnorm"),
                            ),
                            placeholder="Spec name",
                            id="spec",
                        )

                        yield Label("Output Dir", classes="field-label")
                        yield Input(
                            value="",
                            placeholder="Output dir (optional)",
                            id="output-dir",
                        )

                        with Horizontal(classes="spacer"):
                            yield Checkbox("Dry-run", value=False, id="integrate-dry-run")
                            yield Checkbox(
                                "Clean git",
                                value=self._config.get("require_clean_git", False),
                                id="integrate-require-clean",
                            )

                        with Horizontal(classes="spacer"):
                            yield Button("Run", id="run", variant="primary")
                            yield Button("Clear", id="clear")

                    # Right: output panel
                    with Vertical(id="output-panel"):
                        yield Label("Output", classes="output-header")
                        yield LogView(id="log-view")

                        # Result display (hidden until there is a result)
                        yield Static("", id="result-placeholder")

                        # History
                        yield Label("  History", classes="panel-section-title")
                        yield HistoryPane(id="history-pane")
                        with Horizontal(classes="spacer"):
                            yield Input(value="", placeholder="#", id="history-id")
                            yield Button("Rerun", id="rerun-history")
                            yield Button("View", id="view-history")

        # Agent section at the bottom
        with Vertical(id="agent-section"):
            with Horizontal(id="agent-header"):
                yield Label("Agent Mode", classes="agent-title")
                with Horizontal(id="agent-controls"):
                    yield Button("Launch", id="launch-agent", variant="primary")
                    yield Button("Stop", id="stop-agent", variant="error")
                    yield AgentStatus(id="agent-status")
            yield TextArea("", id="agent-logs", read_only=True)

        # Prompt bar
        with Horizontal(id="prompt-bar"):
            yield Input(
                value="",
                placeholder="> Type request... (Ctrl+K for commands, Ctrl+? for help)",
                id="prompt-input",
            )

        yield Footer()

    # -----------------------------------------------------------------------
    # Mount / unmount
    # -----------------------------------------------------------------------

    def on_mount(self) -> None:
        self._refresh_action_state()
        self._update_agent_status("Offline")
        self._set_phase("idle")

        # Show welcome on first launch
        if self._config.get("show_welcome", True):
            self.call_later(self._maybe_show_welcome)

    def _maybe_show_welcome(self) -> None:
        marker = self._config_dir / ".welcome_seen"
        if not marker.exists():
            self.push_screen(WelcomeScreen())
            try:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text("")
            except Exception:
                pass

    def on_unmount(self) -> None:
        if self._agent_process and self._agent_process.poll() is None:
            self._agent_process.terminate()

    # -----------------------------------------------------------------------
    # Request capture / apply
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Action state management
    # -----------------------------------------------------------------------

    def _refresh_action_state(self) -> None:
        action = self.query_one("#action", Select).value
        is_search = action == "search"
        needs_spec = action in {"map", "generate", "integrate"}
        supports_output_dir = action == "generate"
        is_integrate = action == "integrate"

        for field_id, widget_cls in [
            ("query", Input),
            ("search-arxiv", Checkbox),
            ("search-web", Checkbox),
            ("search-language", Input),
            ("search-max-results", Input),
        ]:
            try:
                self.query_one(f"#{field_id}", widget_cls).disabled = not is_search
            except Exception:
                pass

        try:
            self.query_one("#spec", Input).disabled = not needs_spec
            self.query_one("#output-dir", Input).disabled = not supports_output_dir
            self.query_one("#integrate-dry-run", Checkbox).disabled = not is_integrate
            self.query_one("#integrate-require-clean", Checkbox).disabled = not is_integrate
        except Exception:
            pass

        # Sync sidebar selection
        try:
            sidebar = self.query_one(Sidebar)
            if isinstance(action, str):
                sidebar.set_selected(action)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # History management
    # -----------------------------------------------------------------------

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

        try:
            history_pane = self.query_one(HistoryPane)
            history_pane.add_entry(record["id"], action, status, duration)
            self.query_one("#history-id", Input).value = str(record["id"])
        except Exception:
            pass

        self._saved_context["last_action"] = action
        self._saved_context["last_repo"] = request.get("repo_path", "")
        self._saved_context["last_spec"] = request.get("spec", "")
        self._save_context()

    def _resolve_history(self, raw: str) -> dict[str, Any] | None:
        if not self._run_history:
            return None
        if not raw:
            return self._run_history[0]
        try:
            return next((r for r in self._run_history if r["id"] == int(raw)), None)
        except ValueError:
            return None

    # -----------------------------------------------------------------------
    # Button states
    # -----------------------------------------------------------------------

    def _disable_run_buttons(self) -> None:
        for btn_id in [
            "run",
            "rerun-history",
            "view-history",
        ]:
            try:
                self.query_one(f"#{btn_id}", Button).disabled = True
            except Exception:
                pass

    def _enable_run_buttons(self) -> None:
        for btn_id in [
            "run",
            "rerun-history",
            "view-history",
        ]:
            try:
                self.query_one(f"#{btn_id}", Button).disabled = False
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Quick actions
    # -----------------------------------------------------------------------

    def _execute_quick(self, action: str) -> None:
        repo = self.query_one("#repo-path", Input).value.strip()

        if not repo:
            self._set_status("Error: Set repository path first", "error")
            self._log_to_view(["Error: Repository path is required"])
            return

        valid, err = self._validate_repo_path(repo)
        if not valid:
            self._set_status(f"Error: {err}", "error")
            self._log_to_view([f"Error: {err}"])
            return

        self.query_one("#action", Select).value = action
        self._set_phase("validating")
        self._run_workflow()

    # -----------------------------------------------------------------------
    # Core workflow execution
    # -----------------------------------------------------------------------

    def _run_workflow(self, override: dict[str, Any] | None = None) -> None:
        req = override or self._capture_request()
        action = req.get("action", "analyze")
        repo = req.get("repo_path", "")
        spec = req.get("spec", "")

        # Validate
        valid, err = self._validate_repo_path(repo)
        if not valid:
            self._set_status(f"Error: {err}", "error")
            self._log_to_view([f"Error: {err}"])
            return

        if spec:
            valid, err = self._validate_spec(spec)
            if not valid:
                self._set_status(f"Error: {err}", "error")
                self._log_to_view([f"Error: {err}"])
                return

        # Git check
        if req.get("integrate_require_clean"):
            dirty, err = self._check_git_status(repo)
            if dirty:
                self._set_status("Warning: Uncommitted changes", "warning")
                self._log_to_view([f"Warning: {err}"])

        self._disable_run_buttons()
        self._set_status(f"Running '{action}'...", "info")

        try:
            status_bar = self.query_one(StatusBar)
            status_bar.start_timer()
        except Exception:
            pass

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
        self._log_to_view([f"Started: {action} on {repo}"])

    # -----------------------------------------------------------------------
    # Event handlers
    # -----------------------------------------------------------------------

    @on(Sidebar.ActionSelected)
    def on_sidebar_action(self, msg: Sidebar.ActionSelected) -> None:
        action = msg.action
        if action in (
            "analyze",
            "suggest",
            "search",
            "specs",
            "map",
            "generate",
            "validate",
            "integrate",
        ):
            self.query_one("#action", Select).value = action
            self._refresh_action_state()
        elif action == "quick-analyze":
            self._execute_quick("analyze")
        elif action == "quick-suggest":
            self._execute_quick("suggest")
        elif action == "quick-integrate":
            self._execute_quick("integrate")

    @on(Select.Changed, "#action")
    def on_action_change(self) -> None:
        self._refresh_action_state()

    @on(Button.Pressed, "#run")
    def on_run(self) -> None:
        self._run_workflow()

    @on(Button.Pressed, "#clear")
    def on_clear(self) -> None:
        self.action_clear_logs()

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
            self._log_to_view([f"Run #{record['id']}: {record['action']} - {record['status']}"])

    @on(Button.Pressed, "#launch-agent")
    def on_launch(self) -> None:
        if self._agent_process and self._agent_process.poll() is None:
            self._log_to_legacy("agent-logs", ["Agent already running"])
            return

        project_root = Path(__file__).resolve().parents[4]
        agent_dir = project_root / "agent"

        if not agent_dir.exists():
            self._update_agent_status("Error")
            self._log_to_legacy("agent-logs", [f"Agent directory not found: {agent_dir}"])
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
            self._log_to_legacy("agent-logs", [f"Failed to launch: {exc}"])
            return

        self._log_to_legacy(
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
        self._log_to_legacy("agent-logs", [msg.line])
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
        if not self._agent_process or self._agent_process.poll() is None:
            self._log_to_legacy("agent-logs", ["No running agent"])
            self._update_agent_status("Offline")
            return
        self._agent_running = False
        self._agent_process.terminate()
        self._update_agent_status("Offline")
        self._log_to_legacy("agent-logs", ["Agent stopped"])

    @on(TaskCompleted)
    def on_task_done(self, msg: TaskCompleted) -> None:
        self._set_phase("complete")
        self._enable_run_buttons()

        # Show result in log view
        if msg.error:
            self._log_to_view([f"Error: {msg.error}"])
            self._set_status(f"Failed ({msg.title})", "error")
        else:
            # Log a compact result summary
            result_summary = json.dumps(msg.result, indent=2, default=str)
            # Truncate very long results
            if len(result_summary) > 2000:
                result_summary = result_summary[:2000] + "\n..."
            self._log_to_view([f"Complete: {msg.title}", result_summary])
            self._set_status(f"Done ({msg.title})", "success")

        if not self._live_logs_enabled:
            self._log_to_view(msg.logs)
        self._live_logs_enabled = False

        # Update timer
        try:
            status_bar = self.query_one(StatusBar)
            status_bar.update_timer()
        except Exception:
            pass

        # Record history
        duration = max(0.0, time.perf_counter() - self._active_run_started_at)
        if self._active_run_request:
            self._append_history(
                self._active_run_request.get("action", "unknown"),
                "Done" if msg.error is None else "Failed",
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
        self._log_to_view([msg.line])

        # Update timer during execution
        try:
            status_bar = self.query_one(StatusBar)
            status_bar.update_timer()
        except Exception:
            pass

    @on(Input.Submitted, "#prompt-input")
    def on_prompt(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return

        self._command_history.append(prompt)
        self._history_index = len(self._command_history)
        event.input.value = ""

        if prompt.lower() in ("help", "?"):
            self._log_to_legacy(
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
            self._log_to_legacy("agent-logs", [f"User: {prompt}"])

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

        self._log_to_legacy("agent-logs", [f"User: {prompt}"])

        try:
            self._agent_stdin.write(prompt + "\n")
            self._agent_stdin.flush()
        except Exception as exc:
            self._log_to_legacy("agent-logs", [f"Error: {exc}"])

    # -----------------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------------

    def action_run_selected(self) -> None:
        self._run_workflow()

    def action_clear_logs(self) -> None:
        try:
            log_view = self.query_one(LogView)
            log_view.clear_logs()
        except Exception:
            pass

    def action_quick_action_analyze(self) -> None:
        self._execute_quick("analyze")

    def action_quick_action_suggest(self) -> None:
        self._execute_quick("suggest")

    def action_quick_action_integrate(self) -> None:
        self._execute_quick("integrate")

    def action_command_palette(self) -> None:
        """Open the command palette overlay."""

        def handle_result(result: str | None) -> None:
            if result is None:
                return
            if result == "quit":
                self.exit()
            elif result == "clear":
                self.action_clear_logs()
            elif result in (
                "analyze",
                "suggest",
                "search",
                "specs",
                "map",
                "generate",
                "validate",
                "integrate",
            ):
                self.query_one("#action", Select).value = result
                self._refresh_action_state()

        self.push_screen(CommandPalette(), handle_result)

    def action_help(self) -> None:
        """Show keyboard shortcuts overlay."""
        self.push_screen(HelpOverlay())

    def action_handle_escape(self) -> None:
        """Handle Esc key: first press warns, second stops agent."""
        t = time.time()
        if t - self._last_escape_time > 2.0:
            self._escape_pressed_count = 0
            self._escape_warning_shown = False
        self._last_escape_time = t
        self._escape_pressed_count += 1

        if self._escape_pressed_count == 1:
            self._set_status("Press ESC again to stop agent...", "warning")
            self._escape_warning_shown = True
        elif self._escape_pressed_count >= 2 and self._escape_warning_shown:
            self._set_status("Ready", "info")
            self.on_stop()
            self._escape_pressed_count = 0
            self._escape_warning_shown = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_tui() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app = ScholarDevClawApp()
    app.run()


if __name__ == "__main__":
    run_tui()
