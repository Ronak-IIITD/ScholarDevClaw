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


# ANSI colors for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    # Background
    BG_BLACK = "\033[40m"
    BG_BLUE = "\033[44m"
    BG_CYAN = "\033[46m"
    BG_MAGENTA = "\033[45m"


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


class LoadingComplete(Message):
    pass


class ScholarDevClawApp(App[None]):
    TITLE = "ScholarDevClaw"
    SUB_TITLE = "Research → Code Assistant"

    PHASES = [
        ("idle", "💤 Ready", 0),
        ("analyzing", "🔍 Analyzing repository...", 0.15),
        ("research", "📚 Fetching research papers...", 0.3),
        ("mapping", "🗺 Mapping code patterns...", 0.5),
        ("generating", "⚡ Generating patches...", 0.7),
        ("validating", "✅ Running validation...", 0.85),
        ("complete", "✨ Complete!", 1.0),
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
    /* ========================================
       PREMIUM DARK THEME V3
       Enhanced with Animations + Glows
       ======================================== */

    Screen {
        layout: vertical;
        background: #030712;
        color: #e2e8f0;
    }

    Header {
        background: linear-gradient(90deg, #0c1929 0%, #1e1b4b 50%, #0c1929 100%);
        color: #f1f5f9;
        dock: top;
        height: 3;
    }

    Header > .header--clock {
        color: #22d3ee;
        text-style: bold;
    }

    Footer {
        background: #0c1929;
        color: #475569;
        dock: bottom;
        height: 1;
    }

    #main-container {
        height: 100%;
        padding: 1 2;
        background: #030712;
    }

    .glass-panel {
        background: rgba(12, 25, 41, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.1);
        backdrop-blur: 20px;
        border-radius: 20px;
        padding: 1;
    }

    .section-title {
        text-style: bold;
        color: #22d3ee;
        text-shadow: 0 0 25px rgba(34, 211, 238, 0.5);
        margin-bottom: 1;
    }

    .section-subtitle {
        color: #475569;
        margin-bottom: 1;
    }

    /* Panels */
    #wizard-panel {
        width: 32%;
        border: solid rgba(59, 130, 246, 0.25);
        background: rgba(12, 25, 41, 0.6);
        border-radius: 20px;
        padding: 1 1;
        margin-right: 1;
    }

    #output-panel {
        width: 68%;
        border: solid rgba(139, 92, 246, 0.25);
        background: rgba(12, 25, 41, 0.6);
        border-radius: 20px;
        padding: 1 1;
    }

    #agent-panel {
        height: 30%;
        dock: bottom;
        border: solid rgba(34, 211, 238, 0.35);
        background: rgba(8, 47, 73, 0.35);
        border-radius: 20px 20px 0 0;
        margin: 0 2;
        padding: 1;
    }

    /* Animated progress bar */
    #phase-container {
        dock: top;
        height: 3;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 10px;
        margin-bottom: 1;
    }

    #phase-progress {
        width: 0%;
        height: 100%;
        background: linear-gradient(90deg, #06b6d4, #3b82f6, #8b5cf6, #06b6d4);
        background-size: 300% 100%;
        border-radius: 10px;
    }

    #phase-progress.animated {
        animation: gradient-shift 2s ease infinite;
    }

    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    #phase-label {
        dock: top;
        height: 1;
        color: #22d3ee;
        text-style: bold;
    }

    /* Loading spinner */
    #spinner {
        dock: top;
        height: 1;
        color: #22d3ee;
        display: none;
    }

    #spinner.visible {
        display: block;
    }

    /* Result with syntax highlighting */
    #result {
        height: 10;
        border: none;
        background: rgba(3, 7, 18, 0.95);
        border-radius: 16px;
        padding: 0 1;
    }

    #logs {
        height: 7;
        border: none;
        background: rgba(3, 7, 18, 0.95);
        border-radius: 16px;
        margin-top: 1;
    }

    #history {
        height: 4;
        border: none;
        background: rgba(3, 7, 18, 0.95);
        border-radius: 16px;
        margin-top: 1;
    }

    #agent-logs {
        height: 1fr;
        border: none;
        background: rgba(3, 7, 18, 0.98);
        border-radius: 16px;
    }

    /* Prompt bar */
    #prompt-bar {
        dock: bottom;
        height: 3;
        background: rgba(12, 25, 41, 0.98);
        border-top: 1px solid rgba(34, 211, 238, 0.15);
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
        color: #1e3a5f;
    }

    /* Quick actions */
    #quick-actions {
        dock: top;
        height: 3;
        background: rgba(12, 25, 41, 0.5);
        padding: 0 1;
    }

    /* Inputs */
    Input {
        background: rgba(3, 7, 18, 0.85);
        color: #e2e8f0;
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 14px;
        padding: 0 1;
    }

    Input:focus {
        border: 1px solid #22d3ee;
        box-shadow: 0 0 25px rgba(34, 211, 238, 0.3);
    }

    Input::placeholder {
        color: #1e3a5f;
    }

    Select {
        background: rgba(3, 7, 18, 0.85);
        color: #e2e8f0;
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 14px;
    }

    Select:focus {
        border: 1px solid #22d3ee;
    }

    Checkbox {
        color: #475569;
    }

    Checkbox:focus {
        color: #22d3ee;
    }

    /* Buttons */
    Button {
        border: none;
        background: rgba(59, 130, 246, 0.12);
        color: #60a5fa;
        border-radius: 14px;
        padding: 0 1;
        min-width: 12;
    }

    Button:hover {
        background: rgba(59, 130, 246, 0.25);
        border: 1px solid #3b82f6;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.35);
    }

    Button.-primary {
        background: linear-gradient(135deg, #0891b2 0%, #3b82f6 100%);
        color: #ffffff;
        text-style: bold;
        border: none;
    }

    Button.-primary:hover {
        box-shadow: 0 0 30px rgba(14, 165, 233, 0.6);
    }

    Button.-success {
        background: linear-gradient(135deg, #047857 0%, #10b981 100%);
        color: #ffffff;
        text-style: bold;
    }

    Button.-success:hover {
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.6);
    }

    Button.-error {
        background: linear-gradient(135deg, #b91c1c 0%, #ef4444 100%);
        color: #ffffff;
        text-style: bold;
    }

    Button.-error:hover {
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.6);
    }

    Button:disabled {
        opacity: 0.25;
    }

    .quick-btn {
        background: rgba(139, 92, 246, 0.12);
        color: #a78bfa;
        border-radius: 12px;
        padding: 0 1;
        min-width: 10;
    }

    .quick-btn:hover {
        background: rgba(139, 92, 246, 0.3);
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.45);
    }

    Label {
        color: #475569;
    }

    #run-status {
        dock: top;
        height: 1;
        background: rgba(12, 25, 41, 0.9);
        color: #22d3ee;
        text-style: bold;
        padding: 0 1;
        border-radius: 8px;
    }

    .spaced { margin-top: 1; }
    .spaced-small { margin-top: 0; }

    Pretty {
        background: transparent;
        color: #e2e8f0;
    }

    TextArea {
        background: rgba(3, 7, 18, 0.9);
        color: #64748b;
        border: none;
        border-radius: 16px;
    }

    TextArea:focus {
        border: 1px solid rgba(34, 211, 238, 0.15);
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
        ("ctrl+e", "run_selected", "Run"),
        ("escape", "handle_escape", "Stop"),
        ("ctrl+l", "clear_logs", "Clear"),
        ("ctrl+a", "quick_action_analyze", "Analyze"),
        ("ctrl+s", "quick_action_suggest", "Suggest"),
        ("ctrl+i", "quick_action_integrate", "Integrate"),
        ("ctrl+h", "show_help", "Help"),
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
        """Load saved context from file."""
        try:
            if self._context_file.exists():
                self._saved_context = json.loads(self._context_file.read_text())
        except Exception:
            self._saved_context = {}

    def _save_context(self) -> None:
        """Save context to file."""
        try:
            self._context_file.parent.mkdir(parents=True, exist_ok=True)
            self._context_file.write_text(json.dumps(self._saved_context, indent=2))
        except Exception:
            pass

    def _update_saved_context(self, key: str, value: Any) -> None:
        """Update and persist context."""
        self._saved_context[key] = value
        self._save_context()

    def _get_completions(self, partial: str) -> list[str]:
        """Get tab completions for input."""
        completions = []
        partial_lower = partial.lower()

        # Common commands
        commands = [
            "analyze",
            "suggest",
            "integrate",
            "search",
            "map",
            "generate",
            "validate",
            "specs",
            "set repo",
            "set spec",
            "help",
            "exit",
            "context",
            "clear",
        ]
        completions.extend([c for c in commands if c.startswith(partial_lower)])

        # Spec names
        completions.extend([s for s in self.AVAILABLE_SPECS if s.startswith(partial_lower)])

        # Paths (simple)
        if partial.startswith("/") or partial.startswith("."):
            try:
                base = Path(partial).parent if "/" in partial else Path(".")
                if base.exists():
                    for p in base.iterdir():
                        name = p.name
                        if name.startswith(Path(partial).name):
                            completions.append(str(p))
            except Exception:
                pass

        return completions[:10]

    def _parse_natural_command(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """Parse natural language into structured command."""
        prompt_lower = prompt.strip().lower()
        ctx: dict[str, Any] = {}
        command = "help"

        # Extract repo path
        path_match = re.search(r"(?:to|on|in|at|for)\s+([/\w~.][^\s]+)", prompt)
        if path_match:
            ctx["repo_path"] = path_match.group(1)

        # Extract spec
        for spec in self.AVAILABLE_SPECS:
            if spec in prompt_lower:
                ctx["spec"] = spec
                break

        # Detect intent
        if any(kw in prompt_lower for kw in ["analyze", "scan", "inspect", "examine"]):
            command = "analyze"
        elif any(kw in prompt_lower for kw in ["suggest", "recommend", "improvement", "ideas"]):
            command = "suggest"
        elif any(kw in prompt_lower for kw in ["integrate", "apply", "implement", "add to"]):
            command = "integrate"
        elif any(kw in prompt_lower for kw in ["search", "find", "look for"]):
            command = "search"
            q_match = re.search(r'search(?:ing)?\s+(?:for\s+)?["\']?(.+)', prompt_lower)
            if q_match:
                ctx["query"] = q_match.group(1).strip().strip("\"'")
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
        """Update phase with animation."""
        self._current_phase = phase
        for phase_name, label, progress in self.PHASES:
            if phase_name == phase:
                try:
                    bar = self.query_one("#phase-progress")
                    bar.styles.width = f"{int(progress * 100)}%"
                    bar.add_class("animated") if progress > 0 else bar.remove_class("animated")

                    label_widget = self.query_one("#phase-label")
                    label_widget.update(label)
                except Exception:
                    pass
                break

    def _format_rich_output(self, data: dict[str, Any]) -> str:
        """Format output with colors and structure."""
        lines = []

        # Title
        title = data.get("title", "")
        if title:
            lines.append(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}▶ {title}{Colors.RESET}")
            lines.append("")

        # Status
        error = data.get("error")
        if error:
            lines.append(f"{Colors.BRIGHT_RED}✗ Error: {error}{Colors.RESET}")
        elif data.get("result"):
            lines.append(f"{Colors.BRIGHT_GREEN}✓ Success{Colors.RESET}")

        # Key info
        result = data.get("result", {})
        if isinstance(result, dict):
            # Languages
            if "languages" in result:
                langs = result["languages"]
                lines.append(
                    f"{Colors.CYAN}Languages:{Colors.RESET} {', '.join(langs) if isinstance(langs, list) else langs}"
                )

            # Specs
            if "spec" in result:
                lines.append(f"{Colors.MAGENTA}Spec:{Colors.RESET} {result['spec']}")

            # Suggestions count
            if "suggestions" in result:
                count = len(result["suggestions"])
                lines.append(f"{Colors.YELLOW}Suggestions:{Colors.RESET} {count} found")

            # Written files
            gen = result.get("generation") or result
            if "written_files" in gen:
                files = gen["written_files"]
                lines.append(f"{Colors.GREEN}Files written:{Colors.RESET}")
                for f in (files if isinstance(files, list) else [])[:5]:
                    lines.append(f"  • {f}")

            # Validation
            if "scorecard" in result:
                sc = result["scorecard"]
                summary = sc.get("summary", "unknown")
                status_color = Colors.BRIGHT_GREEN if summary == "pass" else Colors.BRIGHT_YELLOW
                lines.append(f"{status_color}Validation:{Colors.RESET} {summary}")

        return "\n".join(lines) if lines else json.dumps(data, indent=2)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # Phase progress
        yield Label("💤 Ready", id="phase-label")
        with Horizontal(id="phase-container"):
            yield Label("", id="phase-progress")

        with Horizontal(id="main-container"):
            # Left Panel
            with Vertical(id="wizard-panel", classes="glass-panel"):
                yield Label("⚡ Workflow", classes="section-title")
                yield Label("Pipeline configuration", classes="section-subtitle")
                yield Select(
                    self.action_mode_options,
                    value=self._saved_context.get("last_action", "analyze"),
                    id="action",
                )
                yield Input(
                    value=self._saved_context.get("last_repo", str(Path.cwd())),
                    placeholder="/path/to/repo",
                    id="repo-path",
                )
                yield Input(
                    value=self._saved_context.get("last_query", "layer normalization"),
                    placeholder="Search query...",
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
                yield Label("Status: Ready", id="run-status")
                with Horizontal(classes="spaced"):
                    yield Button("▶ Run", id="run", variant="success")
                    yield Button("↺", id="clear", variant="default", tooltip="Clear")

            # Right Panel
            with Vertical(id="output-panel", classes="glass-panel"):
                yield Label("📊 Results", classes="section-title")
                yield Pretty({}, id="result")
                yield Label("Logs", classes="section-title")
                yield TextArea("", id="logs", read_only=True)
                yield Label("History", classes="section-title")
                yield TextArea("No runs yet.", id="history", read_only=True)
                with Horizontal(classes="spaced-small"):
                    yield Input(value="", placeholder="#", id="history-id")
                    yield Button("↻", id="rerun-history", variant="primary")
                    yield Button("👁", id="view-history", variant="default")

        # Agent Panel
        with Vertical(id="agent-panel", classes="glass-panel"):
            yield Label("🤖 Agent Mode", classes="section-title")
            with Horizontal(id="quick-actions"):
                yield Button("🚀 Launch", id="launch-agent", variant="primary")
                yield Button("⏹ Stop", id="stop-agent", variant="error")
                yield Button("📋 Analyze", id="quick-analyze", classes="quick-btn")
                yield Button("💡 Suggest", id="quick-suggest", classes="quick-btn")
                yield Button("🔗 Integrate", id="quick-integrate", classes="quick-btn")
                yield Button("❓ Help", id="show-help", classes="quick-btn")
                yield Label("", id="agent-status", classes="agent-status offline")
            yield TextArea("", id="agent-logs", read_only=True)

        # Prompt Bar
        with Horizontal(id="prompt-bar"):
            yield Input(
                value="",
                placeholder="> Type naturally... (Try: 'apply rmsnorm to /path')",
                id="prompt-input",
            )

        yield Footer()

    def _append_logs(self, widget_id: str, lines: list[str]) -> None:
        area = self.query_one(f"#{widget_id}", TextArea)
        current = area.text
        merged = (current + "\n" if current else "") + "\n".join(lines)
        area.load_text(merged)
        # Auto-scroll
        area.action_cursor_bottom_end()

    def _payload_compat_messages(
        self, payload: dict[str, Any], *, expected_types: set[str] | None = None
    ) -> list[str]:
        report = evaluate_payload_compatibility(payload, expected_types=expected_types)
        lines = []
        lines.extend([f"⚠️ {item}" for item in report.issues])
        lines.extend([f"📝 {item}" for item in report.warnings])
        lines.extend([f"ℹ️ {item}" for item in report.notes])
        return lines

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

        # Save context
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
            status.update("● Online")
            status.add_class("online")
            status.remove_class("offline")
        else:
            status.update("○ Offline")
            status.add_class("offline")
            status.remove_class("online")

    def _execute_quick_action(self, action: str) -> None:
        repo_path = self.query_one("#repo-path", Input).value.strip()

        if not repo_path:
            self._append_logs("agent-logs", ["⚠️ Set repository path first"])
            return

        self.query_one("#action", Select).value = action
        self._set_phase("analyzing")
        self._run_selected_workflow()

    def action_show_help(self) -> None:
        self._append_logs(
            "agent-logs",
            [
                f"{Colors.BRIGHT_CYAN}📖 Keyboard Shortcuts:{Colors.RESET}",
                "  Ctrl+A - Quick Analyze",
                "  Ctrl+S - Quick Suggest",
                "  Ctrl+I - Quick Integrate",
                "  Ctrl+R - Run workflow",
                "  Ctrl+L - Clear logs",
                "  Esc   - Stop agent",
                "",
                f"{Colors.BRIGHT_YELLOW}💡 Natural Language:{Colors.RESET}",
                "  'apply rmsnorm to /path'",
                "  'suggest improvements for /repo'",
                "  'search for attention mechanism'",
            ],
        )

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

    @on(Button.Pressed, "#show-help")
    def on_show_help(self) -> None:
        self.action_show_help()

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

        self.query_one("#run-status", Label).update(f"⚡ Running '{action}'...")
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
        self._append_logs("logs", [f"🚀 Starting: {action} on {repo_path or 'default'}"])

    @on(TaskCompleted)
    def on_task_completed(self, message: TaskCompleted) -> None:
        self._set_phase("complete")

        # Rich output
        payload = {"title": message.title, "error": message.error, "result": message.result}
        self._format_rich_output(payload)  # Generate formatted text
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

        # Re-enable buttons
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

        status = (
            f"{Colors.BRIGHT_GREEN}✓ Done{Colors.RESET}"
            if message.error is None
            else f"{Colors.BRIGHT_RED}✗ Failed{Colors.RESET}"
        )
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
            self._append_logs("agent-logs", ["❌ Agent folder not found"])
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
            self._append_logs("agent-logs", [f"❌ Launch failed: {exc}"])
            return

        self._append_logs(
            "agent-logs",
            [
                f"{Colors.BRIGHT_GREEN}🚀 Agent ready!{Colors.RESET}",
                f"{Colors.CYAN}Commands:{Colors.RESET} analyze, suggest, integrate, search...",
                f"{Colors.YELLOW}Try:{Colors.RESET} 'apply rmsnorm to /path/to/repo'",
                f"{Colors.DIM}Type 'exit' to stop, 'help' for more{Colors.RESET}",
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

        self._command_history.append(prompt)
        self._history_index = len(self._command_history)
        event.input.value = ""

        # Help command
        if prompt.lower() in ("help", "?"):
            self.action_show_help()
            return

        if not self._agent_running or not self._agent_stdin:
            # Parse and run locally
            command, ctx = self._parse_natural_command(prompt)
            self._append_logs("agent-logs", [f"👤 {prompt}"])

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

        # Send to agent
        if prompt.lower() in ("exit", "quit"):
            self.on_stop_agent()
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
