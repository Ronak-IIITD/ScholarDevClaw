"""ScholarDevClaw TUI — clean terminal interface.

Three-zone layout inspired by Claude Code:
  1. Main output area (top) — logs, results, chat
  2. Contextual config bar (middle) — action-specific fields
  3. Prompt bar (bottom) — command input with status

Navigation: ctrl+k command palette, ctrl+h help, type /commands.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import (
    Button,
    Checkbox,
    Header,
    Input,
    Label,
    Select,
    Static,
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
    ChatLog,
    HistoryPane,
    LogView,
    PhaseTracker,
    PromptInput,
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


class AgentEvent(Message):
    def __init__(self, role: str, content: str):
        super().__init__()
        self.role = role
        self.content = content


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


class ScholarDevClawApp(App[None]):
    TITLE = "ScholarDevClaw"
    SUB_TITLE = "Research-to-Code Agent"

    PHASES = [
        ("idle", "Ready", 0),
        ("validating", "Validating...", 0.05),
        ("analyzing", "Analyzing...", 0.2),
        ("research", "Researching...", 0.4),
        ("mapping", "Mapping...", 0.55),
        ("generating", "Generating...", 0.7),
        ("validating_patches", "Verifying...", 0.85),
        ("complete", "Done", 1.0),
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
    # CSS — Catppuccin Mocha with stronger workspace hierarchy
    # -----------------------------------------------------------------------

    CSS = """
    $surface: #1e1e2e;
    $surface-dark: #181825;
    $surface-elevated: #24273a;
    $panel: #11111b;
    $panel-soft: #151822;
    $border: #313244;
    $border-light: #45475a;
    $text: #cdd6f4;
    $text-dim: #a6adc8;
    $text-muted: #7f849c;
    $accent: #89b4fa;
    $accent-soft: #74c7ec;
    $success: #a6e3a1;
    $error: #f38ba8;
    $warning: #f9e2af;
    $text-inv: #11111b;

    Screen {
        layout: vertical;
        background: $panel;
        color: $text;
    }

    Header {
        background: $surface-dark;
        color: $text-dim;
        dock: top;
        height: 1;
        text-align: center;
        border-bottom: solid $border;
    }

    Label {
        color: $text-dim;
    }

    #main-area {
        width: 100%;
        height: 1fr;
        layout: vertical;
        padding: 1 2 1 2;
    }

    #hero-panel {
        width: 100%;
        height: auto;
        min-height: 8;
        layout: horizontal;
        background: $surface;
        border: solid $border-light;
        padding: 1 2;
        margin-bottom: 1;
    }

    #hero-copy {
        width: 1fr;
        padding-right: 2;
    }

    #hero-kicker {
        color: $accent-soft;
        text-style: bold;
    }

    #hero-title {
        color: $text;
        text-style: bold;
    }

    #hero-subtitle {
        color: $text-dim;
        margin-top: 1;
    }

    #hero-meta {
        width: 34;
        min-width: 28;
        background: $surface-dark;
        border-left: solid $border;
        padding: 0 0 0 2;
    }

    .hero-meta-label {
        color: $text-muted;
        margin-top: 1;
    }

    .hero-meta-value {
        color: $text;
        background: $panel;
        border: solid $border;
        padding: 0 1;
    }

    .surface-card {
        width: 100%;
        background: $surface;
        border: solid $border;
        padding: 1;
    }

    .surface-heading {
        width: 100%;
        height: auto;
        layout: horizontal;
        margin-bottom: 1;
    }

    .surface-title {
        width: 1fr;
        color: $text;
        text-style: bold;
    }

    .surface-caption {
        width: auto;
        color: $text-muted;
    }

    #log-card {
        height: 1fr;
        margin-bottom: 1;
    }

    #history-card {
        height: auto;
    }

    #config-bar {
        width: 100%;
        height: auto;
        max-height: 22;
        background: $surface-dark;
        border-top: solid $border-light;
        padding: 1 2 2 2;
        overflow-y: auto;
    }

    #config-bar.collapsed {
        height: 0;
        max-height: 0;
        padding: 0;
        border: none;
    }

    #config-header {
        width: 100%;
        height: auto;
        layout: horizontal;
        margin-bottom: 1;
    }

    #config-header-copy {
        width: 1fr;
        padding-right: 1;
    }

    #config-title {
        color: $text;
        text-style: bold;
    }

    #action-context {
        color: $text-dim;
        margin-top: 1;
    }

    #config-header-chips {
        width: auto;
        height: auto;
        padding-top: 1;
    }

    .config-pill {
        width: auto;
        background: $surface-elevated;
        border: solid $border;
        color: $text-dim;
        padding: 0 1;
        margin-left: 1;
    }

    #validation-hint {
        width: 100%;
        background: $surface;
        border: solid $border;
        color: $text-muted;
        padding: 0 1;
        margin-bottom: 1;
    }

    #validation-hint.hint-ready { color: $text-muted; }
    #validation-hint.hint-warning { color: $warning; }
    #validation-hint.hint-error { color: $error; }
    #validation-hint.hint-running { color: $accent; }

    #config-fields {
        width: 100%;
        height: auto;
        layout: horizontal;
    }

    .config-group {
        width: 28;
        min-width: 22;
        height: auto;
        background: $surface;
        border: solid $border;
        padding: 1 1 0 1;
        margin-right: 1;
    }

    .config-label {
        color: $text-muted;
        margin-bottom: 1;
        text-style: bold;
    }

    #config-bar Input,
    #config-bar Select {
        width: 100%;
        height: 3;
        background: $panel-soft;
        color: $text;
        border: solid $border;
        padding: 0 1;
        margin-bottom: 1;
    }

    #config-bar Input:focus,
    #config-bar Select:focus {
        border: solid $accent;
    }

    #config-bar Checkbox {
        color: $text-dim;
        width: auto;
        margin-right: 2;
        margin-bottom: 1;
    }

    #config-actions {
        width: auto;
        min-width: 20;
        background: transparent;
        border: none;
        padding: 0;
        margin-left: 1;
    }

    #config-actions Button {
        height: 3;
        min-width: 11;
        margin-right: 1;
    }

    #prompt-zone {
        width: 100%;
        height: auto;
        background: $surface-dark;
        border-top: solid $border-light;
        padding: 0 2 1 2;
        dock: bottom;
    }

    #prompt-shell {
        width: 100%;
        height: auto;
        background: $surface;
        border: solid $border;
        padding: 0 1;
        margin-top: 1;
    }

    #prompt-row {
        width: 100%;
        height: 3;
        layout: horizontal;
        content-align: center middle;
    }

    #prompt-prefix {
        width: 6;
        color: $accent-soft;
        text-style: bold;
        content-align: center middle;
    }

    #prompt-input {
        width: 1fr;
        height: 3;
        background: transparent;
        color: $text;
        border: none;
        padding: 0;
    }

    #prompt-meta {
        width: auto;
        height: 3;
        content-align: center middle;
    }

    #prompt-meta .chip {
        width: auto;
        background: $surface-dark;
        border: solid $border;
        color: $text-muted;
        padding: 0 1;
        margin-left: 1;
    }

    #next-action-chip {
        color: $accent;
        text-style: bold;
    }

    #prompt-helper {
        width: 100%;
        color: $text-muted;
        margin-bottom: 1;
    }

    #status-bar {
        width: 100%;
        height: 1;
        background: $panel;
        border-top: solid $border-light;
        padding: 0 2;
        dock: bottom;
    }

    #chat-workspace {
        display: none;
        width: 100%;
        height: 0;
        layout: horizontal;
        padding: 1 2;
    }

    Screen.chat-mode #main-area {
        display: none;
    }

    Screen.chat-mode #config-bar {
        display: none;
    }

    Screen.chat-mode #chat-workspace {
        display: block;
        height: 1fr;
    }

    #chat-main {
        width: 1fr;
        height: 100%;
        background: $surface;
        border: solid $border;
        padding: 1;
        margin-right: 1;
    }

    #chat-sidebar {
        width: 30;
        min-width: 26;
        max-width: 36;
        height: 100%;
        background: $surface;
        border: solid $border;
        padding: 1 1 1 2;
    }

    #chat-sidebar .cs-title {
        color: $accent-soft;
        text-style: bold;
        margin-bottom: 1;
    }

    #chat-sidebar .cs-label {
        color: $text-muted;
        margin-top: 1;
    }

    #chat-sidebar .cs-value {
        color: $text;
        background: $surface-dark;
        border: solid $border;
        padding: 0 1;
    }

    Button {
        border: solid $border;
        background: $surface-elevated;
        color: $text;
        padding: 0 1;
        min-width: 8;
    }

    Button:hover {
        background: $border-light;
    }

    Button.-primary {
        background: $accent;
        color: $text-inv;
        text-style: bold;
        border: none;
    }

    Button.-primary:hover {
        background: #b4befe;
    }

    Button.-success {
        background: $success;
        color: $text-inv;
        text-style: bold;
        border: none;
    }

    Button.-error {
        background: $error;
        color: $text-inv;
        text-style: bold;
        border: none;
    }

    Button:disabled {
        opacity: 0.35;
    }

    LogView .log-empty,
    ChatLog .chat-empty {
        color: $text-muted;
        padding: 1 1;
    }

    PromptInput {
        background: transparent;
        color: $text;
        border: none;
    }

    PromptInput:focus {
        border: none;
    }

    PromptInput.multiline {
        height: 4;
    }
    """

    # -----------------------------------------------------------------------
    # Key bindings — minimal, memorable
    # -----------------------------------------------------------------------

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+r", "run_selected", "Run"),
        ("ctrl+shift+r", "rerun_last", "Rerun"),
        ("ctrl+k", "command_palette", "Commands"),
        ("ctrl+h", "help", "Help"),
        ("ctrl+l", "clear_logs", "Clear"),
        ("ctrl+n", "new_session", "New"),
        ("ctrl+e", "export_log", "Export"),
        ("ctrl+p", "focus_prompt", "Prompt"),
        ("ctrl+g", "focus_history", "History"),
        ("tab", "focus_next", "Next"),
        ("shift+tab", "focus_prev", "Prev"),
        ("ctrl+o", "toggle_config", "Config"),
        ("escape", "handle_escape", "Back"),
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

    ACTION_GUIDE: dict[str, dict[str, str]] = {
        "analyze": {
            "summary": "Scan repository structure, languages, and entry points.",
            "hint": "Needs: repository path",
        },
        "suggest": {
            "summary": "Find high-impact research improvements for this codebase.",
            "hint": "Needs: repository path",
        },
        "search": {
            "summary": "Search local specs, plus optional arXiv/web sources.",
            "hint": "Needs: query (repo optional)",
        },
        "specs": {
            "summary": "List available research specs and categories.",
            "hint": "No required inputs",
        },
        "map": {
            "summary": "Map selected research spec to concrete code targets.",
            "hint": "Needs: repository path, spec",
        },
        "generate": {
            "summary": "Generate patches for a mapped research spec.",
            "hint": "Needs: repository path, spec",
        },
        "validate": {
            "summary": "Run tests/benchmarks and summarize confidence.",
            "hint": "Needs: repository path",
        },
        "integrate": {
            "summary": "Run full pipeline: analyze → suggest → map → generate → validate.",
            "hint": "Needs: repository path (spec optional)",
        },
    }

    ACTION_FIELD_VISIBILITY: dict[str, dict[str, bool]] = {
        "analyze": {
            "query": False,
            "search_sources": False,
            "spec": False,
            "output": False,
            "integrate": False,
        },
        "suggest": {
            "query": False,
            "search_sources": False,
            "spec": False,
            "output": False,
            "integrate": False,
        },
        "search": {
            "query": True,
            "search_sources": True,
            "spec": False,
            "output": False,
            "integrate": False,
        },
        "specs": {
            "query": False,
            "search_sources": False,
            "spec": False,
            "output": False,
            "integrate": False,
        },
        "map": {
            "query": False,
            "search_sources": False,
            "spec": True,
            "output": False,
            "integrate": False,
        },
        "generate": {
            "query": False,
            "search_sources": False,
            "spec": True,
            "output": True,
            "integrate": False,
        },
        "validate": {
            "query": False,
            "search_sources": False,
            "spec": False,
            "output": False,
            "integrate": False,
        },
        "integrate": {
            "query": False,
            "search_sources": False,
            "spec": True,
            "output": True,
            "integrate": True,
        },
    }

    ACTION_REQUIRES_REPO = {"analyze", "suggest", "map", "generate", "validate", "integrate"}
    ACTION_REQUIRES_SPEC = {"map", "generate"}
    ACTION_REQUIRES_QUERY = {"search"}

    FOCUS_ORDER = [
        "#action",
        "#repo-path",
        "#spec",
        "#query",
        "#model-provider",
        "#run",
        "#prompt-input",
        "#log-view",
        "#run-history",
    ]

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()
        self._agent_process: subprocess.Popen[str] | None = None
        self._live_logs_enabled = False
        self._live_log_count = 0
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
        self._chat_mode = False
        self._config_visible = True
        self._command_history: list[str] = []
        self._history_index = -1
        self._run_in_progress = False
        self._workflow_steps = {
            "analyze": (1, 6),
            "suggest": (2, 6),
            "search": (2, 6),
            "map": (3, 6),
            "generate": (4, 6),
            "validate": (5, 6),
            "integrate": (6, 6),
            "specs": (1, 1),
        }

        # Config
        self._config_dir = Path.home() / ".scholardevclaw"
        self._config_file = self._config_dir / "config.json"
        self._context_file = self._config_dir / "tui_context.json"
        self._config = self._load_config()
        self._saved_context: dict[str, Any] = self._load_context()

    def _set_chat_mode(self, enabled: bool) -> None:
        self._chat_mode = enabled
        if enabled:
            self.add_class("chat-mode")
            self._set_status("Chat session active", "accent")
            self._set_next_action("send a prompt")
        else:
            self.remove_class("chat-mode")
            self._set_status("Ready — choose an action and run", "info")
            self._set_next_action("press Ctrl+R")
        try:
            self.query_one("#chat-mode-value", Label).update(
                "agent: running" if self._agent_running else "agent: idle"
            )
        except Exception:
            pass

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
        try:
            label = next((item[1] for item in self.PHASES if item[0] == phase), phase.title())
            self.query_one("#hero-phase-value", Label).update(label)
        except Exception:
            pass

    def _set_status(self, message: str, level: str = "info") -> None:
        try:
            status_bar = self.query_one(StatusBar)
            status_bar.set_status(message, level)
        except Exception:
            pass
        try:
            self.query_one("#hero-status-value", Label).update(message)
        except Exception:
            pass

    def _update_agent_status(self, status: str) -> None:
        try:
            agent_dot = self.query_one(AgentStatus)
            agent_dot.set_status(status)
            status_bar = self.query_one(StatusBar)
            mode = "running" if status == "Online" else "idle"
            if not self._run_in_progress:
                status_bar.set_center(f"agent: {mode}")
            self.query_one("#chat-mode-value", Label).update(f"agent: {mode}")
        except Exception:
            pass

    def _log_to_view(self, lines: list[str]) -> None:
        """Append lines to the LogView widget."""
        try:
            log_view = self.query_one(LogView)
            log_view.add_logs(lines)
        except Exception:
            pass

    def _sync_phase_from_log_line(self, line: str) -> None:
        text = line.lower()
        if any(tok in text for tok in ("preflight", "validate", "checking")):
            self._set_phase("validating")
        elif any(tok in text for tok in ("analyz", "scan")):
            self._set_phase("analyzing")
        elif any(tok in text for tok in ("research", "search", "suggest")):
            self._set_phase("research")
        elif "map" in text:
            self._set_phase("mapping")
        elif any(tok in text for tok in ("patch", "generat")):
            self._set_phase("generating")
        elif any(tok in text for tok in ("benchmark", "test", "validating patch", "verify")):
            self._set_phase("validating_patches")

    def _log_to_legacy(self, widget_id: str, lines: list[str]) -> None:
        """Backward-compatible sink: writes to rich chat log."""
        for line in lines:
            role = "system"
            low = line.lower()
            if line.startswith("User:"):
                role = "user"
            elif widget_id == "agent-logs":
                role = "agent"
                if low.startswith(
                    (
                        "error:",
                        "agent launched",
                        "commands:",
                        "type 'help'",
                        "agent stopped",
                        "no running agent",
                        "failed",
                    )
                ):
                    role = "system"
            self._add_chat(role, line)

    def _add_chat(self, role: str, content: str) -> None:
        try:
            chat = self.query_one("#chat-main-log", ChatLog)
            chat.add_entry(role, content)
        except Exception:
            pass

    def _resolve_model_provider(self) -> tuple[str | None, str | None]:
        raw = self.query_one("#model-provider", Select).value
        if not isinstance(raw, str) or raw == "auto":
            return None, None
        if ":" not in raw:
            return None, None
        provider, model = raw.split(":", 1)
        mapping = {
            "github": "github_copilot",
            "openai": "openai",
            "anthropic": "anthropic",
        }
        provider = mapping.get(provider, provider)
        return provider, model

    def _apply_provider_env(self) -> dict[str, str | None]:
        provider, model = self._resolve_model_provider()
        prev_provider = os.environ.get("SCHOLARDEVCLAW_API_PROVIDER")
        prev_model = os.environ.get("SCHOLARDEVCLAW_API_MODEL")
        if provider:
            os.environ["SCHOLARDEVCLAW_API_PROVIDER"] = provider
            os.environ["SCHOLARDEVCLAW_API_MODEL"] = model or ""
        return {
            "provider": prev_provider,
            "model": prev_model,
        }

    def _restore_provider_env(self, prev: dict[str, str | None]) -> None:
        prev_provider = prev.get("provider")
        if prev_provider is None:
            os.environ.pop("SCHOLARDEVCLAW_API_PROVIDER", None)
        else:
            os.environ["SCHOLARDEVCLAW_API_PROVIDER"] = prev_provider
        prev_model = prev.get("model")
        if prev_model is None:
            os.environ.pop("SCHOLARDEVCLAW_API_MODEL", None)
        else:
            os.environ["SCHOLARDEVCLAW_API_MODEL"] = prev_model

    def _refresh_provider_chip(self) -> None:
        provider, model = self._resolve_model_provider()

        def _env_for_provider(name: str) -> str:
            mapping = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "github_copilot": "GITHUB_TOKEN",
            }
            return mapping.get(name, "")

        try:
            chip = self.query_one("#provider-chip", Label)
            chat_provider = self.query_one("#chat-provider-value", Label)
            chat_build = self.query_one("#chat-build-value", Label)
            config_provider = self.query_one("#config-provider-pill", Label)
            if provider and model:
                key_env = _env_for_provider(provider)
                has_key = bool(key_env and os.environ.get(key_env, "").strip())
                status = "connected" if has_key else "no key"
                chip.update(f"{provider}:{model} ({status})")
                chat_provider.update(f"{provider} ({status})")
                chat_build.update(model)
                config_provider.update(f"model {provider}:{model}")
            else:
                chip.update("auto")
                chat_provider.update("auto")
                chat_build.update("auto")
                config_provider.update("model auto")
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Natural language command parsing
    # -----------------------------------------------------------------------

    def _parse_natural_command(self, prompt: str) -> tuple[str, dict[str, Any]]:
        prompt_lower = prompt.strip().lower()
        ctx: dict[str, Any] = {}
        command = "analyze"

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
    # Compose — clean 3-zone layout
    # -----------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # -- Zone 1: Main output area --
        with Vertical(id="main-area"):
            with Horizontal(id="hero-panel"):
                with Vertical(id="hero-copy"):
                    yield Label("CONTROL SURFACE", id="hero-kicker")
                    yield Label("Research-to-code workflows with a cleaner operator view.", id="hero-title")
                    yield Label(
                        "Run analysis, map papers, generate patches, and validate impact from one keyboard-first workspace.",
                        id="hero-subtitle",
                    )
                with Vertical(id="hero-meta"):
                    yield Label("Current action", classes="hero-meta-label")
                    yield Label("Analyze", classes="hero-meta-value", id="hero-action-value")
                    yield Label("Pipeline phase", classes="hero-meta-label")
                    yield Label("Ready", classes="hero-meta-value", id="hero-phase-value")
                    yield Label("Workspace status", classes="hero-meta-label")
                    yield Label("Ready", classes="hero-meta-value", id="hero-status-value")

            with Vertical(classes="surface-card", id="log-card"):
                with Horizontal(classes="surface-heading"):
                    yield Label("Workflow output", classes="surface-title", id="log-title")
                    yield Label("Live execution stream", classes="surface-caption")
                yield PhaseTracker(id="phase-tracker")
                yield LogView(id="log-view")

            with Vertical(classes="surface-card", id="history-card"):
                with Horizontal(classes="surface-heading"):
                    yield Label("Run history", classes="surface-title", id="history-title")
                    yield Label("↑/↓ select · Enter rerun", classes="surface-caption")
                yield HistoryPane(id="run-history")

        # -- Chat overlay (hidden until agent mode) --
        with Horizontal(id="chat-workspace"):
            with Vertical(id="chat-main"):
                yield ChatLog(id="chat-main-log")
            with Vertical(id="chat-sidebar"):
                yield Label("Session", classes="cs-title")
                yield Label("Mode", classes="cs-label")
                yield Label("idle", classes="cs-value", id="chat-mode-value")
                yield Label("Provider", classes="cs-label")
                yield Label("auto", classes="cs-value", id="chat-provider-value")
                yield Label("Build", classes="cs-label")
                yield Label("auto", classes="cs-value", id="chat-build-value")
                yield Label("Agent", classes="cs-label")
                yield AgentStatus(id="agent-status")
                yield Static("")
                yield Button("Launch Agent", id="launch-agent", variant="primary")
                yield Button("Stop Agent", id="stop-agent", variant="error")

        # -- Zone 2: Contextual config bar --
        with Vertical(id="config-bar"):
            with Horizontal(id="config-header"):
                with Vertical(id="config-header-copy"):
                    yield Label("Workflow configuration", id="config-title")
                    yield Label("", id="action-context")
                with Horizontal(id="config-header-chips"):
                    yield Label("action analyze", id="config-action-pill", classes="config-pill")
                    yield Label("model auto", id="config-provider-pill", classes="config-pill")
            yield Label("", id="validation-hint")
            with Horizontal(id="config-fields"):
                # Action selector
                with Vertical(classes="config-group", id="group-action"):
                    yield Label("Action", classes="config-label")
                    yield Select(
                        self.action_mode_options,
                        value=self._saved_context.get(
                            "last_action",
                            self._config.get("default_action", "analyze"),
                        ),
                        id="action",
                    )

                # Repo path
                with Vertical(classes="config-group", id="group-repo"):
                    yield Label("Repository", classes="config-label")
                    yield Input(
                        value=self._saved_context.get(
                            "last_repo",
                            self._config.get("default_repo", str(Path.cwd())),
                        ),
                        placeholder="/path/to/repo",
                        id="repo-path",
                    )

                # Spec (for map/generate/integrate)
                with Vertical(classes="config-group", id="group-spec"):
                    yield Label("Spec", classes="config-label")
                    yield Input(
                        value=self._saved_context.get(
                            "last_spec",
                            self._config.get("default_spec", "rmsnorm"),
                        ),
                        placeholder="rmsnorm",
                        id="spec",
                    )

                # Search query (for search)
                with Vertical(classes="config-group", id="group-query"):
                    yield Label("Query", classes="config-label")
                    yield Input(
                        value=self._saved_context.get("last_query", "layer normalization"),
                        placeholder="search query",
                        id="query",
                    )

                # Search options (for search)
                with Vertical(classes="config-group", id="group-search-sources"):
                    yield Label("Sources", classes="config-label")
                    with Horizontal():
                        yield Checkbox("arXiv", value=False, id="search-arxiv")
                        yield Checkbox("Web", value=False, id="search-web")

                # Output dir (for generate)
                with Vertical(classes="config-group", id="group-output"):
                    yield Label("Output", classes="config-label")
                    yield Input(
                        value="",
                        placeholder="output dir",
                        id="output-dir",
                    )

                # Options (for integrate)
                with Vertical(classes="config-group", id="group-integrate"):
                    yield Label("Options", classes="config-label")
                    with Horizontal():
                        yield Checkbox("Dry-run", value=False, id="integrate-dry-run")
                        yield Checkbox(
                            "Clean git",
                            value=self._config.get("require_clean_git", False),
                            id="integrate-require-clean",
                        )

                # Model provider
                with Vertical(classes="config-group", id="group-model"):
                    yield Label("Model", classes="config-label")
                    yield Select(
                        [
                            ("auto", "auto"),
                            ("openai:gpt-5", "openai:gpt-5"),
                            ("anthropic:claude-sonnet-4", "anthropic:claude-sonnet-4"),
                            ("github:copilot", "github:copilot"),
                        ],
                        value="auto",
                        id="model-provider",
                    )

                # Run / Clear
                with Vertical(id="config-actions"):
                    yield Label("", classes="config-label")
                    with Horizontal():
                        yield Button("Run", id="run", variant="primary")
                        yield Button("Clear", id="clear")

        # -- Zone 3: Prompt bar --
        with Vertical(id="prompt-zone"):
            with Vertical(id="prompt-shell"):
                with Horizontal(id="prompt-row"):
                    yield Label("› run", id="prompt-prefix")
                    yield PromptInput(
                        value="",
                        placeholder="Describe what to run…  (Ctrl+K commands · Ctrl+H help)",
                        id="prompt-input",
                    )
                    with Horizontal(id="prompt-meta"):
                        yield Label("ready", id="next-action-chip", classes="chip")
                        yield Label("auto", id="provider-chip", classes="chip")
                yield Label(
                    "Natural language works here: analyze ./repo, search flash attention, integrate rmsnorm in ./repo",
                    id="prompt-helper",
                )

        # -- Status bar --
        yield StatusBar(id="status-bar")

    # -----------------------------------------------------------------------
    # Mount / unmount
    # -----------------------------------------------------------------------

    def on_mount(self) -> None:
        self._refresh_action_state()
        self._update_agent_status("Offline")
        self._refresh_provider_chip()
        self._set_phase("idle")
        self._set_status("Ready — choose an action and run", "info")
        self._log_to_view(
            [
                "ScholarDevClaw ready.",
                "",
                "Fast keys",
                "  Ctrl+R        run selected workflow",
                "  Ctrl+Shift+R  rerun latest",
                "  Ctrl+K        command palette",
                "  Ctrl+H        keyboard guide",
                "  Ctrl+P / Ctrl+G   jump prompt / history",
                "  Tab / Shift+Tab   move focus",
                "",
                "Type naturally in prompt: analyze ./my-project",
            ]
        )
        try:
            status_bar = self.query_one(StatusBar)
            status_bar.set_center("agent: idle")
            status_bar.set_step(0, 0)
            status_bar.set_status("Ready — choose an action and run", "info")
        except Exception:
            pass

        self._render_history()
        self.call_later(self.action_focus_prompt)

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

    def _current_action(self) -> str:
        raw = self.query_one("#action", Select).value
        if isinstance(raw, str) and raw in self.ACTION_GUIDE:
            return raw
        return "analyze"

    def _capture_request(self) -> dict[str, Any]:
        action = self._current_action()
        return {
            "action": action,
            "repo_path": self.query_one("#repo-path", Input).value.strip(),
            "query": self.query_one("#query", Input).value.strip(),
            "include_arxiv": self.query_one("#search-arxiv", Checkbox).value,
            "include_web": self.query_one("#search-web", Checkbox).value,
            "spec": self.query_one("#spec", Input).value.strip(),
            "output_dir": self.query_one("#output-dir", Input).value.strip() or None,
            "integrate_dry_run": self.query_one("#integrate-dry-run", Checkbox).value,
            "integrate_require_clean": self.query_one("#integrate-require-clean", Checkbox).value,
            "max_results": int(self._config.get("max_results", 10)),
            "model_provider": self.query_one("#model-provider", Select).value,
        }

    def _refresh_contextual_copy(self) -> None:
        action = self._current_action()
        meta = self.ACTION_GUIDE.get(action, self.ACTION_GUIDE["analyze"])
        try:
            self.query_one("#action-context", Label).update(
                f"[bold]{action}[/] · {meta['summary']}  [dim]({meta['hint']})[/]"
            )
        except Exception:
            pass
        label = action.replace("_", " ").title()
        try:
            self.query_one("#hero-action-value", Label).update(label)
        except Exception:
            pass
        try:
            self.query_one("#config-action-pill", Label).update(f"action {action}")
        except Exception:
            pass

    def _set_next_action(self, text: str) -> None:
        try:
            self.query_one("#next-action-chip", Label).update(text)
        except Exception:
            pass

    def _refresh_validation_hint(self) -> None:
        hint_level = "ready"
        if self._run_in_progress:
            try:
                label = self.query_one("#validation-hint", Label)
                label.update("Workflow running · live logs and phase progress are updating")
                label.remove_class("hint-ready", "hint-warning", "hint-error")
                label.add_class("hint-running")
            except Exception:
                pass
            self._set_next_action("running")
            return

        req = self._capture_request()
        action = req.get("action", "analyze")
        missing: list[str] = []

        if action in self.ACTION_REQUIRES_REPO and not req.get("repo_path"):
            missing.append("repository")
        if action in self.ACTION_REQUIRES_SPEC and not req.get("spec"):
            missing.append("spec")
        if action in self.ACTION_REQUIRES_QUERY and not req.get("query"):
            missing.append("query")

        if missing:
            hint = f"Missing: {', '.join(missing)} · press ctrl+r after completing required inputs"
            hint_level = "error"
            self._set_next_action("complete required fields")
        elif action == "search" and not (req.get("include_arxiv") or req.get("include_web")):
            hint = "Ready · search will use local spec index only (enable arXiv/web for external sources)"
            hint_level = "warning"
            self._set_next_action("run local search")
        else:
            hint = "Ready · press Ctrl+R to run · use ↑/↓ for prompt history"
            self._set_next_action("press Ctrl+R")

        try:
            label = self.query_one("#validation-hint", Label)
            label.update(hint)
            label.remove_class("hint-ready", "hint-warning", "hint-error", "hint-running")
            label.add_class(f"hint-{hint_level}")
        except Exception:
            pass

    def _validate_request_inputs(self, req: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
        action = req.get("action", "analyze")
        repo = str(req.get("repo_path") or "").strip()
        spec = str(req.get("spec") or "").strip()
        query = str(req.get("query") or "").strip()
        include_arxiv = bool(req.get("include_arxiv", False))
        include_web = bool(req.get("include_web", False))

        errors: list[str] = []
        warnings: list[str] = []

        if action in self.ACTION_REQUIRES_REPO:
            valid, err = self._validate_repo_path(repo)
            if not valid:
                errors.append(err)

        if action in self.ACTION_REQUIRES_QUERY and not query:
            errors.append("Search query is required for Search")

        if action in self.ACTION_REQUIRES_SPEC and not spec:
            errors.append("Spec is required for this action")
        elif action in self.ACTION_REQUIRES_SPEC:
            valid, err = self._validate_spec(spec)
            if not valid:
                errors.append(err)
        elif action == "integrate" and spec:
            valid, err = self._validate_spec(spec)
            if not valid:
                errors.append(err)
        elif spec:
            valid, err = self._validate_spec(spec)
            if not valid:
                warnings.append(err)

        if action == "search" and not (include_arxiv or include_web):
            warnings.append("Only local spec index will be used (arXiv/Web disabled)")

        if action == "integrate" and req.get("integrate_require_clean") and repo:
            dirty, err = self._check_git_status(repo)
            if dirty:
                warnings.append(err)

        return not errors, errors, warnings

    def _apply_request(self, request: dict[str, Any]) -> None:
        self.query_one("#action", Select).value = request.get("action", "analyze")
        self.query_one("#repo-path", Input).value = request.get("repo_path", "")
        self.query_one("#query", Input).value = request.get("query", "")
        self.query_one("#search-arxiv", Checkbox).value = bool(request.get("include_arxiv", False))
        self.query_one("#search-web", Checkbox).value = bool(request.get("include_web", False))
        self.query_one("#spec", Input).value = request.get("spec", "")
        self.query_one("#output-dir", Input).value = request.get("output_dir") or ""
        mp = request.get("model_provider", "auto")
        if isinstance(mp, str):
            try:
                self.query_one("#model-provider", Select).value = mp
            except Exception:
                self.query_one("#model-provider", Select).value = "auto"
        self.query_one("#integrate-dry-run", Checkbox).value = bool(
            request.get("integrate_dry_run", False)
        )
        self.query_one("#integrate-require-clean", Checkbox).value = bool(
            request.get("integrate_require_clean", False)
        )
        self._refresh_provider_chip()
        self._refresh_action_state()
        self._refresh_validation_hint()

    # -----------------------------------------------------------------------
    # Action state management — show/hide contextual fields
    # -----------------------------------------------------------------------

    def _refresh_action_state(self) -> None:
        action = self._current_action()
        visibility = self.ACTION_FIELD_VISIBILITY.get(
            action, self.ACTION_FIELD_VISIBILITY["analyze"]
        )

        group_visibility = {
            "#group-query": visibility["query"],
            "#group-search-sources": visibility["search_sources"],
            "#group-spec": visibility["spec"],
            "#group-output": visibility["output"],
            "#group-integrate": visibility["integrate"],
        }

        for selector, is_visible in group_visibility.items():
            try:
                self.query_one(selector).display = is_visible
            except Exception:
                pass

        for field_id, widget_cls, visible in [
            ("query", Input, visibility["query"]),
            ("search-arxiv", Checkbox, visibility["search_sources"]),
            ("search-web", Checkbox, visibility["search_sources"]),
            ("spec", Input, visibility["spec"]),
            ("output-dir", Input, visibility["output"]),
            ("integrate-dry-run", Checkbox, visibility["integrate"]),
            ("integrate-require-clean", Checkbox, visibility["integrate"]),
        ]:
            try:
                self.query_one(f"#{field_id}", widget_cls).disabled = not visible
            except Exception:
                pass

        self._refresh_contextual_copy()
        self._refresh_validation_hint()

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
            "finished_at": datetime.now().strftime("%H:%M:%S"),
            "request": request,
            "title": title,
            "result": result,
            "error": error,
        }
        self._next_run_id += 1
        self._run_history.insert(0, record)
        self._run_history = self._run_history[: self._history_limit]

        self._saved_context["last_action"] = action
        self._saved_context["last_repo"] = request.get("repo_path", "")
        self._saved_context["last_spec"] = request.get("spec", "")
        self._saved_context["last_query"] = request.get("query", "")
        self._save_context()

        self._render_history()

    def _render_history(self) -> None:
        try:
            history = self.query_one("#run-history", HistoryPane)
            history.clear_history()
            for entry in reversed(self._run_history):
                request = entry.get("request", {}) if isinstance(entry.get("request"), dict) else {}
                history.add_entry(
                    entry.get("id", 0),
                    str(entry.get("action", "unknown")),
                    str(entry.get("status", "Unknown")),
                    float(entry.get("duration_s", 0.0)),
                    repo=str(request.get("repo_path", "")),
                    spec=str(request.get("spec", "")),
                    finished_at=str(entry.get("finished_at", "--:--:--")),
                )
        except Exception:
            pass

    def _find_history_record(self, run_id: int) -> dict[str, Any] | None:
        for record in self._run_history:
            if int(record.get("id", -1)) == run_id:
                return record
        return None

    # -----------------------------------------------------------------------
    # Button states
    # -----------------------------------------------------------------------

    def _disable_run_buttons(self) -> None:
        for btn_id in ["run"]:
            try:
                self.query_one(f"#{btn_id}", Button).disabled = True
            except Exception:
                pass
        self._set_form_disabled(True)

    def _enable_run_buttons(self) -> None:
        for btn_id in ["run"]:
            try:
                self.query_one(f"#{btn_id}", Button).disabled = False
            except Exception:
                pass
        self._set_form_disabled(False)

    def _set_form_disabled(self, disabled: bool) -> None:
        field_map = [
            ("action", Select),
            ("repo-path", Input),
            ("spec", Input),
            ("query", Input),
            ("search-arxiv", Checkbox),
            ("search-web", Checkbox),
            ("output-dir", Input),
            ("integrate-dry-run", Checkbox),
            ("integrate-require-clean", Checkbox),
            ("model-provider", Select),
        ]
        for field_id, widget_cls in field_map:
            try:
                widget = self.query_one(f"#{field_id}", widget_cls)
                if widget.display:
                    widget.disabled = disabled
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Core workflow execution
    # -----------------------------------------------------------------------

    def _run_workflow(self, override: dict[str, Any] | None = None) -> None:
        if self._run_in_progress:
            self._set_status("Workflow already running — wait for completion", "warning")
            self._log_to_view(["Warning: run request ignored because another run is active"])
            return

        req = override or self._capture_request()
        req = dict(req)
        action = str(req.get("action", "analyze"))
        repo = str(req.get("repo_path", ""))
        spec = str(req.get("spec", ""))
        req["run_id"] = self._next_run_id

        self._set_phase("validating")

        ok, errors, warnings = self._validate_request_inputs(req)
        if not ok:
            self._set_phase("idle")
            self._set_status("Cannot start — required inputs missing", "error")
            for err in errors:
                self._log_to_view([f"Error: {err}"])
            self._refresh_validation_hint()
            return

        for warning in warnings:
            self._log_to_view([f"Warning: {warning}"])
        if warnings:
            self._set_status("Starting with warnings — review output", "warning")

        if action == "search":
            repo = "(global)"

        self._run_in_progress = True
        self._live_log_count = 0

        self._disable_run_buttons()
        self._set_status(f"Running {action}…", "accent")
        self._set_next_action("running")

        try:
            status_bar = self.query_one(StatusBar)
            status_bar.start_timer()
            step = self._workflow_steps.get(action, (0, 0))
            status_bar.set_step(step[0], step[1])
            status_bar.set_center("workflow: running")
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

            prev_env = self._apply_provider_env()
            try:
                if action == "analyze":
                    result = run_analyze(str(req.get("repo_path", "")), log_callback=_emit)
                elif action == "suggest":
                    result = run_suggest(str(req.get("repo_path", "")), log_callback=_emit)
                elif action == "search":
                    result = run_search(
                        str(req.get("query") or "layer normalization"),
                        include_arxiv=req.get("include_arxiv", False),
                        include_web=req.get("include_web", False),
                        language="python",
                        max_results=max(1, int(req.get("max_results", 10))),
                        log_callback=_emit,
                    )
                elif action == "map":
                    result = run_map(
                        str(req.get("repo_path", "")),
                        spec or "rmsnorm",
                        log_callback=_emit,
                    )
                elif action == "generate":
                    result = run_generate(
                        str(req.get("repo_path", "")),
                        spec or "rmsnorm",
                        output_dir=req.get("output_dir"),
                        log_callback=_emit,
                    )
                elif action == "validate":
                    result = run_validate(str(req.get("repo_path", "")), log_callback=_emit)
                elif action == "integrate":
                    result = run_integrate(
                        str(req.get("repo_path", "")),
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
            finally:
                self._restore_provider_env(prev_env)

        threading.Thread(target=_run, daemon=True).start()
        self._log_to_view([f"Started {action} · target {repo}"])

    # -----------------------------------------------------------------------
    # Event handlers
    # -----------------------------------------------------------------------

    @on(Select.Changed, "#action")
    def on_action_change(self) -> None:
        self._refresh_action_state()

    @on(Select.Changed, "#model-provider")
    def on_model_provider_change(self) -> None:
        self._refresh_provider_chip()

    @on(Input.Changed, "#repo-path")
    @on(Input.Changed, "#spec")
    @on(Input.Changed, "#query")
    def on_field_changed(self) -> None:
        self._refresh_validation_hint()

    @on(Checkbox.Changed, "#search-arxiv")
    @on(Checkbox.Changed, "#search-web")
    @on(Checkbox.Changed, "#integrate-require-clean")
    def on_toggle_changed(self) -> None:
        self._refresh_validation_hint()

    @on(HistoryPane.RunSelected)
    def on_history_selected(self, msg: HistoryPane.RunSelected) -> None:
        record = self._find_history_record(msg.run_id)
        if not record:
            self._set_status(f"Run #{msg.run_id:02d} not found", "warning")
            return

        request = record.get("request")
        if not isinstance(request, dict):
            self._set_status(f"Run #{msg.run_id:02d} has no request payload", "warning")
            return

        if self._run_in_progress:
            self._set_status("Cannot rerun while another workflow is active", "warning")
            return

        self._apply_request(request)
        self._set_status(f"Re-running run #{msg.run_id:02d}", "accent")
        self._set_next_action("running")
        self._run_workflow(dict(request))

    @on(Button.Pressed, "#run")
    def on_run(self) -> None:
        self._run_workflow()

    @on(Button.Pressed, "#clear")
    def on_clear(self) -> None:
        self.action_clear_logs()

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
            env = os.environ.copy()
            provider, model = self._resolve_model_provider()
            if provider:
                env["SCHOLARDEVCLAW_API_PROVIDER"] = provider
            if model:
                env["SCHOLARDEVCLAW_API_MODEL"] = model
            if not env.get("OPENCLAW_TOKEN"):
                env["OPENCLAW_TOKEN"] = "dev-local-token"
            self._agent_process = subprocess.Popen(
                ["bun", "run", "start", "--repl"],
                cwd=agent_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
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
        if provider and model:
            self._add_chat("system", f"model selected: {provider}:{model}")

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
        if self._agent_process and self._agent_process.poll() is None:
            self._agent_running = False
            self._agent_process.terminate()
            self._update_agent_status("Offline")
            self._log_to_legacy("agent-logs", ["Agent stopped"])
        else:
            self._log_to_legacy("agent-logs", ["No running agent"])
            self._update_agent_status("Offline")

    @on(TaskCompleted)
    def on_task_done(self, msg: TaskCompleted) -> None:
        self._run_in_progress = False
        self._set_phase("idle" if msg.error else "complete")
        self._enable_run_buttons()

        if msg.error:
            self._log_to_view([f"Error: {msg.error}"])
            self._set_status(f"Run failed · {msg.title}", "error")
            self._set_next_action("fix and rerun")
        else:
            result_summary = json.dumps(msg.result, indent=2, default=str)
            if len(result_summary) > 2000:
                result_summary = result_summary[:2000] + "\n..."
            self._log_to_view(["", f"=== {msg.title} complete ===", result_summary])
            self._set_status(f"Run complete · {msg.title}", "success")
            self._set_next_action("press Ctrl+Shift+R to rerun")

        if not self._live_logs_enabled:
            self._log_to_view(msg.logs)
        self._live_logs_enabled = False

        try:
            status_bar = self.query_one(StatusBar)
            status_bar.update_timer()
            status_bar.stop_timer()
            status_bar.set_center("workflow: idle")
        except Exception:
            pass

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

            run_id = int(self._active_run_request.get("run_id", 0) or 0)
            if run_id > 0:
                self._log_to_view([f"Run #{run_id:02d} finished in {duration:.1f}s"])

        self._active_run_request = None
        self._active_run_started_at = 0.0
        self._refresh_validation_hint()

    @on(TaskLog)
    def on_log(self, msg: TaskLog) -> None:
        self._sync_phase_from_log_line(msg.line)
        self._log_to_view([msg.line])
        self._live_log_count += 1
        try:
            status_bar = self.query_one(StatusBar)
            status_bar.set_center(f"workflow: running · {self._live_log_count} events")
            status_bar.update_timer()
        except Exception:
            pass

    @on(Input.Submitted, "#prompt-input")
    def on_prompt(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return

        self._set_chat_mode(True)

        if prompt.startswith("/"):
            cmd = prompt[1:].strip().lower()
            event.input.value = ""
            if cmd in {"commands", "cmd", "palette"}:
                self.action_command_palette()
                return
            if cmd in {"new", "new-session", "reset"}:
                self.action_new_session()
                return
            if cmd in {"export", "export-log"}:
                self.action_export_log()
                return
            if cmd in {"clear", "cls"}:
                self.action_clear_logs()
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

    @on(PromptInput.HistoryPrev)
    def on_prompt_history_prev(self) -> None:
        if not self._command_history:
            return
        self._history_index = max(0, self._history_index - 1)
        self.query_one("#prompt-input", PromptInput).value = self._command_history[
            self._history_index
        ]

    @on(PromptInput.HistoryNext)
    def on_prompt_history_next(self) -> None:
        if not self._command_history:
            return
        self._history_index = min(len(self._command_history), self._history_index + 1)
        prompt_input = self.query_one("#prompt-input", PromptInput)
        if self._history_index >= len(self._command_history):
            prompt_input.value = ""
        else:
            prompt_input.value = self._command_history[self._history_index]

    # -----------------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------------

    def _resolve_focus_order(self) -> list[Any]:
        widgets: list[Any] = []
        for selector in self.FOCUS_ORDER:
            try:
                widget = self.query_one(selector)
            except Exception:
                continue
            try:
                if getattr(widget, "disabled", False):
                    continue
            except Exception:
                continue
            if getattr(widget, "display", True):
                widgets.append(widget)
        return widgets

    def _cycle_focus(self, reverse: bool = False) -> None:
        ordered = self._resolve_focus_order()
        if not ordered:
            return

        current = self.focused
        current_index = -1
        for i, widget in enumerate(ordered):
            if widget is current:
                current_index = i
                break

        if current_index < 0:
            target = ordered[-1] if reverse else ordered[0]
        else:
            step = -1 if reverse else 1
            target = ordered[(current_index + step) % len(ordered)]

        try:
            target.focus()
        except Exception:
            pass

    def action_focus_next(self) -> None:
        self._cycle_focus(reverse=False)

    def action_focus_prev(self) -> None:
        self._cycle_focus(reverse=True)

    def action_focus_prompt(self) -> None:
        try:
            self.query_one("#prompt-input", PromptInput).focus()
        except Exception:
            pass

    def action_focus_history(self) -> None:
        try:
            self.query_one("#run-history", HistoryPane).focus()
        except Exception:
            pass

    def action_run_selected(self) -> None:
        self._run_workflow()

    def action_rerun_last(self) -> None:
        if not self._run_history:
            self._set_status("No previous run to rerun", "warning")
            self._set_next_action("run your first workflow")
            return
        if self._run_in_progress:
            self._set_status("Cannot rerun while another workflow is active", "warning")
            return

        latest = self._run_history[0]
        request = latest.get("request") if isinstance(latest.get("request"), dict) else None
        if not request:
            self._set_status("Last run has no stored request", "warning")
            return

        self._apply_request(request)
        self._set_status(f"Re-running latest action: {latest.get('action', 'unknown')}", "accent")
        self._set_next_action("running")
        self._run_workflow(dict(request))

    def action_clear_logs(self) -> None:
        try:
            log_view = self.query_one(LogView)
            log_view.clear_logs()
        except Exception:
            pass
        try:
            self.query_one(ChatLog).clear_entries()
        except Exception:
            pass

    def action_new_session(self) -> None:
        self.action_clear_logs()
        self._run_history.clear()
        self._render_history()
        self._active_run_request = None
        self._active_run_started_at = 0.0
        self._run_in_progress = False
        self._set_phase("idle")
        self._set_chat_mode(False)
        self._set_status("New session started", "success")
        self._log_to_view(["Session cleared. Ready."])
        self._set_next_action("press Ctrl+R")
        self._refresh_validation_hint()

    def action_export_log(self) -> None:
        export_dir = Path.cwd() / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = export_dir / f"tui-log-{stamp}.md"
        try:
            chat = self.query_one(ChatLog)
            path.write_text(chat.export_markdown())
            self._add_chat("system", f"exported log: `{path}`")
            self._set_status(f"Exported log to {path}", "success")
        except Exception as exc:
            self._set_status(f"Export failed: {exc}", "error")

    def action_toggle_config(self) -> None:
        """Toggle config bar visibility."""
        try:
            config_bar = self.query_one("#config-bar")
            self._config_visible = not self._config_visible
            config_bar.display = self._config_visible
            self._set_status(
                "Config panel visible" if self._config_visible else "Config panel hidden",
                "info",
            )
        except Exception:
            pass

    def action_command_palette(self) -> None:
        """Open the command palette overlay."""

        def handle_result(result: str | None) -> None:
            if result is None:
                return
            if result == "quit":
                self.exit()
            elif result == "clear":
                self.action_clear_logs()
            elif result == "new_session":
                self.action_new_session()
            elif result == "export_log":
                self.action_export_log()
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
