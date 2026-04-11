"""Command-first terminal UI for ScholarDevClaw."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Input, Static

from scholardevclaw.application.pipeline import (
    run_analyze,
    run_generate,
    run_integrate,
    run_map,
    run_search,
    run_suggest,
    run_validate,
)
from scholardevclaw.auth.store import AuthStore
from scholardevclaw.auth.types import AuthProvider
from scholardevclaw.llm.client import DEFAULT_MODELS, LLMAPIError, LLMClient, LLMConfigError
from scholardevclaw.security.path_policy import enforce_allowed_repo_path

from .screens import CommandPalette, HelpOverlay, ProviderSetupScreen
from .theme import COLORS as TUI_COLORS
from .widgets import HistoryPane, LogView, PhaseTracker, PromptInput, StatusBar

logger = logging.getLogger(__name__)

MODES = ("analyze", "search", "edit")
SUPPORTED_TUI_PROVIDERS = {
    "openrouter": AuthProvider.OPENROUTER,
    "ollama": AuthProvider.OLLAMA,
}
DEFAULT_OPENROUTER_MODEL = DEFAULT_MODELS[AuthProvider.OPENROUTER]
MODE_HINTS = {
    "analyze": [
        "Hint -> /run analyze ./repo",
        "Hint -> /ask what this repo does",
        "Hint -> runs",
        "Hint -> suggest ./repo",
        "Hint -> validate ./repo",
    ],
    "search": [
        "Hint -> /run search layer normalization",
        "Hint -> /ask papers on flash attention",
        "Hint -> runs",
        "Hint -> search flash attention",
        "Hint -> setup",
    ],
    "edit": [
        "Hint -> /run map ./repo rmsnorm",
        "Hint -> /ask implement RMSNorm",
        "Hint -> runs",
        "Hint -> generate ./repo rmsnorm",
        "Hint -> integrate ./repo rmsnorm",
    ],
}
MODE_COMMANDS = {
    "analyze": [
        "/run analyze ./repo",
        "/ask explain this repository",
        "analyze ./repo",
        "suggest ./repo",
        "validate ./repo",
        "set dir ./repo",
        "set provider openrouter",
        f"set model {DEFAULT_OPENROUTER_MODEL}",
        "runs",
        "run show 1",
        "run rerun 1",
        ":search",
        ":edit",
    ],
    "search": [
        "/run search layer normalization",
        "/ask find fast inference ideas",
        "search layer normalization",
        "search flash attention",
        "set mode search",
        "chat find fast inference ideas",
        "runs",
        "run show 1",
        "run rerun 1",
        "setup",
        ":analyze",
        ":edit",
    ],
    "edit": [
        "/run map ./repo rmsnorm",
        "/ask how should I patch this file",
        "map ./repo rmsnorm",
        "generate ./repo rmsnorm",
        "integrate ./repo rmsnorm",
        "set dir ./repo",
        "chat how should I patch this file",
        "runs",
        "run show 1",
        "run rerun 1",
        ":analyze",
        ":search",
    ],
}
GLOBAL_COMMANDS = [
    "setup",
    "providers",
    "status",
    "/ask hello",
    "/run analyze ./repo",
    "/run generate ./repo rmsnorm",
    "chat hello",
    "set mode analyze",
    "set mode search",
    "set mode edit",
    "set provider openrouter",
    "set provider ollama",
    f"set model {DEFAULT_OPENROUTER_MODEL}",
    "set dir ./repo",
    "runs",
    "run show 1",
    "run rerun 1",
    ":analyze",
    ":search",
    ":edit",
    "help",
    "clear",
]
SPEC_COMMANDS = {
    "rmsnorm",
    "flashattention",
    "swiglu",
    "geglu",
    "gqa",
    "rope",
    "preln_transformer",
    "qknorm",
}
PROGRESS_LABELS = {
    "analyze": "Scanning repository...",
    "suggest": "Finding improvements...",
    "search": "Searching research...",
    "map": "Mapping research to code...",
    "generate": "Generating patch artifacts...",
    "validate": "Validating repository...",
    "integrate": "Running integration workflow...",
    "chat": "Thinking...",
}
WORKFLOW_ACTIONS = {
    "analyze",
    "suggest",
    "search",
    "map",
    "generate",
    "validate",
    "integrate",
}
RUN_CONTEXT_LIMIT = 8
RUN_PERSIST_LIMIT = 20
NATURAL_ACTION_ROUTING_ENV = "SCHOLARDEVCLAW_TUI_ENABLE_NATURAL_ACTION_ROUTING"
AUTO_MODEL_FALLBACK_ENV = "SCHOLARDEVCLAW_TUI_AUTO_MODEL_FALLBACK"
CHAT_SYSTEM_PROMPTS = {
    "analyze": (
        "You are ScholarDevClaw, a terse coding assistant inside a terminal UI. "
        "In analyze mode, prioritize repository architecture/debug/execution questions. "
        "For casual conversation, still respond naturally and briefly. "
        "Keep answers short, concrete, and developer-focused."
    ),
    "search": (
        "You are ScholarDevClaw, a terse research assistant inside a terminal UI. "
        "In search mode, answer with concise research directions, paper names, and implementation tradeoffs. "
        "For casual conversation, respond naturally and briefly."
    ),
    "edit": (
        "You are ScholarDevClaw, a terse coding assistant inside a terminal UI. "
        "In edit mode, focus on code changes, implementation advice, and safe next actions. "
        "For casual conversation, respond naturally and briefly."
    ),
}


class RunLifecycleState(str, Enum):
    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CHATTING = "chatting"


RUN_STATE_LABELS = {
    RunLifecycleState.IDLE: "IDLE",
    RunLifecycleState.QUEUED: "QUEUED",
    RunLifecycleState.RUNNING: "RUNNING",
    RunLifecycleState.COMPLETED: "COMPLETED",
    RunLifecycleState.FAILED: "FAILED",
    RunLifecycleState.CANCELLED: "CANCELLED",
    RunLifecycleState.CHATTING: "CHATTING",
}


@dataclass
class TUIRuntimeState:
    provider: str = "setup"
    model: str = ""
    directory: str = "."
    models_by_provider: dict[str, str] = field(default_factory=dict)
    recent_run_artifacts: list[dict[str, Any]] = field(default_factory=list)
    replay_map: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class RunArtifact:
    run_id: int
    action: str
    status: str
    repo_path: str
    spec: str
    query: str = ""
    duration_seconds: float = 0.0
    terminal_state: str = RunLifecycleState.IDLE.value
    summary_lines: list[str] = field(default_factory=list)


class TaskLog(Message):
    def __init__(self, token: int, line: str):
        super().__init__()
        self.token = token
        self.line = line


class TaskCompleted(Message):
    def __init__(self, token: int, action: str, result: Any, request: dict[str, Any]):
        super().__init__()
        self.token = token
        self.action = action
        self.result = result
        self.request = request


class ChatDelta(Message):
    def __init__(self, token: int, content: str):
        super().__init__()
        self.token = token
        self.content = content


class TaskCancelledError(RuntimeError):
    """Raised when a running TUI task is cancelled cooperatively."""


class ScholarDevClawApp(App[None]):
    """Keyboard-first shell UI."""

    BINDINGS = [
        ("ctrl+c", "cancel_task", "Cancel"),
        ("ctrl+j", "open_command_palette", "Palette"),
        ("ctrl+k", "clear_screen", "Clear"),
        ("ctrl+h", "show_help", "Help"),
        ("escape", "handle_escape", "ESC"),
    ]

    CSS = """
    Screen {
        layout: vertical;
        background: $background;
        color: $text;
        padding: 0 1;
    }

    #header {
        height: 1;
        color: $accent;
        text-style: bold;
    }

    .separator {
        height: 1;
        color: $border;
    }

    #command-meta {
        height: auto;
        color: $text-muted;
    }

    #prompt-input {
        height: 1;
        border: none;
        background: transparent;
        color: $text;
        padding: 0;
    }

    #prompt-input:focus {
        border: none;
        background: transparent;
    }
    """

    STYLES = {
        "background": TUI_COLORS["background"],
        "text": TUI_COLORS["text"],
        "accent": TUI_COLORS["accent"],
        "border": TUI_COLORS["border"],
        "surface": TUI_COLORS["surface"],
        "text-muted": TUI_COLORS["text-muted"],
        "success": TUI_COLORS["success"],
        "warning": TUI_COLORS["warning"],
        "error": TUI_COLORS["error"],
    }

    def __init__(self) -> None:
        super().__init__()
        self._mode = "analyze"
        self._provider = "setup"
        self._model = ""
        self._directory = "."
        self._status_level = "info"
        self._command_history: list[str] = []
        self._history_index = 0
        self._history_draft = ""
        self._suggestions: list[str] = []
        self._hint_index = 0
        self._context_hints: list[str] = []
        self._task_token = 0
        self._active_token = 0
        self._running_action: str | None = None
        self._active_request: dict[str, Any] | None = None
        self._run_state = RunLifecycleState.IDLE
        self._last_escape_time = 0.0
        self._escape_pressed_count = 0
        self._escape_warning_shown = False
        self._line_progress = 0
        self._chat_history: list[dict[str, str]] = []
        self._chat_preview = ""
        self._session_input_tokens = 0
        self._session_output_tokens = 0
        self._last_total_tokens = 0
        self._cancel_events: dict[int, threading.Event] = {}
        self._task_threads: dict[int, threading.Thread] = {}
        self._run_started_at: dict[int, float] = {}
        self._run_replay_map: dict[int, dict[str, Any]] = {}
        self._recent_run_artifacts: list[RunArtifact] = []
        self._models_by_provider: dict[str, str] = {}
        self._load_runtime_state()
        if not self._model and self._provider in SUPPORTED_TUI_PROVIDERS:
            self._model = DEFAULT_MODELS[SUPPORTED_TUI_PROVIDERS[self._provider]]

    def compose(self) -> ComposeResult:
        yield Static("ScholarDevClaw", id="header")
        yield Static("────────────────────────", classes="separator")
        yield StatusBar(id="status-bar")
        yield Static("────────────────────────", classes="separator")
        yield PhaseTracker(id="phase-tracker")
        yield Static("────────────────────────", classes="separator")
        yield LogView(id="main-output")
        yield Static("────────────────────────", classes="separator")
        yield HistoryPane(id="history-pane")
        yield Static("────────────────────────", classes="separator")
        with Vertical():
            yield Static("", id="command-meta")
            yield PromptInput(placeholder="> ", id="prompt-input")

    def on_mount(self) -> None:
        if self._directory in {"", "."}:
            self._directory = os.getcwd()
        self._startup_preflight()
        self._sync_status_bar()
        self._transition_run_state(RunLifecycleState.IDLE, action="system", detail="ready")
        self._hydrate_history_from_recent_artifacts()
        self._update_command_meta()
        self.set_interval(6.0, self._rotate_hint)
        self.query_one("#prompt-input", PromptInput).focus()
        self.set_timer(0.01, self._maybe_show_setup)

    def _hydrate_history_from_recent_artifacts(self) -> None:
        if not self._recent_run_artifacts:
            return
        pane = self.query_one("#history-pane", HistoryPane)
        pane.clear_history()
        for artifact in self._recent_run_artifacts[-RUN_PERSIST_LIMIT:]:
            try:
                pane.add_entry(
                    run_id=artifact.run_id,
                    action=artifact.action,
                    status=artifact.status,
                    duration=max(0.0, artifact.duration_seconds),
                    repo=artifact.repo_path,
                    spec=artifact.spec,
                    finished_at="--:--:--",
                )
            except Exception:
                break

    # ------------------------------------------------------------------
    # Runtime configuration
    # ------------------------------------------------------------------

    def _runtime_state_path(self) -> Path:
        store = AuthStore(enable_audit=False, enable_rate_limit=False)
        return store.store_dir / "tui.json"

    def _load_runtime_state(self) -> None:
        state = TUIRuntimeState()
        config_path = self._runtime_state_path()
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text())
                models_by_provider = data.get("models_by_provider")
                if not isinstance(models_by_provider, dict):
                    models_by_provider = {}
                recent_run_artifacts = data.get("recent_run_artifacts")
                if not isinstance(recent_run_artifacts, list):
                    recent_run_artifacts = []
                replay_map = data.get("replay_map")
                if not isinstance(replay_map, dict):
                    replay_map = {}
                state = TUIRuntimeState(
                    provider=str(data.get("provider", state.provider)),
                    model=str(data.get("model", state.model)),
                    directory=str(data.get("directory", state.directory)),
                    models_by_provider={
                        str(k): str(v)
                        for k, v in models_by_provider.items()
                        if isinstance(k, str) and isinstance(v, str)
                    },
                    recent_run_artifacts=[item for item in recent_run_artifacts],
                    replay_map={str(k): v for k, v in replay_map.items()},
                )
            except Exception:
                pass

        env_provider = os.environ.get("SCHOLARDEVCLAW_API_PROVIDER", "").strip().lower()
        env_model = os.environ.get("SCHOLARDEVCLAW_API_MODEL", "").strip()
        if env_provider in SUPPORTED_TUI_PROVIDERS:
            state.provider = env_provider
        if env_model:
            state.model = env_model

        if state.provider == "setup":
            try:
                auth_status = AuthStore(enable_audit=False, enable_rate_limit=False).get_status()
                if auth_status.provider in SUPPORTED_TUI_PROVIDERS:
                    state.provider = str(auth_status.provider)
            except Exception:
                pass

        self._provider = state.provider
        self._model = state.model
        self._directory = state.directory or "."
        self._models_by_provider = dict(state.models_by_provider)
        self._recent_run_artifacts = self._deserialize_recent_run_artifacts(
            state.recent_run_artifacts
        )
        self._run_replay_map = self._deserialize_replay_map(state.replay_map)
        if self._run_replay_map:
            self._task_token = max(self._run_replay_map)

    @staticmethod
    def _coerce_run_state(value: str | RunLifecycleState) -> RunLifecycleState:
        raw = (
            value.value
            if isinstance(value, RunLifecycleState)
            else str(value or "").strip().lower()
        )
        try:
            return RunLifecycleState(raw)
        except ValueError:
            return RunLifecycleState.IDLE

    @staticmethod
    def _status_to_terminal_state(status: str) -> RunLifecycleState:
        normalized = str(status or "").strip().lower()
        if normalized == "success":
            return RunLifecycleState.COMPLETED
        if normalized == "cancelled":
            return RunLifecycleState.CANCELLED
        if normalized == "failed":
            return RunLifecycleState.FAILED
        return RunLifecycleState.IDLE

    @staticmethod
    def _sanitize_request_for_persistence(request: dict[str, Any]) -> dict[str, Any]:
        allowed_keys = {
            "action",
            "repo_path",
            "spec",
            "query",
            "include_arxiv",
            "include_web",
            "integrate_require_clean",
        }
        payload: dict[str, Any] = {}
        for key in allowed_keys:
            value = request.get(key)
            if value is None:
                continue
            if isinstance(value, (str, bool, int, float)):
                payload[key] = value
        return payload

    def _deserialize_recent_run_artifacts(self, rows: list[Any]) -> list[RunArtifact]:
        artifacts: list[RunArtifact] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            run_id_value = row.get("run_id")
            if isinstance(run_id_value, bool):
                continue
            if isinstance(run_id_value, int):
                run_id = run_id_value
            elif isinstance(run_id_value, float):
                run_id = int(run_id_value)
            elif isinstance(run_id_value, str):
                try:
                    run_id = int(run_id_value.strip())
                except Exception:
                    continue
            else:
                continue
            status = str(row.get("status", "")).strip() or "Unknown"
            terminal_state = self._coerce_run_state(
                str(row.get("terminal_state", RunLifecycleState.IDLE.value))
            )
            summary_lines_raw = row.get("summary_lines", [])
            summary_lines = (
                [str(line).strip() for line in summary_lines_raw if str(line).strip()][:4]
                if isinstance(summary_lines_raw, list)
                else []
            )
            try:
                duration_seconds = float(row.get("duration_seconds", 0.0) or 0.0)
            except Exception:
                duration_seconds = 0.0
            artifacts.append(
                RunArtifact(
                    run_id=run_id,
                    action=str(row.get("action", "")).strip() or "unknown",
                    status=status,
                    repo_path=str(row.get("repo_path", "") or ""),
                    spec=str(row.get("spec", "") or ""),
                    query=str(row.get("query", "") or ""),
                    duration_seconds=max(0.0, duration_seconds),
                    terminal_state=terminal_state.value,
                    summary_lines=summary_lines,
                )
            )
        artifacts.sort(key=lambda item: item.run_id)
        return artifacts[-RUN_PERSIST_LIMIT:]

    def _deserialize_replay_map(self, replay_map: dict[str, Any]) -> dict[int, dict[str, Any]]:
        loaded: dict[int, dict[str, Any]] = {}
        for run_id_raw, entry in replay_map.items():
            try:
                run_id = int(run_id_raw)
            except Exception:
                continue
            if not isinstance(entry, dict):
                continue
            action = str(entry.get("action", "")).strip().lower()
            if not action or action == "chat":
                continue
            command = str(entry.get("command", "") or "").strip()
            request = self._sanitize_request_for_persistence(dict(entry.get("request") or {}))
            terminal_state = self._coerce_run_state(
                str(entry.get("terminal_state", RunLifecycleState.IDLE.value))
            )
            try:
                duration_seconds = float(entry.get("duration_seconds", 0.0) or 0.0)
            except Exception:
                duration_seconds = 0.0
            loaded[run_id] = {
                "command": command,
                "action": action,
                "request": request,
                "terminal_state": terminal_state.value,
                "status": str(entry.get("status", "")).strip() or "Unknown",
                "duration_seconds": max(0.0, duration_seconds),
                "summary_lines": [
                    str(line).strip()
                    for line in list(entry.get("summary_lines") or [])
                    if str(line).strip()
                ][:4],
            }
        for stale in sorted(loaded)[:-RUN_PERSIST_LIMIT]:
            loaded.pop(stale, None)
        return loaded

    def _model_for_provider(self, provider: str) -> str:
        if provider in self._models_by_provider and self._models_by_provider[provider].strip():
            return self._models_by_provider[provider].strip()
        auth_provider = SUPPORTED_TUI_PROVIDERS.get(provider)
        if auth_provider is not None:
            return DEFAULT_MODELS[auth_provider]
        return self._model or ""

    def _remember_model_for_provider(self, provider: str, model: str) -> None:
        provider_name = (provider or "").strip().lower()
        model_name = (model or "").strip()
        if not provider_name or not model_name:
            return
        if provider_name in SUPPORTED_TUI_PROVIDERS:
            self._models_by_provider[provider_name] = model_name

    def _save_runtime_state(self) -> None:
        config_path = self._runtime_state_path()
        payload = {
            "provider": self._provider,
            "model": self._model,
            "directory": self._directory,
            "models_by_provider": self._models_by_provider,
            "recent_run_artifacts": self._serialize_recent_run_artifacts(),
            "replay_map": self._serialize_replay_map(),
        }
        try:
            config_path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            message = f"Warning: failed to save TUI runtime state ({exc})"
            logger.warning(message)
            self._notify_runtime_state_warning(message)

    def _serialize_recent_run_artifacts(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for artifact in self._recent_run_artifacts[-RUN_PERSIST_LIMIT:]:
            if artifact.action == "chat":
                continue
            rows.append(
                {
                    "run_id": artifact.run_id,
                    "action": artifact.action,
                    "status": artifact.status,
                    "repo_path": artifact.repo_path,
                    "spec": artifact.spec,
                    "query": artifact.query,
                    "duration_seconds": artifact.duration_seconds,
                    "terminal_state": self._coerce_run_state(artifact.terminal_state).value,
                    "summary_lines": artifact.summary_lines[:4],
                }
            )
        return rows[-RUN_PERSIST_LIMIT:]

    def _serialize_replay_map(self) -> dict[str, dict[str, Any]]:
        compact: dict[str, dict[str, Any]] = {}
        run_ids = sorted(self._run_replay_map)
        if len(run_ids) > RUN_PERSIST_LIMIT:
            run_ids = run_ids[-RUN_PERSIST_LIMIT:]
        for run_id in run_ids:
            entry = dict(self._run_replay_map.get(run_id) or {})
            action = str(entry.get("action", "") or "").strip().lower()
            if not action or action == "chat":
                continue
            compact[str(run_id)] = {
                "command": str(entry.get("command", "") or "").strip(),
                "action": action,
                "request": self._sanitize_request_for_persistence(dict(entry.get("request") or {})),
                "terminal_state": self._coerce_run_state(
                    str(entry.get("terminal_state", RunLifecycleState.IDLE.value))
                ).value,
                "status": str(entry.get("status", "") or "").strip(),
                "duration_seconds": float(entry.get("duration_seconds", 0.0) or 0.0),
                "summary_lines": [
                    str(line).strip()
                    for line in list(entry.get("summary_lines") or [])
                    if str(line).strip()
                ][:4],
            }
        return compact

    def _pretty_directory(self) -> str:
        try:
            return str(Path(self._directory).expanduser()).replace(str(Path.home()), "~", 1)
        except Exception:
            return self._directory

    def _resolve_auth_provider(self) -> AuthProvider | None:
        return SUPPORTED_TUI_PROVIDERS.get(self._provider)

    def _get_saved_key_for_provider(self, provider: AuthProvider) -> str | None:
        try:
            store = AuthStore(enable_audit=False, enable_rate_limit=False)
            config = store.get_config()
            active = config.get_active_key(provider)
            if active and active.provider == provider and active.is_valid():
                value = (active.key or "").strip()
                if value:
                    return value

            # Fallback for older stores where default selection may be stale.
            for key in store.list_api_keys():
                if key.provider == provider and key.is_valid():
                    value = (key.key or "").strip()
                    if value:
                        return value
        except Exception:
            return None
        return None

    def _provider_has_credentials(self, provider: str | None = None) -> bool:
        provider_name = provider or self._provider
        auth_provider = SUPPORTED_TUI_PROVIDERS.get(provider_name)
        if auth_provider is None:
            return False
        if auth_provider == AuthProvider.OLLAMA:
            return True
        if (os.environ.get(auth_provider.env_var_name) or "").strip():
            return True
        return bool(self._get_saved_key_for_provider(auth_provider))

    def _llm_ready(self) -> bool:
        return (
            self._provider in SUPPORTED_TUI_PROVIDERS
            and bool(self._model)
            and self._provider_has_credentials()
        )

    def _ollama_reachable(self, timeout: float = 1.0) -> bool:
        host = (os.environ.get("OLLAMA_HOST") or "").strip() or (
            AuthProvider.OLLAMA.default_base_url or "http://localhost:11434"
        )
        try:
            resp = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def _startup_preflight(self) -> None:
        changed = False

        # Ensure directory exists.
        try:
            current_dir = str(Path(self._directory).expanduser())
            if not Path(current_dir).exists() or not Path(current_dir).is_dir():
                self._directory = os.getcwd()
                changed = True
                self._append_output(
                    f"Startup: previous directory missing; using {self._pretty_directory()}",
                    "warning",
                )
        except Exception:
            self._directory = os.getcwd()
            changed = True

        # Normalize provider/model.
        if self._provider not in SUPPORTED_TUI_PROVIDERS and self._provider != "setup":
            self._provider = "setup"
            changed = True

        if self._provider in SUPPORTED_TUI_PROVIDERS:
            target_model = self._model or self._model_for_provider(self._provider)
            if target_model != self._model:
                self._model = target_model
                changed = True
            self._remember_model_for_provider(self._provider, self._model)

            if self._provider == "ollama" and not self._ollama_reachable(timeout=1.0):
                self._append_output(
                    "Startup: Ollama selected but not reachable at OLLAMA_HOST. "
                    "Run `ollama serve` or `set provider openrouter`.",
                    "warning",
                )
                self._set_status("Ollama unavailable", "warning")

        if changed:
            self._save_runtime_state()

    def _maybe_show_setup(self) -> None:
        if self._llm_ready():
            return
        self._open_setup()

    def _open_setup(self) -> None:
        self.push_screen(
            ProviderSetupScreen(
                provider=self._provider
                if self._provider in SUPPORTED_TUI_PROVIDERS
                else "openrouter",
                model=self._model or DEFAULT_MODELS[AuthProvider.OPENROUTER],
                has_saved_key=self._provider_has_credentials("openrouter"),
            ),
            self._apply_setup_result,
        )

    def _apply_setup_result(self, result: dict[str, str] | None) -> None:
        if result is None:
            self._append_output("LLM setup skipped", "warning")
            self._set_status("Offline mode", "warning")
            return

        ok, message = self._save_provider_setup(
            result.get("provider", ""),
            result.get("model", ""),
            result.get("api_key", ""),
        )
        if not ok:
            self._append_output(f"Error: {message}", "error")
            self._set_status("Setup failed", "error")
            return

        self._append_output(f"Provider: {self._provider}", "accent")
        self._append_output(f"Model: {self._model}")
        self._set_status("LLM ready", "success")
        self._sync_status_bar()
        self._update_command_meta()

    def _save_provider_setup(
        self, provider: str, model: str, api_key: str = ""
    ) -> tuple[bool, str]:
        provider_name = provider.strip().lower()
        auth_provider = SUPPORTED_TUI_PROVIDERS.get(provider_name)
        if auth_provider is None:
            return False, "Provider must be openrouter or ollama"
        if not model.strip():
            return False, "Model is required"

        store = AuthStore(enable_audit=False, enable_rate_limit=False)
        existing = None
        for key in store.list_api_keys():
            if key.provider == auth_provider:
                existing = key
                if api_key and key.key == api_key:
                    break

        try:
            if auth_provider == AuthProvider.OPENROUTER:
                if api_key:
                    if existing and existing.key == api_key:
                        store.set_default_key(existing.id)
                    else:
                        store.add_api_key(
                            api_key,
                            "openrouter-tui",
                            auth_provider,
                            set_default=True,
                            validate=True,
                            metadata={"source": "tui"},
                        )
                    os.environ[auth_provider.env_var_name] = api_key
                else:
                    saved = self._get_saved_key_for_provider(auth_provider)
                    env_key = (os.environ.get(auth_provider.env_var_name) or "").strip()
                    if saved:
                        os.environ[auth_provider.env_var_name] = saved
                    elif env_key:
                        os.environ[auth_provider.env_var_name] = env_key
                    else:
                        return False, "OpenRouter requires an API key"
            else:
                if existing is not None:
                    store.set_default_key(existing.id)
                else:
                    store.add_api_key(
                        "ollama-local",
                        "ollama-local",
                        auth_provider,
                        set_default=True,
                        metadata={"source": "tui"},
                    )
                os.environ.setdefault(
                    "OLLAMA_HOST", auth_provider.default_base_url or "http://localhost:11434"
                )
        except Exception as exc:
            return False, str(exc)

        self._provider = provider_name
        self._model = model.strip()
        self._remember_model_for_provider(provider_name, self._model)
        self._save_runtime_state()
        return True, "OK"

    # ------------------------------------------------------------------
    # Validation helpers used by tests and command execution
    # ------------------------------------------------------------------

    def _validate_repo_path(self, path: str) -> tuple[bool, str]:
        if not path or not path.strip():
            return False, "Repository path is required"
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            return False, "Repository not found"
        if not resolved.is_dir():
            return False, "Repository path must be a directory"
        try:
            enforce_allowed_repo_path(resolved)
        except PermissionError as exc:
            return False, str(exc)
        return True, ""

    def _validate_spec(self, spec: str) -> tuple[bool, str]:
        if not spec:
            return False, "Specification is required"
        if spec not in SPEC_COMMANDS:
            return False, "Unknown spec"
        return True, ""

    def _check_git_status(self, path: str) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except Exception:
            return False, ""
        if result.returncode != 0:
            return False, ""
        if result.stdout.strip():
            return True, "Repository has uncommitted changes"
        return False, ""

    def _resolve_model_provider(self) -> tuple[str | None, str | None]:
        if self._provider in SUPPORTED_TUI_PROVIDERS:
            return self._provider, self._model or DEFAULT_MODELS[
                SUPPORTED_TUI_PROVIDERS[self._provider]
            ]
        if not self._model or self._model == "auto":
            return None, None
        if ":" in self._model:
            provider, model = self._model.split(":", 1)
            return provider or None, model or None
        return "openai", self._model

    def _apply_provider_env(self) -> dict[str, str | None]:
        prev = {
            "SCHOLARDEVCLAW_API_PROVIDER": os.environ.get("SCHOLARDEVCLAW_API_PROVIDER"),
            "SCHOLARDEVCLAW_API_MODEL": os.environ.get("SCHOLARDEVCLAW_API_MODEL"),
            "SCHOLARDEVCLAW_API_KEY": os.environ.get("SCHOLARDEVCLAW_API_KEY"),
        }
        provider, model = self._resolve_model_provider()
        auth_provider = self._resolve_auth_provider()
        provider_env_var = auth_provider.env_var_name if auth_provider else None
        if provider_env_var:
            prev[provider_env_var] = os.environ.get(provider_env_var)
        if provider is None:
            os.environ.pop("SCHOLARDEVCLAW_API_PROVIDER", None)
            os.environ.pop("SCHOLARDEVCLAW_API_MODEL", None)
            os.environ.pop("SCHOLARDEVCLAW_API_KEY", None)
        else:
            os.environ["SCHOLARDEVCLAW_API_PROVIDER"] = provider
            if model:
                os.environ["SCHOLARDEVCLAW_API_MODEL"] = model
            else:
                os.environ.pop("SCHOLARDEVCLAW_API_MODEL", None)
            key = None
            if auth_provider == AuthProvider.OLLAMA:
                os.environ.setdefault(
                    "OLLAMA_HOST", auth_provider.default_base_url or "http://localhost:11434"
                )
            elif provider_env_var:
                assert auth_provider is not None
                key = self._get_saved_key_for_provider(auth_provider) or os.environ.get(
                    provider_env_var
                )
            if provider_env_var and key:
                os.environ[provider_env_var] = key
                os.environ["SCHOLARDEVCLAW_API_KEY"] = key
        return prev

    def _restore_provider_env(self, prev: dict[str, str | None]) -> None:
        for key, value in prev.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _parse_natural_command(self, prompt: str) -> tuple[str, dict[str, Any]]:
        lower = prompt.lower().strip()
        tokens = prompt.strip().split()
        ctx: dict[str, Any] = {}
        normalized = lower
        prefixes = (
            "please ",
            "can you ",
            "could you ",
            "would you ",
            "lets ",
            "let's ",
            "run ",
        )
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix) :].strip()
                    changed = True
                    break

        action = "chat"
        action_heads = {
            "integrate": "integrate",
            "generate": "generate",
            "patch": "generate",
            "validate": "validate",
            "benchmark": "validate",
            "suggest": "suggest",
            "search": "search",
            "find paper": "search",
            "map": "map",
            "analyze": "analyze",
            "analyse": "analyze",
            "scan": "analyze",
        }
        for head, resolved in action_heads.items():
            if normalized == head or normalized.startswith(f"{head} "):
                action = resolved
                break

        for token in tokens:
            if token.startswith("./") or token.startswith("/") or token.startswith("../"):
                ctx["repo_path"] = token
                break

        for token in tokens:
            if token.lower() in SPEC_COMMANDS:
                ctx["spec"] = token.lower()
                break

        if action == "search":
            query = prompt
            for prefix in ("search", "find", "look up"):
                if lower.startswith(prefix):
                    query = prompt[len(prefix) :].strip()
                    break
            ctx["query"] = query.strip()
        elif action == "chat":
            ctx["prompt"] = prompt.strip()

        return action, ctx

    def _validate_request_inputs(self, req: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
        action = str(req.get("action", ""))
        repo = str(req.get("repo_path", "") or "")
        spec = str(req.get("spec", "") or "")
        query = str(req.get("query", "") or "")
        include_arxiv = bool(req.get("include_arxiv", False))
        include_web = bool(req.get("include_web", False))
        errors: list[str] = []
        warnings: list[str] = []

        if action in {"analyze", "suggest", "map", "generate", "validate", "integrate"}:
            valid, err = self._validate_repo_path(repo)
            if not valid:
                errors.append(err)

        if action in {"map", "generate", "integrate"}:
            valid, err = self._validate_spec(spec)
            if not valid:
                errors.append(err)

        if action == "search" and not query.strip():
            errors.append("Search query is required")

        if action == "search" and not include_arxiv and not include_web:
            warnings.append("Only local spec index will be searched")

        if action == "integrate" and req.get("integrate_require_clean"):
            dirty, err = self._check_git_status(repo)
            if dirty and err:
                warnings.append(err)

        return not errors, errors, warnings

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _sync_status_bar(self) -> None:
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.set_context(
            mode=self._mode,
            provider=self._provider,
            model=self._model or "unset",
            directory=self._pretty_directory(),
        )
        status_bar.set_usage(
            session_tokens=self._session_input_tokens + self._session_output_tokens,
            last_tokens=self._last_total_tokens,
        )

    def _set_phase(self, phase: str) -> None:
        try:
            self.query_one("#phase-tracker", PhaseTracker).set_phase(phase)
        except Exception:
            pass

    def _phase_for_action(self, action: str) -> str:
        return {
            "analyze": "analyzing",
            "suggest": "research",
            "search": "research",
            "map": "mapping",
            "generate": "generating",
            "validate": "validating_patches",
            "integrate": "validating_patches",
            "chat": "research",
        }.get(action, "validating")

    def _phase_for_run_state(self, state: RunLifecycleState, action: str | None = None) -> str:
        if state == RunLifecycleState.QUEUED:
            return "validating"
        if state in {RunLifecycleState.RUNNING, RunLifecycleState.CHATTING}:
            return self._phase_for_action(action or self._running_action or "analyze")
        if state == RunLifecycleState.COMPLETED:
            return "complete"
        return "idle"

    def _transition_run_state(
        self,
        state: RunLifecycleState,
        *,
        action: str | None = None,
        detail: str = "",
    ) -> None:
        self._run_state = self._coerce_run_state(state)
        self._set_phase(self._phase_for_run_state(self._run_state, action))
        action_name = (action or self._running_action or "run").replace("_", " ")
        label = RUN_STATE_LABELS[self._run_state]
        if detail:
            message = f"Run state: {label} | {action_name} | {detail}"
        else:
            message = f"Run state: {label} | {action_name}"
        level = {
            RunLifecycleState.IDLE: "info",
            RunLifecycleState.QUEUED: "accent",
            RunLifecycleState.RUNNING: "accent",
            RunLifecycleState.CHATTING: "accent",
            RunLifecycleState.COMPLETED: "success",
            RunLifecycleState.FAILED: "error",
            RunLifecycleState.CANCELLED: "warning",
        }[self._run_state]
        self._set_status(message, level)

    def _notify_runtime_state_warning(self, message: str) -> None:
        try:
            self._append_output(message, "warning")
            self._set_status("Runtime State Warning", "warning")
        except Exception:
            pass

    @staticmethod
    def _completion_status_text(action: str, status: str) -> str:
        action_label = action.replace("_", " ").title()
        status_label = status.replace("_", " ").title()
        return f"{action_label} {status_label}"

    def _add_history_entry(
        self,
        *,
        run_id: int,
        action: str,
        status: str,
        duration: float,
        terminal_state: RunLifecycleState,
        summary_lines: list[str] | None = None,
        request: dict[str, Any] | None = None,
        command: str | None = None,
    ) -> None:
        req = request or {}
        repo = str(req.get("repo_path", "") or "")
        spec = str(req.get("spec", "") or "")
        try:
            self.query_one("#history-pane", HistoryPane).add_entry(
                run_id=run_id,
                action=action,
                status=status,
                duration=max(0.0, duration),
                repo=repo,
                spec=spec,
                finished_at=time.strftime("%H:%M:%S"),
            )
        except Exception:
            pass

        self._run_replay_map[run_id] = {
            "command": command,
            "action": action,
            "request": self._sanitize_request_for_persistence(dict(req)),
            "status": status,
            "duration_seconds": max(0.0, duration),
            "terminal_state": self._coerce_run_state(terminal_state).value,
            "summary_lines": [
                str(line).strip() for line in list(summary_lines or []) if str(line).strip()
            ][:4],
        }
        for stale in sorted(self._run_replay_map)[:-RUN_PERSIST_LIMIT]:
            self._run_replay_map.pop(stale, None)

    def _get_recent_runs(self, limit: int = RUN_CONTEXT_LIMIT) -> list[RunArtifact]:
        return list(self._recent_run_artifacts[-limit:])[::-1]

    def _find_run_artifact(self, run_id: int) -> RunArtifact | None:
        for artifact in reversed(self._recent_run_artifacts):
            if artifact.run_id == run_id:
                return artifact
        return None

    def _render_runs_compact(self) -> list[str]:
        runs = self._get_recent_runs(limit=RUN_CONTEXT_LIMIT)
        if not runs:
            return ["No recent runs recorded"]
        return [
            (f"#{run.run_id} {run.action:<9} {run.status:<9} {max(0.0, run.duration_seconds):.1f}s")
            for run in runs
        ]

    def _render_run_details(self, run_id: int) -> list[str]:
        artifact = self._find_run_artifact(run_id)
        if artifact is None:
            return [f"Run #{run_id} not found"]

        replay = dict(self._run_replay_map.get(run_id) or {})
        request = dict(replay.get("request") or {})
        lines = [
            f"Run #{artifact.run_id}",
            f"Action: {artifact.action}",
            f"Status: {artifact.status}",
            f"Terminal state: {self._coerce_run_state(artifact.terminal_state).value}",
            f"Duration: {max(0.0, artifact.duration_seconds):.1f}s",
            f"Repo: {request.get('repo_path') or artifact.repo_path or 'n/a'}",
            f"Spec: {request.get('spec') or artifact.spec or 'n/a'}",
            f"Query: {request.get('query') or artifact.query or 'n/a'}",
        ]
        if request:
            request_parts: list[str] = []
            for key in ("action", "repo_path", "spec", "query", "include_arxiv", "include_web"):
                if key in request:
                    request_parts.append(f"{key}={request[key]}")
            if request_parts:
                lines.append("Request: " + ", ".join(request_parts))
        lines.append("Summary:")
        for line in artifact.summary_lines or ["(none)"]:
            lines.append(f"- {line}")
        return lines

    def _set_status(self, message: str, level: str = "info") -> None:
        self._status_level = level
        self.query_one("#status-bar", StatusBar).set_status(message, level)

    def _append_output(self, line: str, level: str = "auto") -> None:
        self.query_one("#main-output", LogView).add_log(line, level)

    def _clear_output(self) -> None:
        self.query_one("#main-output", LogView).clear_logs()

    def _record_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        self._session_input_tokens += max(0, input_tokens)
        self._session_output_tokens += max(0, output_tokens)
        self._last_total_tokens = max(0, input_tokens + output_tokens)
        self._sync_status_bar()

    @staticmethod
    def _env_flag_enabled(name: str, *, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        stripped = text.strip()
        if not stripped:
            return 0
        return max(1, len(stripped) // 4)

    def _set_progress(self, action: str, fraction: float, label: str | None = None) -> None:
        text = label or PROGRESS_LABELS.get(action, "Working...")
        line = f"{text} {self._progress_bar(fraction)}"
        self.query_one("#main-output", LogView).set_progress(line, "system")

    def _clear_progress(self) -> None:
        self.query_one("#main-output", LogView).clear_progress()

    def _set_live_text(self, text: str, level: str = "info") -> None:
        self.query_one("#main-output", LogView).set_progress(text, level)

    def _progress_bar(self, fraction: float) -> str:
        width = 10
        clamped = max(0.0, min(1.0, fraction))
        filled = int(round(width * clamped))
        return f"[{'█' * filled}{'░' * (width - filled)}] {int(clamped * 100):>3d}%"

    def _emit_progress(self, action: str, fraction: float, label: str | None = None) -> None:
        self._set_progress(action, fraction, label)

    def _rotate_hint(self) -> None:
        if self._suggestions:
            return
        self._hint_index = (self._hint_index + 1) % len(MODE_HINTS[self._mode])
        self._update_command_meta()

    def _all_commands(self) -> list[str]:
        command_dir = self._directory if self._directory not in {"", "."} else "./repo"
        contextual = [
            f"/run analyze {command_dir}",
            f"/run generate {command_dir} rmsnorm",
            "/ask explain this repository",
            f"analyze {command_dir}",
            f"suggest {command_dir}",
            f"validate {command_dir}",
            f"map {command_dir} rmsnorm",
            f"generate {command_dir} rmsnorm",
            f"integrate {command_dir} rmsnorm",
            "setup",
            "providers",
            "status",
            "runs",
            "run show 1",
            "run rerun 1",
            "chat hello",
        ]
        commands = MODE_COMMANDS[self._mode] + contextual + self._context_hints + GLOBAL_COMMANDS
        return list(dict.fromkeys(commands))

    @staticmethod
    def _fuzzy_score(prompt: str, candidate: str) -> tuple[int, int, int]:
        needle = prompt.lower().strip()
        hay = candidate.lower()
        if not needle:
            return (0, 0, 0)

        if hay == needle:
            return (7, len(needle), -len(candidate))
        if hay.startswith(needle):
            return (6, len(needle), -len(candidate))

        tokens = hay.replace(":", " ").split()
        for token in tokens:
            if token.startswith(needle):
                return (5, len(needle), -len(candidate))
        if f" {needle}" in hay:
            return (4, len(needle), -len(candidate))

        cursor = 0
        matches = 0
        for char in hay:
            if cursor < len(needle) and char == needle[cursor]:
                cursor += 1
                matches += 1
        if cursor == len(needle):
            return (3, matches, -len(candidate))

        overlap = sum(1 for char in set(needle) if char in hay)
        if overlap:
            return (2, overlap, -len(candidate))
        return (0, 0, -len(candidate))

    def _compute_suggestions(self, prompt: str) -> list[str]:
        prompt = prompt.strip()
        commands = self._all_commands()
        if not prompt:
            return []
        scored = [(self._fuzzy_score(prompt, candidate), candidate) for candidate in commands]
        ranked = [
            candidate
            for score, candidate in sorted(scored, key=lambda item: item[0], reverse=True)
            if score[0] > 0
        ]
        return ranked[:3]

    def _update_command_meta(self) -> None:
        widget = self.query_one("#command-meta", Static)
        if self._suggestions:
            lines = []
            for idx, suggestion in enumerate(self._suggestions[:3]):
                prefix = "Suggestion ->" if idx == 0 else "             "
                if idx == 0:
                    lines.append(f"{prefix} [bold $accent]{suggestion}[/]")
                else:
                    lines.append(f"{prefix} [dim]{suggestion}[/]")
            widget.update("\n".join(lines))
            return
        if self._context_hints:
            lines = []
            for idx, hint in enumerate(self._context_hints[:3]):
                prefix = "Next ->" if idx == 0 else "       "
                style = "[bold $accent]" if idx == 0 else "[dim]"
                lines.append(f"{prefix} {style}{hint}[/]")
            widget.update("\n".join(lines))
            return
        widget.update(f"[dim]{MODE_HINTS[self._mode][self._hint_index]}[/]")

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        self._hint_index = 0
        self._context_hints = []
        self._sync_status_bar()
        self._set_status(f"Mode set to {mode}", "accent")
        self._append_output(f"Mode: {mode}", "accent")
        self._update_command_meta()

    def _suggest_next_commands(
        self,
        action: str,
        payload: dict[str, Any],
        request: dict[str, Any],
    ) -> list[str]:
        repo = str(request.get("repo_path") or self._directory)
        spec = str(request.get("spec") or "rmsnorm")
        if action == "analyze":
            return [f"suggest {repo}", f"validate {repo}", ":edit"]
        if action == "suggest":
            top_spec = (
                payload.get("suggestions", [{}])[0].get("spec")
                or payload.get("suggestions", [{}])[0].get("id")
                or spec
            )
            return [f"map {repo} {str(top_spec).lower()}", f"generate {repo} {spec}", ":search"]
        if action == "search":
            return [":edit", f"map {repo} {spec}", "search flash attention"]
        if action == "map":
            return [f"generate {repo} {spec}", f"validate {repo}", ":edit"]
        if action == "generate":
            return [f"validate {repo}", f"integrate {repo} {spec}", ":analyze"]
        if action == "validate":
            return [f"integrate {repo} {spec}", f"analyze {repo}", ":search"]
        if action == "integrate":
            return [f"validate {repo}", f"analyze {repo}", ":search"]
        if action == "chat":
            if self._mode == "search":
                return ["search flash attention", f"analyze {repo}", ":edit"]
            if self._mode == "edit":
                return [f"map {repo} rmsnorm", f"generate {repo} rmsnorm", ":analyze"]
            return [f"analyze {repo}", f"suggest {repo}", ":search"]
        return []

    # ------------------------------------------------------------------
    # Command parsing and execution
    # ------------------------------------------------------------------

    def _build_request(self, command: str) -> tuple[str | None, dict[str, Any]]:
        raw = command.strip()
        if not raw:
            return None, {}

        if raw.startswith(":"):
            payload = raw[1:].strip()
            if payload in MODES:
                return "set_mode", {"mode": payload}
            if payload.startswith("mode "):
                return "set_mode", {"mode": payload.split(" ", 1)[1].strip()}
            if payload.startswith("model "):
                return "set_model", {"model": payload.split(" ", 1)[1].strip()}
            if payload.startswith("provider "):
                return "set_provider", {"provider": payload.split(" ", 1)[1].strip()}
            if payload.startswith("dir "):
                return "set_dir", {"directory": payload.split(" ", 1)[1].strip()}
            if payload == "setup":
                return "setup", {}
            return None, {}

        parts = raw.split()
        head = parts[0].lower()

        if head == "runs":
            return "runs", {}

        if head == "run" and len(parts) >= 2:
            subcommand = parts[1].strip().lower()
            if subcommand == "show":
                if len(parts) < 3:
                    return None, {}
                try:
                    run_id = int(parts[2])
                except ValueError:
                    return None, {}
                return "run_show", {"run_id": run_id}
            if subcommand == "rerun":
                if len(parts) < 3:
                    return None, {}
                try:
                    run_id = int(parts[2])
                except ValueError:
                    return None, {}
                return "run_rerun", {"run_id": run_id}

        if head == "/ask":
            return "chat", {"action": "chat", "prompt": raw[len(parts[0]) :].strip()}

        if head == "/run":
            if len(parts) < 2:
                return None, {}
            namespaced_action = parts[1].strip().lower()
            if namespaced_action not in WORKFLOW_ACTIONS:
                return None, {}
            remainder = raw.split(maxsplit=2)
            namespaced_command = namespaced_action
            if len(remainder) == 3 and remainder[2].strip():
                namespaced_command = f"{namespaced_action} {remainder[2].strip()}"
            return self._build_request(namespaced_command)

        if head == "set" and len(parts) >= 3:
            key = parts[1].lower()
            value = " ".join(parts[2:]).strip()
            if key == "mode":
                return "set_mode", {"mode": value}
            if key == "provider":
                return "set_provider", {"provider": value}
            if key == "model":
                return "set_model", {"model": value}
            if key == "key":
                return "set_key", {"api_key": value}
            if key == "dir":
                return "set_dir", {"directory": value}

        if head in {"help", "clear", "quit", "setup", "providers", "status"}:
            return head, {}

        if head == "chat":
            return "chat", {"action": "chat", "prompt": raw[len(parts[0]) :].strip()}

        if head == "search":
            return "search", {
                "action": "search",
                "query": raw[len(parts[0]) :].strip(),
                "include_arxiv": True,
                "include_web": False,
            }

        if head in {"analyze", "suggest", "validate"}:
            repo_path = self._directory
            if len(parts) > 1:
                candidate = parts[1].strip().lower()
                if candidate not in {
                    "this",
                    "current",
                    "the",
                    "repo",
                    "repository",
                    "project",
                    "here",
                }:
                    repo_path = parts[1]
            return head, {"action": head, "repo_path": repo_path}

        if head in {"map", "generate", "integrate"}:
            repo_path = self._directory
            spec = ""
            remaining_tokens = parts[1:]
            if remaining_tokens:
                first_token = remaining_tokens[0].strip()
                candidate = first_token.lower()
                if candidate in {
                    "this",
                    "current",
                    "the",
                    "repo",
                    "repository",
                    "project",
                    "here",
                }:
                    repo_path = self._directory
                    remaining_tokens = remaining_tokens[1:]
                elif candidate in SPEC_COMMANDS:
                    spec = candidate
                    remaining_tokens = remaining_tokens[1:]
                else:
                    repo_path = first_token
                    remaining_tokens = remaining_tokens[1:]
            if not spec:
                spec = self._extract_spec_from_tokens(remaining_tokens)
            return head, {"action": head, "repo_path": repo_path, "spec": spec}

        action, ctx = self._parse_natural_command(raw)
        if self._env_flag_enabled(NATURAL_ACTION_ROUTING_ENV, default=False):
            if action == "chat":
                return "chat", {"action": "chat", "prompt": ctx.get("prompt", raw)}
            ctx.setdefault("action", action)
            if action == "search":
                ctx.setdefault("include_arxiv", True)
                ctx.setdefault("include_web", False)
            if action != "search":
                ctx.setdefault("repo_path", self._directory)
            return action, ctx

        if action == "chat":
            return "chat", {"action": "chat", "prompt": ctx.get("prompt", raw)}
        return "chat", {"action": "chat", "prompt": raw}

    def _summarize_result(self, action: str, payload: dict[str, Any]) -> list[str]:
        if action == "analyze":
            return [
                f"Languages: {', '.join(payload.get('languages', [])) or 'None'}",
                f"Frameworks: {', '.join(payload.get('frameworks', [])) or 'None'}",
                f"Entry points: {len(payload.get('entry_points', []))}",
                "Next -> suggest ./repo",
            ]
        if action == "suggest":
            suggestions = payload.get("suggestions", [])
            if not suggestions:
                return ["No suggestions found"]
            top = suggestions[0]
            title = top.get("paper", {}).get("title", "unknown")
            return [
                f"Suggestions: {len(suggestions)}",
                f"Top match: {title}",
                "Next -> map ./repo rmsnorm",
            ]
        if action == "search":
            return [
                f"Local specs: {len(payload.get('local', []))}",
                f"arXiv results: {len(payload.get('arxiv', []))}",
                f"Web repos: {len(payload.get('web', {}).get('github_repos', []))}",
            ]
        if action == "map":
            return [
                f"Targets: {payload.get('target_count', 0)}",
                f"Strategy: {payload.get('strategy', 'none')}",
                f"Confidence: {payload.get('confidence', 0)}%",
                "Next -> generate ./repo rmsnorm",
            ]
        if action == "generate":
            return [
                f"Branch: {payload.get('branch_name', '')}",
                f"New files: {len(payload.get('new_files', []))}",
                f"Transformations: {len(payload.get('transformations', []))}",
                "Next -> validate ./repo",
            ]
        if action == "validate":
            scorecard = payload.get("scorecard", {})
            return [
                f"Stage: {payload.get('stage', 'unknown')}",
                f"Passed: {'Yes' if payload.get('passed') else 'No'}",
                f"Summary: {scorecard.get('summary', '')}",
            ]
        if action == "integrate":
            validation = payload.get("validation", {})
            return [
                f"Spec: {payload.get('spec', 'auto')}",
                f"Validation passed: {'Yes' if validation.get('passed') else 'No'}",
                f"Rollback snapshot: {payload.get('rollback_snapshot_id', 'n/a')}",
            ]
        if action == "chat":
            return []
        return []

    def _record_run_artifact(
        self,
        *,
        run_id: int,
        action: str,
        status: str,
        terminal_state: RunLifecycleState,
        duration: float,
        request: dict[str, Any],
        summary_lines: list[str],
    ) -> None:
        if action == "chat" or status not in {"Success", "Failed", "Cancelled"}:
            return
        artifact = RunArtifact(
            run_id=run_id,
            action=action,
            status=status,
            repo_path=str(request.get("repo_path", "") or ""),
            spec=str(request.get("spec", "") or ""),
            query=str(request.get("query", "") or ""),
            duration_seconds=max(0.0, duration),
            terminal_state=self._coerce_run_state(terminal_state).value,
            summary_lines=[line.strip() for line in summary_lines if str(line).strip()][:4],
        )
        self._recent_run_artifacts.append(artifact)
        if len(self._recent_run_artifacts) > RUN_PERSIST_LIMIT:
            self._recent_run_artifacts = self._recent_run_artifacts[-RUN_PERSIST_LIMIT:]
        self._save_runtime_state()

    def _build_recent_run_context(self) -> str:
        if not self._recent_run_artifacts:
            return "(none)"

        lines: list[str] = []
        for artifact in self._recent_run_artifacts[-RUN_CONTEXT_LIMIT:]:
            repo_part = f" repo={artifact.repo_path}" if artifact.repo_path else ""
            spec_part = f" spec={artifact.spec}" if artifact.spec else ""
            summary = " | ".join(artifact.summary_lines) if artifact.summary_lines else "no summary"
            lines.append(
                f"- run #{artifact.run_id}: {artifact.action} [{artifact.status}]{repo_part}{spec_part} :: {summary}"
            )
        return "\n".join(lines)

    def _build_chat_system_prompt(self) -> str:
        base = CHAT_SYSTEM_PROMPTS[self._mode]
        repo_snapshot = self._build_repo_snapshot()
        run_context = self._build_recent_run_context()
        return (
            f"{base} "
            f"Current working directory: {self._pretty_directory()}. "
            "If the user asks to run a repo workflow, mention the exact shell command they can run here. "
            "For short greetings, reply naturally in one short sentence and avoid repo/tooling details unless asked. "
            "For conversational messages (for example: how are you, thanks), respond naturally and briefly without forcing repository context. "
            "Do not mention repository details or cwd unless the user asks about repo/workflow/code tasks. "
            "Grounding rule: only claim facts present in the run context, repo snapshot, or user prompt. "
            "If unknown, say unknown and suggest one concrete `/run ...` command. "
            "Only claim frameworks, libraries, or architecture details when they are explicitly present in the repo snapshot below or user-provided context. "
            "If uncertain, say so briefly and suggest the next concrete command. "
            "Do not pretend you already executed commands unless the transcript shows it. "
            f"Recent run context:\n{run_context}\n"
            f"Repo snapshot: {repo_snapshot}"
        )

    def _format_chat_error(self, exc: Exception) -> str:
        if isinstance(exc, LLMAPIError) and exc.status_code == 429:
            return "Rate limit reached (429). Retry in 30-60s or run `set provider ollama`."
        if isinstance(exc, LLMAPIError) and exc.status_code == 401:
            return "Authentication failed (401). Run `setup` and paste a valid API key."
        if isinstance(exc, LLMAPIError) and exc.status_code == 402:
            return "Provider credits issue (402). Check billing or run `set provider ollama`."
        if isinstance(exc, LLMAPIError) and exc.status_code == 0:
            detail = (exc.detail or "").lower()
            if "no parseable" in detail or "empty" in detail:
                return (
                    "DEGRADED: chat transport returned empty/invalid stream. "
                    "Retry, check provider transport, or run `set model <provider-model>`."
                )
        if isinstance(exc, LLMAPIError) and exc.status_code in {400, 404}:
            detail = (exc.detail or "").lower()
            if "model" in detail or "not found" in detail or "does not exist" in detail:
                return "Model unavailable for this provider. Run `set model <provider-model>` or switch provider."
        if isinstance(exc, LLMConfigError):
            return str(exc)
        return "LLM request failed. Retry, run `setup`, or switch provider."

    @staticmethod
    def _make_result(
        *,
        ok: bool,
        payload: dict[str, Any] | None = None,
        error: str = "",
        logs: list[str] | None = None,
    ) -> Any:
        return type(
            "Result",
            (),
            {
                "ok": ok,
                "payload": payload or {},
                "error": error,
                "logs": logs or ([] if not error else [error]),
            },
        )()

    def _chat_result_from_text(self, prompt: str, response_text: str) -> Any:
        content = (response_text or "").strip()
        if not content:
            message = (
                "DEGRADED: chat transport produced empty response/output. "
                "Retry, verify provider health, or run `set model <provider-model>`."
            )
            return self._make_result(ok=False, payload={}, error=message, logs=[message])

        input_tokens = self._estimate_tokens(prompt)
        output_tokens = self._estimate_tokens(content)
        return self._make_result(
            ok=True,
            payload={
                "content": content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            error="",
            logs=[],
        )

    @staticmethod
    def _extract_spec_from_tokens(tokens: list[str]) -> str:
        skip = {
            "this",
            "current",
            "the",
            "repo",
            "repository",
            "project",
            "here",
            "in",
            "on",
            "for",
        }
        for token in tokens:
            value = token.strip().lower()
            if not value or value in skip:
                continue
            return value
        return ""

    def _build_repo_snapshot(self) -> str:
        try:
            root = Path(self._directory).expanduser().resolve()
            if not root.exists() or not root.is_dir():
                return f"root={self._pretty_directory()} (missing)"

            entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            top_entries = [f"{p.name}/" if p.is_dir() else p.name for p in entries[:14]]
            markers = [
                "pyproject.toml",
                "package.json",
                "requirements.txt",
                "Cargo.toml",
                "go.mod",
                "docker-compose.yml",
                "Dockerfile",
            ]
            present_markers = [name for name in markers if (root / name).exists()]
            top = ", ".join(top_entries) if top_entries else "(empty)"
            marker_text = ", ".join(present_markers) if present_markers else "none"
            return f"root={root}; top={top}; markers={marker_text}"
        except Exception:
            return f"root={self._pretty_directory()}; snapshot unavailable"

    def _get_llm_client(self) -> LLMClient:
        auth_provider = self._resolve_auth_provider()
        if auth_provider is None:
            raise LLMConfigError("Run `setup` to choose OpenRouter or Ollama first.")

        model = self._model or DEFAULT_MODELS[auth_provider]
        if auth_provider == AuthProvider.OLLAMA:
            os.environ.setdefault(
                "OLLAMA_HOST", auth_provider.default_base_url or "http://localhost:11434"
            )
            return LLMClient.from_provider(auth_provider, api_key="", model=model)

        key = self._get_saved_key_for_provider(auth_provider) or os.environ.get(
            auth_provider.env_var_name,
            "",
        )
        key = key.strip()
        if not key:
            raise LLMConfigError("No OpenRouter key found. Run `setup` and paste your key.")
        os.environ[auth_provider.env_var_name] = key
        return LLMClient.from_provider(auth_provider, api_key=key, model=model)

    def _record_command(self, command: str) -> None:
        if command and (not self._command_history or self._command_history[-1] != command):
            self._command_history.append(command)
        self._history_index = len(self._command_history)

    def _is_task_cancelled(self, token: int) -> bool:
        event = self._cancel_events.get(token)
        return bool(event and event.is_set())

    def _start_task(
        self,
        action: str,
        request: dict[str, Any],
        *,
        command: str | None = None,
    ) -> None:
        if self._running_action is not None:
            self._set_status("Run state: RUNNING | another task in progress", "warning")
            return

        ok, errors, warnings = self._validate_request_inputs(request)
        if not ok:
            for error in errors:
                self._append_output(f"Error: {error}", "error")
            self._set_status("Command rejected", "error")
            return

        for warning in warnings:
            self._append_output(f"Warning: {warning}", "warning")

        self._task_token += 1
        self._active_token = self._task_token
        request_payload = dict(request)
        if command:
            request_payload["_original_command"] = command
        self._running_action = action
        self._active_request = request_payload
        self._run_started_at[self._active_token] = time.perf_counter()
        self._line_progress = 0
        self._context_hints = []
        self._cancel_events[self._active_token] = threading.Event()
        self._transition_run_state(RunLifecycleState.QUEUED, action=action)
        self.query_one("#status-bar", StatusBar).start_timer()
        self._transition_run_state(RunLifecycleState.RUNNING, action=action)
        self._emit_progress(action, 0.05)
        self._update_command_meta()

        thread = threading.Thread(
            target=self._run_task_in_thread,
            args=(self._active_token, action, request_payload),
            daemon=True,
        )
        self._task_threads[self._active_token] = thread
        thread.start()

    def _start_chat(self, prompt: str, *, command: str | None = None) -> None:
        if self._running_action is not None:
            self._set_status("Run state: RUNNING | another task in progress", "warning")
            return
        if not self._llm_ready():
            self._append_output("Error: configure OpenRouter or Ollama first", "error")
            self._open_setup()
            self._set_status("LLM setup required", "warning")
            return

        self._task_token += 1
        self._active_token = self._task_token
        request_payload = {"action": "chat", "prompt": prompt}
        if command:
            request_payload["_original_command"] = command
        self._running_action = "chat"
        self._active_request = request_payload
        self._run_started_at[self._active_token] = time.perf_counter()
        self._chat_preview = ""
        self._context_hints = []
        self._cancel_events[self._active_token] = threading.Event()
        self._transition_run_state(RunLifecycleState.CHATTING, action="chat")
        self.query_one("#status-bar", StatusBar).start_timer()
        self._set_live_text("Thinking...", "system")
        self._update_command_meta()

        thread = threading.Thread(
            target=self._run_chat_in_thread,
            args=(self._active_token, prompt, request_payload),
            daemon=True,
        )
        self._task_threads[self._active_token] = thread
        thread.start()

    def _run_task_in_thread(self, token: int, action: str, request: dict[str, Any]) -> None:
        previous_env = self._apply_provider_env()
        try:
            if self._is_task_cancelled(token):
                raise TaskCancelledError("Task cancelled")

            sink_out = io.StringIO()
            sink_err = io.StringIO()

            def _log_callback(line: str) -> None:
                if self._is_task_cancelled(token):
                    raise TaskCancelledError("Task cancelled")
                self.call_from_thread(self.post_message, TaskLog(token, line))

            with contextlib.redirect_stdout(sink_out), contextlib.redirect_stderr(sink_err):
                if action == "analyze":
                    result = run_analyze(request["repo_path"], log_callback=_log_callback)
                elif action == "suggest":
                    result = run_suggest(request["repo_path"], log_callback=_log_callback)
                elif action == "search":
                    result = run_search(
                        request["query"],
                        include_arxiv=bool(request.get("include_arxiv")),
                        include_web=bool(request.get("include_web")),
                        log_callback=_log_callback,
                    )
                elif action == "map":
                    result = run_map(
                        request["repo_path"], request["spec"], log_callback=_log_callback
                    )
                elif action == "generate":
                    result = run_generate(
                        request["repo_path"],
                        request["spec"],
                        log_callback=_log_callback,
                    )
                elif action == "validate":
                    result = run_validate(request["repo_path"], log_callback=_log_callback)
                elif action == "integrate":
                    result = run_integrate(
                        request["repo_path"],
                        request["spec"],
                        log_callback=_log_callback,
                    )
                else:
                    raise RuntimeError(f"Unsupported action: {action}")
        except TaskCancelledError:
            result = type(
                "Result",
                (),
                {
                    "ok": False,
                    "payload": {"cancelled": True},
                    "error": "Task cancelled",
                    "logs": [],
                },
            )()
        except Exception as exc:
            result = type(
                "Result",
                (),
                {"ok": False, "payload": {}, "error": str(exc), "logs": [str(exc)]},
            )()
        finally:
            self._restore_provider_env(previous_env)
            self._task_threads.pop(token, None)
            self._cancel_events.pop(token, None)

        self.call_from_thread(self.post_message, TaskCompleted(token, action, result, request))

    def _run_chat_in_thread(self, token: int, prompt: str, request: dict[str, Any]) -> None:
        response_text = ""
        client: LLMClient | None = None
        try:
            if self._is_task_cancelled(token):
                raise TaskCancelledError("Task cancelled")

            sink_out = io.StringIO()
            sink_err = io.StringIO()
            with contextlib.redirect_stdout(sink_out), contextlib.redirect_stderr(sink_err):
                client = self._get_llm_client()
                messages = self._chat_history[-8:] + [{"role": "user", "content": prompt}]
                for chunk in client.chat_stream(
                    prompt,
                    messages=messages,
                    system=self._build_chat_system_prompt(),
                    model=self._model,
                    max_tokens=2048,
                    temperature=0.2,
                ):
                    if token != self._active_token or self._is_task_cancelled(token):
                        raise TaskCancelledError("Task cancelled")
                    if chunk.delta:
                        response_text += chunk.delta
                        self.call_from_thread(self.post_message, ChatDelta(token, response_text))
            result = self._chat_result_from_text(prompt, response_text)
        except TaskCancelledError:
            result = self._make_result(
                ok=False,
                payload={"cancelled": True},
                error="Task cancelled",
                logs=[],
            )
        except LLMAPIError as exc:
            detail = (exc.detail or "").lower()
            looks_like_bad_model = exc.status_code in {400, 404} and (
                "model" in detail or "not found" in detail or "does not exist" in detail
            )
            auto_fallback = self._env_flag_enabled(AUTO_MODEL_FALLBACK_ENV, default=False)

            if looks_like_bad_model and auto_fallback and self._provider in SUPPORTED_TUI_PROVIDERS:
                fallback_model = DEFAULT_MODELS[SUPPORTED_TUI_PROVIDERS[self._provider]]
                if fallback_model and fallback_model != self._model:
                    try:
                        if client is not None:
                            client.close()
                            client = None

                        self.call_from_thread(
                            self.post_message,
                            TaskLog(
                                token,
                                f"Model unavailable; retrying once with fallback '{fallback_model}'.",
                            ),
                        )

                        with (
                            contextlib.redirect_stdout(io.StringIO()),
                            contextlib.redirect_stderr(io.StringIO()),
                        ):
                            client = self._get_llm_client()
                            messages = self._chat_history[-8:] + [
                                {"role": "user", "content": prompt}
                            ]
                            response_text = ""
                            for chunk in client.chat_stream(
                                prompt,
                                messages=messages,
                                system=self._build_chat_system_prompt(),
                                model=fallback_model,
                                max_tokens=2048,
                                temperature=0.2,
                            ):
                                if token != self._active_token or self._is_task_cancelled(token):
                                    raise TaskCancelledError("Task cancelled")
                                if chunk.delta:
                                    response_text += chunk.delta
                                    self.call_from_thread(
                                        self.post_message, ChatDelta(token, response_text)
                                    )
                        result = self._chat_result_from_text(prompt, response_text)
                    except TaskCancelledError:
                        result = self._make_result(
                            ok=False,
                            payload={"cancelled": True},
                            error="Task cancelled",
                            logs=[],
                        )
                    except Exception as fallback_exc:
                        error_message = self._format_chat_error(fallback_exc)
                        result = self._make_result(ok=False, payload={}, error=error_message)
                else:
                    error_message = self._format_chat_error(exc)
                    result = self._make_result(ok=False, payload={}, error=error_message)
            elif looks_like_bad_model and not auto_fallback:
                result = self._make_result(
                    ok=False,
                    payload={},
                    error=(
                        "DEGRADED: model lookup failed and automatic fallback is disabled "
                        f"({AUTO_MODEL_FALLBACK_ENV}=false). "
                        "Run `set model <provider-model>` or enable fallback explicitly."
                    ),
                )
            else:
                error_message = self._format_chat_error(exc)
                result = self._make_result(ok=False, payload={}, error=error_message)
        except (LLMConfigError, Exception) as exc:
            error_message = self._format_chat_error(exc)
            result = self._make_result(ok=False, payload={}, error=error_message)
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
            self._task_threads.pop(token, None)
            self._cancel_events.pop(token, None)

        self.call_from_thread(
            self.post_message,
            TaskCompleted(token, "chat", result, request),
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    @on(Input.Changed, "#prompt-input")
    def on_prompt_changed(self, event: Input.Changed) -> None:
        self._suggestions = self._compute_suggestions(event.value)
        self._update_command_meta()

    def _execute_action_request(
        self,
        action: str,
        request: dict[str, Any],
        *,
        command: str | None = None,
    ) -> None:
        if action == "help":
            self.push_screen(HelpOverlay())
            return
        if action == "clear":
            self.action_clear_screen()
            return
        if action == "setup":
            self._open_setup()
            return
        if action == "providers":
            self._append_output("Providers: openrouter, ollama")
            self._append_output("Use `setup` to configure credentials and model IDs.")
            return
        if action == "status":
            self._append_output(f"Mode: {self._mode}")
            self._append_output(f"Provider: {self._provider}")
            self._append_output(f"Model: {self._model or 'unset'}")
            self._append_output(f"Directory: {self._pretty_directory()}")
            self._append_output(f"Run state: {RUN_STATE_LABELS[self._run_state]}")
            self._append_output(
                f"Session tokens: {self._session_input_tokens + self._session_output_tokens}"
            )
            return
        if action == "runs":
            for line in self._render_runs_compact():
                self._append_output(line)
            self._set_status("Recent runs listed", "info")
            return
        if action == "run_show":
            run_id = int(request.get("run_id", 0) or 0)
            detail_lines = self._render_run_details(run_id)
            level = "warning" if detail_lines and "not found" in detail_lines[0].lower() else "info"
            for line in detail_lines:
                self._append_output(line, level if line == detail_lines[0] else "info")
            if level == "warning":
                self._set_status(f"Run #{run_id} not found", "warning")
            else:
                self._set_status(f"Run #{run_id} details", "info")
            return
        if action == "run_rerun":
            run_id = int(request.get("run_id", 0) or 0)
            self._rerun_history_item(run_id)
            return
        if action == "quit":
            self.exit()
            return
        if action == "set_mode":
            mode = request.get("mode", "")
            if mode not in MODES:
                self._append_output("Error: mode must be analyze, search, or edit", "error")
                return
            self._set_mode(str(mode))
            return
        if action == "set_provider":
            provider = str(request.get("provider", "") or "").strip().lower()
            if provider not in SUPPORTED_TUI_PROVIDERS:
                self._append_output("Error: provider must be openrouter or ollama", "error")
                return
            self._provider = provider
            self._model = self._model_for_provider(provider)
            self._remember_model_for_provider(provider, self._model)
            self._save_runtime_state()
            self._context_hints = []
            self._sync_status_bar()
            self._set_status(f"Provider set to {provider}", "accent")
            if not self._provider_has_credentials(provider):
                self._open_setup()
            self._update_command_meta()
            return
        if action == "set_model":
            self._model = str(request.get("model", "") or "")
            self._remember_model_for_provider(self._provider, self._model)
            self._save_runtime_state()
            self._context_hints = []
            self._sync_status_bar()
            self._set_status(f"Model set to {self._model}", "accent")
            self._update_command_meta()
            return
        if action == "set_key":
            provider = self._provider if self._provider in SUPPORTED_TUI_PROVIDERS else "openrouter"
            ok, message = self._save_provider_setup(
                provider,
                self._model or DEFAULT_MODELS[SUPPORTED_TUI_PROVIDERS[provider]],
                str(request.get("api_key", "") or ""),
            )
            if not ok:
                self._append_output(f"Error: {message}", "error")
                self._set_status("Key rejected", "error")
                return
            self._sync_status_bar()
            self._set_status("Key Saved", "success")
            return
        if action == "set_dir":
            directory = str(request.get("directory", "") or "")
            valid, err = self._validate_repo_path(directory)
            if not valid:
                self._append_output(f"Error: {err}", "error")
                self._set_status("Directory Rejected", "error")
                return
            self._directory = directory
            self._save_runtime_state()
            self._context_hints = []
            self._sync_status_bar()
            self._set_status(f"Directory set to {directory}", "accent")
            self._update_command_meta()
            return
        if action == "chat":
            self._start_chat(str(request.get("prompt", "") or command or ""), command=command)
            return

        self._start_task(action, request, command=command)

    def _execute_command(
        self,
        command: str,
        *,
        record_history: bool = True,
        echo: bool = True,
    ) -> None:
        command_text = command.strip()
        if not command_text:
            return

        prompt = self.query_one("#prompt-input", PromptInput)
        prompt.value = ""
        if record_history:
            self._record_command(command_text)
        if echo:
            self._append_output(f"> {command_text}", "accent")
        self._suggestions = []
        self._update_command_meta()

        action, request = self._build_request(command_text)
        if action is None:
            self._append_output("Error: command not understood", "error")
            self._set_status("Unknown Command", "error")
            return

        self._execute_action_request(action, request, command=command_text)

    def _rerun_history_item(self, run_id: int) -> None:
        if self._running_action is not None:
            self._set_status("Run state: RUNNING | cannot rerun while active", "warning")
            self._append_output("Warning: finish or cancel current task before rerun", "warning")
            return

        replay = self._run_replay_map.get(run_id)
        if not replay:
            self._append_output(f"Warning: no replay data for run #{run_id}", "warning")
            self._set_status("Run rerun unavailable", "warning")
            return

        command = str(replay.get("command") or "").strip()
        if command:
            self._execute_command(command, record_history=False, echo=True)
            return

        action = str(replay.get("action") or "").strip()
        request = dict(replay.get("request") or {})
        if not action:
            self._append_output(f"Warning: invalid replay payload for run #{run_id}", "warning")
            self._set_status("Run rerun unavailable", "warning")
            return

        self._append_output(f"> rerun #{run_id}", "accent")
        self._execute_action_request(action, request, command=None)

    @on(Input.Submitted, "#prompt-input")
    def on_prompt_submitted(self, event: Input.Submitted) -> None:
        command = event.value.strip()
        if not command:
            return
        self._execute_command(command)

    @on(HistoryPane.RunSelected)
    def on_history_run_selected(self, message: HistoryPane.RunSelected) -> None:
        self._rerun_history_item(int(message.run_id))

    @on(PromptInput.HistoryPrev)
    def on_history_prev(self) -> None:
        if not self._command_history:
            return
        prompt = self.query_one("#prompt-input", PromptInput)
        if self._history_index >= len(self._command_history):
            self._history_draft = prompt.value
        self._history_index = max(0, self._history_index - 1)
        prompt.value = self._command_history[self._history_index]
        prompt.cursor_position = len(prompt.value)

    @on(PromptInput.HistoryNext)
    def on_history_next(self) -> None:
        if not self._command_history:
            return
        self._history_index = min(len(self._command_history), self._history_index + 1)
        prompt = self.query_one("#prompt-input", PromptInput)
        if self._history_index >= len(self._command_history):
            prompt.value = self._history_draft
        else:
            prompt.value = self._command_history[self._history_index]
        prompt.cursor_position = len(prompt.value)

    @on(PromptInput.AutoComplete)
    def on_autocomplete(self) -> None:
        if not self._suggestions:
            return
        prompt = self.query_one("#prompt-input", PromptInput)
        prompt.value = self._suggestions[0]
        prompt.cursor_position = len(prompt.value)
        self._suggestions = self._compute_suggestions(prompt.value)
        self._update_command_meta()

    @on(PromptInput.SuggestionDismiss)
    def on_suggestion_dismiss(self) -> None:
        self._suggestions = []
        self._update_command_meta()

    @on(ChatDelta)
    def on_chat_delta(self, message: ChatDelta) -> None:
        if message.token != self._active_token or self._running_action != "chat":
            return
        self._chat_preview = message.content
        self.query_one("#status-bar", StatusBar).update_timer()
        self._set_live_text(f"Assistant: {message.content}", "info")

    @on(TaskLog)
    def on_task_log(self, message: TaskLog) -> None:
        if message.token != self._active_token:
            return
        self._line_progress += 1
        fraction = min(0.9, 0.1 + (self._line_progress * 0.08))
        line = (message.line or "").strip()
        if line:
            lower = line.lower()
            # Show meaningful workflow logs (not only errors) so users can follow progress.
            if any(term in lower for term in ("error", "failed", "warning")):
                self._append_output(line)
            elif (
                any(
                    marker in lower
                    for marker in (
                        "starting",
                        "running",
                        "analyz",
                        "search",
                        "map",
                        "generat",
                        "validat",
                        "integrat",
                        "done",
                        "completed",
                    )
                )
                and self._line_progress <= 20
            ):
                self._append_output(line, "system")
        self.query_one("#status-bar", StatusBar).update_timer()
        self._set_progress(self._running_action or "analyze", fraction)

    @on(TaskCompleted)
    def on_task_completed(self, message: TaskCompleted) -> None:
        if message.token != self._active_token:
            return

        self.query_one("#status-bar", StatusBar).stop_timer()
        self._running_action = None
        self._active_request = None
        duration = 0.0
        started_at = self._run_started_at.pop(message.token, None)
        if started_at is not None:
            duration = max(0.0, time.perf_counter() - started_at)

        result = message.result
        if result.ok and message.action == "chat":
            self._clear_progress()
            content = str(result.payload.get("content", "") or "").strip()
            if content:
                lines = content.splitlines()
                self._append_output(f"Assistant: {lines[0]}")
                for line in lines[1:]:
                    self._append_output(line)
                self._chat_history.extend(
                    [
                        {"role": "user", "content": message.request.get("prompt", "")},
                        {"role": "assistant", "content": content},
                    ]
                )
                self._chat_history = self._chat_history[-12:]
            self._record_token_usage(
                int(result.payload.get("input_tokens", 0)),
                int(result.payload.get("output_tokens", 0)),
            )
            self._context_hints = self._suggest_next_commands(
                "chat", result.payload, message.request
            )
            self._transition_run_state(RunLifecycleState.COMPLETED, action="chat")
            self._add_history_entry(
                run_id=message.token,
                action=message.action,
                status="Success",
                duration=duration,
                terminal_state=RunLifecycleState.COMPLETED,
                summary_lines=[f"Assistant: {content.splitlines()[0]}"] if content else [],
                request=message.request,
                command=str(message.request.get("_original_command") or "") or None,
            )
        elif result.ok:
            summary_lines = self._summarize_result(message.action, result.payload)
            self._set_progress(message.action, 1.0)
            self._clear_progress()
            self._append_output(
                f"{PROGRESS_LABELS.get(message.action, 'Done')} {self._progress_bar(1.0)}",
                "success",
            )
            for line in summary_lines:
                self._append_output(line)
            self._context_hints = self._suggest_next_commands(
                message.action, result.payload, message.request
            )
            self._transition_run_state(RunLifecycleState.COMPLETED, action=message.action)
            self._add_history_entry(
                run_id=message.token,
                action=message.action,
                status="Success",
                duration=duration,
                terminal_state=RunLifecycleState.COMPLETED,
                summary_lines=summary_lines,
                request=message.request,
                command=str(message.request.get("_original_command") or "") or None,
            )
            self._record_run_artifact(
                run_id=message.token,
                action=message.action,
                status="Success",
                terminal_state=RunLifecycleState.COMPLETED,
                duration=duration,
                request=message.request,
                summary_lines=summary_lines,
            )
        else:
            payload = dict(getattr(result, "payload", {}) or {})
            summary_lines = self._summarize_result(message.action, payload)
            self._clear_progress()
            status = "Failed"
            terminal_state = RunLifecycleState.FAILED
            if bool(payload.get("cancelled")) or str(result.error or "").lower().startswith(
                "task cancelled"
            ):
                self._append_output("Task cancelled", "warning")
                self._transition_run_state(RunLifecycleState.CANCELLED, action=message.action)
                status = "Cancelled"
                terminal_state = RunLifecycleState.CANCELLED
                self._add_history_entry(
                    run_id=message.token,
                    action=message.action,
                    status="Cancelled",
                    duration=duration,
                    terminal_state=RunLifecycleState.CANCELLED,
                    summary_lines=summary_lines,
                    request=message.request,
                    command=str(message.request.get("_original_command") or "") or None,
                )
            else:
                self._append_output(f"Error: {result.error or 'command failed'}", "error")
                self._transition_run_state(RunLifecycleState.FAILED, action=message.action)
                self._add_history_entry(
                    run_id=message.token,
                    action=message.action,
                    status="Failed",
                    duration=duration,
                    terminal_state=RunLifecycleState.FAILED,
                    summary_lines=summary_lines,
                    request=message.request,
                    command=str(message.request.get("_original_command") or "") or None,
                )
            self._record_run_artifact(
                run_id=message.token,
                action=message.action,
                status=status,
                terminal_state=terminal_state,
                duration=duration,
                request=message.request,
                summary_lines=summary_lines,
            )
            self._context_hints = []

        self._update_command_meta()
        self.query_one("#prompt-input", PromptInput).focus()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_cancel_task(self) -> None:
        if self._running_action is None:
            self.exit()
            return
        cancel_event = self._cancel_events.get(self._active_token)
        if cancel_event is not None:
            cancel_event.set()
        self._append_output("Cancel requested (stopping current task)...", "warning")
        self._set_status("Run state: RUNNING | cancel requested", "warning")

    def action_clear_screen(self) -> None:
        self._clear_output()
        self._context_hints = []
        self._update_command_meta()
        self._set_status("Screen cleared", "accent")

    def action_show_help(self) -> None:
        self.push_screen(HelpOverlay())

    def action_open_command_palette(self) -> None:
        self.push_screen(CommandPalette(), self._on_command_palette_result)

    def _on_command_palette_result(self, command: str | None) -> None:
        if not command:
            return
        self._execute_command(command)

    def action_handle_escape(self) -> None:
        now = time.time()
        if now - self._last_escape_time > 1.0:
            self._escape_pressed_count = 0
            self._escape_warning_shown = False
        self._last_escape_time = now
        self._escape_pressed_count += 1

        if self._suggestions:
            self._suggestions = []
            self._update_command_meta()
            return

        if self._escape_pressed_count == 1:
            self._escape_warning_shown = True
            self._set_status("Press ESC again to stop agent...", "warning")
            return

        self._escape_pressed_count = 0
        self._escape_warning_shown = False
        self._set_status("Ready", "info")
        self.on_stop()

    def on_stop(self) -> None:
        # Cancel any still-running tasks before dropping state.
        for event in self._cancel_events.values():
            event.set()
        self._running_action = None
        self._active_request = None
        self._run_state = RunLifecycleState.IDLE


def run_tui() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    app = ScholarDevClawApp()
    app.run()


if __name__ == "__main__":
    run_tui()
