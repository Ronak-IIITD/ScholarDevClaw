"""Command-first terminal UI for ScholarDevClaw."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
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
from scholardevclaw.execution import ReproducibilityScorer, SandboxRunner, SelfHealingLoop
from scholardevclaw.execution.scorer import ReproducibilityReport
from scholardevclaw.generation import CodeOrchestrator
from scholardevclaw.generation.models import GenerationResult
from scholardevclaw.ingestion import PaperIngester
from scholardevclaw.ingestion.models import PaperDocument
from scholardevclaw.llm.client import DEFAULT_MODELS, LLMAPIError, LLMClient, LLMConfigError
from scholardevclaw.planning import ImplementationPlan, ImplementationPlanner
from scholardevclaw.product.trust_report import write_paper_workflow_reports
from scholardevclaw.security.path_policy import enforce_allowed_repo_path
from scholardevclaw.understanding import PaperUnderstanding, UnderstandingAgent

from .screens import (
    CommandPalette,
    ExecutionScreen,
    GenerationScreen,
    HelpOverlay,
    PaperIngestionScreen,
    PlanningScreen,
    ProductScreen,
    ProviderSetupScreen,
    UnderstandingScreen,
)
from .theme import COLORS as TUI_COLORS
from .widgets import HistoryPane, LogView, PhaseTracker, PromptInput, RunInspector, StatusBar

logger = logging.getLogger(__name__)

MODES = ("analyze", "search", "edit")
PAPER_WORKFLOW_NAME = "Paper to Code"
PAPER_WORKFLOW_ALIASES = ("paper", "paper-to-code", "papertocode", "from-paper")
SUPPORTED_TUI_PROVIDERS = {
    "anthropic": AuthProvider.ANTHROPIC,
    "openai": AuthProvider.OPENAI,
    "gemini": AuthProvider.GEMINI,
    "grok": AuthProvider.GROK,
    "moonshot": AuthProvider.MOONSHOT,
    "glm": AuthProvider.GLM,
    "minimax": AuthProvider.MINIMAX,
    "openrouter": AuthProvider.OPENROUTER,
    "ollama": AuthProvider.OLLAMA,
    "groq": AuthProvider.GROQ,
    "mistral": AuthProvider.MISTRAL,
    "deepseek": AuthProvider.DEEPSEEK,
    "cohere": AuthProvider.COHERE,
    "together": AuthProvider.TOGETHER,
    "fireworks": AuthProvider.FIREWORKS,
}
DEFAULT_TUI_PROVIDER = "openrouter"
DEFAULT_OPENROUTER_MODEL = DEFAULT_MODELS[AuthProvider.OPENROUTER]
MODE_HINTS = {
    "analyze": [
        "Hint -> paper arxiv:1706.03762",
        "Hint -> /run analyze ./repo",
        "Hint -> /ask what this repo does",
        "Hint -> runs",
        "Hint -> inspect",
        "Hint -> run events 1",
        "Hint -> suggest ./repo",
        "Hint -> validate ./repo",
    ],
    "search": [
        "Hint -> paper ./paper.pdf",
        "Hint -> /run search layer normalization",
        "Hint -> /ask papers on flash attention",
        "Hint -> runs",
        "Hint -> inspect",
        "Hint -> run events 1",
        "Hint -> search flash attention",
        "Hint -> setup",
    ],
    "edit": [
        "Hint -> from-paper arxiv:1706.03762",
        "Hint -> /run map ./repo rmsnorm",
        "Hint -> /ask implement RMSNorm",
        "Hint -> runs",
        "Hint -> inspect",
        "Hint -> run events 1",
        "Hint -> generate ./repo rmsnorm",
        "Hint -> integrate ./repo rmsnorm",
    ],
}
MODE_COMMANDS = {
    "analyze": [
        "paper",
        "paper arxiv:1706.03762",
        "/run analyze ./repo",
        "/ask explain this repository",
        "analyze ./repo",
        "suggest ./repo",
        "validate ./repo",
        "set dir ./repo",
        "set provider openrouter",
        "set provider anthropic",
        f"set model {DEFAULT_OPENROUTER_MODEL}",
        "runs",
        "inspect",
        "run show 1",
        "run events 1",
        "run rerun 1",
        ":search",
        ":edit",
    ],
    "search": [
        "paper",
        "paper ./paper.pdf",
        "/run search layer normalization",
        "/ask find fast inference ideas",
        "search layer normalization",
        "search flash attention",
        "set mode search",
        "chat find fast inference ideas",
        "runs",
        "inspect",
        "run show 1",
        "run events 1",
        "run rerun 1",
        "setup",
        ":analyze",
        ":edit",
    ],
    "edit": [
        "paper",
        "from-paper arxiv:1706.03762",
        "/run map ./repo rmsnorm",
        "/ask how should I patch this file",
        "map ./repo rmsnorm",
        "generate ./repo rmsnorm",
        "integrate ./repo rmsnorm",
        "set dir ./repo",
        "chat how should I patch this file",
        "runs",
        "inspect",
        "run show 1",
        "run events 1",
        "run rerun 1",
        ":analyze",
        ":search",
    ],
}
GLOBAL_COMMANDS = [
    "paper",
    "paper arxiv:1706.03762",
    "paper ./paper.pdf",
    "from-paper arxiv:1706.03762",
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
    "set provider anthropic",
    "set provider ollama",
    f"set model {DEFAULT_OPENROUTER_MODEL}",
    "set dir ./repo",
    "runs",
    "inspect",
    "run show 1",
    "run events 1",
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
RUN_EVENT_LIMIT_PER_RUN = 120
RUN_EVENT_RUN_LIMIT = 20
NATURAL_ACTION_ROUTING_ENV = "SCHOLARDEVCLAW_TUI_ENABLE_NATURAL_ACTION_ROUTING"
AUTO_MODEL_FALLBACK_ENV = "SCHOLARDEVCLAW_TUI_AUTO_MODEL_FALLBACK"
TUI_APPROVAL_GATES_ENV = "SCHOLARDEVCLAW_TUI_APPROVAL_GATES"
APPROVAL_REVIEW_POLL_SECONDS = 0.2
APPROVAL_REVIEW_TIMEOUT_SECONDS = 300.0
PHASE9_HEALING_ENV = "SCHOLARDEVCLAW_TUI_PHASE9_ENABLE_HEALING"
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
    run_events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


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
    failure_code: str = ""
    error: str = ""
    summary_lines: list[str] = field(default_factory=list)


@dataclass
class RunEvent:
    run_id: int
    seq: int
    timestamp: float
    type: str
    phase: str = ""
    state: str = ""
    message: str = ""
    level: str = "info"
    payload: dict[str, Any] | None = None


@dataclass
class Phase9WorkflowState:
    active: bool = False
    source: str = ""
    provider: str = ""
    model: str = ""
    api_key: str = ""
    llm_client: LLMClient | None = None
    work_dir: Path | None = None
    output_dir: Path | None = None
    paper_document: PaperDocument | None = None
    understanding: PaperUnderstanding | None = None
    plan: ImplementationPlan | None = None
    generation_result: GenerationResult | None = None
    execution_report: Any | None = None
    reproducibility_report: ReproducibilityReport | None = None
    healing_payload: dict[str, Any] | None = None
    generation_orchestrator: CodeOrchestrator | None = None
    cancel_event: threading.Event | None = None
    auto_approve: bool = False


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
        ("ctrl+i", "focus_inspector", "Inspector"),
        ("ctrl+h", "show_help", "Help"),
        ("escape", "handle_escape", "ESC"),
        ("ctrl+p", "open_paper_ingestion", "Paper"),
        ("ctrl+u", "open_understanding", "Understanding"),
        ("ctrl+l", "open_planning", "Planning"),
        ("ctrl+g", "open_generation", "Generation"),
        ("ctrl+e", "open_execution", "Execution"),
        ("ctrl+r", "open_product", "Product"),
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

    #workspace {
        width: 100%;
        height: 1fr;
    }

    #main-pane {
        width: 2fr;
        height: 1fr;
    }

    #side-pane {
        width: 1fr;
        min-width: 36;
        height: 1fr;
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

    def __init__(self, *, yes_mode: bool = False) -> None:
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
        self._run_events: dict[int, list[RunEvent]] = {}
        self._run_event_seq: dict[int, int] = {}
        self._chat_event_accumulator: dict[int, str] = {}
        self._chat_event_chunks: dict[int, int] = {}
        self._inspector_run_id: int | None = None
        self._inspector_lines: list[str] = []
        self._models_by_provider: dict[str, str] = {}
        self._approval_lock = threading.Lock()
        self._pending_integrate_reviews: dict[str, dict[str, Any]] = {}
        self._phase9_workflow = Phase9WorkflowState()
        self._yes_mode = yes_mode
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
        with Horizontal(id="workspace"):
            with Vertical(id="main-pane"):
                yield LogView(id="main-output")
            with Vertical(id="side-pane"):
                yield RunInspector(id="run-inspector")
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
        self._refresh_run_inspector()
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
                run_events = data.get("run_events")
                if not isinstance(run_events, dict):
                    run_events = {}
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
                    run_events={str(k): v for k, v in run_events.items()},
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
        self._run_events, self._run_event_seq = self._deserialize_run_events(state.run_events)
        candidate_ids = list(self._run_replay_map) + list(self._run_events)
        if candidate_ids:
            self._task_token = max(candidate_ids)

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
            failure_code = str(row.get("failure_code", "") or "").strip()
            error = self._trim_event_message(str(row.get("error", "") or ""), limit=220)
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
                    failure_code=failure_code,
                    error=error,
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
                "failure_code": str(entry.get("failure_code", "") or "").strip(),
                "error": self._trim_event_message(str(entry.get("error", "") or ""), limit=220),
                "summary_lines": [
                    str(line).strip()
                    for line in list(entry.get("summary_lines") or [])
                    if str(line).strip()
                ][:4],
            }
        for stale in sorted(loaded)[:-RUN_PERSIST_LIMIT]:
            loaded.pop(stale, None)
        return loaded

    @staticmethod
    def _trim_event_message(text: str, *, limit: int = 220) -> str:
        value = str(text or "").replace("\n", " ").strip()
        if len(value) <= limit:
            return value
        return value[: max(0, limit - 1)].rstrip() + "…"

    @staticmethod
    def _sanitize_event_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        clean: dict[str, Any] = {}
        for key, value in payload.items():
            key_name = str(key).strip()
            if not key_name:
                continue
            if isinstance(value, (bool, int, float)):
                clean[key_name] = value
            elif isinstance(value, str):
                clean[key_name] = ScholarDevClawApp._trim_event_message(value, limit=180)
        return clean or None

    def _deserialize_run_events(
        self, rows: dict[str, Any]
    ) -> tuple[dict[int, list[RunEvent]], dict[int, int]]:
        loaded: dict[int, list[RunEvent]] = {}
        seq_map: dict[int, int] = {}
        for run_id_raw, events_raw in rows.items():
            try:
                run_id = int(str(run_id_raw).strip())
            except Exception:
                continue
            if not isinstance(events_raw, list):
                continue
            decoded: list[RunEvent] = []
            for row in events_raw:
                if not isinstance(row, dict):
                    continue
                try:
                    seq = int(row.get("seq", 0))
                    timestamp = float(row.get("timestamp", 0.0) or 0.0)
                except Exception:
                    continue
                if seq <= 0 or timestamp <= 0:
                    continue
                if not math.isfinite(timestamp):
                    continue
                event_type = str(row.get("type", "")).strip() or "event"
                decoded.append(
                    RunEvent(
                        run_id=run_id,
                        seq=seq,
                        timestamp=timestamp,
                        type=event_type,
                        phase=str(row.get("phase", "") or "").strip(),
                        state=str(row.get("state", "") or "").strip(),
                        message=self._trim_event_message(str(row.get("message", "") or "")),
                        level=str(row.get("level", "info") or "info").strip() or "info",
                        payload=self._sanitize_event_payload(
                            row.get("payload") if isinstance(row.get("payload"), dict) else None
                        ),
                    )
                )
            if not decoded:
                continue
            decoded.sort(key=lambda item: item.seq)
            trimmed = decoded[-RUN_EVENT_LIMIT_PER_RUN:]
            loaded[run_id] = trimmed
            seq_map[run_id] = max(item.seq for item in trimmed)

        for stale in sorted(loaded)[:-RUN_EVENT_RUN_LIMIT]:
            loaded.pop(stale, None)
            seq_map.pop(stale, None)
        return loaded, seq_map

    def _append_run_event(
        self,
        run_id: int,
        event_type: str,
        *,
        phase: str = "",
        state: str = "",
        message: str = "",
        level: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> RunEvent:
        next_seq = self._run_event_seq.get(run_id, 0) + 1
        event = RunEvent(
            run_id=run_id,
            seq=next_seq,
            timestamp=time.time(),
            type=str(event_type or "event").strip() or "event",
            phase=str(phase or "").strip(),
            state=str(state or "").strip(),
            message=self._trim_event_message(message),
            level=str(level or "info").strip() or "info",
            payload=self._sanitize_event_payload(payload),
        )
        bucket = self._run_events.setdefault(run_id, [])
        bucket.append(event)
        if len(bucket) > RUN_EVENT_LIMIT_PER_RUN:
            self._run_events[run_id] = bucket[-RUN_EVENT_LIMIT_PER_RUN:]
        self._run_event_seq[run_id] = next_seq

        for stale in sorted(self._run_events)[:-RUN_EVENT_RUN_LIMIT]:
            self._run_events.pop(stale, None)
            self._run_event_seq.pop(stale, None)
            self._chat_event_accumulator.pop(stale, None)
            self._chat_event_chunks.pop(stale, None)
        return event

    def _serialize_run_events(self) -> dict[str, list[dict[str, Any]]]:
        compact: dict[str, list[dict[str, Any]]] = {}
        for run_id in sorted(self._run_events)[-RUN_EVENT_RUN_LIMIT:]:
            rows: list[dict[str, Any]] = []
            for event in self._run_events.get(run_id, [])[-RUN_EVENT_LIMIT_PER_RUN:]:
                rows.append(
                    {
                        "run_id": run_id,
                        "seq": int(event.seq),
                        "timestamp": float(event.timestamp),
                        "type": str(event.type),
                        "phase": str(event.phase),
                        "state": str(event.state),
                        "message": self._trim_event_message(event.message),
                        "level": str(event.level),
                        "payload": self._sanitize_event_payload(event.payload),
                    }
                )
            if rows:
                compact[str(run_id)] = rows
        return compact

    @staticmethod
    def _event_state_phase_label(event: RunEvent) -> str:
        state = event.state.strip()
        phase = event.phase.strip()
        if state and phase:
            return f"{state}/{phase}"
        return state or phase or "-"

    def _render_run_events(self, run_id: int, limit: int | None = None) -> list[str]:
        events = list(self._run_events.get(run_id) or [])
        if not events:
            return [f"Run #{run_id} has no recorded events"]
        if limit is not None and limit > 0:
            events = events[-limit:]
        return [
            f"{event.seq:03d} {event.type:<14} {self._event_state_phase_label(event):<24} {event.message or '-'}"
            for event in events
        ]

    def _record_task_log_event(self, run_id: int, line: str) -> None:
        message = self._trim_event_message(line, limit=180)
        if not message:
            return
        self._append_run_event(
            run_id,
            "log.line",
            phase=self._phase_for_run_state(self._run_state, self._running_action),
            state=self._run_state.value,
            message=message,
            level="info",
        )

    def _record_chat_delta_event(self, run_id: int, full_content: str) -> None:
        text = str(full_content or "")
        if not text:
            return
        self._chat_event_accumulator[run_id] = text
        chunks = self._chat_event_chunks.get(run_id, 0) + 1
        self._chat_event_chunks[run_id] = chunks
        should_sample = chunks % 6 == 0 or len(text) <= 80 or len(text) % 120 < 20
        if not should_sample:
            return
        snippet = self._trim_event_message(text[-160:], limit=160)
        self._append_run_event(
            run_id,
            "chat.delta",
            phase=self._phase_for_run_state(self._run_state, "chat"),
            state=self._run_state.value,
            message=f"…{snippet}" if len(text) > len(snippet) else snippet,
            level="info",
            payload={"chars": len(text), "chunks": chunks},
        )

    def _flush_chat_delta_event(self, run_id: int) -> None:
        content = self._chat_event_accumulator.pop(run_id, "")
        chunks = self._chat_event_chunks.pop(run_id, 0)
        if not content:
            return
        snippet = self._trim_event_message(content[-180:], limit=180)
        self._append_run_event(
            run_id,
            "chat.delta.final",
            phase=self._phase_for_run_state(self._run_state, "chat"),
            state=self._run_state.value,
            message=f"…{snippet}" if len(content) > len(snippet) else snippet,
            level="info",
            payload={"chars": len(content), "chunks": chunks},
        )

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
            "run_events": self._serialize_run_events(),
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
                    "failure_code": artifact.failure_code,
                    "error": self._trim_event_message(artifact.error, limit=220),
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
                "failure_code": str(entry.get("failure_code", "") or "").strip(),
                "error": self._trim_event_message(str(entry.get("error", "") or ""), limit=220),
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

    def _supported_provider_names(self) -> str:
        return ", ".join(SUPPORTED_TUI_PROVIDERS)

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
                    "Run `ollama serve`, `setup`, or switch provider with `set provider <name>`.",
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
        setup_provider = (
            self._provider if self._provider in SUPPORTED_TUI_PROVIDERS else DEFAULT_TUI_PROVIDER
        )
        self.push_screen(
            ProviderSetupScreen(
                provider=setup_provider,
                model=self._model or self._model_for_provider(setup_provider),
                has_saved_key=self._provider_has_credentials(setup_provider),
                supported_providers=SUPPORTED_TUI_PROVIDERS,
                has_saved_key_by_provider={
                    provider_name: self._provider_has_credentials(provider_name)
                    for provider_name in SUPPORTED_TUI_PROVIDERS
                },
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

    def _on_paper_ingestion_result(self, result: dict[str, Any] | None) -> None:
        if result is None:
            self._append_output("Paper workflow dismissed", "warning")
            return
        source = ""
        if isinstance(result, dict):
            source = str(result.get("source", "") or "").strip()
        if not source:
            self._append_output("Paper workflow cancelled (no source provided)", "warning")
            return
        self._start_phase9_workflow(source)

    def _on_understanding_result(self, result: dict[str, Any] | str | None) -> None:
        if result is None:
            self._append_output("Understanding dismissed", "warning")
            return
        decision = result if isinstance(result, str) else str(result.get("decision", "unknown"))
        self._append_output(f"Decision: {decision}")

    def _on_planning_result(self, result: dict[str, Any] | str | None) -> None:
        if result is None:
            self._append_output("Planning dismissed", "warning")
            return
        decision = result if isinstance(result, str) else str(result.get("decision", "unknown"))
        self._append_output(f"Decision: {decision}")

    def _on_generation_result(self, result: dict[str, Any] | None) -> None:
        if result is None:
            self._append_output("Generation dismissed", "warning")
            return
        self._append_output(f"Generation result: {result}")

    def _on_execution_result(self, result: dict[str, Any] | None) -> None:
        if result is None:
            self._append_output("Execution dismissed", "warning")
            return
        self._append_output(f"Execution result: {result}")

    def _on_product_result(self, result: dict[str, Any] | None) -> None:
        if result is None:
            self._append_output("Product dismissed", "warning")
            return
        self._append_output("Product ready for review", "success")
        self._append_output(f"Product result: {result}")

    # ------------------------------------------------------------------
    # Paper-to-code TUI workflow
    # ------------------------------------------------------------------

    def _phase9_ui_call(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        try:
            self.call_from_thread(callback, *args, **kwargs)
        except Exception:
            pass

    def _phase9_log(self, line: str, level: str = "info") -> None:
        self._phase9_ui_call(self._append_output, line, level)

    def _phase9_status(self, message: str, level: str = "info") -> None:
        self._phase9_ui_call(self._set_status, message, level)

    def _phase9_set_phase(self, phase: str) -> None:
        self._phase9_ui_call(self._set_phase, phase)

    @staticmethod
    def _phase9_slugify(source: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", source).strip("-").lower()
        return (slug[:48] or "paper").strip("-") or "paper"

    def _phase9_is_cancelled(self, state: Phase9WorkflowState) -> bool:
        return bool(state.cancel_event and state.cancel_event.is_set())

    def _phase9_raise_if_cancelled(self, state: Phase9WorkflowState) -> None:
        if self._phase9_is_cancelled(state):
            raise TaskCancelledError("Phase-9 workflow cancelled")

    def _phase9_model_and_key(self) -> tuple[str, str, str]:
        if self._provider not in SUPPORTED_TUI_PROVIDERS:
            raise ValueError(
                "Provider is not configured. Run `setup` and choose a provider/model before "
                "running the Phase-9 workflow."
            )
        provider_name = self._provider
        auth_provider = SUPPORTED_TUI_PROVIDERS[provider_name]
        model = (
            self._model.strip()
            or self._model_for_provider(provider_name)
            or DEFAULT_MODELS[auth_provider]
        )
        if not model:
            raise ValueError(
                "Model is missing. Set one with `set model <id>` and retry Phase-9 workflow."
            )
        if auth_provider == AuthProvider.OLLAMA:
            os.environ.setdefault(
                "OLLAMA_HOST",
                auth_provider.default_base_url or "http://localhost:11434",
            )
            return provider_name, model, ""

        api_key = (
            self._get_saved_key_for_provider(auth_provider)
            or (os.environ.get(auth_provider.env_var_name) or "").strip()
            or (os.environ.get("SCHOLARDEVCLAW_API_KEY") or "").strip()
        )
        if not api_key:
            raise ValueError(
                "API key is missing. Set "
                f"{auth_provider.env_var_name} (or SCHOLARDEVCLAW_API_KEY) before "
                "running the Phase-9 workflow."
            )
        return provider_name, model, api_key

    def _phase9_prepare_dirs(self, source: str) -> tuple[Path, Path]:
        root = Path(self._directory).expanduser().resolve()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        workflow_dir = (
            root / ".scholardevclaw_phase9" / f"{timestamp}-{self._phase9_slugify(source)}"
        )
        output_dir = workflow_dir / "generated"
        workflow_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        return workflow_dir, output_dir

    def _phase9_call_screen(self, screen: Any, method_name: str, *args: Any, **kwargs: Any) -> None:
        def _invoke() -> None:
            method = getattr(screen, method_name, None)
            if callable(method):
                with contextlib.suppress(Exception):
                    method(*args, **kwargs)

        self._phase9_ui_call(_invoke)

    def _phase9_wait_for_understanding_decision(
        self,
        state: Phase9WorkflowState,
        understanding: PaperUnderstanding,
    ) -> bool:
        if state.auto_approve or os.environ.get(
            "SCHOLARDEVCLAW_TUI_PHASE9_AUTO_APPROVE", ""
        ).strip().lower() in {"1", "true", "yes"}:
            self._phase9_log("Auto-approved (--yes mode)", "accent")
            return True
        done = threading.Event()
        decision: dict[str, bool] = {"proceed": True}

        def _callback(result: dict[str, Any] | str | None) -> None:
            if result is None:
                decision["proceed"] = False
            elif isinstance(result, str):
                decision["proceed"] = result.strip().lower() in {"proceed", "approve", "approved"}
            else:
                value = str(result.get("decision", "")).strip().lower()
                decision["proceed"] = value in {"proceed", "approve", "approved"}
            done.set()

        self._phase9_ui_call(
            self.push_screen,
            UnderstandingScreen(understanding=understanding),
            _callback,
        )
        self._phase9_log("Understanding ready. Confirm to continue.", "accent")

        while not done.wait(0.1):
            self._phase9_raise_if_cancelled(state)
        return bool(decision["proceed"])

    def _phase9_wait_for_planning_approval(
        self,
        state: Phase9WorkflowState,
        plan: ImplementationPlan,
    ) -> bool:
        if state.auto_approve or os.environ.get(
            "SCHOLARDEVCLAW_TUI_PHASE9_AUTO_APPROVE", ""
        ).strip().lower() in {"1", "true", "yes"}:
            self._phase9_log("Auto-approved (--yes mode)", "accent")
            return True
        done = threading.Event()
        decision: dict[str, bool] = {"approved": False}

        def _callback(result: dict[str, Any] | str | None) -> None:
            if result is None:
                decision["approved"] = False
            elif isinstance(result, str):
                decision["approved"] = result.strip().lower() in {
                    "approve",
                    "approved",
                    "true",
                    "yes",
                }
            else:
                if "approved" in result:
                    decision["approved"] = bool(result.get("approved"))
                else:
                    value = str(result.get("decision", "")).strip().lower()
                    decision["approved"] = value in {"approve", "approved", "true", "yes"}
            done.set()

        self._phase9_ui_call(self.push_screen, PlanningScreen(plan=plan), _callback)
        self._phase9_log("Approval gate: approve plan to start generation.", "warning")

        while not done.wait(0.1):
            self._phase9_raise_if_cancelled(state)
        return bool(decision["approved"])

    def _phase9_ingest_paper(self, state: Phase9WorkflowState) -> PaperDocument:
        assert state.work_dir is not None
        self._phase9_set_phase("paper_ingest")
        self._phase9_status(f"{PAPER_WORKFLOW_NAME}: ingesting paper", "accent")
        self._phase9_log(f"[1/7] Ingesting paper source: {state.source}", "accent")
        ingester = PaperIngester()
        return ingester.ingest(state.source, state.work_dir / "paper")

    def _phase9_understand_paper(self, state: Phase9WorkflowState) -> PaperUnderstanding:
        assert state.paper_document is not None
        self._phase9_set_phase("paper_understand")
        self._phase9_status(f"{PAPER_WORKFLOW_NAME}: understanding paper", "accent")
        self._phase9_log("[2/7] Understanding paper...", "accent")
        agent = UnderstandingAgent(
            api_key=state.api_key,
            model=state.model,
            provider=state.provider,
        )
        return agent.understand(state.paper_document)

    def _phase9_plan_implementation(self, state: Phase9WorkflowState) -> ImplementationPlan:
        assert state.paper_document is not None
        assert state.understanding is not None
        self._phase9_set_phase("paper_plan")
        self._phase9_status(f"{PAPER_WORKFLOW_NAME}: planning implementation", "accent")
        self._phase9_log("[3/7] Planning implementation...", "accent")
        planner = ImplementationPlanner(
            api_key=state.api_key,
            model=state.model,
            provider=state.provider,
        )
        return planner.plan(state.understanding, state.paper_document)

    def _phase9_generate_code(self, state: Phase9WorkflowState) -> GenerationResult:
        plan = state.plan
        understanding = state.understanding
        output_dir = state.output_dir
        if plan is None or understanding is None or output_dir is None:
            raise RuntimeError("Generation prerequisites are missing")

        module_ids = [module.id for module in plan.modules if module.id.strip()]
        generation_screen = GenerationScreen(module_ids=module_ids)
        self._phase9_ui_call(self.push_screen, generation_screen, self._on_generation_result)
        self._phase9_set_phase("paper_generate")
        self._phase9_status(f"{PAPER_WORKFLOW_NAME}: generating code", "accent")
        self._phase9_log(f"[4/7] Generating {len(module_ids)} modules...", "accent")

        orchestrator = CodeOrchestrator(
            api_key=state.api_key,
            model=state.model,
            client=state.llm_client,
        )
        state.generation_orchestrator = orchestrator

        done = threading.Event()
        result_box: dict[str, GenerationResult] = {}
        error_box: dict[str, Exception] = {}
        logged_modules: set[str] = set()

        def _generate() -> None:
            try:
                result_box["result"] = orchestrator.generate_sync(
                    plan=plan,
                    understanding=understanding,
                    output_dir=output_dir,
                    max_parallel=4,
                )
            except Exception as exc:  # pragma: no cover - runtime dependent
                error_box["error"] = exc
            finally:
                done.set()

        threading.Thread(target=_generate, daemon=True).start()
        ticker = 0
        while not done.wait(0.2):
            self._phase9_raise_if_cancelled(state)
            ticker += 1
            for module in plan.modules:
                module_file = output_dir / module.file_path
                test_file = output_dir / module.test_file_path
                progress = 0.0
                if module_file.exists():
                    progress = 0.6
                if module_file.exists() and test_file.exists():
                    progress = 1.0
                    if module.id not in logged_modules:
                        logged_modules.add(module.id)
                        self._phase9_call_screen(
                            generation_screen,
                            "append_module_log",
                            module.id,
                            f"Generated artifacts for {module.id}",
                        )
                self._phase9_call_screen(
                    generation_screen,
                    "set_module_progress",
                    module.id,
                    progress,
                )
            if module_ids:
                self._phase9_call_screen(
                    generation_screen,
                    "append_module_log",
                    module_ids[min(len(module_ids) - 1, ticker % len(module_ids))],
                    f"Generating... tick {ticker}",
                )

        if "error" in error_box:
            raise RuntimeError(str(error_box["error"]))
        result = result_box.get("result")
        if result is None:
            raise RuntimeError("Generation did not return a result")

        for row in result.module_results:
            self._phase9_call_screen(generation_screen, "set_module_progress", row.module_id, 1.0)
            self._phase9_call_screen(
                generation_screen,
                "append_module_log",
                row.module_id,
                f"Generated: {row.file_path} / {row.test_file_path}",
            )
            if row.final_errors:
                for error in row.final_errors:
                    self._phase9_call_screen(
                        generation_screen,
                        "add_syntax_error",
                        row.module_id,
                        error,
                    )

        self._phase9_call_screen(generation_screen, "set_generation_result", result)
        self._phase9_log(f"Generation success rate: {result.success_rate:.0%}", "success")
        return result

    @staticmethod
    def _phase9_extract_test_status(stdout: str) -> list[tuple[str, bool]]:
        statuses: list[tuple[str, bool]] = []
        for line in stdout.splitlines():
            stripped = line.strip()
            if "::" not in stripped:
                continue
            if stripped.endswith("PASSED"):
                statuses.append((stripped.rsplit(" ", 1)[0], True))
            elif stripped.endswith("FAILED"):
                statuses.append((stripped.rsplit(" ", 1)[0], False))
        return statuses

    def _phase9_execute_tests(self, state: Phase9WorkflowState) -> tuple[Any, GenerationResult]:
        output_dir = state.output_dir
        generation_result = state.generation_result
        plan = state.plan
        understanding = state.understanding
        if output_dir is None or generation_result is None or plan is None or understanding is None:
            raise RuntimeError("Execution prerequisites are missing")

        execution_screen = ExecutionScreen()
        self._phase9_ui_call(self.push_screen, execution_screen, self._on_execution_result)
        self._phase9_set_phase("paper_execute")
        self._phase9_status(f"{PAPER_WORKFLOW_NAME}: executing tests", "accent")
        self._phase9_log("[5/7] Running sandbox tests...", "accent")

        runner = SandboxRunner()
        report = runner.run_tests(output_dir)
        state.healing_payload = None
        for line in (report.stdout or "").splitlines()[-200:]:
            self._phase9_call_screen(execution_screen, "append_pytest_output", line)
        for test_name, passed in self._phase9_extract_test_status(report.stdout or ""):
            self._phase9_call_screen(
                execution_screen,
                "set_test_status",
                test_name,
                passed=passed,
            )

        healing_enabled = self._env_flag_enabled(PHASE9_HEALING_ENV, default=False)
        if healing_enabled and not report.success:
            initial_failed_tests = report.tests_failed
            initial_error_tests = report.tests_errors
            self._phase9_log("Sandbox tests failed; starting self-healing...", "warning")
            orchestrator = state.generation_orchestrator or CodeOrchestrator(
                api_key=state.api_key,
                model=state.model,
                client=state.llm_client,
            )
            healer = SelfHealingLoop(orchestrator, runner)
            generation_result = healer.heal(generation_result, plan, understanding)
            for row in healer.round_reports:
                round_id = int(row.get("round", 0) or 0)
                total = max(1, len(healer.round_reports))
                self._phase9_call_screen(
                    execution_screen,
                    "set_healing_round",
                    round_id,
                    total_rounds=total,
                )
                self._phase9_call_screen(
                    execution_screen,
                    "append_pytest_output",
                    (
                        f"[heal round {round_id}] passed={row.get('tests_passed', 0)} "
                        f"failed={row.get('tests_failed', 0)} errors={row.get('tests_errors', 0)}"
                    ),
                )
            report = runner.run_tests(output_dir)
            for line in (report.stdout or "").splitlines()[-120:]:
                self._phase9_call_screen(execution_screen, "append_pytest_output", line)
            state.healing_payload = {
                "round_count": len(healer.round_reports),
                "round_reports": healer.round_reports,
                "initial_failed_tests": initial_failed_tests,
                "initial_error_tests": initial_error_tests,
                "final_failed_tests": report.tests_failed,
                "final_error_tests": report.tests_errors,
            }

        self._phase9_log(
            (
                f"Execution complete: passed={report.tests_passed} "
                f"failed={report.tests_failed} errors={report.tests_errors}"
            ),
            "success" if report.success else "warning",
        )
        return report, generation_result

    def _phase9_score_reproducibility(
        self,
        state: Phase9WorkflowState,
        execution_report: Any,
    ) -> ReproducibilityReport:
        assert state.understanding is not None
        self._phase9_set_phase("paper_score")
        self._phase9_status(f"{PAPER_WORKFLOW_NAME}: scoring reproducibility", "accent")
        self._phase9_log("[6/7] Scoring reproducibility...", "accent")
        scorer = ReproducibilityScorer(
            api_key=state.api_key,
            model=state.model,
            provider=state.provider,
            client=state.llm_client,
        )
        report = scorer.score(state.understanding, execution_report)
        self._phase9_log(
            f"Reproducibility: {report.score:.0%} ({report.verdict})",
            "success" if report.score >= 0.5 else "warning",
        )
        return report

    def _phase9_present_product(self, state: Phase9WorkflowState) -> None:
        assert state.output_dir is not None
        self._phase9_set_phase("paper_product")
        if (
            state.paper_document is not None
            and state.understanding is not None
            and state.plan is not None
            and state.generation_result is not None
            and state.execution_report is not None
            and state.reproducibility_report is not None
        ):
            try:
                workflow_reports = write_paper_workflow_reports(
                    source=state.source,
                    document=state.paper_document,
                    understanding=state.understanding,
                    plan=state.plan,
                    generation_result=state.generation_result,
                    execution_report=state.execution_report,
                    reproducibility_report=state.reproducibility_report,
                    project_dir=state.output_dir,
                    healing_payload=state.healing_payload,
                )
                self._phase9_log(
                    f"Trust report: {workflow_reports.trust_report_markdown_path}",
                    "accent",
                )
                self._phase9_log(
                    (
                        "Traceability coverage: "
                        f"{workflow_reports.traceability_report.coverage_score:.0%} "
                        f"({workflow_reports.traceability_report.mapped_equations}/"
                        f"{workflow_reports.traceability_report.total_equations})"
                    ),
                    (
                        "success"
                        if workflow_reports.traceability_report.coverage_score >= 0.5
                        or workflow_reports.traceability_report.total_equations == 0
                        else "warning"
                    ),
                )
            except (OSError, RuntimeError, ValueError) as exc:
                self._phase9_log(f"Warning: failed to write trust artifacts: {exc}", "warning")
        install_command = f'pip install -e "{state.output_dir}"'
        self._phase9_ui_call(
            self.push_screen,
            ProductScreen(output_dir=state.output_dir, install_command=install_command),
            self._on_product_result,
        )

        files = [path for path in sorted(state.output_dir.rglob("*")) if path.is_file()][:12]
        self._phase9_log("[7/7] Product artifacts ready", "success")
        self._phase9_log(f"Output directory: {state.output_dir}", "accent")
        if files:
            self._phase9_log("File tree preview:", "accent")
            for path in files:
                self._phase9_log(f"  - {path.relative_to(state.output_dir)}")
        self._phase9_log(f"Install command: {install_command}", "accent")

    def _phase9_run_flow(self, state: Phase9WorkflowState) -> str:
        self._phase9_raise_if_cancelled(state)
        state.paper_document = self._phase9_ingest_paper(state)
        self._phase9_raise_if_cancelled(state)

        state.understanding = self._phase9_understand_paper(state)
        self._phase9_raise_if_cancelled(state)
        proceed = self._phase9_wait_for_understanding_decision(state, state.understanding)
        if not proceed:
            self._phase9_log("Understanding rejected. Workflow stopped before planning.", "warning")
            return "stopped"

        state.plan = self._phase9_plan_implementation(state)
        self._phase9_raise_if_cancelled(state)

        approved = self._phase9_wait_for_planning_approval(state, state.plan)
        if not approved:
            self._phase9_log("Planning approval rejected. Generation was not started.", "warning")
            return "stopped"

        state.generation_result = self._phase9_generate_code(state)
        self._phase9_raise_if_cancelled(state)

        execution_report, generation_result = self._phase9_execute_tests(state)
        state.execution_report = execution_report
        state.generation_result = generation_result
        self._phase9_raise_if_cancelled(state)

        state.reproducibility_report = self._phase9_score_reproducibility(state, execution_report)
        self._phase9_raise_if_cancelled(state)
        self._phase9_present_product(state)
        return "completed"

    def _run_phase9_workflow_thread(self, state: Phase9WorkflowState) -> None:
        try:
            provider_name, model, api_key = self._phase9_model_and_key()
            state.provider = provider_name
            state.model = model
            state.api_key = api_key
            state.llm_client = LLMClient.from_provider(
                provider_name,
                api_key=api_key,
                model=model,
            )
            self._phase9_log(
                f"Starting {PAPER_WORKFLOW_NAME} workflow for source: {state.source}",
                "accent",
            )
            outcome = self._phase9_run_flow(state)
            if self._phase9_is_cancelled(state):
                self._phase9_set_phase("idle")
                self._phase9_status(f"{PAPER_WORKFLOW_NAME} workflow cancelled", "warning")
            elif outcome == "stopped":
                self._phase9_set_phase("idle")
                self._phase9_status(f"{PAPER_WORKFLOW_NAME} workflow stopped", "warning")
            else:
                self._phase9_set_phase("complete")
                self._phase9_status(f"{PAPER_WORKFLOW_NAME} workflow complete", "success")
        except TaskCancelledError:
            self._phase9_set_phase("idle")
            self._phase9_log(f"{PAPER_WORKFLOW_NAME} workflow cancelled", "warning")
            self._phase9_status(f"{PAPER_WORKFLOW_NAME} workflow cancelled", "warning")
        except Exception as exc:
            self._phase9_set_phase("idle")
            self._phase9_log(
                f"{PAPER_WORKFLOW_NAME} workflow failed: {exc}",
                "error",
            )
            self._phase9_status(f"{PAPER_WORKFLOW_NAME} workflow failed", "error")
        finally:
            if state.llm_client is not None:
                with contextlib.suppress(Exception):
                    state.llm_client.close()
                state.llm_client = None
            state.active = False
            if self._phase9_workflow is state:
                self._phase9_workflow.active = False
                self._phase9_workflow.cancel_event = None

    def _start_phase9_workflow(self, source: str) -> None:
        if self._phase9_workflow.active:
            self._append_output(f"{PAPER_WORKFLOW_NAME} workflow already running", "warning")
            self._set_status(f"{PAPER_WORKFLOW_NAME} workflow already running", "warning")
            return
        if self._running_action is not None:
            self._append_output(
                f"Finish current command task before starting {PAPER_WORKFLOW_NAME} workflow",
                "warning",
            )
            self._set_status("Command task running", "warning")
            return

        source_text = str(source or "").strip()
        if not source_text:
            self._append_output("Paper source is required", "warning")
            self._set_status(f"{PAPER_WORKFLOW_NAME} source missing", "warning")
            return

        work_dir, output_dir = self._phase9_prepare_dirs(source_text)
        workflow_state = Phase9WorkflowState(
            active=True,
            source=source_text,
            work_dir=work_dir,
            output_dir=output_dir,
            cancel_event=threading.Event(),
            auto_approve=self._yes_mode,
        )
        self._phase9_workflow = workflow_state
        self._set_phase("paper_ingest")
        self._append_output(f"{PAPER_WORKFLOW_NAME} work directory: {work_dir}", "accent")
        self._set_status(f"{PAPER_WORKFLOW_NAME} workflow running", "accent")

        thread = threading.Thread(
            target=self._run_phase9_workflow_thread,
            args=(workflow_state,),
            daemon=True,
        )
        thread.start()

    def _save_provider_setup(
        self, provider: str, model: str, api_key: str = ""
    ) -> tuple[bool, str]:
        provider_name = provider.strip().lower()
        auth_provider = SUPPORTED_TUI_PROVIDERS.get(provider_name)
        if auth_provider is None:
            return False, f"Provider must be one of: {self._supported_provider_names()}"
        if not model.strip():
            return False, "Model is required"

        store = AuthStore(enable_audit=False, enable_rate_limit=False)
        normalized_api_key = api_key.strip()
        existing = None
        for key in store.list_api_keys():
            if key.provider == auth_provider:
                existing = key
                if normalized_api_key and key.key == normalized_api_key:
                    break

        try:
            if auth_provider.requires_api_key:
                if normalized_api_key:
                    if existing and existing.key == normalized_api_key:
                        store.set_default_key(existing.id)
                    else:
                        store.add_api_key(
                            normalized_api_key,
                            f"{provider_name}-tui",
                            auth_provider,
                            set_default=True,
                            validate=True,
                            metadata={"source": "tui"},
                        )
                    os.environ[auth_provider.env_var_name] = normalized_api_key
                else:
                    saved = self._get_saved_key_for_provider(auth_provider)
                    env_key = (os.environ.get(auth_provider.env_var_name) or "").strip()
                    if saved:
                        os.environ[auth_provider.env_var_name] = saved
                    elif env_key:
                        os.environ[auth_provider.env_var_name] = env_key
                    else:
                        return False, f"{auth_provider.display_name} requires an API key"
            else:
                if existing is not None:
                    store.set_default_key(existing.id)
                else:
                    store.add_api_key(
                        f"{provider_name}-local",
                        f"{provider_name}-local",
                        auth_provider,
                        set_default=True,
                        metadata={"source": "tui"},
                    )
                os.environ.setdefault(
                    auth_provider.env_var_name,
                    auth_provider.default_base_url or "http://localhost:11434",
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
        failure_code: str = "",
        error: str = "",
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
            "failure_code": str(failure_code or "").strip(),
            "error": self._trim_event_message(str(error or ""), limit=220),
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
            f"Failure code: {replay.get('failure_code') or artifact.failure_code or 'n/a'}",
        ]
        error_text = str(replay.get("error") or artifact.error or "").strip()
        if error_text:
            lines.append(f"Error: {self._trim_event_message(error_text, limit=220)}")
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

    @staticmethod
    def _classify_failure_code(error_text: str, *, cancelled: bool = False) -> str:
        text = str(error_text or "").strip().lower()
        if cancelled or "cancelled" in text:
            return "E_CANCELLED_BY_USER"
        if "429" in text or "rate limit" in text:
            return "E_LLM_RATE_LIMIT"
        if any(token in text for token in ("401", "auth", "api key", "unauthorized", "forbidden")):
            return "E_LLM_AUTH"
        if any(
            token in text
            for token in (
                "model unavailable",
                "model lookup failed",
                "model not found",
                "does not exist",
                "unknown model",
            )
        ):
            return "E_LLM_MODEL"
        if any(
            token in text
            for token in (
                "config",
                "setup",
                "no key found",
                "requires an api key",
                "llm setup",
                "provider",
            )
        ):
            return "E_LLM_CONFIG"
        return "E_RUNTIME_EXCEPTION"

    def _latest_run_id_for_inspector(self) -> int | None:
        if self._active_token and self._running_action is not None:
            return self._active_token
        if self._recent_run_artifacts:
            return self._recent_run_artifacts[-1].run_id
        if self._run_replay_map:
            return max(self._run_replay_map)
        return None

    def _build_run_inspector_snapshot(self) -> dict[str, Any] | None:
        run_id = self._latest_run_id_for_inspector()
        if run_id is None:
            return None

        replay = dict(self._run_replay_map.get(run_id) or {})
        events = self._render_run_events(run_id, limit=4)
        event_lines = [] if (events and "no recorded events" in events[0].lower()) else events

        if self._running_action is not None and run_id == self._active_token:
            active_req = dict(self._active_request or {})
            started_at = self._run_started_at.get(run_id)
            duration = 0.0
            if started_at is not None:
                duration = max(0.0, time.perf_counter() - started_at)
            return {
                "run_id": run_id,
                "action": self._running_action,
                "status": RUN_STATE_LABELS[self._run_state].title(),
                "duration": duration,
                "terminal_state": self._run_state.value,
                "failure_code": "",
                "error": "",
                "repo": str(active_req.get("repo_path", "") or ""),
                "spec": str(active_req.get("spec", "") or ""),
                "query": str(active_req.get("query", "") or active_req.get("prompt", "") or ""),
                "summary_lines": list(replay.get("summary_lines") or []),
                "event_lines": event_lines,
            }

        artifact = self._find_run_artifact(run_id)
        if artifact is None:
            req = dict(replay.get("request") or {})
            return {
                "run_id": run_id,
                "action": str(replay.get("action", "unknown") or "unknown"),
                "status": str(replay.get("status", "Unknown") or "Unknown"),
                "duration": float(replay.get("duration_seconds", 0.0) or 0.0),
                "terminal_state": self._coerce_run_state(
                    str(replay.get("terminal_state", RunLifecycleState.IDLE.value))
                ).value,
                "failure_code": str(replay.get("failure_code", "") or ""),
                "error": str(replay.get("error", "") or ""),
                "repo": str(req.get("repo_path", "") or ""),
                "spec": str(req.get("spec", "") or ""),
                "query": str(req.get("query", "") or ""),
                "summary_lines": list(replay.get("summary_lines") or []),
                "event_lines": event_lines,
            }

        req = dict(replay.get("request") or {})
        return {
            "run_id": artifact.run_id,
            "action": artifact.action,
            "status": artifact.status,
            "duration": max(0.0, artifact.duration_seconds),
            "terminal_state": self._coerce_run_state(artifact.terminal_state).value,
            "failure_code": artifact.failure_code or str(replay.get("failure_code", "") or ""),
            "error": artifact.error or str(replay.get("error", "") or ""),
            "repo": str(req.get("repo_path") or artifact.repo_path or ""),
            "spec": str(req.get("spec") or artifact.spec or ""),
            "query": str(req.get("query") or artifact.query or ""),
            "summary_lines": list(artifact.summary_lines or replay.get("summary_lines") or []),
            "event_lines": event_lines,
        }

    def _refresh_run_inspector(self) -> list[str]:
        review = self._pending_review_for_display()
        if review is not None:
            token = int(review.get("token", 0) or 0)
            stage = str(review.get("stage") or "patch_application")
            hunks = list(review.get("hunks") or [])
            decisions = {
                str(key): self._normalize_hunk_decision(str(value))
                for key, value in dict(review.get("decisions") or {}).items()
            }
            self._inspector_run_id = token if token > 0 else None
            self._inspector_lines = RunInspector.render_review_lines(
                stage=stage,
                hunks=hunks,
                decisions=decisions,
            )
            with contextlib.suppress(Exception):
                inspector = self.query_one("#run-inspector", RunInspector)
                inspector.set_review(
                    token=token,
                    stage=stage,
                    hunks=hunks,
                    decisions=decisions,
                    run_id=self._inspector_run_id,
                )
            return list(self._inspector_lines)

        snapshot = self._build_run_inspector_snapshot()
        if snapshot is None:
            self._inspector_run_id = None
            self._inspector_lines = ["Run Inspector: no runs yet"]
            with contextlib.suppress(Exception):
                self.query_one("#run-inspector", RunInspector).set_placeholder(
                    self._inspector_lines[0]
                )
            return list(self._inspector_lines)

        lines = RunInspector.render_snapshot_lines(snapshot)
        self._inspector_run_id = int(snapshot.get("run_id", 0) or 0)
        self._inspector_lines = list(lines)
        with contextlib.suppress(Exception):
            self.query_one("#run-inspector", RunInspector).set_lines(
                lines,
                run_id=self._inspector_run_id if self._inspector_run_id > 0 else None,
            )
        return list(lines)

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
    def _parse_approval_value(raw: str | None) -> bool | None:
        if raw is None:
            return None
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "y", "on", "approve", "approved", "allow"}:
            return True
        if value in {"0", "false", "no", "n", "off", "reject", "rejected", "deny"}:
            return False
        return None

    def _approval_gates_enabled(self) -> bool:
        return self._env_flag_enabled(TUI_APPROVAL_GATES_ENV, default=False)

    def _approval_env_decision(self, stage: str) -> bool | None:
        normalized_stage = stage.strip().upper().replace("-", "_")
        stage_key = f"SCHOLARDEVCLAW_TUI_APPROVAL_{normalized_stage}"
        stage_value = self._parse_approval_value(os.environ.get(stage_key))
        if stage_value is not None:
            return stage_value
        return self._parse_approval_value(os.environ.get("SCHOLARDEVCLAW_TUI_APPROVAL_DEFAULT"))

    @staticmethod
    def _approval_stage_message(stage: str) -> str:
        if stage == "patch_application":
            return "Apply generated patch artifacts to temporary validation copy?"
        if stage == "impact_acceptance":
            return "Accept post-validation impact and finalize integration?"
        return f"Approve stage '{stage}'?"

    @staticmethod
    def _normalize_hunk_decision(value: str) -> str:
        decision = str(value or "").strip().lower()
        if decision in {"accept", "reject", "regenerate"}:
            return decision
        return "pending"

    @classmethod
    def _compact_hunk_summary(cls, hunk: dict[str, Any], index: int) -> dict[str, str]:
        hunk_id = str(
            hunk.get("id")
            or hunk.get("hunk_id")
            or hunk.get("key")
            or hunk.get("index")
            or (index + 1)
        ).strip()
        if not hunk_id:
            hunk_id = str(index + 1)
        return {
            "key": hunk_id,
            "id": hunk_id,
            "file": RunInspector._trim(
                str(hunk.get("file") or hunk.get("file_path") or hunk.get("path") or "?"),
                40,
            ),
            "header": RunInspector._trim(
                str(hunk.get("header") or hunk.get("summary") or ""),
                80,
            ),
        }

    @classmethod
    def _compact_hunk_summaries(cls, hunks: list[Any]) -> list[dict[str, str]]:
        summaries: list[dict[str, str]] = []
        for index, hunk in enumerate(hunks):
            if not isinstance(hunk, dict):
                continue
            summaries.append(cls._compact_hunk_summary(hunk, index))
        return summaries

    @staticmethod
    def _review_key(token: int, stage: str) -> str:
        return f"{int(token)}:{str(stage).strip()}"

    @staticmethod
    def _review_decision_counts(*, total: int, decisions: dict[str, str]) -> dict[str, int]:
        accepted = sum(1 for value in decisions.values() if value == "accept")
        rejected = sum(1 for value in decisions.values() if value == "reject")
        regenerated = sum(1 for value in decisions.values() if value == "regenerate")
        pending = max(0, total - accepted - rejected - regenerated)
        return {
            "accepted": accepted,
            "rejected": rejected,
            "regenerated": regenerated,
            "pending": pending,
        }

    def _set_pending_integrate_review(
        self,
        *,
        token: int,
        stage: str,
        hunks: list[dict[str, str]],
    ) -> dict[str, Any]:
        key = self._review_key(token, stage)
        with self._approval_lock:
            event = threading.Event()
            decisions = {hunk["key"]: "pending" for hunk in hunks if hunk.get("key")}
            entry = {
                "token": int(token),
                "stage": str(stage),
                "hunks": list(hunks),
                "decisions": decisions,
                "event": event,
                "submitted": None,
                "created_at": time.monotonic(),
            }
            self._pending_integrate_reviews[key] = entry
            return entry

    def _get_pending_integrate_review(self, token: int, stage: str) -> dict[str, Any] | None:
        key = self._review_key(token, stage)
        with self._approval_lock:
            entry = self._pending_integrate_reviews.get(key)
            if entry is None:
                return None
            return dict(entry)

    def _pending_review_for_display(self) -> dict[str, Any] | None:
        with self._approval_lock:
            candidates = [
                dict(item)
                for item in self._pending_integrate_reviews.values()
                if isinstance(item, dict)
            ]
        if not candidates:
            return None
        active_candidates = [
            item for item in candidates if int(item.get("token", 0) or 0) == self._active_token
        ]
        target = active_candidates or candidates
        target.sort(key=lambda item: float(item.get("created_at", 0.0) or 0.0), reverse=True)
        return target[0]

    def _update_pending_review_decisions(
        self,
        *,
        token: int,
        stage: str,
        decisions: dict[str, Any],
    ) -> dict[str, Any] | None:
        key = self._review_key(token, stage)
        with self._approval_lock:
            entry = self._pending_integrate_reviews.get(key)
            if not isinstance(entry, dict):
                return None
            merged = dict(entry.get("decisions") or {})
            for hunk_key, raw_decision in decisions.items():
                candidate = str(hunk_key or "").strip()
                if not candidate:
                    continue
                merged[candidate] = self._normalize_hunk_decision(str(raw_decision))
            entry["decisions"] = merged
            return dict(entry)

    def _submit_pending_review(
        self,
        *,
        token: int,
        stage: str,
        approved: bool,
        hunk_decisions: dict[str, Any],
    ) -> bool:
        key = self._review_key(token, stage)
        with self._approval_lock:
            entry = self._pending_integrate_reviews.get(key)
            if not isinstance(entry, dict):
                return False
            merged = dict(entry.get("decisions") or {})
            for hunk_key, raw_decision in hunk_decisions.items():
                candidate = str(hunk_key or "").strip()
                if not candidate:
                    continue
                merged[candidate] = self._normalize_hunk_decision(str(raw_decision))
            entry["decisions"] = merged
            entry["submitted"] = {
                "approved": bool(approved),
                "hunk_decisions": dict(merged),
            }
            event = entry.get("event")
            if isinstance(event, threading.Event):
                event.set()
            return True

    def _clear_pending_review(self, *, token: int, stage: str) -> None:
        key = self._review_key(token, stage)
        with self._approval_lock:
            self._pending_integrate_reviews.pop(key, None)

    def _clear_pending_reviews_for_token(self, token: int) -> None:
        with self._approval_lock:
            stale = [
                key
                for key, value in self._pending_integrate_reviews.items()
                if int((value or {}).get("token", 0) or 0) == int(token)
            ]
            for key in stale:
                self._pending_integrate_reviews.pop(key, None)

    def _emit_review_counts_status(
        self, review: dict[str, Any], *, submitted: bool = False
    ) -> None:
        hunks = list(review.get("hunks") or [])
        decisions = {
            str(key): self._normalize_hunk_decision(str(value))
            for key, value in dict(review.get("decisions") or {}).items()
        }
        counts = self._review_decision_counts(total=len(hunks), decisions=decisions)
        stage = str(review.get("stage") or "patch_application")
        label = "submitted" if submitted else "pending"
        self._set_status(
            (
                f"Review [{stage}] {label} "
                f"A:{counts['accepted']} X:{counts['rejected']} "
                f"G:{counts['regenerated']} P:{counts['pending']}"
            ),
            "accent" if not submitted else "success",
        )

    def _build_integrate_approval_callback(
        self, token: int, *, input_reader: Callable[[str], str] = input
    ) -> Callable[[str, dict[str, Any]], bool | dict[str, Any]] | None:
        if not self._approval_gates_enabled():
            return None

        def _callback(stage: str, context: dict[str, Any]) -> bool | dict[str, Any]:
            prompt = self._approval_stage_message(stage)
            self.call_from_thread(
                self.post_message,
                TaskLog(
                    token,
                    f"Approval required [{stage}] — {prompt}",
                ),
            )

            env_decision = self._approval_env_decision(stage)
            if env_decision is not None:
                self.call_from_thread(
                    self.post_message,
                    TaskLog(
                        token,
                        f"Approval [{stage}] from env: {'approved' if env_decision else 'rejected'}",
                    ),
                )
                return env_decision

            raw_hunks = list(context.get("hunks") or []) if isinstance(context, dict) else []
            has_hunks = stage == "patch_application" and bool(raw_hunks)
            if has_hunks:
                hunks = self._compact_hunk_summaries(raw_hunks)
                if hunks:
                    review = self._set_pending_integrate_review(
                        token=token, stage=stage, hunks=hunks
                    )
                    self.call_from_thread(self._emit_review_counts_status, review)
                    self.call_from_thread(self._refresh_run_inspector)
                    self.call_from_thread(
                        self.post_message,
                        TaskLog(
                            token,
                            (
                                f"Approval [{stage}] waiting for inspector review "
                                f"({len(hunks)} hunks)"
                            ),
                        ),
                    )

                    if not (sys.stdin.isatty() and sys.stdout.isatty()):
                        self.call_from_thread(
                            self.post_message,
                            TaskLog(
                                token,
                                (f"Approval [{stage}] auto-approved (non-interactive terminal)"),
                            ),
                        )
                        self.call_from_thread(
                            self._submit_pending_review,
                            token=token,
                            stage=stage,
                            approved=True,
                            hunk_decisions={hunk["key"]: "accept" for hunk in hunks},
                        )

                    start = time.monotonic()
                    while True:
                        if self._is_task_cancelled(token):
                            self.call_from_thread(
                                self.post_message,
                                TaskLog(token, f"Approval [{stage}] rejected (task cancelled)"),
                            )
                            self.call_from_thread(
                                self._clear_pending_review, token=token, stage=stage
                            )
                            self.call_from_thread(self._refresh_run_inspector)
                            return False

                        with self._approval_lock:
                            key = self._review_key(token, stage)
                            live = self._pending_integrate_reviews.get(key)
                            if isinstance(live, dict):
                                submitted = dict(live.get("submitted") or {})
                                decisions = dict(live.get("decisions") or {})
                                wait_event = live.get("event")
                            else:
                                submitted = {}
                                decisions = {}
                                wait_event = None

                        if submitted:
                            approved = bool(submitted.get("approved", False))
                            clean_decisions: dict[str, str] = {}
                            for hunk in hunks:
                                hunk_key = str(hunk.get("key") or "").strip()
                                if not hunk_key:
                                    continue
                                normalized = self._normalize_hunk_decision(
                                    str(decisions.get(hunk_key, "pending"))
                                )
                                if normalized == "pending":
                                    normalized = "accept"
                                clean_decisions[hunk_key] = normalized

                            self.call_from_thread(
                                self.post_message,
                                TaskLog(
                                    token,
                                    (
                                        f"Approval [{stage}] "
                                        f"{'approved' if approved else 'rejected'} "
                                        "from inspector"
                                    ),
                                ),
                            )
                            self.call_from_thread(
                                self._clear_pending_review, token=token, stage=stage
                            )
                            self.call_from_thread(self._refresh_run_inspector)
                            return {
                                "approved": approved,
                                "hunk_decisions": clean_decisions,
                            }

                        if wait_event is not None and isinstance(wait_event, threading.Event):
                            wait_event.wait(timeout=APPROVAL_REVIEW_POLL_SECONDS)
                        else:
                            time.sleep(APPROVAL_REVIEW_POLL_SECONDS)

                        if (time.monotonic() - start) >= APPROVAL_REVIEW_TIMEOUT_SECONDS:
                            self.call_from_thread(
                                self.post_message,
                                TaskLog(token, f"Approval [{stage}] rejected (review timeout)"),
                            )
                            self.call_from_thread(
                                self._clear_pending_review, token=token, stage=stage
                            )
                            self.call_from_thread(self._refresh_run_inspector)
                            return False

            if not (sys.stdin.isatty() and sys.stdout.isatty()):
                self.call_from_thread(
                    self.post_message,
                    TaskLog(
                        token,
                        f"Approval [{stage}] auto-approved (non-interactive terminal)",
                    ),
                )
                return True

            try:
                response = input_reader(f"{prompt} [y/N]: ")
            except EOFError:
                self.call_from_thread(
                    self.post_message,
                    TaskLog(token, f"Approval [{stage}] rejected (EOF while prompting)"),
                )
                return False

            parsed = self._parse_approval_value(response)
            decision = bool(parsed) if parsed is not None else False
            self.call_from_thread(
                self.post_message,
                TaskLog(
                    token,
                    f"Approval [{stage}] {'approved' if decision else 'rejected'}",
                ),
            )
            return decision

        return _callback

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
            "paper",
            "paper arxiv:1706.03762",
            "paper ./paper.pdf",
            "from-paper arxiv:1706.03762",
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
            "inspect",
            "run show 1",
            "run events 1",
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

        if head == "inspect":
            return "inspect", {}

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
            if subcommand == "events":
                if len(parts) < 3:
                    return "run_events", {}
                try:
                    run_id = int(parts[2])
                except ValueError:
                    return "run_events", {}
                parsed: dict[str, Any] = {"run_id": run_id}
                if len(parts) >= 4:
                    try:
                        parsed["limit"] = int(parts[3])
                    except ValueError:
                        return "run_events", parsed
                return "run_events", parsed

        if head in PAPER_WORKFLOW_ALIASES:
            return "paper_workflow", {"source": raw[len(parts[0]) :].strip()}

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
        failure_code: str = "",
        error: str = "",
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
            failure_code=str(failure_code or "").strip(),
            error=self._trim_event_message(str(error or ""), limit=220),
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
            raise LLMConfigError("Run `setup` to choose an LLM provider first.")

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
            raise LLMConfigError(
                f"No {auth_provider.display_name} key found. Run `setup` and paste your key."
            )
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
        self._append_run_event(
            self._active_token,
            "run.accepted",
            phase=self._phase_for_action(action),
            state=RunLifecycleState.IDLE.value,
            message=f"Accepted {action}",
            level="info",
        )
        self._transition_run_state(RunLifecycleState.QUEUED, action=action)
        self._append_run_event(
            self._active_token,
            "run.queued",
            phase=self._phase_for_run_state(RunLifecycleState.QUEUED, action),
            state=RunLifecycleState.QUEUED.value,
            message=f"Queued {action}",
            level="info",
        )
        self.query_one("#status-bar", StatusBar).start_timer()
        self._transition_run_state(RunLifecycleState.RUNNING, action=action)
        self._append_run_event(
            self._active_token,
            "run.running",
            phase=self._phase_for_run_state(RunLifecycleState.RUNNING, action),
            state=RunLifecycleState.RUNNING.value,
            message=f"Running {action}",
            level="info",
        )
        self._emit_progress(action, 0.05)
        self._update_command_meta()
        self._refresh_run_inspector()

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
            self._append_output(
                f"Error: configure an LLM provider first ({self._supported_provider_names()})",
                "error",
            )
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
        self._append_run_event(
            self._active_token,
            "run.accepted",
            phase=self._phase_for_action("chat"),
            state=RunLifecycleState.IDLE.value,
            message="Accepted chat prompt",
            level="info",
        )
        self._append_run_event(
            self._active_token,
            "run.queued",
            phase=self._phase_for_run_state(RunLifecycleState.QUEUED, "chat"),
            state=RunLifecycleState.QUEUED.value,
            message="Queued chat prompt",
            level="info",
        )
        self._transition_run_state(RunLifecycleState.CHATTING, action="chat")
        self._append_run_event(
            self._active_token,
            "run.running",
            phase=self._phase_for_run_state(RunLifecycleState.CHATTING, "chat"),
            state=RunLifecycleState.CHATTING.value,
            message="Streaming chat response",
            level="info",
        )
        self.query_one("#status-bar", StatusBar).start_timer()
        self._set_live_text("Thinking...", "system")
        self._update_command_meta()
        self._refresh_run_inspector()

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
                    approval_callback = self._build_integrate_approval_callback(token)
                    result = run_integrate(
                        request["repo_path"],
                        request["spec"],
                        log_callback=_log_callback,
                        approval_callback=approval_callback,
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
            self._append_output(f"Providers: {self._supported_provider_names()}")
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
        if action == "inspect":
            lines = self._refresh_run_inspector()
            for idx, line in enumerate(lines):
                self._append_output(line, "info" if idx > 0 else "accent")
            self._set_status("Run inspector snapshot", "info")
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
        if action == "run_events":
            run_id_raw = request.get("run_id")
            if run_id_raw is None:
                self._append_output(
                    "Warning: run id required. Usage: run events <id> [limit]", "warning"
                )
                self._set_status("Run events requires id", "warning")
                return
            run_id = int(run_id_raw)
            limit_raw = request.get("limit")
            limit: int | None = None
            if isinstance(limit_raw, int) and limit_raw > 0:
                limit = limit_raw
            lines = self._render_run_events(run_id, limit=limit)
            level = "warning" if lines and "no recorded events" in lines[0].lower() else "info"
            for idx, line in enumerate(lines):
                self._append_output(line, level if idx == 0 else "info")
            if level == "warning":
                self._set_status(f"Run #{run_id} has no events", "warning")
            elif limit is not None:
                self._set_status(f"Run #{run_id} events (last {limit})", "info")
            else:
                self._set_status(f"Run #{run_id} events", "info")
            return
        if action == "paper_workflow":
            self._open_paper_workflow(str(request.get("source", "") or ""))
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
                self._append_output(
                    f"Error: provider must be one of: {self._supported_provider_names()}",
                    "error",
                )
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
            provider = (
                self._provider
                if self._provider in SUPPORTED_TUI_PROVIDERS
                else DEFAULT_TUI_PROVIDER
            )
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

        self._append_run_event(
            run_id,
            "run.rerun_invoked",
            phase="validating",
            state=self._coerce_run_state(
                str(replay.get("terminal_state", RunLifecycleState.IDLE.value))
            ).value,
            message=f"Rerun requested for run #{run_id}",
            level="info",
        )
        self._save_runtime_state()
        self._refresh_run_inspector()

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

    @on(RunInspector.InspectorAction)
    def on_inspector_action(self, message: RunInspector.InspectorAction) -> None:
        if message.action in {"review_update", "review_submit"}:
            payload = dict(getattr(message, "payload", {}) or {})
            token = int(payload.get("token", self._active_token) or 0)
            stage = str(payload.get("stage", "patch_application") or "patch_application")
            decisions = dict(payload.get("hunk_decisions") or {})

            review = self._update_pending_review_decisions(
                token=token,
                stage=stage,
                decisions=decisions,
            )
            if review is None:
                self._append_output("Warning: no pending review request", "warning")
                self._set_status("No pending review", "warning")
                with contextlib.suppress(Exception):
                    self.query_one("#prompt-input", PromptInput).focus()
                return

            if message.action == "review_submit":
                approved = bool(payload.get("approved", True))
                self._submit_pending_review(
                    token=token,
                    stage=stage,
                    approved=approved,
                    hunk_decisions=decisions,
                )
                latest = self._get_pending_integrate_review(token, stage)
                if latest is not None:
                    self._emit_review_counts_status(latest, submitted=True)
                    counts = self._review_decision_counts(
                        total=len(list(latest.get("hunks") or [])),
                        decisions=dict(latest.get("decisions") or {}),
                    )
                    self._append_output(
                        (
                            f"Review submitted [{stage}] "
                            f"A:{counts['accepted']} X:{counts['rejected']} "
                            f"G:{counts['regenerated']} P:{counts['pending']}"
                        ),
                        "success",
                    )
            else:
                self._emit_review_counts_status(review)

            self._refresh_run_inspector()
            if message.action == "review_submit":
                with contextlib.suppress(Exception):
                    self.query_one("#prompt-input", PromptInput).focus()
            return

        run_id = int(message.run_id or 0)
        if run_id <= 0:
            self._append_output("Warning: no run selected in inspector", "warning")
            self._set_status("Inspector has no run selected", "warning")
            with contextlib.suppress(Exception):
                self.query_one("#prompt-input", PromptInput).focus()
            return

        if message.action == "show":
            self._execute_action_request(
                "run_show", {"run_id": run_id}, command=f"run show {run_id}"
            )
        elif message.action == "rerun":
            self._execute_action_request(
                "run_rerun", {"run_id": run_id}, command=f"run rerun {run_id}"
            )
        else:
            self._execute_action_request(
                "run_events", {"run_id": run_id}, command=f"run events {run_id}"
            )

        with contextlib.suppress(Exception):
            self.query_one("#prompt-input", PromptInput).focus()

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
        self._record_chat_delta_event(message.token, message.content)
        self.query_one("#status-bar", StatusBar).update_timer()
        self._set_live_text(f"Assistant: {message.content}", "info")
        self._refresh_run_inspector()

    @on(TaskLog)
    def on_task_log(self, message: TaskLog) -> None:
        if message.token != self._active_token:
            return
        self._line_progress += 1
        fraction = min(0.9, 0.1 + (self._line_progress * 0.08))
        line = (message.line or "").strip()
        if line:
            self._record_task_log_event(message.token, line)
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
        self._refresh_run_inspector()

    @on(TaskCompleted)
    def on_task_completed(self, message: TaskCompleted) -> None:
        if message.token != self._active_token:
            return

        self._flush_chat_delta_event(message.token)

        self.query_one("#status-bar", StatusBar).stop_timer()
        self._running_action = None
        self._active_request = None
        duration = 0.0
        started_at = self._run_started_at.pop(message.token, None)
        if started_at is not None:
            duration = max(0.0, time.perf_counter() - started_at)

        result = message.result
        failure_code = ""
        failure_error = ""
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
            self._append_run_event(
                message.token,
                "run.completed",
                phase=self._phase_for_run_state(RunLifecycleState.COMPLETED, "chat"),
                state=RunLifecycleState.COMPLETED.value,
                message="Chat completed",
                level="success",
                payload={"duration_seconds": round(duration, 3)},
            )
            self._add_history_entry(
                run_id=message.token,
                action=message.action,
                status="Success",
                duration=duration,
                terminal_state=RunLifecycleState.COMPLETED,
                failure_code="",
                error="",
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
            self._append_run_event(
                message.token,
                "run.completed",
                phase=self._phase_for_run_state(RunLifecycleState.COMPLETED, message.action),
                state=RunLifecycleState.COMPLETED.value,
                message=f"{message.action} completed",
                level="success",
                payload={"duration_seconds": round(duration, 3)},
            )
            self._add_history_entry(
                run_id=message.token,
                action=message.action,
                status="Success",
                duration=duration,
                terminal_state=RunLifecycleState.COMPLETED,
                failure_code="",
                error="",
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
                failure_code="",
                error="",
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
                failure_error = str(result.error or "Task cancelled")
                failure_code = self._classify_failure_code(failure_error, cancelled=True)
                self._append_run_event(
                    message.token,
                    "run.cancelled",
                    phase=self._phase_for_run_state(RunLifecycleState.CANCELLED, message.action),
                    state=RunLifecycleState.CANCELLED.value,
                    message=f"{message.action} cancelled",
                    level="warning",
                    payload={
                        "duration_seconds": round(duration, 3),
                        "failure_code": failure_code,
                    },
                )
                status = "Cancelled"
                terminal_state = RunLifecycleState.CANCELLED
                self._add_history_entry(
                    run_id=message.token,
                    action=message.action,
                    status="Cancelled",
                    duration=duration,
                    terminal_state=RunLifecycleState.CANCELLED,
                    failure_code=failure_code,
                    error=failure_error,
                    summary_lines=summary_lines,
                    request=message.request,
                    command=str(message.request.get("_original_command") or "") or None,
                )
            else:
                failure_error = str(result.error or "command failed")
                failure_code = self._classify_failure_code(failure_error)
                self._append_output(f"Error: {failure_error}", "error")
                self._transition_run_state(RunLifecycleState.FAILED, action=message.action)
                self._append_run_event(
                    message.token,
                    "run.failed",
                    phase=self._phase_for_run_state(RunLifecycleState.FAILED, message.action),
                    state=RunLifecycleState.FAILED.value,
                    message=self._trim_event_message(failure_error, limit=180),
                    level="error",
                    payload={
                        "duration_seconds": round(duration, 3),
                        "failure_code": failure_code,
                    },
                )
                self._add_history_entry(
                    run_id=message.token,
                    action=message.action,
                    status="Failed",
                    duration=duration,
                    terminal_state=RunLifecycleState.FAILED,
                    failure_code=failure_code,
                    error=failure_error,
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
                failure_code=failure_code,
                error=failure_error,
                summary_lines=summary_lines,
            )
            self._context_hints = []

        self._update_command_meta()
        self._clear_pending_reviews_for_token(message.token)
        self._save_runtime_state()
        self._refresh_run_inspector()
        self.query_one("#prompt-input", PromptInput).focus()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_cancel_task(self) -> None:
        if self._running_action is None:
            if self._phase9_workflow.active and self._phase9_workflow.cancel_event is not None:
                self._phase9_workflow.cancel_event.set()
                self._append_output(
                    f"Cancel requested ({PAPER_WORKFLOW_NAME} workflow)...",
                    "warning",
                )
                self._set_status(f"{PAPER_WORKFLOW_NAME} cancellation requested", "warning")
                return
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

    def _open_paper_workflow(self, source: str = "") -> None:
        source_text = str(source or "").strip()
        if source_text:
            self._start_phase9_workflow(source_text)
            return
        self.push_screen(PaperIngestionScreen(), self._on_paper_ingestion_result)

    def action_open_paper_ingestion(self) -> None:
        self._open_paper_workflow()

    def action_open_understanding(self) -> None:
        self.push_screen(UnderstandingScreen(), self._on_understanding_result)

    def action_open_planning(self) -> None:
        self.push_screen(PlanningScreen(), self._on_planning_result)

    def action_open_generation(self) -> None:
        self.push_screen(GenerationScreen(), self._on_generation_result)

    def action_open_execution(self) -> None:
        self.push_screen(ExecutionScreen(), self._on_execution_result)

    def action_open_product(self) -> None:
        self.push_screen(ProductScreen(), self._on_product_result)

    def action_focus_inspector(self) -> None:
        try:
            self.query_one("#run-inspector", RunInspector).focus()
            self._set_status("Inspector focused (j/k navigate, enter/r/s/e action)", "accent")
        except Exception:
            self._set_status("Inspector unavailable", "warning")

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
        with self._approval_lock:
            for value in self._pending_integrate_reviews.values():
                maybe_event = (value or {}).get("event") if isinstance(value, dict) else None
                if isinstance(maybe_event, threading.Event):
                    maybe_event.set()
            self._pending_integrate_reviews.clear()
        self._running_action = None
        self._active_request = None
        self._run_state = RunLifecycleState.IDLE


def run_tui(*, yes_mode: bool = False) -> None:
    if yes_mode:
        os.environ["SCHOLARDEVCLAW_TUI_APPROVAL_GATES"] = "false"
        os.environ["SCHOLARDEVCLAW_TUI_PHASE9_AUTO_APPROVE"] = "true"
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    app = ScholarDevClawApp()
    app.run()


if __name__ == "__main__":
    run_tui()
