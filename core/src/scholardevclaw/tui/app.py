"""Command-first terminal UI for ScholarDevClaw."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

from .screens import HelpOverlay, ProviderSetupScreen
from .widgets import LogView, PromptInput, StatusBar

logger = logging.getLogger(__name__)

MODES = ("analyze", "search", "edit")
SUPPORTED_TUI_PROVIDERS = {
    "openrouter": AuthProvider.OPENROUTER,
    "ollama": AuthProvider.OLLAMA,
}
MODE_HINTS = {
    "analyze": [
        "Hint -> analyze ./repo",
        "Hint -> ask what this repo does",
        "Hint -> suggest ./repo",
        "Hint -> validate ./repo",
    ],
    "search": [
        "Hint -> search layer normalization",
        "Hint -> ask for papers on flash attention",
        "Hint -> search flash attention",
        "Hint -> setup",
    ],
    "edit": [
        "Hint -> map ./repo rmsnorm",
        "Hint -> ask how to implement RMSNorm",
        "Hint -> generate ./repo rmsnorm",
        "Hint -> integrate ./repo rmsnorm",
    ],
}
MODE_COMMANDS = {
    "analyze": [
        "analyze ./repo",
        "suggest ./repo",
        "validate ./repo",
        "set dir ./repo",
        "set provider openrouter",
        "set model anthropic/claude-sonnet-4",
        ":search",
        ":edit",
    ],
    "search": [
        "search layer normalization",
        "search flash attention",
        "set mode search",
        "chat find fast inference ideas",
        "setup",
        ":analyze",
        ":edit",
    ],
    "edit": [
        "map ./repo rmsnorm",
        "generate ./repo rmsnorm",
        "integrate ./repo rmsnorm",
        "set dir ./repo",
        "chat how should I patch this file",
        ":analyze",
        ":search",
    ],
}
GLOBAL_COMMANDS = [
    "setup",
    "providers",
    "status",
    "chat hello",
    "set mode analyze",
    "set mode search",
    "set mode edit",
    "set provider openrouter",
    "set provider ollama",
    "set model anthropic/claude-sonnet-4",
    "set dir ./repo",
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
CHAT_SYSTEM_PROMPTS = {
    "analyze": (
        "You are ScholarDevClaw, a terse coding assistant inside a terminal UI. "
        "In analyze mode, help the user understand the repository, architecture, and likely next shell commands. "
        "Keep answers short, concrete, and developer-focused."
    ),
    "search": (
        "You are ScholarDevClaw, a terse research assistant inside a terminal UI. "
        "In search mode, answer with concise research directions, paper names, and implementation tradeoffs."
    ),
    "edit": (
        "You are ScholarDevClaw, a terse coding assistant inside a terminal UI. "
        "In edit mode, focus on code changes, implementation advice, and safe next actions."
    ),
}


@dataclass
class TUIRuntimeState:
    provider: str = "setup"
    model: str = ""
    directory: str = "."


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


class ScholarDevClawApp(App[None]):
    """Keyboard-first shell UI."""

    BINDINGS = [
        ("ctrl+c", "cancel_task", "Cancel"),
        ("ctrl+k", "clear_screen", "Clear"),
        ("ctrl+h", "show_help", "Help"),
    ]

    CSS = """
    Screen {
        layout: vertical;
        background: #0b0f12;
        color: #d7dee7;
        padding: 0 1;
    }

    #header {
        height: 1;
        color: #7dd3fc;
        text-style: bold;
    }

    .separator {
        height: 1;
        color: #475569;
    }

    #command-meta {
        height: auto;
        color: #94a3b8;
    }

    #prompt-input {
        height: 1;
        border: none;
        background: transparent;
        color: #e2e8f0;
        padding: 0;
    }

    #prompt-input:focus {
        border: none;
        background: transparent;
    }
    """

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
        self._last_escape_time = 0.0
        self._escape_pressed_count = 0
        self._escape_warning_shown = False
        self._line_progress = 0
        self._chat_history: list[dict[str, str]] = []
        self._chat_preview = ""
        self._session_input_tokens = 0
        self._session_output_tokens = 0
        self._last_total_tokens = 0
        self._load_runtime_state()
        if not self._model and self._provider in SUPPORTED_TUI_PROVIDERS:
            self._model = DEFAULT_MODELS[SUPPORTED_TUI_PROVIDERS[self._provider]]

    def compose(self) -> ComposeResult:
        yield Static("ScholarDevClaw", id="header")
        yield Static("────────────────────────", classes="separator")
        yield StatusBar(id="status-bar")
        yield Static("────────────────────────", classes="separator")
        yield LogView(id="main-output")
        yield Static("────────────────────────", classes="separator")
        with Vertical():
            yield Static("", id="command-meta")
            yield PromptInput(placeholder="> ", id="prompt-input")

    def on_mount(self) -> None:
        if self._directory in {"", "."}:
            self._directory = os.getcwd()
        self._sync_status_bar()
        self._set_status("Ready", "info")
        self._update_command_meta()
        self.set_interval(6.0, self._rotate_hint)
        self.query_one("#prompt-input", PromptInput).focus()
        self.set_timer(0.01, self._maybe_show_setup)

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
                state = TUIRuntimeState(
                    provider=str(data.get("provider", state.provider)),
                    model=str(data.get("model", state.model)),
                    directory=str(data.get("directory", state.directory)),
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

    def _save_runtime_state(self) -> None:
        config_path = self._runtime_state_path()
        payload = {
            "provider": self._provider,
            "model": self._model,
            "directory": self._directory,
        }
        config_path.write_text(json.dumps(payload, indent=2))

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
        self._save_runtime_state()
        return True, "OK"

    # ------------------------------------------------------------------
    # Validation helpers used by tests and command execution
    # ------------------------------------------------------------------

    def _validate_repo_path(self, path: str) -> tuple[bool, str]:
        if not path or not path.strip():
            return False, "Repository path is required"
        resolved = Path(path).expanduser()
        if not resolved.exists():
            return False, "Repository not found"
        if not resolved.is_dir():
            return False, "Repository path must be a directory"
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
        for prefix in (
            "please ",
            "can you ",
            "could you ",
            "would you ",
            "lets ",
            "let's ",
            "run ",
        ):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :].strip()
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
            f"analyze {command_dir}",
            f"suggest {command_dir}",
            f"validate {command_dir}",
            f"map {command_dir} rmsnorm",
            f"generate {command_dir} rmsnorm",
            f"integrate {command_dir} rmsnorm",
            "setup",
            "providers",
            "status",
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
                    lines.append(f"{prefix} [bold #7dd3fc]{suggestion}[/]")
                else:
                    lines.append(f"{prefix} [dim]{suggestion}[/]")
            widget.update("\n".join(lines))
            return
        if self._context_hints:
            lines = []
            for idx, hint in enumerate(self._context_hints[:3]):
                prefix = "Next ->" if idx == 0 else "       "
                style = "[bold #7dd3fc]" if idx == 0 else "[dim]"
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
                    "repo",
                    "repository",
                    "project",
                    "here",
                }:
                    repo_path = parts[1]
            return head, {"action": head, "repo_path": repo_path}

        if head in {"map", "generate", "integrate"}:
            repo_path = parts[1] if len(parts) > 1 else self._directory
            spec = parts[2].lower() if len(parts) > 2 else ""
            return head, {"action": head, "repo_path": repo_path, "spec": spec}

        action, ctx = self._parse_natural_command(raw)
        if action == "chat":
            return "chat", {"action": "chat", "prompt": ctx.get("prompt", raw)}
        ctx.setdefault("action", action)
        if action == "search":
            ctx.setdefault("include_arxiv", True)
            ctx.setdefault("include_web", False)
        if action != "search":
            ctx.setdefault("repo_path", self._directory)
        return action, ctx

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

    @staticmethod
    def _is_greeting_prompt(prompt: str) -> bool:
        text = prompt.strip().lower()
        return text in {
            "hi",
            "hello",
            "hey",
            "yo",
            "hii",
            "hiya",
            "good morning",
            "good afternoon",
            "good evening",
        }

    def _build_chat_system_prompt(self, prompt: str = "") -> str:
        base = CHAT_SYSTEM_PROMPTS[self._mode]
        repo_snapshot = self._build_repo_snapshot()
        greeting_rule = (
            "If the user sends only a short greeting, reply naturally in one short friendly sentence and avoid repo/tooling details. "
            if self._is_greeting_prompt(prompt)
            else ""
        )
        return (
            f"{base} "
            f"Current working directory: {self._pretty_directory()}. "
            "If the user asks to run a repo workflow, mention the exact shell command they can run here. "
            "Only claim frameworks, libraries, or architecture details when they are explicitly present in the repo snapshot below or user-provided context. "
            f"{greeting_rule}"
            "If uncertain, say so briefly and suggest the next concrete command. "
            "Do not pretend you already executed commands unless the transcript shows it. "
            f"Repo snapshot: {repo_snapshot}"
        )

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

    def _start_task(self, action: str, request: dict[str, Any]) -> None:
        if self._running_action is not None:
            self._set_status("Task already running", "warning")
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
        self._running_action = action
        self._active_request = request
        self._line_progress = 0
        self._context_hints = []
        self.query_one("#status-bar", StatusBar).start_timer()
        self._set_status(f"Running {action}", "accent")
        self._emit_progress(action, 0.05)
        self._update_command_meta()

        thread = threading.Thread(
            target=self._run_task_in_thread,
            args=(self._active_token, action, request),
            daemon=True,
        )
        thread.start()

    def _start_chat(self, prompt: str) -> None:
        if self._running_action is not None:
            self._set_status("Task already running", "warning")
            return
        if not self._llm_ready():
            self._append_output("Error: configure OpenRouter or Ollama first", "error")
            self._open_setup()
            self._set_status("LLM setup required", "warning")
            return

        self._task_token += 1
        self._active_token = self._task_token
        self._running_action = "chat"
        self._active_request = {"action": "chat", "prompt": prompt}
        self._chat_preview = ""
        self._context_hints = []
        self.query_one("#status-bar", StatusBar).start_timer()
        self._set_status(f"Chatting with {self._provider}", "accent")
        self._set_live_text("Thinking...", "system")
        self._update_command_meta()

        thread = threading.Thread(
            target=self._run_chat_in_thread,
            args=(self._active_token, prompt),
            daemon=True,
        )
        thread.start()

    def _run_task_in_thread(self, token: int, action: str, request: dict[str, Any]) -> None:
        previous_env = self._apply_provider_env()
        try:
            sink_out = io.StringIO()
            sink_err = io.StringIO()

            def _log_callback(line: str) -> None:
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
        except Exception as exc:
            result = type(
                "Result",
                (),
                {"ok": False, "payload": {}, "error": str(exc), "logs": [str(exc)]},
            )()
        finally:
            self._restore_provider_env(previous_env)

        self.call_from_thread(self.post_message, TaskCompleted(token, action, result, request))

    def _run_chat_in_thread(self, token: int, prompt: str) -> None:
        response_text = ""
        try:
            sink_out = io.StringIO()
            sink_err = io.StringIO()
            user_prompt = prompt.strip()
            if self._is_greeting_prompt(user_prompt):
                if user_prompt.startswith("good "):
                    response_text = f"{user_prompt.title()} 👋"
                elif user_prompt in {"yo", "hey", "hiya"}:
                    response_text = "Hey 👋"
                else:
                    response_text = "Hi 👋"

                result = type(
                    "Result",
                    (),
                    {
                        "ok": True,
                        "payload": {
                            "content": response_text,
                            "input_tokens": 0,
                            "output_tokens": 0,
                        },
                        "error": "",
                        "logs": [],
                    },
                )()
                self.call_from_thread(
                    self.post_message,
                    TaskCompleted(token, "chat", result, {"action": "chat", "prompt": prompt}),
                )
                return

            with contextlib.redirect_stdout(sink_out), contextlib.redirect_stderr(sink_err):
                client = self._get_llm_client()
                messages = self._chat_history[-8:] + [{"role": "user", "content": prompt}]
                for chunk in client.chat_stream(
                    prompt,
                    messages=messages,
                    system=self._build_chat_system_prompt(prompt),
                    model=self._model,
                    max_tokens=2048,
                    temperature=0.2,
                ):
                    if token != self._active_token:
                        client.close()
                        return
                    if chunk.delta:
                        response_text += chunk.delta
                        self.call_from_thread(self.post_message, ChatDelta(token, response_text))
                client.close()
            input_tokens = self._estimate_tokens(prompt)
            output_tokens = self._estimate_tokens(response_text)
            result = type(
                "Result",
                (),
                {
                    "ok": True,
                    "payload": {
                        "content": response_text.strip(),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                    "error": "",
                    "logs": [],
                },
            )()
        except (LLMAPIError, LLMConfigError, Exception) as exc:
            result = type(
                "Result",
                (),
                {"ok": False, "payload": {}, "error": str(exc), "logs": [str(exc)]},
            )()

        self.call_from_thread(
            self.post_message,
            TaskCompleted(token, "chat", result, {"action": "chat", "prompt": prompt}),
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    @on(Input.Changed, "#prompt-input")
    def on_prompt_changed(self, event: Input.Changed) -> None:
        self._suggestions = self._compute_suggestions(event.value)
        self._update_command_meta()

    @on(Input.Submitted, "#prompt-input")
    def on_prompt_submitted(self, event: Input.Submitted) -> None:
        command = event.value.strip()
        if not command:
            return
        prompt = self.query_one("#prompt-input", PromptInput)
        prompt.value = ""
        self._record_command(command)
        self._append_output(f"> {command}", "accent")
        self._suggestions = []
        self._update_command_meta()

        action, request = self._build_request(command)
        if action is None:
            self._append_output("Error: command not understood", "error")
            self._set_status("Unknown command", "error")
            return

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
            self._append_output(
                f"Session tokens: {self._session_input_tokens + self._session_output_tokens}"
            )
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
            if not self._model:
                self._model = DEFAULT_MODELS[SUPPORTED_TUI_PROVIDERS[provider]]
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
            self._set_status("Key saved", "success")
            return
        if action == "set_dir":
            directory = str(request.get("directory", "") or "")
            valid, err = self._validate_repo_path(directory)
            if not valid:
                self._append_output(f"Error: {err}", "error")
                self._set_status("Directory rejected", "error")
                return
            self._directory = directory
            self._save_runtime_state()
            self._context_hints = []
            self._sync_status_bar()
            self._set_status(f"Directory set to {directory}", "accent")
            self._update_command_meta()
            return
        if action == "chat":
            self._start_chat(str(request.get("prompt", "") or command))
            return

        self._start_task(action, request)

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
        if line and any(term in line.lower() for term in ("error", "failed", "warning")):
            self._append_output(line)
        self.query_one("#status-bar", StatusBar).update_timer()
        self._set_progress(self._running_action or "analyze", fraction)

    @on(TaskCompleted)
    def on_task_completed(self, message: TaskCompleted) -> None:
        if message.token != self._active_token:
            return

        self.query_one("#status-bar", StatusBar).stop_timer()
        self._running_action = None
        self._active_request = None

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
            self._set_status("chat complete", "success")
        elif result.ok:
            self._set_progress(message.action, 1.0)
            self._clear_progress()
            self._append_output(
                f"{PROGRESS_LABELS.get(message.action, 'Done')} {self._progress_bar(1.0)}",
                "success",
            )
            for line in self._summarize_result(message.action, result.payload):
                self._append_output(line)
            self._context_hints = self._suggest_next_commands(
                message.action, result.payload, message.request
            )
            self._set_status(f"{message.action} complete", "success")
        else:
            self._clear_progress()
            self._append_output(f"Error: {result.error or 'command failed'}", "error")
            self._context_hints = []
            self._set_status(f"{message.action} failed", "error")

        self._update_command_meta()
        self.query_one("#prompt-input", PromptInput).focus()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_cancel_task(self) -> None:
        if self._running_action is None:
            self.exit()
            return
        self._task_token += 1
        self._active_token = self._task_token
        self._clear_progress()
        self._append_output("Cancel requested", "warning")
        self._running_action = None
        self._active_request = None
        self.query_one("#status-bar", StatusBar).stop_timer()
        self._set_status("Task cancelled", "warning")

    def action_clear_screen(self) -> None:
        self._clear_output()
        self._context_hints = []
        self._update_command_meta()
        self._set_status("Screen cleared", "accent")

    def action_show_help(self) -> None:
        self.push_screen(HelpOverlay())

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
        self._running_action = None
        self._active_request = None


def run_tui() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    app = ScholarDevClawApp()
    app.run()


if __name__ == "__main__":
    run_tui()
