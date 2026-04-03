"""Command-first terminal UI for ScholarDevClaw."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from difflib import get_close_matches
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

from .screens import HelpOverlay
from .widgets import LogView, PromptInput, StatusBar

logger = logging.getLogger(__name__)

MODES = ("analyze", "search", "edit")
MODE_HINTS = {
    "analyze": [
        "Hint -> analyze ./repo",
        "Hint -> suggest ./repo",
        "Hint -> validate ./repo",
    ],
    "search": [
        "Hint -> search layer normalization",
        "Hint -> search flash attention",
        "Hint -> set model auto",
    ],
    "edit": [
        "Hint -> map ./repo rmsnorm",
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
        "set model auto",
        ":search",
        ":edit",
    ],
    "search": [
        "search layer normalization",
        "search flash attention",
        "set mode search",
        "set model auto",
        ":analyze",
        ":edit",
    ],
    "edit": [
        "map ./repo rmsnorm",
        "generate ./repo rmsnorm",
        "integrate ./repo rmsnorm",
        "set dir ./repo",
        "set model auto",
        ":analyze",
        ":search",
    ],
}
GLOBAL_COMMANDS = [
    "set mode analyze",
    "set mode search",
    "set mode edit",
    "set model auto",
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
}


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
        self._model = "auto"
        self._directory = "."
        self._status_level = "info"
        self._command_history: list[str] = []
        self._history_index = 0
        self._suggestions: list[str] = []
        self._hint_index = 0
        self._task_token = 0
        self._active_token = 0
        self._running_action: str | None = None
        self._active_request: dict[str, Any] | None = None
        self._last_escape_time = 0.0
        self._escape_pressed_count = 0
        self._escape_warning_shown = False
        self._line_progress = 0

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
        self._sync_status_bar()
        self._set_status("Ready", "info")
        self._update_command_meta()
        self.set_interval(6.0, self._rotate_hint)
        self.query_one("#prompt-input", PromptInput).focus()

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
        }
        provider, model = self._resolve_model_provider()
        if provider is None:
            os.environ.pop("SCHOLARDEVCLAW_API_PROVIDER", None)
            os.environ.pop("SCHOLARDEVCLAW_API_MODEL", None)
        else:
            os.environ["SCHOLARDEVCLAW_API_PROVIDER"] = provider
            if model:
                os.environ["SCHOLARDEVCLAW_API_MODEL"] = model
            else:
                os.environ.pop("SCHOLARDEVCLAW_API_MODEL", None)
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

        if "integrate" in lower:
            action = "integrate"
        elif "generate" in lower or "patch" in lower:
            action = "generate"
        elif "validate" in lower or "benchmark" in lower:
            action = "validate"
        elif "suggest" in lower or "improvement" in lower:
            action = "suggest"
        elif "search" in lower or "find paper" in lower:
            action = "search"
        elif "map" in lower:
            action = "map"
        elif "analyze" in lower or "scan" in lower:
            action = "analyze"
        else:
            action = self._mode if self._mode in MODES else "analyze"

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
        self.query_one("#status-bar", StatusBar).set_context(
            mode=self._mode,
            model=self._model,
            directory=self._directory,
        )

    def _set_status(self, message: str, level: str = "info") -> None:
        self._status_level = level
        self.query_one("#status-bar", StatusBar).set_status(message, level)

    def _append_output(self, line: str, level: str = "auto") -> None:
        self.query_one("#main-output", LogView).add_log(line, level)

    def _clear_output(self) -> None:
        self.query_one("#main-output", LogView).clear_logs()

    def _progress_bar(self, fraction: float) -> str:
        width = 10
        clamped = max(0.0, min(1.0, fraction))
        filled = int(round(width * clamped))
        return f"[{'█' * filled}{'░' * (width - filled)}] {int(clamped * 100):>3d}%"

    def _emit_progress(self, action: str, fraction: float, label: str | None = None) -> None:
        text = label or PROGRESS_LABELS.get(action, "Working...")
        self._append_output(text, "accent")
        self._append_output(self._progress_bar(fraction), "system")

    def _rotate_hint(self) -> None:
        if self._suggestions:
            return
        self._hint_index = (self._hint_index + 1) % len(MODE_HINTS[self._mode])
        self._update_command_meta()

    def _all_commands(self) -> list[str]:
        commands = MODE_COMMANDS[self._mode] + GLOBAL_COMMANDS
        return list(dict.fromkeys(commands))

    def _compute_suggestions(self, prompt: str) -> list[str]:
        prompt = prompt.strip()
        commands = self._all_commands()
        if not prompt:
            return []

        ranked: list[str] = []
        for candidate in commands:
            lower_candidate = candidate.lower()
            lower_prompt = prompt.lower()
            if lower_candidate.startswith(lower_prompt):
                ranked.append(candidate)
        for candidate in commands:
            lower_candidate = candidate.lower()
            lower_prompt = prompt.lower()
            if lower_prompt in lower_candidate and candidate not in ranked:
                ranked.append(candidate)
        for candidate in get_close_matches(prompt, commands, n=3, cutoff=0.3):
            if candidate not in ranked:
                ranked.append(candidate)
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
        widget.update(f"[dim]{MODE_HINTS[self._mode][self._hint_index]}[/]")

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        self._hint_index = 0
        self._sync_status_bar()
        self._set_status(f"Mode set to {mode}", "accent")
        self._append_output(f"Mode: {mode}", "accent")
        self._update_command_meta()

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
            if payload.startswith("dir "):
                return "set_dir", {"directory": payload.split(" ", 1)[1].strip()}
            return None, {}

        parts = raw.split()
        head = parts[0].lower()

        if head == "set" and len(parts) >= 3:
            key = parts[1].lower()
            value = " ".join(parts[2:]).strip()
            if key == "mode":
                return "set_mode", {"mode": value}
            if key == "model":
                return "set_model", {"model": value}
            if key == "dir":
                return "set_dir", {"directory": value}

        if head in {"help", "clear", "quit"}:
            return head, {}

        if head == "search":
            return "search", {
                "action": "search",
                "query": raw[len(parts[0]) :].strip(),
                "include_arxiv": True,
                "include_web": False,
            }

        if head in {"analyze", "suggest", "validate"}:
            repo_path = parts[1] if len(parts) > 1 else self._directory
            return head, {"action": head, "repo_path": repo_path}

        if head in {"map", "generate", "integrate"}:
            repo_path = parts[1] if len(parts) > 1 else self._directory
            spec = parts[2].lower() if len(parts) > 2 else ""
            return head, {"action": head, "repo_path": repo_path, "spec": spec}

        action, ctx = self._parse_natural_command(raw)
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
        return []

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
        self.query_one("#status-bar", StatusBar).start_timer()
        self._set_status(f"Running {action}", "accent")
        self._emit_progress(action, 0.05)

        thread = threading.Thread(
            target=self._run_task_in_thread,
            args=(self._active_token, action, request),
            daemon=True,
        )
        thread.start()

    def _run_task_in_thread(self, token: int, action: str, request: dict[str, Any]) -> None:
        previous_env = self._apply_provider_env()
        try:
            def _log_callback(line: str) -> None:
                self.call_from_thread(self.post_message, TaskLog(token, line))

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
                result = run_map(request["repo_path"], request["spec"], log_callback=_log_callback)
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
        if action == "set_model":
            self._model = str(request.get("model", "auto") or "auto")
            self._sync_status_bar()
            self._set_status(f"Model set to {self._model}", "accent")
            return
        if action == "set_dir":
            directory = str(request.get("directory", "") or "")
            valid, err = self._validate_repo_path(directory)
            if not valid:
                self._append_output(f"Error: {err}", "error")
                self._set_status("Directory rejected", "error")
                return
            self._directory = directory
            self._sync_status_bar()
            self._set_status(f"Directory set to {directory}", "accent")
            return

        self._start_task(action, request)

    @on(PromptInput.HistoryPrev)
    def on_history_prev(self) -> None:
        if not self._command_history:
            return
        self._history_index = max(0, self._history_index - 1)
        self.query_one("#prompt-input", PromptInput).value = self._command_history[self._history_index]

    @on(PromptInput.HistoryNext)
    def on_history_next(self) -> None:
        if not self._command_history:
            return
        self._history_index = min(len(self._command_history), self._history_index + 1)
        prompt = self.query_one("#prompt-input", PromptInput)
        if self._history_index >= len(self._command_history):
            prompt.value = ""
        else:
            prompt.value = self._command_history[self._history_index]

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

    @on(TaskLog)
    def on_task_log(self, message: TaskLog) -> None:
        if message.token != self._active_token:
            return
        self._line_progress += 1
        fraction = min(0.9, 0.1 + (self._line_progress * 0.08))
        self._append_output(message.line)
        self.query_one("#status-bar", StatusBar).update_timer()
        self._append_output(self._progress_bar(fraction), "system")

    @on(TaskCompleted)
    def on_task_completed(self, message: TaskCompleted) -> None:
        if message.token != self._active_token:
            return

        self.query_one("#status-bar", StatusBar).stop_timer()
        self._running_action = None
        self._active_request = None

        result = message.result
        if result.ok:
            self._append_output(self._progress_bar(1.0), "success")
            for line in self._summarize_result(message.action, result.payload):
                self._append_output(line)
            self._set_status(f"{message.action} complete", "success")
        else:
            self._append_output(f"Error: {result.error or 'command failed'}", "error")
            self._set_status(f"{message.action} failed", "error")

        self.query_one("#prompt-input", PromptInput).focus()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_cancel_task(self) -> None:
        if self._running_action is None:
            self._set_status("No task running", "warning")
            return
        self._task_token += 1
        self._active_token = self._task_token
        self._append_output("Cancel requested", "warning")
        self._running_action = None
        self._active_request = None
        self.query_one("#status-bar", StatusBar).stop_timer()
        self._set_status("Task cancelled", "warning")

    def action_clear_screen(self) -> None:
        self._clear_output()
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app = ScholarDevClawApp()
    app.run()


if __name__ == "__main__":
    run_tui()
