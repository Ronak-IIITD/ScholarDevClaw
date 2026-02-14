from __future__ import annotations

import subprocess
import threading
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
    CSS = """
    Screen {
        layout: vertical;
    }

    #main-row {
        height: 1fr;
    }

    #wizard {
        width: 45%;
        border: solid $primary;
        padding: 1;
    }

    #output {
        width: 55%;
        border: solid $secondary;
        padding: 1;
    }

    #agent-row {
        height: 14;
        border: solid $accent;
        padding: 1;
    }

    #agent-logs {
        height: 1fr;
    }

    .section-title {
        text-style: bold;
        margin-bottom: 1;
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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
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
                yield Label("Status: Idle", id="run-status")
                with Horizontal(classes="spaced"):
                    yield Button("Run", id="run", variant="success")
                    yield Button("Clear output", id="clear", variant="default")
            with Vertical(id="output"):
                yield Label("Result", classes="section-title")
                yield Pretty({}, id="result")
                yield Label("Execution logs", classes="section-title")
                yield TextArea("", id="logs", read_only=True)

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

    def _refresh_action_input_state(self) -> None:
        action = self.query_one("#action", Select).value
        is_search = action == "search"
        needs_spec = action in {"map", "generate", "integrate"}
        supports_output_dir = action == "generate"

        self.query_one("#query", Input).disabled = not is_search
        self.query_one("#search-arxiv", Checkbox).disabled = not is_search
        self.query_one("#search-web", Checkbox).disabled = not is_search
        self.query_one("#search-language", Input).disabled = not is_search
        self.query_one("#search-max-results", Input).disabled = not is_search
        self.query_one("#spec", Input).disabled = not needs_spec
        self.query_one("#output-dir", Input).disabled = not supports_output_dir

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

    @on(Select.Changed, "#action")
    def on_action_changed(self) -> None:
        self._refresh_action_input_state()

    def _run_selected_workflow(self) -> None:
        action = self.query_one("#action", Select).value
        repo_path = self.query_one("#repo-path", Input).value.strip()
        query = self.query_one("#query", Input).value.strip()
        include_arxiv = self.query_one("#search-arxiv", Checkbox).value
        include_web = self.query_one("#search-web", Checkbox).value
        search_language = self.query_one("#search-language", Input).value.strip() or "python"
        max_results_raw = self.query_one("#search-max-results", Input).value.strip() or "10"
        spec = self.query_one("#spec", Input).value.strip()
        output_dir = self.query_one("#output-dir", Input).value.strip() or None

        try:
            max_results = max(1, int(max_results_raw))
        except ValueError:
            max_results = 10

        run_button = self.query_one("#run", Button)
        run_button.disabled = True
        self.query_one("#run-status", Label).update(f"Status: Running '{action}'...")
        self._live_logs_enabled = True

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
                result = run_integrate(repo_path, spec or None, log_callback=_emit)
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
        if not self._live_logs_enabled:
            self._append_logs("logs", message.logs)
        self._live_logs_enabled = False
        self.query_one("#run", Button).disabled = False
        status = "Done" if message.error is None else "Failed"
        self.query_one("#run-status", Label).update(f"Status: {status} ({message.title})")

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
