"""
Agent REPL — interactive and single-shot command interface.

Supports both the legacy StreamingAgentEngine and the new SmartAgentEngine.
The SmartAgentEngine is the default and provides:
- Query classification (trivial/simple/moderate/complex)
- Token budget management
- Memory-enriched context
- Quality reflection with retry
"""

from __future__ import annotations

import asyncio
import sys
from typing import Callable, Any, Coroutine

import rich
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table

from .engine import StreamingAgentEngine, StreamEvent, StreamEventType, AgentMode, AgentResponse


class StreamingAgentREPL:
    """Interactive REPL that streams agent output to the terminal."""

    def __init__(
        self,
        engine: StreamingAgentEngine | None = None,
        smart: bool = True,
        repo_path: str | None = None,
    ):
        if smart and engine is None:
            # Use the SmartAgentEngine by default
            from .smart_engine import SmartAgentEngine

            self._smart_engine = SmartAgentEngine()
            self.engine = None  # not needed when smart_engine is active
        else:
            self._smart_engine = None
            self.engine = engine or StreamingAgentEngine()

        self.console = Console()
        self.running = False
        self._initial_repo = repo_path
        self._terminal_mode = False
        self._slash_commands = {
            "/help": "Show help",
            "/exit": "Exit the app",
            "/status": "View status",
            "/terminal": "Enter terminal mode",
            "/clear": "Clear screen",
            "/new": "Start new session",
            "/sessions": "List sessions",
            "/repo": "Switch repo: /repo <path>",
            "/agents": "List available agents",
            "/models": "Show model info",
            "/review": "Review changes",
            "/mcp": "Show MCP status",
            "/run": "Run a terminal command: /run <cmd>",
            "/git": "Git helper: /git <args>",
            "/docker": "Docker helper: /docker <args>",
            "/compose": "Docker compose: /compose <args>",
            "/test": "Run tests (smart)",
            "/build": "Run intelligent build/test",
        }
        self._dangerous_fragments = [
            "rm -rf /",
            "rm -rf *",
            "git reset --hard",
            "git push --force",
            "docker system prune",
            "docker volume rm",
            "mkfs",
            "dd if=",
        ]

    def print_welcome(self) -> None:
        welcome = """
# ScholarDevClaw Agent

Your AI-powered research-to-code assistant.

**Quick Start:**
- `analyze <path>` - Analyze a repository
- `integrate <spec>` - Apply research improvements
- `suggest` - Get improvement suggestions
- `help` - See all commands
- `terminal` - Enter terminal mode (super powers)
- `!<cmd>` - Run a terminal command inline (e.g., !ls)
- `/help` - Show slash commands

**Examples:**
- analyze ./my-project
- integrate rmsnorm
- suggest improvements

Type your request or 'exit' to quit.
"""
        self.console.print(Panel(welcome.strip(), title="ScholarDevClaw", border_style="blue"))

    def run(self) -> None:
        self.running = True

        # Create session
        if self._smart_engine:
            session = self._smart_engine.create_session(self._initial_repo)
        else:
            assert self.engine is not None
            session = self.engine.create_session(self._initial_repo)

        self.print_welcome()

        while self.running:
            try:
                if self._terminal_mode:
                    user_input = self._terminal_prompt()
                else:
                    user_input = Prompt.ask("\n[bold blue]ScholarDevClaw[/bold blue]", default="")

                if not user_input.strip():
                    continue

                # Slash command handling
                if user_input.strip().startswith("/"):
                    handled = self._handle_slash_command(user_input.strip())
                    if handled:
                        continue

                # Inline terminal command
                if user_input.strip().startswith("!"):
                    cmd = user_input.strip()[1:].strip()
                    asyncio.run(self._process_terminal(cmd))
                    continue

                # Enter terminal mode
                if user_input.strip().lower() in ["terminal", "shell", "console"]:
                    self._terminal_mode = True
                    self.console.print("Entered terminal mode. Type 'exit' to leave.", style="cyan")
                    asyncio.run(self._process_terminal(None))
                    continue

                # Exit terminal mode
                if self._terminal_mode and user_input.strip().lower() in ["exit", "quit", "back"]:
                    self._terminal_mode = False
                    self.console.print("Exited terminal mode.", style="cyan")
                    continue

                if user_input.lower() in ["exit", "quit", "q"]:
                    self.console.print("Goodbye!")
                    break

                if user_input.lower() == "clear":
                    self.console.clear()
                    continue

                if self._terminal_mode:
                    asyncio.run(self._process_terminal(user_input))
                else:
                    asyncio.run(self._process_streaming(user_input))

            except KeyboardInterrupt:
                self.console.print("\nGoodbye!")
                break
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"\nError: {str(e)}", style="red")

        self.running = False

    async def _process_streaming(self, user_input: str) -> None:
        """Process user input with real-time streaming."""
        try:
            if self._smart_engine:
                async for event in self._smart_engine.stream_smart(user_input):
                    self._handle_stream_event(event)
            else:
                assert self.engine is not None
                async for event in self.engine.stream_events(user_input):
                    self._handle_stream_event(event)
        except Exception as e:
            self.console.print(f"\nError: {str(e)}", style="red")

    async def _process_terminal(self, command: str | None) -> None:
        """Process terminal command using SmartAgentEngine terminal mode."""
        try:
            if not self._smart_engine:
                self.console.print("Terminal mode requires SmartAgentEngine", style="red")
                return

            result = await self._smart_engine._exec_terminal(command)
            if result.message:
                self.console.print(result.message)
        except Exception as e:
            self.console.print(f"\nTerminal error: {str(e)}", style="red")

    def _terminal_prompt(self) -> str:
        """Return terminal-style prompt."""
        if not self._smart_engine:
            return Prompt.ask("\n[bold blue]ScholarDevClaw[/bold blue]", default="")

        prompt = self._smart_engine.terminal.get_prompt()
        return Prompt.ask(f"\n{prompt}", default="")

    def _handle_slash_command(self, command: str) -> bool:
        """Handle slash commands like /help, /status, /sessions."""
        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd == "/help":
            table = Table(title="Slash Commands")
            table.add_column("Command", style="cyan")
            table.add_column("Description", style="green")
            for k, v in self._slash_commands.items():
                table.add_row(k, v)
            self.console.print(table)
            return True

        if cmd in ("/exit", "/quit"):
            self.console.print("Goodbye!")
            self.running = False
            return True

        if cmd == "/clear":
            self.console.clear()
            return True

        if cmd == "/status":
            if self._smart_engine:
                self.console.print(self._smart_engine._build_status_text())
                return True
            return False

        if cmd == "/terminal":
            self._terminal_mode = True
            self.console.print("Entered terminal mode. Type 'exit' to leave.", style="cyan")
            asyncio.run(self._process_terminal(None))
            return True

        if cmd == "/new":
            if self._smart_engine:
                self._smart_engine.create_session(self._initial_repo)
                self.console.print("New session created", style="green")
                return True
            return False

        if cmd == "/sessions":
            if self._smart_engine:
                sessions = self._smart_engine.sessions
                if not sessions:
                    self.console.print("No sessions yet.")
                else:
                    for sid, sess in sessions.items():
                        repo = sess.repo_path or "(no repo)"
                        marker = "*" if self._smart_engine.current_session == sess else " "
                        self.console.print(f"{marker} {sid} — {repo}")
                return True
            return False

        if cmd == "/repo":
            if not args:
                self.console.print("Usage: /repo <path>", style="yellow")
                return True
            path = " ".join(args)
            if self._smart_engine and self._smart_engine.switch_repo(path):
                self.console.print(f"Switched repo to {path}", style="green")
            else:
                self.console.print(f"Failed to switch repo to {path}", style="red")
            return True

        if cmd == "/run":
            if not args:
                self.console.print("Usage: /run <command>", style="yellow")
                return True
            command = " ".join(args)
            if self._confirm_dangerous(command):
                self._run_async(self._process_terminal(command))
            return True

        if cmd == "/git":
            if not args:
                self.console.print("Usage: /git <args>", style="yellow")
                return True
            command = "git " + " ".join(args)
            if self._confirm_dangerous(command):
                self._run_async(self._process_terminal(command))
            return True

        if cmd == "/docker":
            if not args:
                self.console.print("Usage: /docker <args>", style="yellow")
                return True
            command = "docker " + " ".join(args)
            if self._confirm_dangerous(command):
                self._run_async(self._process_terminal(command))
            return True

        if cmd == "/compose":
            if not args:
                self.console.print("Usage: /compose <args>", style="yellow")
                return True
            command = "docker compose " + " ".join(args)
            if self._confirm_dangerous(command):
                self._run_async(self._process_terminal(command))
            return True

        if cmd == "/test":
            self._run_async(self._process_streaming("test"))
            return True

        if cmd == "/build":
            self._run_async(self._process_streaming("do it"))
            return True

        if cmd in self._slash_commands:
            # Placeholder for future commands
            self.console.print(f"{cmd}: {self._slash_commands[cmd]}")
            return True

        return False

    def _run_async(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Run coroutine safely from sync context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
            return

        # If already in event loop, create task
        loop.create_task(coro)

    def _confirm_dangerous(self, command: str) -> bool:
        """Prompt confirmation for potentially destructive commands."""
        lower = command.lower()
        if any(fragment in lower for fragment in self._dangerous_fragments):
            confirm = Prompt.ask(
                f"Dangerous command detected. Type 'yes' to continue: {command}",
                default="no",
            )
            return confirm.strip().lower() == "yes"
        return True

    def _handle_stream_event(self, event: StreamEvent) -> None:
        """Handle a single streaming event with rich formatting."""

        if event.type == StreamEventType.START:
            pass  # silent start

        elif event.type == StreamEventType.PROGRESS:
            self.console.print(event.message, style="dim")

        elif event.type == StreamEventType.OUTPUT:
            if event.message.startswith("#"):
                md = Markdown(event.message)
                self.console.print(md)
            elif event.message.startswith("  -") or event.message.startswith("  *"):
                self.console.print(event.message, style="cyan")
            else:
                self.console.print(event.message)

        elif event.type == StreamEventType.ERROR:
            self.console.print(f"Error: {event.message}", style="red")

        elif event.type == StreamEventType.COMPLETE:
            self.console.print(event.message, style="dim")

        elif event.type == StreamEventType.SUGGESTION:
            self.console.print(f"  Try: {event.message}", style="yellow")


# Alias for backwards compatibility
AgentREPL = StreamingAgentREPL


def run_agent_repl(repo_path: str | None = None) -> None:
    """Launch the interactive REPL with SmartAgentEngine."""
    repl = StreamingAgentREPL(smart=True, repo_path=repo_path)
    repl.run()


def run_agent_command(
    user_input: str,
    repo_path: str | None = None,
) -> AgentResponse:
    """
    Run a single agent command and return the full response.

    Unlike the old implementation, this actually collects output from stream
    events and returns meaningful content instead of a generic placeholder.
    """
    from .smart_engine import SmartAgentEngine

    engine = SmartAgentEngine()
    session = engine.create_session(repo_path)

    async def _run() -> AgentResponse:
        return await engine.process(user_input)

    return asyncio.run(_run())
