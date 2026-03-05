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
from typing import Callable

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
