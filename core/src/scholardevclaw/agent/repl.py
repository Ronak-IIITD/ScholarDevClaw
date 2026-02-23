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
    def __init__(self, engine: StreamingAgentEngine | None = None):
        self.engine = engine or StreamingAgentEngine()
        self.console = Console()
        self.running = False

    def print_welcome(self) -> None:
        welcome = """
# 👋 Welcome to ScholarDevClaw Agent

Your AI-powered research-to-code assistant.

**Quick Start:**
- `analyze <path>` - Analyze a repository
- `integrate <spec>` - Apply research improvements
- `suggest` - Get improvement suggestions
- `help` - See all commands

**Examples:**
- analyze ./my-project
- integrate rmsnorm
- suggest improvements

Type your request or 'exit' to quit.
"""
        self.console.print(Panel(welcome.strip(), title="ScholarDevClaw", border_style="blue"))

    def run(self) -> None:
        self.running = True
        self.engine.create_session()
        self.print_welcome()

        while self.running:
            try:
                user_input = Prompt.ask("\n[bold blue]ScholarDevClaw[/bold blue]", default="")

                if not user_input.strip():
                    continue

                if user_input.lower() in ["exit", "quit", "q"]:
                    self.console.print("👋 Goodbye!")
                    break

                if user_input.lower() == "clear":
                    self.console.clear()
                    continue

                asyncio.run(self._process_streaming(user_input))

            except KeyboardInterrupt:
                self.console.print("\n👋 Goodbye!")
                break
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"\n❌ Error: {str(e)}", style="red")

        self.running = False

    async def _process_streaming(self, user_input: str) -> None:
        """Process user input with real-time streaming."""

        try:
            async for event in self.engine.stream_events(user_input):
                self._handle_stream_event(event)
        except Exception as e:
            self.console.print(f"\n❌ Error: {str(e)}", style="red")

    def _handle_stream_event(self, event: StreamEvent) -> None:
        """Handle a single streaming event."""

        if event.type == StreamEventType.START:
            pass

        elif event.type == StreamEventType.PROGRESS:
            self.console.print(event.message, style="dim")

        elif event.type == StreamEventType.OUTPUT:
            if event.message.startswith("✅") or event.message.startswith("⚠️"):
                self.console.print(
                    event.message, style="green" if "✅" in event.message else "yellow"
                )
            elif event.message.startswith("  •"):
                self.console.print(event.message, style="cyan")
            elif event.message.startswith("#"):
                md = Markdown(event.message)
                self.console.print(md)
            else:
                self.console.print(event.message)

        elif event.type == StreamEventType.ERROR:
            self.console.print(f"❌ {event.message}", style="red")

        elif event.type == StreamEventType.COMPLETE:
            self.console.print("✓ Done", style="dim")

        elif event.type == StreamEventType.SUGGESTION:
            self.console.print(f"💡 Try: {event.message}", style="yellow")


# Alias for backwards compatibility
AgentREPL = StreamingAgentREPL


def run_agent_repl() -> None:
    repl = StreamingAgentREPL()
    repl.run()


def run_agent_command(user_input: str) -> AgentResponse:
    """Run a single command and return response."""
    engine = StreamingAgentEngine()
    engine.create_session()

    # Run synchronously for CLI usage
    async def run():
        events = []
        async for event in engine.stream_events(user_input):
            events.append(event)
        return events

    asyncio.run(run())

    return AgentResponse(ok=True, message="Command executed")
