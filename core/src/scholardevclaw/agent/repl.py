from __future__ import annotations

import asyncio
import os
import sys
from typing import Callable

import rich
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table

from .engine import AgentEngine, AgentMode, AgentResponse


class AgentREPL:
    def __init__(self, engine: AgentEngine | None = None):
        self.engine = engine or AgentEngine()
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
        self.console.print(
            Panel(
                welcome.strip(),
                title="ScholarDevClaw",
                border_style="blue",
            )
        )

    def print_response(self, response: AgentResponse) -> None:
        if response.message:
            if response.message.startswith("#"):
                md = Markdown(response.message)
                self.console.print(md)
            else:
                self.console.print(response.message)

        if response.output and self._should_show_output(response):
            if "languages" in response.output:
                self._print_analysis_output(response.output)
            elif "suggestions" in response.output:
                self._print_suggestions(response.output)
            elif "results" in response.output:
                self._print_search_results(response.output)
            else:
                self.console.print(
                    Panel(
                        str(response.output),
                        style="dim",
                    )
                )

        if response.suggestions:
            self.console.print("\n💡 Suggestions:")
            for s in response.suggestions:
                self.console.print(f"  • {s}")

        if response.next_steps:
            self.console.print("\n🚀 Next steps:")
            for step in response.next_steps:
                self.console.print(f"  • {step}")

        if response.error:
            self.console.print(f"\n❌ Error: {response.error}", style="red")

    def _should_show_output(self, response: AgentResponse) -> bool:
        if not response.output:
            return False
        if response.output and "languages" in response.output:
            return True
        return len(response.output) > 0

    def _print_analysis_output(self, output: dict) -> None:
        table = Table(title="Repository Analysis", show_header=False)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="white")

        if "languages" in output:
            table.add_row("Languages", ", ".join(output["languages"]))
        if "frameworks" in output:
            table.add_row("Frameworks", ", ".join(output.get("frameworks", [])))
        if "file_count" in output:
            table.add_row("Files", str(output["file_count"]))
        if "patterns" in output and output["patterns"]:
            patterns = ", ".join(f"{k} ({len(v)} locations)" for k, v in output["patterns"].items())
            table.add_row("Patterns", patterns)

        self.console.print(table)

    def _print_suggestions(self, output: dict) -> None:
        suggestions = output.get("suggestions", [])
        if not suggestions:
            return

        table = Table(title="Improvement Suggestions")
        table.add_column("Paper", style="cyan")
        table.add_column("Confidence", style="green")
        table.add_column("Category", style="yellow")

        for s in suggestions[:5]:
            paper = s.get("paper", {})
            table.add_row(
                paper.get("title", "Unknown")[:50],
                f"{s.get('confidence', 0):.0%}",
                paper.get("category", "N/A"),
            )

        self.console.print(table)

    def _print_search_results(self, output: dict) -> None:
        results = output.get("results", [])
        if not results:
            return

        table = Table(title="Research Papers")
        table.add_column("Title", style="cyan")
        table.add_column("Category", style="yellow")

        for r in results[:5]:
            table.add_row(
                r.get("title", "Unknown")[:60],
                r.get("category", "N/A"),
            )

        self.console.print(table)

    def run(self) -> None:
        self.running = True
        self.engine.create_session()
        self.print_welcome()

        while self.running:
            try:
                user_input = Prompt.ask(
                    "\n[bold blue]ScholarDevClaw[/bold blue]",
                    default="",
                )

                if not user_input.strip():
                    continue

                if user_input.lower() in ["exit", "quit", "q"]:
                    self.console.print("👋 Goodbye!")
                    break

                if user_input.lower() == "clear":
                    self.console.clear()
                    continue

                response = asyncio.run(self.engine.process(user_input))
                self.print_response(response)

            except KeyboardInterrupt:
                self.console.print("\n👋 Goodbye!")
                break
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"\n❌ Error: {str(e)}", style="red")

        self.running = False

    def run_with_input(self, user_input: str) -> AgentResponse:
        return asyncio.run(self.engine.process(user_input))


def run_agent_repl() -> None:
    repl = AgentREPL()
    repl.run()


def run_agent_command(user_input: str) -> AgentResponse:
    engine = AgentEngine()
    engine.create_session()
    repl = AgentREPL(engine)
    return repl.run_with_input(user_input)
