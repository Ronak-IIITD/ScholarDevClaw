from __future__ import annotations

import asyncio
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, AsyncGenerator


class AgentMode(str, Enum):
    INTERACTIVE = "interactive"
    STREAMING = "streaming"
    BACKGROUND = "background"


class StreamEventType(str, Enum):
    START = "start"
    PROGRESS = "progress"
    OUTPUT = "output"
    ERROR = "error"
    COMPLETE = "complete"
    SUGGESTION = "suggestion"


@dataclass
class StreamEvent:
    type: StreamEventType
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentMessage:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    ok: bool
    message: str
    output: dict[str, Any] | None = None
    error: str | None = None
    suggestions: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)


@dataclass
class AgentSession:
    id: str
    repo_path: str | None = None
    messages: list[AgentMessage] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_active: str = field(
        default_factory=lambda: datetime.now().isoformat())
    context: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, metadata: dict | None = None) -> None:
        self.messages.append(
            AgentMessage(
                role=role,
                content=content,
                metadata=metadata or {},
            )
        )
        self.last_active = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "repo_path": self.repo_path,
            "messages_count": len(self.messages),
            "created_at": self.created_at,
            "last_active": self.last_active,
            "context": self.context,
        }


class StreamingAgentEngine:
    def __init__(self):
        self.sessions: dict[str, AgentSession] = {}
        self.current_session: AgentSession | None = None

    def create_session(self, repo_path: str | None = None) -> AgentSession:
        session_id = str(uuid.uuid4())[:12]
        session = AgentSession(id=session_id, repo_path=repo_path)
        self.sessions[session_id] = session
        self.current_session = session
        return session

    def get_session(self, session_id: str) -> AgentSession | None:
        return self.sessions.get(session_id)

    def set_current_session(self, session_id: str) -> bool:
        session = self.sessions.get(session_id)
        if session:
            self.current_session = session
            return True
        return False

    def switch_repo(self, repo_path: str) -> bool:
        if not self.current_session:
            return False

        path = Path(repo_path).expanduser().resolve()
        if not path.exists():
            return False

        self.current_session.repo_path = str(path)
        return True

    async def stream_events(
        self,
        user_input: str,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Generator that yields streaming events for real-time UI updates."""

        if not self.current_session:
            self.create_session()

        session = self.current_session
        if session:
            session.add_message("user", user_input)

        # Emit start event
        yield StreamEvent(
            type=StreamEventType.START,
            message="Processing your request...",
            data={"user_input": user_input},
        )

        try:
            command = self._parse_command(user_input)

            if not command:
                # Handle as natural language
                async for event in self._handle_nl_streaming(user_input):
                    yield event
                return

            action = command["action"]
            target = command.get("target")

            # Emit command start
            yield StreamEvent(
                type=StreamEventType.PROGRESS,
                message=f"Executing: {action}",
                data={"action": action, "target": target},
            )

            # Execute based on action
            if action == "analyze":
                async for event in self._stream_analyze(target):
                    yield event
            elif action == "integrate":
                async for event in self._stream_integrate(target):
                    yield event
            elif action == "search":
                async for event in self._stream_search(target or ""):
                    yield event
            elif action == "suggest":
                async for event in self._stream_suggest():
                    yield event
            elif action == "validate":
                async for event in self._stream_validate():
                    yield event
            elif action == "security":
                async for event in self._stream_security(target):
                    yield event
            elif action == "help":
                for event in self._stream_help():
                    yield event
            else:
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    message=f"Unknown command: {action}",
                )

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message=str(e),
                data={"exception": type(e).__name__},
            )

    async def _stream_analyze(self, repo_path: str | None) -> AsyncGenerator[StreamEvent, None]:
        from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

        if not repo_path:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message="No repository path provided",
            )
            return

        path = Path(repo_path).expanduser().resolve()

        yield StreamEvent(
            type=StreamEventType.PROGRESS,
            message=f"📊 Analyzing {path.name}...",
            data={"path": str(path)},
        )

        # Simulate some progress
        yield StreamEvent(
            type=StreamEventType.PROGRESS,
            message="🔍 Scanning files...",
        )

        try:
            analyzer = TreeSitterAnalyzer(path)
            result = analyzer.analyze()

            if self.current_session:
                self.current_session.add_message(
                    "assistant", f"Analyzed {path}", {"action": "analyze"}
                )

            yield StreamEvent(
                type=StreamEventType.OUTPUT,
                message=f"✅ Found {len(result.languages)} languages",
                data={
                    "languages": result.languages,
                    "frameworks": result.frameworks,
                    "file_count": sum(s.file_count for s in result.language_stats),
                },
            )

            if result.frameworks:
                yield StreamEvent(
                    type=StreamEventType.OUTPUT,
                    message=f"🔧 Frameworks: {', '.join(result.frameworks)}",
                )

            # Suggestions
            yield StreamEvent(
                type=StreamEventType.SUGGESTION,
                message="suggest - Get improvement suggestions",
            )
            yield StreamEvent(
                type=StreamEventType.SUGGESTION,
                message="integrate rmsnorm - Apply RMSNorm optimization",
            )

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message=f"Analysis failed: {str(e)}",
            )

    async def _stream_integrate(self, spec: str | None) -> AsyncGenerator[StreamEvent, None]:
        from scholardevclaw.application.pipeline import run_integrate

        if not self.current_session or not self.current_session.repo_path:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message="No repository set. Run 'analyze <path>' first.",
            )
            return

        if not spec:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message="No spec provided. Use: integrate <rmsnorm|swiglu|flashattention>",
            )
            return

        repo_path = self.current_session.repo_path

        yield StreamEvent(
            type=StreamEventType.PROGRESS,
            message=f"🚀 Integrating {spec} into {Path(repo_path).name}...",
        )
        yield StreamEvent(type=StreamEventType.PROGRESS, message="📝 Generating patch...")

        try:
            result = await asyncio.to_thread(run_integrate, repo_path, spec)

            if self.current_session:
                self.current_session.add_message(
                    "assistant", f"Integrated {spec}", {
                        "action": "integrate", "spec": spec}
                )

            if result.ok:
                yield StreamEvent(
                    type=StreamEventType.OUTPUT,
                    message=f"✅ Successfully integrated {spec}!",
                )
                yield StreamEvent(
                    type=StreamEventType.SUGGESTION,
                    message="validate - Test the changes",
                )
            else:
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    message=f"Integration failed: {result.error}",
                )

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message=f"Integration error: {str(e)}",
            )

    async def _stream_search(self, query: str) -> AsyncGenerator[StreamEvent, None]:
        from scholardevclaw.research_intelligence.extractor import ResearchExtractor

        if not query:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message="No search query provided",
            )
            return

        yield StreamEvent(
            type=StreamEventType.PROGRESS,
            message=f"🔍 Searching for: {query}",
        )

        try:
            extractor = ResearchExtractor()
            results = extractor.search_by_keyword(query, max_results=5)

            yield StreamEvent(
                type=StreamEventType.OUTPUT,
                message=f"Found {len(results)} results",
                data={"results": results},
            )

            for r in results:
                yield StreamEvent(type=StreamEventType.OUTPUT, message=f"  • {r['title']}", data=r)

            if results:
                spec_name = results[0].get("name", "rmsnorm")
                yield StreamEvent(
                    type=StreamEventType.SUGGESTION,
                    message=f"integrate {spec_name}",
                )

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message=f"Search failed: {str(e)}",
            )

    async def _stream_suggest(self) -> AsyncGenerator[StreamEvent, None]:
        from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

        if not self.current_session or not self.current_session.repo_path:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message="No repository set. Run 'analyze <path>' first.",
            )
            return

        yield StreamEvent(
            type=StreamEventType.PROGRESS,
            message="💡 Analyzing for improvements...",
        )

        try:
            analyzer = TreeSitterAnalyzer(Path(self.current_session.repo_path))
            suggestions = analyzer.suggest_research_papers()

            yield StreamEvent(
                type=StreamEventType.OUTPUT,
                message=f"Found {len(suggestions)} improvement opportunities",
            )

            for s in suggestions[:3]:
                yield StreamEvent(
                    type=StreamEventType.OUTPUT,
                    message=f"  • {s['paper']['title']
                                   } ({s['confidence']:.0%})",
                    data=s,
                )
                yield StreamEvent(
                    type=StreamEventType.SUGGESTION,
                    message=f"integrate {s['paper']['name']}",
                )

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message=f"Suggestion failed: {str(e)}",
            )

    async def _stream_validate(self) -> AsyncGenerator[StreamEvent, None]:
        from scholardevclaw.application.pipeline import run_validate

        if not self.current_session or not self.current_session.repo_path:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message="No repository set",
            )
            return

        yield StreamEvent(
            type=StreamEventType.PROGRESS,
            message="🧪 Running validation tests...",
        )

        try:
            result = await asyncio.to_thread(run_validate, self.current_session.repo_path)

            if result.ok:
                yield StreamEvent(
                    type=StreamEventType.OUTPUT,
                    message="✅ Validation passed!",
                )
            else:
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    message=f"❌ Validation failed: {result.error}",
                )

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message=f"Validation error: {str(e)}",
            )

    async def _stream_security(self, repo_path: str | None) -> AsyncGenerator[StreamEvent, None]:
        from scholardevclaw.security import SecurityScanner

        target_path = repo_path or (
            self.current_session.repo_path if self.current_session else None
        )

        if not target_path:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message="No repository set",
            )
            return

        yield StreamEvent(
            type=StreamEventType.PROGRESS,
            message="🔒 Running security scan...",
        )

        try:
            scanner = SecurityScanner()
            result = scanner.scan(target_path)

            if result.passed:
                yield StreamEvent(
                    type=StreamEventType.OUTPUT,
                    message="✅ Security scan passed - no issues found!",
                )
            else:
                yield StreamEvent(
                    type=StreamEventType.OUTPUT,
                    message=f"⚠️ Found {result.total_findings} issues",
                )
                yield StreamEvent(
                    type=StreamEventType.OUTPUT,
                    message=f"   High: {result.high_severity_count}, Medium: {
                        result.medium_severity_count
                    }, Low: {result.low_severity_count}",
                )

        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message=f"Security scan error: {str(e)}",
            )

    def _stream_help(self) -> list[StreamEvent]:
        help_text = """
📖 **ScholarDevClaw Commands**

**analyze <path>** - Analyze a repository
**integrate <spec>** - Apply research improvement
  - rmsnorm, swiglu, flashattention, rope

**suggest** - Get AI improvement suggestions
**search <query>** - Search research papers

**validate** - Run validation tests
**security** - Run security scan

**help** - Show this message
"""
        return [
            StreamEvent(
                type=StreamEventType.OUTPUT,
                message=help_text.strip(),
            ),
            StreamEvent(
                type=StreamEventType.SUGGESTION,
                message="analyze ./my-project",
            ),
        ]

    async def _handle_nl_streaming(self, user_input: str) -> AsyncGenerator[StreamEvent, None]:
        user_lower = user_input.lower()

        if any(word in user_lower for word in ["hello", "hi", "hey", "start"]):
            yield StreamEvent(
                type=StreamEventType.OUTPUT,
                message="""👋 Hello! I'm ScholarDevClaw.

I can help you:
• Analyze repositories: analyze ./my-project
• Apply improvements: integrate rmsnorm
• Find research: search normalization
• Get suggestions: suggest

What would you like to do?""",
            )
            return

        yield StreamEvent(
            type=StreamEventType.ERROR,
            message="I didn't understand that. Try 'help' for available commands.",
        )
        yield StreamEvent(
            type=StreamEventType.SUGGESTION,
            message="help",
        )

    # Keep old methods for compatibility
    async def process(
        self,
        user_input: str,
        *,
        mode: AgentMode = AgentMode.INTERACTIVE,
        stream_callback: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        if not self.current_session:
            self.create_session()

        session = self.current_session
        if session:
            session.add_message("user", user_input)

        try:
            command = self._parse_command(user_input)

            if command:
                result = await self._execute_command(command, stream_callback)
                return result
            else:
                return await self._handle_natural_language(user_input, stream_callback)
        except Exception as e:
            return AgentResponse(
                ok=False,
                message="An error occurred",
                error=str(e),
            )

    def _parse_command(self, user_input: str) -> dict[str, Any] | None:
        user_input_lower = user_input.lower().strip()

        patterns = {
            "analyze": ["analyze", "analyse", "look at", "examine", "check"],
            "integrate": ["integrate", "apply", "implement", "add", "install"],
            "search": ["search", "find", "look for", "research"],
            "suggest": ["suggest", "recommend", "what can", "improvements"],
            "validate": ["validate", "test", "run tests", "verify"],
            "rollback": ["rollback", "revert", "undo"],
            "security": ["security", "scan", "vulnerability"],
            "context": ["context", "history", "what do you know"],
            "help": ["help", "commands", "what can you do"],
        }

        for action, keywords in patterns.items():
            for keyword in keywords:
                if keyword in user_input_lower:
                    target = self._extract_target(user_input, action)
                    return {"action": action, "target": target}

        return None

    def _extract_target(self, user_input: str, action: str) -> str | None:
        parts = user_input.split()
        spec_names = {"rmsnorm", "swiglu",
                      "flashattention", "rope", "gqa", "mixture"}

        if action == "search":
            for i, part in enumerate(parts):
                if part.lower() == "search" and i + 1 < len(parts):
                    query_parts = parts[i + 1:]
                    while query_parts and query_parts[0].lower() in {"for", "about", "on"}:
                        query_parts = query_parts[1:]
                    if query_parts:
                        return " ".join(query_parts)
            return None

        for i, part in enumerate(parts):
            if part.lower() in spec_names:
                return part.lower()
            if part.startswith(("/", "./", "../", "~")):
                path = Path(part).expanduser()
                if path.exists():
                    return str(path.resolve())
            if i > 0 and parts[i - 1].lower() in {"to", "for"}:
                if Path(part).expanduser().exists():
                    return str(Path(part).expanduser().resolve())

        if len(parts) > 1:
            last_arg = parts[-1]
            path = Path(last_arg).expanduser()
            if path.exists():
                return str(path.resolve())

        return None

    async def _execute_command(
        self,
        command: dict[str, Any],
        stream_callback: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        action = command["action"]
        target = command.get("target")

        if action == "help":
            return self._get_help()

        no_repo_commands = {"search", "help"}

        if action not in no_repo_commands:
            if not self.current_session or not self.current_session.repo_path:
                if action != "analyze" or not target:
                    return AgentResponse(
                        ok=False,
                        message="No repository set",
                        error="Please specify a repository first",
                        suggestions=["analyze ./my-project"],
                    )

        repo_path = self.current_session.repo_path if self.current_session else None

        try:
            if action == "analyze":
                return await self._cmd_analyze(target or repo_path, stream_callback)
            elif action == "integrate":
                return await self._cmd_integrate(repo_path, target, stream_callback)
            elif action == "search":
                return await self._cmd_search(target or "", stream_callback)
            elif action == "suggest":
                return await self._cmd_suggest(repo_path, stream_callback)
            elif action == "validate":
                return await self._cmd_validate(repo_path, stream_callback)
            elif action == "rollback":
                return await self._cmd_rollback(repo_path, target, stream_callback)
            elif action == "security":
                return await self._cmd_security(repo_path or target, stream_callback)
            elif action == "context":
                return self._cmd_context()
            else:
                return AgentResponse(ok=False, message=f"Unknown action: {action}")
        except Exception as e:
            return AgentResponse(ok=False, message=f"Error executing {action}", error=str(e))

    async def _cmd_analyze(self, repo_path: str | None, stream_callback) -> AgentResponse:
        from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

        if not repo_path:
            return AgentResponse(ok=False, message="No repository path provided")

        path = Path(repo_path).expanduser().resolve()
        if not path.exists():
            return AgentResponse(ok=False, message=f"Repository not found: {repo_path}")

        if stream_callback:
            stream_callback(f"📊 Analyzing repository: {path}\n")

        analyzer = TreeSitterAnalyzer(path)
        result = analyzer.analyze()

        if self.current_session:
            self.current_session.add_message("assistant", f"Analyzed {
                                             path}", {"action": "analyze"})

        return AgentResponse(
            ok=True,
            message=f"Analysis complete! Found {
                len(result.languages)} languages",
            output={"languages": result.languages,
                    "frameworks": result.frameworks},
            suggestions=["suggest - Get improvements", "integrate rmsnorm"],
        )

    async def _cmd_integrate(
        self, repo_path: str | None, spec: str | None, stream_callback
    ) -> AgentResponse:
        from scholardevclaw.application.pipeline import run_integrate

        if not repo_path:
            return AgentResponse(ok=False, message="No repository set")
        if not spec:
            return AgentResponse(
                ok=False, message="No spec provided", suggestions=["integrate rmsnorm"]
            )

        if stream_callback:
            stream_callback(f"🚀 Integrating {spec}...\n")

        result = await asyncio.to_thread(run_integrate, repo_path, spec)

        if self.current_session:
            self.current_session.add_message(
                "assistant", f"Integrated {spec}", {"action": "integrate"}
            )

        return AgentResponse(
            ok=result.ok,
            message=f"Integration {'successful' if result.ok else 'failed'}",
            error=result.error,
        )

    async def _cmd_search(self, query: str, stream_callback) -> AgentResponse:
        from scholardevclaw.research_intelligence.extractor import ResearchExtractor

        if not query:
            return AgentResponse(ok=False, message="No search query")

        extractor = ResearchExtractor()
        results = extractor.search_by_keyword(query, max_results=5)

        return AgentResponse(
            ok=True,
            message=f"Found {len(results)} results",
            output={"results": results},
        )

    async def _cmd_suggest(self, repo_path: str | None, stream_callback) -> AgentResponse:
        from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

        if not repo_path:
            return AgentResponse(ok=False, message="No repository set")

        analyzer = TreeSitterAnalyzer(Path(repo_path))
        suggestions = analyzer.suggest_research_papers()

        return AgentResponse(
            ok=True,
            message=f"Found {len(suggestions)} suggestions",
            output={"suggestions": suggestions},
        )

    async def _cmd_validate(self, repo_path: str | None, stream_callback) -> AgentResponse:
        from scholardevclaw.application.pipeline import run_validate

        if not repo_path:
            return AgentResponse(ok=False, message="No repository set")

        result = await asyncio.to_thread(run_validate, repo_path)
        return AgentResponse(ok=result.ok, message="Validation passed" if result.ok else "Failed")

    async def _cmd_rollback(
        self, repo_path: str | None, snapshot_id: str | None, stream_callback
    ) -> AgentResponse:
        from scholardevclaw.rollback import RollbackManager

        if not repo_path:
            return AgentResponse(ok=False, message="No repository set")

        manager = RollbackManager()
        result = manager.rollback(repo_path, snapshot_id)
        return AgentResponse(ok=result.ok, message="Rollback done" if result.ok else "Failed")

    async def _cmd_security(self, repo_path: str | None, stream_callback) -> AgentResponse:
        from scholardevclaw.security import SecurityScanner

        if not repo_path:
            return AgentResponse(ok=False, message="No repository set")

        scanner = SecurityScanner()
        result = scanner.scan(repo_path)
        return AgentResponse(ok=result.passed, message="Scan done", output=result.to_dict())

    def _cmd_context(self) -> AgentResponse:
        if not self.current_session:
            return AgentResponse(ok=False, message="No active session")
        return AgentResponse(ok=True, message="Session info", output=self.current_session.to_dict())

    async def _handle_natural_language(self, user_input: str, stream_callback) -> AgentResponse:
        user_lower = user_input.lower()

        if any(word in user_lower for word in ["hello", "hi", "hey"]):
            return AgentResponse(
                ok=True,
                message="👋 Hello! I'm ScholarDevClaw. Try 'analyze ./my-project' or 'help'",
            )

        return AgentResponse(ok=False, message="Unknown command", suggestions=["help"])

    def _get_help(self) -> AgentResponse:
        return AgentResponse(
            ok=True,
            message="""
📖 Commands: analyze <path>, integrate <spec>, search <query>, suggest, validate, security, help
            """.strip(),
        )


# Alias for backwards compatibility
AgentEngine = StreamingAgentEngine


def create_agent_engine() -> StreamingAgentEngine:
    return StreamingAgentEngine()
