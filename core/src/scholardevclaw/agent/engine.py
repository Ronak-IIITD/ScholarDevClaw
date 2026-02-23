from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class AgentMode(str, Enum):
    INTERACTIVE = "interactive"
    STREAMING = "streaming"
    BACKGROUND = "background"


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
    last_active: str = field(default_factory=lambda: datetime.now().isoformat())
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


class AgentEngine:
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

    async def process(
        self,
        user_input: str,
        *,
        mode: AgentMode = AgentMode.INTERACTIVE,
        stream_callback: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        if not self.current_session:
            self.create_session()

        # Type narrowing - current_session is guaranteed to exist after create_session
        session = self.current_session
        if session:
            session.add_message("user", user_input)

        try:
            if mode == AgentMode.STREAMING and stream_callback:
                return await self._process_streaming(user_input, stream_callback)
            else:
                return await self._process_normal(user_input)
        except Exception as e:
            return AgentResponse(
                ok=False,
                message="An error occurred",
                error=str(e),
            )

    async def _process_streaming(
        self,
        user_input: str,
        stream_callback: Callable[[str], None],
    ) -> AgentResponse:
        command = self._parse_command(user_input)

        if command:
            if stream_callback:
                stream_callback(f"🔧 Running: {command['action']} {command.get('target', '')}\n\n")

            result = await self._execute_command(command, stream_callback)
            return result
        else:
            return await self._process_normal(user_input, stream_callback)

    async def _process_normal(
        self,
        user_input: str,
        stream_callback: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        command = self._parse_command(user_input)

        if command:
            return await self._execute_command(command, stream_callback)

        return await self._handle_natural_language(user_input, stream_callback)

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

        skip_words = {"the", "a", "an", "to", "in", "for", "with", "on", "at"}
        spec_names = {"rmsnorm", "swiglu", "flashattention", "rope", "gqa", "mixture"}

        # For search, everything after "search" is the query
        if action == "search":
            # Find "search" in parts and return everything after it
            for i, part in enumerate(parts):
                if part.lower() == "search" and i + 1 < len(parts):
                    query_parts = parts[i + 1 :]
                    # Remove common leading words
                    while query_parts and query_parts[0].lower() in {"for", "about", "on"}:
                        query_parts = query_parts[1:]
                    if query_parts:
                        return " ".join(query_parts)
            return None

        for i, part in enumerate(parts):
            if part.lower() in spec_names:
                return part.lower()

            # Check if it's a path (starts with / or ./ or ~ or .)
            if part.startswith(("/", "./", "../", "~")):
                path = Path(part).expanduser()
                if path.exists():
                    return str(path.resolve())

            # Check for "to" or "for" patterns
            if i > 0 and parts[i - 1].lower() in {"to", "for"}:
                if Path(part).expanduser().exists():
                    return str(Path(part).expanduser().resolve())

        # If no path found, check the last argument as a potential path
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

        # Commands that don't require a repo
        no_repo_commands = {"search", "help"}

        if action not in no_repo_commands:
            if not self.current_session or not self.current_session.repo_path:
                if action != "analyze" or not target:
                    return AgentResponse(
                        ok=False,
                        message="No repository set",
                        error="Please specify a repository first (e.g., 'analyze ./my-project')",
                        suggestions=[
                            "analyze ./my-project",
                            "integrate ./repo rmsnorm",
                            "set repo ./my-project",
                            "search normalization",
                        ],
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
                return AgentResponse(
                    ok=False,
                    message=f"Unknown action: {action}",
                )
        except Exception as e:
            return AgentResponse(
                ok=False,
                message=f"Error executing {action}",
                error=str(e),
            )

    async def _cmd_analyze(
        self,
        repo_path: str | None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

        if not repo_path:
            return AgentResponse(
                ok=False,
                message="No repository path provided",
                error="Please specify a repository to analyze",
            )

        path = Path(repo_path).expanduser().resolve()
        if not path.exists():
            return AgentResponse(
                ok=False,
                message=f"Repository not found: {repo_path}",
            )

        if stream_callback:
            stream_callback(f"📊 Analyzing repository: {path}\n")

        analyzer = TreeSitterAnalyzer(path)
        result = analyzer.analyze()

        if self.current_session:
            self.current_session.add_message("assistant", f"Analyzed {path}", {"action": "analyze"})

        response = AgentResponse(
            ok=True,
            message=f"Analysis complete! Found {len(result.languages)} languages",
            output={
                "languages": result.languages,
                "frameworks": result.frameworks,
                "file_count": sum(s.file_count for s in result.language_stats),
                "patterns": result.patterns,
            },
            suggestions=[
                "suggest improvements for this repo",
                "integrate rmsnorm to improve performance",
                "run security scan",
            ],
            next_steps=[
                "suggest - Get improvement suggestions",
                "integrate <spec> - Apply research improvements",
                "security - Run security scan",
            ],
        )

        if stream_callback:
            stream_callback(
                f"✅ Found {len(result.languages)} languages: {', '.join(result.languages)}\n"
            )
            if result.frameworks:
                stream_callback(f"🔧 Frameworks: {', '.join(result.frameworks)}\n")

        return response

    async def _cmd_integrate(
        self,
        repo_path: str | None,
        spec: str | None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        from scholardevclaw.application.pipeline import run_integrate

        if not repo_path:
            return AgentResponse(
                ok=False,
                message="No repository set",
                error="Please set a repository first using 'analyze <path>'",
            )

        if not spec:
            return AgentResponse(
                ok=False,
                message="No spec provided",
                error="Please specify a spec (e.g., 'integrate rmsnorm')",
                suggestions=["integrate rmsnorm", "integrate swiglu", "integrate flashattention"],
            )

        if stream_callback:
            stream_callback(f"🚀 Integrating {spec} into {repo_path}\n\n")

        result = await asyncio.to_thread(
            run_integrate,
            repo_path,
            spec,
            log_callback=stream_callback,
        )

        if self.current_session:
            self.current_session.add_message(
                "assistant", f"Integrated {spec}", {"action": "integrate", "spec": spec}
            )

        if result.ok:
            return AgentResponse(
                ok=True,
                message=f"Successfully integrated {spec}!",
                output=result.payload,
                suggestions=[
                    "validate - Run validation tests",
                    "rollback - Revert if needed",
                ],
                next_steps=[
                    "validate - Verify changes work",
                    "rollback - Undo changes",
                ],
            )
        else:
            return AgentResponse(
                ok=False,
                message=f"Integration failed: {spec}",
                error=result.error,
            )

    async def _cmd_search(
        self,
        query: str,
        stream_callback: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        from scholardevclaw.research_intelligence.extractor import ResearchExtractor

        if not query:
            return AgentResponse(
                ok=False,
                message="No search query provided",
                error="Please specify what to search for",
            )

        if stream_callback:
            stream_callback(f"🔍 Searching for: {query}\n")

        extractor = ResearchExtractor()
        results = extractor.search_by_keyword(query, max_results=5)

        if stream_callback:
            if results:
                stream_callback(f"Found {len(results)} results:\n")
                for r in results:
                    stream_callback(f"  • {r['title']}\n")
            else:
                stream_callback("No results found.\n")

        return AgentResponse(
            ok=True,
            message=f"Found {len(results)} results for '{query}'",
            output={"results": results},
            suggestions=[
                f"integrate {results[0].get('name', results[0].get('algorithm', {}).get('name', 'rmsnorm'))}"
                if results
                else "integrate <spec>",
            ],
        )

    async def _cmd_suggest(
        self,
        repo_path: str | None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer

        if not repo_path:
            return AgentResponse(
                ok=False,
                message="No repository set",
                error="Please analyze a repository first",
            )

        if stream_callback:
            stream_callback("💡 Generating suggestions...\n")

        analyzer = TreeSitterAnalyzer(Path(repo_path))
        suggestions = analyzer.suggest_research_papers()

        if stream_callback:
            if suggestions:
                stream_callback(f"Found {len(suggestions)} improvement opportunities:\n")
                for s in suggestions[:3]:
                    stream_callback(f"  • {s['paper']['title']} ({s['confidence']:.0%})\n")
            else:
                stream_callback("No specific improvements found.\n")

        return AgentResponse(
            ok=True,
            message=f"Found {len(suggestions)} improvement opportunities",
            output={"suggestions": suggestions},
            next_steps=[f"integrate {s['paper']['name']}" for s in suggestions[:2]],
        )

    async def _cmd_validate(
        self,
        repo_path: str | None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        from scholardevclaw.application.pipeline import run_validate

        if not repo_path:
            return AgentResponse(
                ok=False,
                message="No repository set",
            )

        if stream_callback:
            stream_callback("🧪 Running validation...\n")

        result = await asyncio.to_thread(run_validate, repo_path, log_callback=stream_callback)

        if stream_callback:
            stream_callback(f"Validation: {'✅ Passed' if result.ok else '❌ Failed'}\n")

        return AgentResponse(
            ok=result.ok,
            message="Validation passed" if result.ok else "Validation failed",
            output=result.payload,
            error=result.error,
        )

    async def _cmd_rollback(
        self,
        repo_path: str | None,
        snapshot_id: str | None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        from scholardevclaw.rollback import RollbackManager

        if not repo_path:
            return AgentResponse(
                ok=False,
                message="No repository set",
            )

        if stream_callback:
            stream_callback("🔄 Rolling back...\n")

        manager = RollbackManager()
        result = manager.rollback(repo_path, snapshot_id)

        if stream_callback:
            stream_callback(f"Rollback: {'✅ Success' if result.ok else '❌ Failed'}\n")

        return AgentResponse(
            ok=result.ok,
            message="Rollback successful" if result.ok else "Rollback failed",
            error=result.error,
        )

    async def _cmd_security(
        self,
        repo_path: str | None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        from scholardevclaw.security import SecurityScanner

        if not repo_path:
            return AgentResponse(
                ok=False,
                message="No repository set",
            )

        if stream_callback:
            stream_callback("🔒 Running security scan...\n")

        scanner = SecurityScanner()
        result = scanner.scan(repo_path)

        if stream_callback:
            stream_callback(
                f"Security scan: {'✅ Passed' if result.passed else '⚠️ Found issues'}\n"
            )
            stream_callback(
                f"  High: {result.high_severity_count}, Medium: {result.medium_severity_count}\n"
            )

        return AgentResponse(
            ok=result.passed,
            message="Security scan passed" if result.passed else "Security issues found",
            output=result.to_dict(),
        )

    def _cmd_context(self) -> AgentResponse:
        if not self.current_session:
            return AgentResponse(
                ok=False,
                message="No active session",
            )

        session_info = self.current_session.to_dict()

        return AgentResponse(
            ok=True,
            message="Current session info",
            output=session_info,
            next_steps=[
                "analyze <path> - Analyze a repository",
                "help - Show all commands",
            ],
        )

    async def _handle_natural_language(
        self,
        user_input: str,
        stream_callback: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        user_lower = user_input.lower()

        if any(word in user_lower for word in ["hello", "hi", "hey", "start"]):
            return AgentResponse(
                ok=True,
                message="👋 Hello! I'm ScholarDevClaw, your research-to-code assistant.\n\nI can help you:\n- Analyze repositories\n- Apply research improvements (RMSNorm, SwiGLU, etc.)\n- Search for research papers\n- Run security scans\n- Validate changes\n\nJust tell me what you want to do!",
                suggestions=["analyze ./my-project", "help"],
            )

        if "set repo" in user_lower or "cd " in user_lower:
            parts = user_input.split()
            if len(parts) >= 2:
                target = parts[-1]
                if self.switch_repo(target):
                    return AgentResponse(
                        ok=True,
                        message=f"✅ Switched to repository: {target}",
                        suggestions=["analyze", "suggest", "integrate rmsnorm"],
                    )

        return AgentResponse(
            ok=False,
            message="I didn't understand that",
            error="Try a command like 'analyze ./project' or 'help'",
            suggestions=[
                "analyze ./my-project - Analyze a repository",
                "integrate rmsnorm - Apply research improvement",
                "suggest - Get improvement suggestions",
                "help - See all commands",
            ],
        )

    def _get_help(self) -> AgentResponse:
        help_text = """
📖 **ScholarDevClaw Agent Commands**

**Repository Operations:**
- `analyze <path>` - Analyze a repository structure
- `set repo <path>` - Switch to a different repository

**Research Integration:**
- `integrate <spec>` - Apply a research improvement
  - Examples: rmsnorm, swiglu, flashattention, rope

**Analysis:**
- `suggest` - Get AI-powered improvement suggestions
- `search <query>` - Search research papers

**Validation & Safety:**
- `validate` - Run validation tests
- `security` - Run security scan
- `rollback [id]` - Revert changes

**General:**
- `context` - Show current session info
- `help` - Show this help message

**Examples:**
- "analyze ./my-project"
- "integrate rmsnorm"
- "suggest improvements for this repo"
- "run security scan"
"""
        return AgentResponse(
            ok=True,
            message=help_text.strip(),
            suggestions=[
                "analyze ./my-project",
                "integrate rmsnorm",
            ],
        )


def create_agent_engine() -> AgentEngine:
    return AgentEngine()
