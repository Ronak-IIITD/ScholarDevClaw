"""
Smart Agent Engine — the brain that maximizes output quality per token spent.

This module wires together:
- Memory (persistent context across sessions)
- Planning (task decomposition with token budgets)
- Reflection (output quality scoring + iterative improvement)
- Sub-agents (real pipeline calls, not mock data)
- Token budget management (pocket-friendly execution)

Design principles:
1. Classify query complexity first → route to cheapest sufficient path
2. Use memory to avoid redundant work (don't re-analyze same repo)
3. Plan before executing complex tasks
4. Reflect on output quality; retry if below threshold
5. Stream progress to user in real-time
"""

from __future__ import annotations

import asyncio
import os
import platform
import shlex
import subprocess
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .engine import (
    AgentResponse,
    AgentSession,
    StreamEvent,
    StreamEventType,
)
from .memory import (
    AdvancedAgentMemory,
)
from .planning import (
    AdaptivePlanner,
    Plan,
    TaskPriority,
    TaskStatus,
)
from .reflection import (
    AgentReflector,
    QualityRating,
)
from .terminal import (
    create_shell,
)
from .tools import (
    AdvancedToolManager,
)
from .tools import (
    ToolStatus as ToolExecStatus,
)


def redact_sensitive_output(text: str) -> str:
    """Best-effort redaction for common key=value secret patterns."""
    import re

    patterns = [
        r"(?i)\b([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASS|AUTH|COOKIE|SESSION)[A-Z0-9_]*)\s*=\s*([^\s]+)",
    ]
    redacted = text
    for pattern in patterns:
        redacted = re.sub(pattern, lambda m: f"{m.group(1)}=***REDACTED***", redacted)
    return redacted


class OSDetector:
    """Detects and provides OS-specific configuration."""

    def __init__(self):
        self.system = platform.system().lower()
        self.is_windows = self.system == "windows"
        self.is_mac = self.system == "darwin"
        self.is_linux = self.system == "linux"

        # Detect shell (bash, zsh, fish, etc.)
        self._detect_shell()

    def _detect_shell(self) -> None:
        """Detect the current shell."""
        import shutil

        # Check SHELL environment variable
        shell_path = os.environ.get("SHELL", "")
        if shell_path:
            self.shell_name = os.path.basename(shell_path)
        else:
            self.shell_name = "bash"

        # Check if zsh is available
        self.has_zsh = shutil.which("zsh") is not None
        self.has_bash = shutil.which("bash") is not None
        self.has_fish = shutil.which("fish") is not None
        self.has_powershell = (
            shutil.which("pwsh") is not None or shutil.which("powershell") is not None
        )

        # Prefer detected shell if available
        if self.has_zsh:
            self.user_shell = "zsh"
        elif self.has_fish:
            self.user_shell = "fish"
        elif self.has_bash:
            self.user_shell = "bash"
        else:
            self.user_shell = self.shell_name

    @property
    def shell(self) -> str:
        """Get the default shell for this OS."""
        if self.is_windows:
            return "powershell" if self.has_powershell else "cmd"
        return self.user_shell  # zsh, bash, fish, etc.

    @property
    def shell_exe(self) -> str:
        """Get the shell executable."""
        if self.is_windows:
            if self.has_powershell:
                return "powershell"
            return "cmd"
        return self.user_shell

    def _has_powershell(self) -> bool:
        """Check if PowerShell is available."""
        import shutil

        return shutil.which("pwsh") is not None or shutil.which("powershell") is not None

    @property
    def is_zsh(self) -> bool:
        """Check if current shell is ZSH."""
        return self.user_shell == "zsh"

    @property
    def is_bash(self) -> bool:
        """Check if current shell is BASH."""
        return self.user_shell == "bash"

    @property
    def is_fish(self) -> bool:
        """Check if current shell is Fish."""
        return self.user_shell == "fish"

    @property
    def path_separator(self) -> str:
        """Get the path separator for this OS."""
        return "\\" if self.is_windows else "/"

    @property
    def line_ending(self) -> str:
        """Get the line ending for this OS."""
        return "\r\n" if self.is_windows else "\n"

    def get_command_prefix(self) -> str:
        """Get prefix for command execution."""
        return ""  # shell=True handles this

    def which(self, cmd: str) -> str | None:
        """Find command in PATH."""
        import shutil

        return shutil.which(cmd)

    def has_command(self, cmd: str) -> bool:
        """Check if a command is available."""
        return self.which(cmd) is not None

    @property
    def os_name(self) -> str:
        """Human-readable OS name."""
        if self.is_windows:
            return "Windows"
        if self.is_mac:
            return "macOS"
        return "Linux"

    def get_env(self) -> dict:
        """Get OS-specific environment info."""
        return {
            "system": self.system,
            "os_name": self.os_name,
            "shell": self.shell,
            "user_shell": self.user_shell,
            "shell_name": self.shell_name,
            "is_zsh": self.is_zsh,
            "is_bash": self.is_bash,
            "is_fish": self.is_fish,
            "has_zsh": self.has_zsh,
            "has_bash": self.has_bash,
            "has_fish": self.has_fish,
            "path_separator": self.path_separator,
            "has_powershell": self.has_powershell,
        }


# Global OS detector
DETECTED_OS = OSDetector()


# ---------------------------------------------------------------------------
# Query complexity classification
# ---------------------------------------------------------------------------


class QueryComplexity(str, Enum):
    """How complex is the user's query?"""

    TRIVIAL = "trivial"  # help, status, hello → instant, no tools
    SIMPLE = "simple"  # single pipeline call (analyze, search, suggest)
    MODERATE = "moderate"  # 2-3 chained calls (map + generate, suggest + integrate)
    COMPLEX = "complex"  # full pipeline (integrate = analyze+research+map+gen+validate)


@dataclass
class QueryClassification:
    """Result of classifying a user query."""

    complexity: QueryComplexity
    primary_action: str
    secondary_actions: list[str] = field(default_factory=list)
    target: str | None = None
    query_text: str = ""
    estimated_token_cost: int = 0  # rough estimate
    needs_repo: bool = False
    confidence: float = 0.0


@dataclass
class TokenBudget:
    """Token budget manager for pocket-friendly execution."""

    max_tokens: int = 50_000  # total budget per query
    used_tokens: int = 0
    warn_threshold: float = 0.8  # warn at 80% usage
    hard_limit: float = 0.95  # stop at 95%

    # Per-phase budgets (percentage of max)
    phase_budgets: dict[str, float] = field(
        default_factory=lambda: {
            "classification": 0.02,  # 2% for understanding the query
            "memory_retrieval": 0.03,  # 3% for memory lookups
            "planning": 0.05,  # 5% for task planning
            "execution": 0.75,  # 75% for actual work
            "reflection": 0.05,  # 5% for quality assessment
            "retry": 0.10,  # 10% reserve for retries
        }
    )

    def can_spend(self, tokens: int) -> bool:
        return (self.used_tokens + tokens) <= (self.max_tokens * self.hard_limit)

    def spend(self, tokens: int) -> None:
        self.used_tokens += tokens

    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    def usage_pct(self) -> float:
        return self.used_tokens / self.max_tokens if self.max_tokens > 0 else 1.0

    def phase_budget(self, phase: str) -> int:
        pct = self.phase_budgets.get(phase, 0.10)
        return int(self.max_tokens * pct)

    def is_warning(self) -> bool:
        return self.usage_pct() >= self.warn_threshold


@dataclass
class ExecutionResult:
    """Result from executing a single step."""

    ok: bool
    action: str
    output: dict[str, Any] | None = None
    message: str = ""
    error: str | None = None
    tokens_used: int = 0
    duration_ms: int = 0


# ---------------------------------------------------------------------------
# Query Classifier — determines optimal execution path
# ---------------------------------------------------------------------------


class QueryClassifier:
    """Classify user queries to determine the cheapest sufficient execution path."""

    # Action patterns with complexity and token cost estimates
    ACTION_PATTERNS: dict[str, dict[str, Any]] = {
        "help": {
            "keywords": ["help", "commands", "what can you do", "?"],
            "complexity": QueryComplexity.TRIVIAL,
            "token_cost": 100,
            "needs_repo": False,
        },
        "status": {
            "keywords": ["status", "context", "session", "what do you know"],
            "complexity": QueryComplexity.TRIVIAL,
            "token_cost": 100,
            "needs_repo": False,
        },
        "greet": {
            "keywords": ["hello", "hi", "hey", "start", "good morning", "good evening"],
            "complexity": QueryComplexity.TRIVIAL,
            "token_cost": 100,
            "needs_repo": False,
        },
        "analyze": {
            "keywords": ["analyze", "analyse", "examine", "look at", "scan"],
            "complexity": QueryComplexity.SIMPLE,
            "token_cost": 2000,
            "needs_repo": True,
        },
        "search": {
            "keywords": ["search", "find papers", "look for", "research"],
            "complexity": QueryComplexity.SIMPLE,
            "token_cost": 3000,
            "needs_repo": False,
        },
        "suggest": {
            "keywords": ["suggest", "recommend", "what can", "improvements", "improve"],
            "complexity": QueryComplexity.SIMPLE,
            "token_cost": 3000,
            "needs_repo": True,
        },
        "specs": {
            "keywords": ["specs", "specifications", "list specs", "available"],
            "complexity": QueryComplexity.TRIVIAL,
            "token_cost": 500,
            "needs_repo": False,
        },
        "validate": {
            "keywords": ["validate", "test", "run tests", "verify", "check"],
            "complexity": QueryComplexity.SIMPLE,
            "token_cost": 5000,
            "needs_repo": True,
        },
        "security": {
            "keywords": ["security", "scan vulnerabilities", "audit security"],
            "complexity": QueryComplexity.SIMPLE,
            "token_cost": 3000,
            "needs_repo": True,
        },
        "map": {
            "keywords": ["map spec", "map to code", "where to apply"],
            "complexity": QueryComplexity.MODERATE,
            "token_cost": 8000,
            "needs_repo": True,
        },
        "generate": {
            "keywords": ["generate patch", "generate code", "create patch"],
            "complexity": QueryComplexity.MODERATE,
            "token_cost": 10000,
            "needs_repo": True,
        },
        "integrate": {
            "keywords": ["integrate", "apply", "implement", "add", "install"],
            "complexity": QueryComplexity.COMPLEX,
            "token_cost": 25000,
            "needs_repo": True,
        },
        "rollback": {
            "keywords": ["rollback", "revert", "undo"],
            "complexity": QueryComplexity.SIMPLE,
            "token_cost": 2000,
            "needs_repo": True,
        },
        # --- Tool-based actions ---
        "run_command": {
            "keywords": ["run ", "execute ", "shell ", "bash ", "!"],
            "complexity": QueryComplexity.SIMPLE,
            "token_cost": 500,
            "needs_repo": False,
        },
        "read_file": {
            "keywords": ["read file", "show file", "cat ", "open file", "view file"],
            "complexity": QueryComplexity.TRIVIAL,
            "token_cost": 200,
            "needs_repo": False,
        },
        "write_file": {
            "keywords": ["write file", "save file", "create file", "write to"],
            "complexity": QueryComplexity.SIMPLE,
            "token_cost": 300,
            "needs_repo": False,
        },
        "search_code": {
            "keywords": ["grep ", "find in code", "search code", "search files"],
            "complexity": QueryComplexity.SIMPLE,
            "token_cost": 400,
            "needs_repo": False,
        },
        "list_files": {
            "keywords": ["list files", "ls ", "dir ", "list directory", "show files"],
            "complexity": QueryComplexity.TRIVIAL,
            "token_cost": 100,
            "needs_repo": False,
        },
        "git": {
            "keywords": ["git status", "git log", "git diff", "git branch"],
            "complexity": QueryComplexity.TRIVIAL,
            "token_cost": 200,
            "needs_repo": True,
        },
        "analyze_code": {
            "keywords": ["lint", "ruff", "check code quality", "code quality"],
            "complexity": QueryComplexity.SIMPLE,
            "token_cost": 500,
            "needs_repo": True,
        },
        # --- Terminal/shell features ---
        "terminal": {
            "keywords": [
                "terminal",
                "shell",
                "interactive",
                "console",
                "!ls",
                "!cd",
                "!pwd",
                "!echo",
            ],
            "complexity": QueryComplexity.TRIVIAL,
            "token_cost": 100,
            "needs_repo": False,
        },
        # --- Advanced shell features ---
        "run_code": {
            "keywords": ["run code", "execute code", "run file", "python ", "node ", "python3 "],
            "complexity": QueryComplexity.SIMPLE,
            "token_cost": 800,
            "needs_repo": False,
        },
        "run_tests": {
            "keywords": ["test", "pytest", "run tests", "run test", "tests", "check test"],
            "complexity": QueryComplexity.SIMPLE,
            "token_cost": 1000,
            "needs_repo": True,
        },
        "fix_and_test": {
            "keywords": ["fix tests", "fix failing", "auto-fix", "fix and test"],
            "complexity": QueryComplexity.MODERATE,
            "token_cost": 8000,
            "needs_repo": True,
        },
        "intelligent_run": {
            "keywords": ["do it", "fix it", "run it", "execute", "build", "make"],
            "complexity": QueryComplexity.MODERATE,
            "token_cost": 1500,
            "needs_repo": True,
        },
    }

    # Compound query patterns that require multi-step execution
    COMPOUND_PATTERNS: list[dict[str, Any]] = [
        {
            "triggers": ["analyze and suggest", "analyze then suggest"],
            "actions": ["analyze", "suggest"],
            "complexity": QueryComplexity.MODERATE,
        },
        {
            "triggers": ["suggest and integrate", "suggest then integrate", "find and apply"],
            "actions": ["suggest", "integrate"],
            "complexity": QueryComplexity.COMPLEX,
        },
        {
            "triggers": ["full pipeline", "end to end", "everything"],
            "actions": ["analyze", "suggest", "map", "generate", "validate"],
            "complexity": QueryComplexity.COMPLEX,
        },
    ]

    SPEC_NAMES = {
        "rmsnorm",
        "swiglu",
        "flashattention",
        "rope",
        "gqa",
        "mixture",
        "layernorm",
        "gelu",
        "silu",
        "mqa",
    }

    def classify(self, user_input: str) -> QueryClassification:
        """Classify user input into an execution plan."""
        user_lower = user_input.lower().strip()

        # Check compound patterns first (higher specificity)
        for pattern in self.COMPOUND_PATTERNS:
            for trigger in pattern["triggers"]:
                if trigger in user_lower:
                    target = self._extract_target(user_input)
                    return QueryClassification(
                        complexity=pattern["complexity"],
                        primary_action=pattern["actions"][0],
                        secondary_actions=pattern["actions"][1:],
                        target=target,
                        query_text=user_input,
                        estimated_token_cost=sum(
                            self.ACTION_PATTERNS.get(a, {}).get("token_cost", 5000)
                            for a in pattern["actions"]
                        ),
                        needs_repo=True,
                        confidence=0.9,
                    )

        # Check single-action patterns
        for action, config in self.ACTION_PATTERNS.items():
            for keyword in config["keywords"]:
                if keyword in user_lower:
                    target = self._extract_target(user_input)
                    return QueryClassification(
                        complexity=config["complexity"],
                        primary_action=action,
                        target=target,
                        query_text=user_input,
                        estimated_token_cost=config["token_cost"],
                        needs_repo=config["needs_repo"],
                        confidence=0.85,
                    )

        # Fallback: try to infer intent from spec names
        for spec in self.SPEC_NAMES:
            if spec in user_lower:
                return QueryClassification(
                    complexity=QueryComplexity.COMPLEX,
                    primary_action="integrate",
                    target=spec,
                    query_text=user_input,
                    estimated_token_cost=25000,
                    needs_repo=True,
                    confidence=0.7,
                )

        # Unknown query — treat as generic
        return QueryClassification(
            complexity=QueryComplexity.SIMPLE,
            primary_action="unknown",
            query_text=user_input,
            estimated_token_cost=1000,
            confidence=0.3,
        )

    def _extract_target(self, user_input: str) -> str | None:
        """Extract target (repo path, spec name, command, or file path) from user input."""
        parts = user_input.split()
        user_lower = user_input.lower().strip()

        # --- Tool-specific target extraction ---

        # run/execute/shell: everything after the keyword is the command
        for prefix in ["run ", "execute ", "shell ", "bash ", "!"]:
            if user_lower.startswith(prefix):
                remainder = user_input[len(prefix) :].strip()
                if remainder:
                    return remainder

        # read/write/cat/view file: extract file path
        for prefix in [
            "read file ",
            "show file ",
            "cat ",
            "open file ",
            "view file ",
            "write file ",
            "save file ",
            "create file ",
        ]:
            if user_lower.startswith(prefix):
                remainder = user_input[len(prefix) :].strip()
                if remainder:
                    return remainder.split()[0]  # first token as path

        # grep/search code: extract query
        for prefix in ["grep ", "search code ", "search files ", "find in code "]:
            if user_lower.startswith(prefix):
                remainder = user_input[len(prefix) :].strip()
                if remainder:
                    return remainder

        # list/ls/dir: extract directory path
        for prefix in ["ls ", "dir ", "list files ", "list directory ", "show files "]:
            if user_lower.startswith(prefix):
                remainder = user_input[len(prefix) :].strip()
                if remainder:
                    return remainder.split()[0]
                return "."  # default to current dir

        # git: extract the subcommand
        if user_lower.startswith("git "):
            remainder = user_input[4:].strip()
            if remainder:
                return remainder.split()[0]  # status, log, diff, branch

        # lint/ruff
        for prefix in ["lint ", "ruff ", "check code quality ", "code quality "]:
            if user_lower.startswith(prefix):
                remainder = user_input[len(prefix) :].strip()
                if remainder:
                    return remainder.split()[0]

        # --- Existing extraction logic ---

        # Check for spec names
        for part in parts:
            if part.lower() in self.SPEC_NAMES:
                return part.lower()

        # Check for paths
        for part in parts:
            if part.startswith(("/", "./", "../", "~")):
                path = Path(part).expanduser()
                if path.exists():
                    return str(path.resolve())

        # Check last argument as path
        if len(parts) > 1:
            path = Path(parts[-1]).expanduser()
            if path.exists():
                return str(path.resolve())

        # Extract search query
        for prefix in ["search", "find", "look for", "research"]:
            if prefix in user_lower:
                idx = user_lower.index(prefix) + len(prefix)
                remainder = user_input[idx:].strip()
                # Remove common filler words
                for filler in ["for", "about", "on", "papers"]:
                    if remainder.lower().startswith(filler):
                        remainder = remainder[len(filler) :].strip()
                if remainder:
                    return remainder

        return None


# ---------------------------------------------------------------------------
# Smart Agent Engine — the orchestrator
# ---------------------------------------------------------------------------


class SmartAgentEngine:
    """
    Intelligent agent engine that maximizes output quality per token spent.

    Wires together memory, planning, reflection, and real pipeline calls
    with a token budget to stay pocket-friendly.
    """

    def __init__(
        self,
        agent_id: str = "smart-agent",
        max_tokens: int = 50_000,
        quality_threshold: float = 0.6,
        max_retries: int = 2,
    ):
        self.agent_id = agent_id
        self.quality_threshold = quality_threshold
        self.max_retries = max_retries

        # Core components
        self.classifier = QueryClassifier()
        self.memory = AdvancedAgentMemory(agent_id)
        self.planner = AdaptivePlanner()
        self.reflector = AgentReflector()
        self.budget = TokenBudget(max_tokens=max_tokens)
        self.tools = AdvancedToolManager()
        self.terminal = create_shell()  # Advanced terminal with cd, pipes, background jobs
        self.os = DETECTED_OS  # OS detection for cross-platform commands

        # Safety: commands that are blocked even from run_command
        self.DANGEROUS_COMMANDS: set[str] = {
            "rm -rf /",
            "rm -rf /*",
            "mkfs",
            "dd if=",
            ":(){:|:&};:",
            "chmod -R 777 /",
            "shutdown",
            "reboot",
            "halt",
            "init 0",
            "init 6",
        }
        # Windows-specific dangerous commands
        if self.os.is_windows:
            self.DANGEROUS_COMMANDS.update(
                {
                    "format c:",
                    "del /s /q c:",
                    "rmdir /s /q c:",
                    "icacls ... /grant",
                    "takeown /f",
                    "bcdedit /delete",
                    "diskpart",
                    "reg delete",
                    "schtasks /delete",
                    "net user",
                    "net localgroup",
                }
            )

        # Session
        self.sessions: dict[str, AgentSession] = {}
        self.current_session: AgentSession | None = None

        # Execution stats
        self.total_queries = 0
        self.successful_queries = 0
        self.total_tokens_used = 0

    # -----------------------------------------------------------------------
    # Session management
    # -----------------------------------------------------------------------

    def create_session(self, repo_path: str | None = None) -> AgentSession:
        session_id = str(uuid.uuid4())[:12]
        session = AgentSession(id=session_id, repo_path=repo_path)
        self.sessions[session_id] = session
        self.current_session = session

        # Store session start in memory
        self.memory.remember_episode(
            f"Session started: {session_id}",
            outcome="new_session",
            tags=["session", "start"],
        )
        return session

    def switch_repo(self, repo_path: str) -> bool:
        if not self.current_session:
            return False
        path = Path(repo_path).expanduser().resolve()
        if not path.exists():
            return False
        self.current_session.repo_path = str(path)

        # Remember repo context
        self.memory.learn_fact(
            f"User is working on repository: {path.name} at {path}",
            source="session",
            tags=["repo", "context", path.name],
        )
        return True

    # -----------------------------------------------------------------------
    # Main entry point — streaming
    # -----------------------------------------------------------------------

    async def stream_smart(
        self,
        user_input: str,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Main entry: classify → plan → execute → reflect → improve.

        Yields StreamEvents in real-time for UI rendering.
        """
        if not self.current_session:
            self.create_session()

        session = self.current_session
        if session:
            session.add_message("user", user_input)

        self.total_queries += 1

        # Reset budget for this query
        self.budget = TokenBudget(max_tokens=self.budget.max_tokens)

        # --- Phase 1: Classify ---
        yield StreamEvent(
            type=StreamEventType.START,
            message="Processing your request...",
            data={"user_input": user_input},
        )

        classification = self.classifier.classify(user_input)
        self.budget.spend(50)  # classification cost

        yield StreamEvent(
            type=StreamEventType.PROGRESS,
            message=f"Query classified: {classification.complexity.value} "
            f"(action={classification.primary_action}, "
            f"est. cost ~{classification.estimated_token_cost} tokens)",
            data={
                "complexity": classification.complexity.value,
                "action": classification.primary_action,
                "confidence": classification.confidence,
            },
        )

        # --- Phase 2: Memory retrieval (cheap context boost) ---
        memory_context = self._retrieve_memory_context(user_input, classification)
        if memory_context:
            yield StreamEvent(
                type=StreamEventType.PROGRESS,
                message=f"Loaded {len(memory_context)} relevant memories from past sessions",
                data={"memory_count": len(memory_context)},
            )

        # --- Phase 3: Route by complexity ---
        try:
            if classification.complexity == QueryComplexity.TRIVIAL:
                async for event in self._handle_trivial(classification, memory_context):
                    yield event

            elif classification.complexity == QueryComplexity.SIMPLE:
                async for event in self._handle_simple(classification, memory_context):
                    yield event

            elif classification.complexity == QueryComplexity.MODERATE:
                async for event in self._handle_moderate(classification, memory_context):
                    yield event

            else:  # COMPLEX
                async for event in self._handle_complex(classification, memory_context):
                    yield event

            self.successful_queries += 1

        except Exception as e:
            # Store error in memory for learning
            self.memory.remember_episode(
                f"Error on query '{user_input[:80]}': {str(e)}",
                outcome="error",
                tags=["error", classification.primary_action],
            )
            self.reflector.analyze_error(str(e), context=user_input)

            yield StreamEvent(
                type=StreamEventType.ERROR,
                message=str(e),
                data={"exception": type(e).__name__},
            )

        # --- Final: Budget summary ---
        self.total_tokens_used += self.budget.used_tokens
        yield StreamEvent(
            type=StreamEventType.COMPLETE,
            message=f"Done. Tokens used: ~{self.budget.used_tokens:,} / {self.budget.max_tokens:,}",
            data={
                "tokens_used": self.budget.used_tokens,
                "tokens_remaining": self.budget.remaining(),
            },
        )

    # -----------------------------------------------------------------------
    # Synchronous process() for CLI single-shot usage
    # -----------------------------------------------------------------------

    async def process(self, user_input: str) -> AgentResponse:
        """Process a query and return a complete response (non-streaming)."""
        events: list[StreamEvent] = []
        async for event in self.stream_smart(user_input):
            events.append(event)

        # Collect output from events
        outputs = [e for e in events if e.type == StreamEventType.OUTPUT]
        errors = [e for e in events if e.type == StreamEventType.ERROR]
        suggestions = [e.message for e in events if e.type == StreamEventType.SUGGESTION]

        if errors:
            return AgentResponse(
                ok=False,
                message=errors[0].message,
                error=errors[0].message,
                suggestions=suggestions,
            )

        message_parts = [e.message for e in outputs]
        combined_data = {}
        for e in outputs:
            if e.data:
                combined_data.update(e.data)

        return AgentResponse(
            ok=True,
            message="\n".join(message_parts) if message_parts else "Done",
            output=combined_data if combined_data else None,
            suggestions=suggestions,
        )

    # -----------------------------------------------------------------------
    # Complexity handlers
    # -----------------------------------------------------------------------

    async def _handle_trivial(
        self,
        classification: QueryClassification,
        memory_context: list[dict],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Handle trivial queries (help, greet, status) — zero pipeline calls."""
        action = classification.primary_action

        if action == "greet":
            # Personalized greeting with memory
            greeting = self._build_greeting(memory_context)
            yield StreamEvent(type=StreamEventType.OUTPUT, message=greeting)

        elif action == "help":
            yield StreamEvent(
                type=StreamEventType.OUTPUT,
                message=self._build_help_text(),
            )

        elif action == "status":
            yield StreamEvent(
                type=StreamEventType.OUTPUT,
                message=self._build_status_text(),
            )

        elif action == "specs":
            async for event in self._exec_specs():
                yield event

        elif action in ("list_files", "read_file", "git"):
            # Tool-based trivial actions — fast, no pipeline call
            result = await self._execute_action(
                action, classification.target, classification.query_text
            )
            if result.ok:
                yield StreamEvent(
                    type=StreamEventType.OUTPUT,
                    message=result.message,
                    data=result.output or {},
                )
            else:
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    message=result.error or result.message,
                )

        else:
            yield StreamEvent(
                type=StreamEventType.OUTPUT,
                message="I'm here to help. Try 'help' for available commands.",
            )

        self.budget.spend(100)

    async def _handle_simple(
        self,
        classification: QueryClassification,
        memory_context: list[dict],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Handle simple queries — single pipeline call with reflection."""
        action = classification.primary_action
        target = classification.target

        # Resolve target from session/memory if missing
        if classification.needs_repo and not target:
            target = self._resolve_repo_path()
            if not target and action != "search":
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    message="No repository set. Use 'analyze <path>' first, or provide a path.",
                )
                return

        # Check memory for cached results
        cache_hit = self._check_memory_cache(action, target)
        if cache_hit and action in ("analyze", "suggest"):
            yield StreamEvent(
                type=StreamEventType.PROGRESS,
                message="Found recent results in memory — enriching with fresh data...",
            )

        # Execute
        result = await self._execute_action(action, target, classification.query_text)

        if result.ok:
            # Reflect on output quality
            self._reflect_and_score(result, classification)

            # Emit output
            yield StreamEvent(
                type=StreamEventType.OUTPUT,
                message=result.message,
                data=result.output or {},
            )

            # Emit detailed output if available
            async for event in self._format_result_details(action, result):
                yield event

            # Store in memory
            self._store_result_in_memory(action, target, result)

            # Tool orchestration summary
            yield StreamEvent(
                type=StreamEventType.OUTPUT,
                message=self._build_tool_summary(action, result),
            )

            # Emit contextual suggestions
            for sugg in self._generate_suggestions(action, result):
                yield StreamEvent(type=StreamEventType.SUGGESTION, message=sugg)

        else:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message=result.error or result.message,
            )

            # If quality is too low, try to recover
            if self.budget.can_spend(5000):
                recovery = self._suggest_recovery(action, result.error or "")
                if recovery:
                    yield StreamEvent(
                        type=StreamEventType.SUGGESTION,
                        message=recovery,
                    )

            # Retry once with context if allowed
            if self.max_retries > 0 and self.budget.can_spend(5000):
                retry = await self._retry_with_context(action, target, result, classification)
                if retry and retry.ok:
                    yield StreamEvent(
                        type=StreamEventType.OUTPUT,
                        message=f"[Retry] {retry.message}",
                        data=retry.output or {},
                    )
                    async for event in self._format_result_details(action, retry):
                        yield event
                    self._store_result_in_memory(action, target, retry)
                    yield StreamEvent(
                        type=StreamEventType.OUTPUT,
                        message=self._build_tool_summary(action, retry),
                    )
                    for sugg in self._generate_suggestions(action, retry):
                        yield StreamEvent(type=StreamEventType.SUGGESTION, message=sugg)

    async def _handle_moderate(
        self,
        classification: QueryClassification,
        memory_context: list[dict],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Handle moderate queries — 2-3 chained pipeline calls with planning."""
        all_actions = [classification.primary_action] + classification.secondary_actions
        target = classification.target

        if classification.needs_repo and not target:
            target = self._resolve_repo_path()
            if not target:
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    message="No repository set. Use 'analyze <path>' first.",
                )
                return

        # Create plan
        plan = self.planner.create_plan(
            name=f"Execute: {classification.query_text[:50]}",
            description=classification.query_text,
            context={"target": target, "actions": all_actions},
        )

        for action in all_actions:
            depends = [plan.tasks[-1].id] if plan.tasks else []
            self.planner.add_task(
                plan.id,
                description=f"Execute {action}",
                priority=TaskPriority.HIGH,
                depends_on=depends,
            )

        self.budget.spend(200)  # planning cost

        yield StreamEvent(
            type=StreamEventType.PROGRESS,
            message=f"Planned {len(all_actions)} steps: {' → '.join(all_actions)}",
            data={"plan": [a for a in all_actions]},
        )

        # Execute each step with retry support
        prev_result: ExecutionResult | None = None
        for i, action in enumerate(all_actions):
            if not self.budget.can_spend(2000):
                yield StreamEvent(
                    type=StreamEventType.PROGRESS,
                    message="Token budget limit reached. Stopping early.",
                )
                break

            step_target = target
            # For chained actions, use previous result's context
            if prev_result and prev_result.ok and prev_result.output:
                if action == "integrate" and "spec" in (prev_result.output or {}):
                    step_target = prev_result.output["spec"]

            yield StreamEvent(
                type=StreamEventType.PROGRESS,
                message=f"Step {i + 1}/{len(all_actions)}: {action}...",
            )

            result = await self._execute_action(action, step_target, classification.query_text)
            prev_result = result

            if result.ok:
                yield StreamEvent(
                    type=StreamEventType.OUTPUT,
                    message=result.message,
                    data=result.output or {},
                )
                async for event in self._format_result_details(action, result):
                    yield event
                self._store_result_in_memory(action, step_target, result)
                yield StreamEvent(
                    type=StreamEventType.OUTPUT,
                    message=self._build_tool_summary(action, result),
                )
            else:
                # Retry once with context if budget allows
                retry_result = None
                if self.max_retries > 0 and self.budget.can_spend(5000):
                    retry_result = await self._retry_with_context(
                        action, step_target, result, classification
                    )
                if retry_result and retry_result.ok:
                    yield StreamEvent(
                        type=StreamEventType.OUTPUT,
                        message=f"[Retry] {retry_result.message}",
                        data=retry_result.output or {},
                    )
                    async for event in self._format_result_details(action, retry_result):
                        yield event
                    self._store_result_in_memory(action, step_target, retry_result)
                    yield StreamEvent(
                        type=StreamEventType.OUTPUT,
                        message=self._build_tool_summary(action, retry_result),
                    )
                    prev_result = retry_result
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        message=f"Step {action} failed: {result.error or result.message}",
                    )
                    # Try to continue if possible
                    if action == all_actions[-1]:
                        break  # last step failed, nothing to continue

        # Final suggestions
        for sugg in self._generate_suggestions(all_actions[-1], prev_result):
            yield StreamEvent(type=StreamEventType.SUGGESTION, message=sugg)

    async def _handle_complex(
        self,
        classification: QueryClassification,
        memory_context: list[dict],
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Handle complex queries (e.g., integrate) — full pipeline with
        planning, quality gates, and iterative improvement.
        """
        action = classification.primary_action
        target = classification.target

        if classification.needs_repo and not target:
            # For integrate, target is spec name, repo comes from session
            if action == "integrate":
                target = self._extract_spec_from_query(classification.query_text)
                if not target:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        message="No spec provided. Available: rmsnorm, swiglu, flashattention, rope, gqa",
                    )
                    return
            else:
                target = self._resolve_repo_path()
                if not target:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        message="No repository set. Use 'analyze <path>' first.",
                    )
                    return

        repo_path = self._resolve_repo_path()

        # Context probe to enrich understanding
        if self.budget.can_spend(500):
            probe = await self._exec_context_probe(classification.query_text)
            if probe.ok:
                yield StreamEvent(
                    type=StreamEventType.PROGRESS,
                    message="Context probe complete",
                    data=probe.output or {},
                )

        # Plan the complex execution
        yield StreamEvent(
            type=StreamEventType.PROGRESS,
            message=f"Planning complex execution: {action} {target or ''}",
        )

        # Use adaptive planner for complex tasks
        plan = self.planner.decompose_goal(
            f"{action} {target or ''} on {repo_path or 'repo'}",
            context={"action": action, "target": target, "repo": repo_path},
        )
        self.budget.spend(500)

        yield StreamEvent(
            type=StreamEventType.PROGRESS,
            message=f"Decomposed into {len(plan.tasks)} subtasks",
            data={"subtasks": [t.description for t in plan.tasks]},
        )

        # Execute the primary action (which is a full pipeline call)
        result = await self._execute_action(action, target, classification.query_text)

        if result.ok:
            # Reflect on quality
            quality = self._reflect_and_score(result, classification)

            yield StreamEvent(
                type=StreamEventType.OUTPUT,
                message=result.message,
                data=result.output or {},
            )

            async for event in self._format_result_details(action, result):
                yield event

            # Quality gate — if below threshold and budget allows, try to improve
            if (
                quality
                and quality.value < self.quality_threshold * 5  # scale to 1-5
                and self.budget.can_spend(10000)
            ):
                yield StreamEvent(
                    type=StreamEventType.PROGRESS,
                    message="Output quality below threshold — attempting improvement...",
                )
                # Re-run with enhanced context from first attempt
                enhanced_result = await self._retry_with_context(
                    action, target, result, classification
                )
                if enhanced_result and enhanced_result.ok:
                    yield StreamEvent(
                        type=StreamEventType.OUTPUT,
                        message=f"[Improved] {enhanced_result.message}",
                        data=enhanced_result.output or {},
                    )

            self._store_result_in_memory(action, target, result)

            # Generate suggestions
            for sugg in self._generate_suggestions(action, result):
                yield StreamEvent(type=StreamEventType.SUGGESTION, message=sugg)

        else:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message=result.error or result.message,
            )

            # Analyze failure and suggest recovery
            self.reflector.analyze_error(result.error or "", context=classification.query_text)
            recovery = self._suggest_recovery(action, result.error or "")
            if recovery:
                yield StreamEvent(type=StreamEventType.SUGGESTION, message=recovery)

    async def _execute_plan(
        self,
        plan: Plan,
        classification: QueryClassification,
        target: str | None,
    ) -> ExecutionResult:
        """Execute a plan sequentially with retry support."""
        prev_result: ExecutionResult | None = None
        completed_tasks: set[str] = set()

        for i, task in enumerate(plan.tasks):
            if not self.budget.can_spend(2000):
                return ExecutionResult(
                    ok=False,
                    action="plan",
                    error="Token budget limit reached. Stopping early.",
                    tokens_used=0,
                )

            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now().isoformat()

            step_target = target
            if prev_result and prev_result.ok and prev_result.output:
                if "spec" in (prev_result.output or {}):
                    step_target = prev_result.output.get("spec", step_target)

            # Parse action from task description
            action = task.description.replace("Execute ", "").strip()

            result = await self._execute_action(action, step_target, classification.query_text)
            prev_result = result

            if result.ok:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now().isoformat()
                task.result = result
                completed_tasks.add(task.id)

                self._store_result_in_memory(action, step_target, result)
            else:
                task.status = TaskStatus.FAILED
                task.error = result.error or result.message
                task.completed_at = datetime.now().isoformat()

                # Retry logic for failed steps
                if self.max_retries > 0 and self.budget.can_spend(5000):
                    retry_result = await self._retry_with_context(
                        action, step_target, result, classification
                    )
                    if retry_result and retry_result.ok:
                        prev_result = retry_result
                        task.status = TaskStatus.COMPLETED
                        task.result = retry_result
                        completed_tasks.add(task.id)
                        continue

                return result

        return prev_result or ExecutionResult(
            ok=False,
            action="plan",
            error="No tasks executed",
            tokens_used=0,
        )

    # -----------------------------------------------------------------------
    # Execution layer — real pipeline calls
    # -----------------------------------------------------------------------

    async def _execute_action(
        self,
        action: str,
        target: str | None,
        query_text: str,
        classification: QueryClassification | None = None,
    ) -> ExecutionResult:
        """Execute a single action against the real pipeline."""
        start = time.monotonic()

        try:
            if action == "analyze":
                return await self._exec_analyze(target)
            elif action == "search":
                return await self._exec_search(target or query_text)
            elif action == "suggest":
                return await self._exec_suggest(target)
            elif action == "validate":
                return await self._exec_validate(target)
            elif action == "security":
                return await self._exec_security(target)
            elif action == "integrate":
                return await self._exec_integrate(target)
            elif action == "map":
                return await self._exec_map(target)
            elif action == "generate":
                return await self._exec_generate(target)
            elif action == "rollback":
                return await self._exec_rollback(target)
            elif action == "run_command":
                return await self._exec_run_command(target)
            elif action == "read_file":
                return await self._exec_tool_read_file(target)
            elif action == "write_file":
                return await self._exec_tool_write_file(target, query_text)
            elif action == "search_code":
                return await self._exec_tool_search_code(target)
            elif action == "list_files":
                return await self._exec_tool_list_files(target)
            elif action == "git":
                return await self._exec_tool_git(target)
            elif action == "analyze_code":
                return await self._exec_tool_analyze_code(target)
            elif action == "terminal":
                return await self._exec_terminal(target or query_text)
            elif action == "run_code":
                return await self._exec_run_code(target)
            elif action == "run_tests":
                return await self._exec_run_tests(target)
            elif action == "intelligent_run":
                return await self._exec_intelligent_run(query_text)
            elif action == "fix_and_test":
                return await self._exec_fix_and_test(query_text)
            elif action == "unknown":
                return await self._exec_unknown(query_text)
            else:
                return ExecutionResult(ok=False, action=action, error=f"Unknown action: {action}")
        except Exception as e:
            duration = int((time.monotonic() - start) * 1000)
            return ExecutionResult(ok=False, action=action, error=str(e), duration_ms=duration)

    async def _exec_analyze(self, repo_path: str | None) -> ExecutionResult:
        from scholardevclaw.application.pipeline import run_analyze

        if not repo_path:
            repo_path = self._resolve_repo_path()
        if not repo_path:
            return ExecutionResult(ok=False, action="analyze", error="No repository path")

        path = Path(repo_path).expanduser().resolve()
        if not path.exists():
            return ExecutionResult(ok=False, action="analyze", error=f"Path not found: {repo_path}")

        # Set repo path in session
        if self.current_session:
            self.current_session.repo_path = str(path)

        result = await asyncio.to_thread(run_analyze, str(path))
        tokens = 2000
        self.budget.spend(tokens)

        if result.ok:
            return ExecutionResult(
                ok=True,
                action="analyze",
                message=result.title,
                output=result.payload,
                tokens_used=tokens,
            )
        return ExecutionResult(ok=False, action="analyze", error=result.error, tokens_used=tokens)

    async def _exec_search(self, query: str) -> ExecutionResult:
        from scholardevclaw.application.pipeline import run_search

        if not query:
            return ExecutionResult(ok=False, action="search", error="No search query")

        result = await asyncio.to_thread(run_search, query)
        tokens = 3000
        self.budget.spend(tokens)

        if result.ok:
            return ExecutionResult(
                ok=True,
                action="search",
                message=result.title,
                output=result.payload,
                tokens_used=tokens,
            )
        return ExecutionResult(ok=False, action="search", error=result.error, tokens_used=tokens)

    async def _exec_suggest(self, repo_path: str | None) -> ExecutionResult:
        from scholardevclaw.application.pipeline import run_suggest

        repo_path = repo_path or self._resolve_repo_path()
        if not repo_path:
            return ExecutionResult(ok=False, action="suggest", error="No repository set")

        result = await asyncio.to_thread(run_suggest, repo_path)
        tokens = 3000
        self.budget.spend(tokens)

        if result.ok:
            return ExecutionResult(
                ok=True,
                action="suggest",
                message=result.title,
                output=result.payload,
                tokens_used=tokens,
            )
        return ExecutionResult(ok=False, action="suggest", error=result.error, tokens_used=tokens)

    async def _exec_validate(self, repo_path: str | None) -> ExecutionResult:
        from scholardevclaw.application.pipeline import run_validate

        repo_path = repo_path or self._resolve_repo_path()
        if not repo_path:
            return ExecutionResult(ok=False, action="validate", error="No repository set")

        result = await asyncio.to_thread(run_validate, repo_path)
        tokens = 5000
        self.budget.spend(tokens)

        if result.ok:
            return ExecutionResult(
                ok=True,
                action="validate",
                message=result.title,
                output=result.payload,
                tokens_used=tokens,
            )
        return ExecutionResult(ok=False, action="validate", error=result.error, tokens_used=tokens)

    async def _exec_security(self, repo_path: str | None) -> ExecutionResult:
        from scholardevclaw.security import SecurityScanner

        repo_path = repo_path or self._resolve_repo_path()
        if not repo_path:
            return ExecutionResult(ok=False, action="security", error="No repository set")

        scanner = SecurityScanner()
        result = await asyncio.to_thread(scanner.scan, repo_path)
        tokens = 3000
        self.budget.spend(tokens)

        return ExecutionResult(
            ok=True,
            action="security",
            message=f"Security scan: {'passed' if result.passed else f'{result.total_findings} issues found'}",
            output=result.to_dict(),
            tokens_used=tokens,
        )

    async def _exec_integrate(self, spec: str | None) -> ExecutionResult:
        from scholardevclaw.application.pipeline import run_integrate

        repo_path = self._resolve_repo_path()
        if not repo_path:
            return ExecutionResult(ok=False, action="integrate", error="No repository set")
        if not spec:
            return ExecutionResult(ok=False, action="integrate", error="No spec provided")

        result = await asyncio.to_thread(run_integrate, repo_path, spec)
        tokens = 20000
        self.budget.spend(tokens)

        if result.ok:
            return ExecutionResult(
                ok=True,
                action="integrate",
                message=result.title,
                output=result.payload,
                tokens_used=tokens,
            )
        return ExecutionResult(ok=False, action="integrate", error=result.error, tokens_used=tokens)

    async def _exec_map(self, spec: str | None) -> ExecutionResult:
        from scholardevclaw.application.pipeline import run_map

        repo_path = self._resolve_repo_path()
        if not repo_path:
            return ExecutionResult(ok=False, action="map", error="No repository set")
        if not spec:
            return ExecutionResult(ok=False, action="map", error="No spec provided")

        result = await asyncio.to_thread(run_map, repo_path, spec)
        tokens = 8000
        self.budget.spend(tokens)

        if result.ok:
            return ExecutionResult(
                ok=True,
                action="map",
                message=result.title,
                output=result.payload,
                tokens_used=tokens,
            )
        return ExecutionResult(ok=False, action="map", error=result.error, tokens_used=tokens)

    async def _exec_generate(self, spec: str | None) -> ExecutionResult:
        from scholardevclaw.application.pipeline import run_generate

        repo_path = self._resolve_repo_path()
        if not repo_path:
            return ExecutionResult(ok=False, action="generate", error="No repository set")
        if not spec:
            return ExecutionResult(ok=False, action="generate", error="No spec provided")

        result = await asyncio.to_thread(run_generate, repo_path, spec)
        tokens = 10000
        self.budget.spend(tokens)

        if result.ok:
            return ExecutionResult(
                ok=True,
                action="generate",
                message=result.title,
                output=result.payload,
                tokens_used=tokens,
            )
        return ExecutionResult(ok=False, action="generate", error=result.error, tokens_used=tokens)

    async def _exec_rollback(self, snapshot_id: str | None) -> ExecutionResult:
        from scholardevclaw.rollback import RollbackManager

        repo_path = self._resolve_repo_path()
        if not repo_path:
            return ExecutionResult(ok=False, action="rollback", error="No repository set")

        manager = RollbackManager()
        result = await asyncio.to_thread(manager.rollback, repo_path, snapshot_id)
        tokens = 2000
        self.budget.spend(tokens)

        return ExecutionResult(
            ok=True if result.ok else False,
            action="rollback",
            message="Rollback complete" if result.ok else "Rollback failed",
            error=None if result.ok else str(result.error if hasattr(result, "error") else ""),
            tokens_used=tokens,
        )

    async def _exec_specs(self) -> AsyncGenerator[StreamEvent, None]:
        from scholardevclaw.application.pipeline import run_specs

        result = await asyncio.to_thread(run_specs)
        self.budget.spend(500)

        if result.ok:
            yield StreamEvent(
                type=StreamEventType.OUTPUT,
                message=result.title,
                data=result.payload or {},
            )
            specs = (result.payload or {}).get("specs", [])
            if specs:
                for spec in specs[:10]:
                    name = spec.get("name", "unknown") if isinstance(spec, dict) else str(spec)
                    yield StreamEvent(
                        type=StreamEventType.OUTPUT,
                        message=f"  - {name}",
                    )
        else:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                message=result.error or "Failed to list specs",
            )

    async def _exec_unknown(self, query: str) -> ExecutionResult:
        """Handle unknown queries with helpful fallback."""
        return ExecutionResult(
            ok=False,
            action="unknown",
            message="I didn't understand that command.",
            error=f"Unknown query: '{query[:80]}'. Try 'help' for available commands.",
            tokens_used=100,
        )

    # -----------------------------------------------------------------------
    # Tool-based execution methods (wired to AdvancedToolManager)
    # -----------------------------------------------------------------------

    def _is_dangerous_command(self, command: str) -> str | None:
        """Check if a command matches the blocklist. Returns reason or None."""
        cmd_lower = command.lower().strip()
        for dangerous in self.DANGEROUS_COMMANDS:
            if dangerous in cmd_lower:
                return f"Blocked: '{dangerous}' is a destructive command."
        # Block piping to /dev/sda or similar raw devices
        if "/dev/sd" in cmd_lower or "/dev/nvme" in cmd_lower:
            if "dd " in cmd_lower or "mkfs" in cmd_lower:
                return "Blocked: raw device access detected."
        return None

    async def _exec_run_command(self, command: str | None) -> ExecutionResult:
        """Execute a shell command via the tool system with safety checks."""
        if not command:
            return ExecutionResult(
                ok=False, action="run_command", error="No command provided. Usage: run <command>"
            )

        # Safety check
        blocked = self._is_dangerous_command(command)
        if blocked:
            return ExecutionResult(ok=False, action="run_command", error=blocked, tokens_used=50)

        cwd = self._resolve_repo_path() or "."

        # Add OS prefix for display
        os_info = f"[{self.os.os_name}] "

        execution = await self.tools.execute("run_command", command=command, cwd=cwd)
        tokens = 500
        self.budget.spend(tokens)

        if execution.status == ToolExecStatus.SUCCESS and isinstance(execution.result, dict):
            result = execution.result
            ok = result.get("success", False)
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            rc = result.get("returncode", -1)

            output_text = stdout if stdout else stderr
            if not output_text:
                output_text = f"(exit code {rc})"

            return ExecutionResult(
                ok=ok,
                action="run_command",
                message=os_info + output_text[:5000],
                output={
                    "stdout": stdout[:5000],
                    "stderr": stderr[:2000],
                    "returncode": rc,
                    "os": self.os.os_name,
                },
                tokens_used=tokens,
            )

        return ExecutionResult(
            ok=False,
            action="run_command",
            error=execution.error or "Command execution failed",
            tokens_used=tokens,
        )

    async def _exec_tool_read_file(self, file_path: str | None) -> ExecutionResult:
        """Read a file via the tool system."""
        if not file_path:
            return ExecutionResult(
                ok=False, action="read_file", error="No file path provided. Usage: cat <path>"
            )

        # Resolve relative paths against repo
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            repo = self._resolve_repo_path()
            if repo:
                path = Path(repo) / path

        execution = await self.tools.execute("read_file", path=str(path))
        tokens = 200
        self.budget.spend(tokens)

        if execution.status == ToolExecStatus.SUCCESS and isinstance(execution.result, dict):
            result = execution.result
            if result.get("success"):
                content = result.get("content", "")
                # Truncate very large files for output
                display = content[:8000]
                if len(content) > 8000:
                    display += f"\n... ({len(content) - 8000} more characters)"
                return ExecutionResult(
                    ok=True,
                    action="read_file",
                    message=display,
                    output={"path": str(path), "size": len(content)},
                    tokens_used=tokens,
                )
            return ExecutionResult(
                ok=False,
                action="read_file",
                error=result.get("error", "Failed to read file"),
                tokens_used=tokens,
            )

        return ExecutionResult(
            ok=False,
            action="read_file",
            error=execution.error or "File read failed",
            tokens_used=tokens,
        )

    async def _exec_tool_write_file(self, target: str | None, query: str = "") -> ExecutionResult:
        """Write to a file via the tool system (requires explicit path and content)."""
        if not target:
            return ExecutionResult(
                ok=False,
                action="write_file",
                error="No file path provided. Usage: write file <path> <content>",
            )

        # Parse: target is the path, rest of query after path is content
        # For safety, we require content to be explicitly provided
        parts = query.split(maxsplit=3)  # "write file <path> <content>"
        if len(parts) < 4:
            return ExecutionResult(
                ok=False,
                action="write_file",
                error="Usage: write file <path> <content>. Both path and content are required.",
            )

        file_path = parts[2]
        content = parts[3]

        path = Path(file_path).expanduser()
        if not path.is_absolute():
            repo = self._resolve_repo_path()
            if repo:
                path = Path(repo) / path

        execution = await self.tools.execute("write_file", path=str(path), content=content)
        tokens = 300
        self.budget.spend(tokens)

        if execution.status == ToolExecStatus.SUCCESS and isinstance(execution.result, dict):
            result = execution.result
            if result.get("success"):
                return ExecutionResult(
                    ok=True,
                    action="write_file",
                    message=f"Wrote {result.get('bytes', 0)} bytes to {path}",
                    output={"path": str(path), "bytes": result.get("bytes", 0)},
                    tokens_used=tokens,
                )
            return ExecutionResult(
                ok=False,
                action="write_file",
                error=result.get("error", "Failed to write file"),
                tokens_used=tokens,
            )

        return ExecutionResult(
            ok=False,
            action="write_file",
            error=execution.error or "File write failed",
            tokens_used=tokens,
        )

    async def _exec_tool_search_code(self, query: str | None) -> ExecutionResult:
        """Search code via the tool system (grep-like)."""
        if not query:
            return ExecutionResult(
                ok=False,
                action="search_code",
                error="No search query. Usage: grep <pattern>",
            )

        search_path = self._resolve_repo_path() or "."

        execution = await self.tools.execute(
            "search_code", query=query, path=search_path, file_pattern="*"
        )
        tokens = 400
        self.budget.spend(tokens)

        if execution.status == ToolExecStatus.SUCCESS and isinstance(execution.result, dict):
            result = execution.result
            if result.get("success"):
                matches = result.get("matches", 0)
                results_list = result.get("results", [])
                display = f"Found {matches} matches"
                if results_list:
                    display += ":\n" + "\n".join(results_list[:30])
                    if matches > 30:
                        display += f"\n... ({matches - 30} more)"
                return ExecutionResult(
                    ok=True,
                    action="search_code",
                    message=display,
                    output={"matches": matches, "results": results_list[:50]},
                    tokens_used=tokens,
                )
            return ExecutionResult(
                ok=False,
                action="search_code",
                error=result.get("error", "Search failed"),
                tokens_used=tokens,
            )

        return ExecutionResult(
            ok=False,
            action="search_code",
            error=execution.error or "Search failed",
            tokens_used=tokens,
        )

    async def _exec_tool_list_files(self, directory: str | None) -> ExecutionResult:
        """List files in a directory via the tool system."""
        if not directory:
            directory = self._resolve_repo_path() or "."

        path = Path(directory).expanduser()
        if not path.is_absolute():
            repo = self._resolve_repo_path()
            if repo:
                path = Path(repo) / path

        execution = await self.tools.execute("list_directory", path=str(path), recursive=False)
        tokens = 100
        self.budget.spend(tokens)

        if execution.status == ToolExecStatus.SUCCESS and isinstance(execution.result, dict):
            result = execution.result
            if result.get("success"):
                files = result.get("files", [])
                count = result.get("count", len(files))
                display = f"{count} files in {path}:\n" + "\n".join(f"  {f}" for f in files[:50])
                if count > 50:
                    display += f"\n  ... ({count - 50} more)"
                return ExecutionResult(
                    ok=True,
                    action="list_files",
                    message=display,
                    output={"path": str(path), "files": files, "count": count},
                    tokens_used=tokens,
                )
            return ExecutionResult(
                ok=False,
                action="list_files",
                error=result.get("error", "Failed to list directory"),
                tokens_used=tokens,
            )

        return ExecutionResult(
            ok=False,
            action="list_files",
            error=execution.error or "Directory listing failed",
            tokens_used=tokens,
        )

    async def _exec_tool_git(self, operation: str | None) -> ExecutionResult:
        """Execute a git operation via the tool system."""
        valid_ops = {"status", "log", "diff", "branch"}
        if not operation or operation not in valid_ops:
            operation = "status"  # default to status

        execution = await self.tools.execute("git_operation", operation=operation, args={})
        tokens = 200
        self.budget.spend(tokens)

        if execution.status == ToolExecStatus.SUCCESS and isinstance(execution.result, dict):
            result = execution.result
            if result.get("success"):
                output = result.get("output", "")
                return ExecutionResult(
                    ok=True,
                    action="git",
                    message=output[:5000] if output else "(no output)",
                    output={"operation": operation, "raw": output[:5000]},
                    tokens_used=tokens,
                )
            return ExecutionResult(
                ok=False,
                action="git",
                error=result.get("error", f"git {operation} failed"),
                tokens_used=tokens,
            )

        return ExecutionResult(
            ok=False,
            action="git",
            error=execution.error or f"git {operation} failed",
            tokens_used=tokens,
        )

    async def _exec_tool_analyze_code(self, path: str | None) -> ExecutionResult:
        """Run code analysis (ruff) via the tool system."""
        path = path or self._resolve_repo_path() or "."

        execution = await self.tools.execute("analyze_code", path=path, rules={})
        tokens = 500
        self.budget.spend(tokens)

        if execution.status == ToolExecStatus.SUCCESS and isinstance(execution.result, dict):
            result = execution.result
            if result.get("success"):
                issues = result.get("issues", 0)
                output = result.get("output", "")
                msg = result.get("message", "")
                display = f"Code analysis: {issues} issue(s)"
                if msg:
                    display += f" — {msg}"
                if output:
                    display += f"\n{output[:3000]}"
                return ExecutionResult(
                    ok=True,
                    action="analyze_code",
                    message=display,
                    output={"issues": issues, "raw": output[:5000]},
                    tokens_used=tokens,
                )
            return ExecutionResult(
                ok=False,
                action="analyze_code",
                error=result.get("error", "Analysis failed"),
                tokens_used=tokens,
            )

        return ExecutionResult(
            ok=False,
            action="analyze_code",
            error=execution.error or "Code analysis failed",
            tokens_used=tokens,
        )

    # -----------------------------------------------------------------------
    # Terminal mode — advanced shell execution
    # -----------------------------------------------------------------------

    async def _exec_terminal(self, command: str | None) -> ExecutionResult:
        """
        Execute a terminal command with full shell capabilities.

        Supports:
        - cd, pwd, export, alias, history, jobs
        - Pipes: cat file | grep pattern
        - Redirects: echo hi > file
        - Background: long_task &
        - Environment variables: $VAR, ${VAR}
        """
        if not command:
            # Show terminal info
            state = self.terminal.get_state()
            cwd = state.cwd
            return ExecutionResult(
                ok=True,
                action="terminal",
                message=f"Terminal ready\nCWD: {cwd}\nShell: {self.os.user_shell}\nJobs: {len(state.jobs)}",
                output={"cwd": cwd, "shell": self.os.user_shell, "jobs": len(state.jobs)},
                tokens_used=50,
            )

        # Handle terminal-specific commands
        cmd_lower = command.strip().lower()

        # Reset terminal (new session)
        if cmd_lower in ("exit", "quit", "reset"):
            self.terminal = create_shell()
            return ExecutionResult(
                ok=True,
                action="terminal",
                message="Terminal session reset",
                tokens_used=50,
            )

        # Show help
        if cmd_lower in ("help", "?"):
            return ExecutionResult(
                ok=True,
                action="terminal",
                message="""Terminal Commands:
  cd <dir>     Change directory
  pwd          Print working directory
  export VAR=value  Set environment variable
  alias name=cmd   Create alias
  history       Show command history
  jobs          Show background jobs
  !<cmd>        Run command in background

  Pipes:     cmd1 | cmd2
  Redirect:  cmd > file
  Background: cmd &

  exit/reset  Reset terminal session""",
                tokens_used=50,
            )

        # Execute via advanced shell (async-safe)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            result = await self.terminal.run_command_async(command)
        else:
            result = self.terminal.run_command(command)

        # Format output with colors
        output = redact_sensitive_output(str(result.get("output", "")))
        if result.get("returncode", 0) != 0 and not result.get("timed_out"):
            output = self.terminal.colors.error(output)

        tokens = 200
        self.budget.spend(tokens)

        return ExecutionResult(
            ok=result.get("returncode", 0) == 0,
            action="terminal",
            message=output[:5000],
            output={
                "returncode": result.get("returncode"),
                "timed_out": result.get("timed_out"),
                "cwd": self.terminal.state.cwd,
            },
            tokens_used=tokens,
        )

    # -----------------------------------------------------------------------
    # Advanced shell features — like Claude Code, Codex
    # -----------------------------------------------------------------------

    def get_language_runners(self) -> dict[str, str]:
        """Get OS-specific language runners."""
        runners = {
            ".py": "python3",
            ".js": "node",
            ".ts": "npx ts-node",
            ".sh": "bash",
            ".rb": "ruby",
            ".go": "go run",
            ".rs": "cargo run",
            ".java": "java",
            ".c": "gcc",
            ".cpp": "g++",
            ".php": "php",
            ".ps1": "pwsh",  # PowerShell
            ".bat": "cmd /c",  # Windows batch
            ".cmd": "cmd /c",  # Windows command
        }

        # Windows-specific adjustments
        if self.os.is_windows:
            runners[".py"] = "python"  # Windows often uses 'python' not 'python3'
            runners[".sh"] = "bash"  # Requires WSL or Git Bash
            runners[".rb"] = "ruby"  # May not be installed on Windows
            runners[".ps1"] = "powershell -ExecutionPolicy Bypass -File"

        return runners

    LANGUAGE_RUNNERS: dict[str, str] = {}  # Deprecated, use get_language_runners()

    TEST_PATTERNS: dict[str, list[str]] = {
        "pytest": ["pytest", "-v", "--tb=short"],
        "unittest": ["python3", "-m", "unittest", "discover"],
        "jest": ["npx", "jest", "--passWithNoTests"],
        "vitest": ["npx", "vitest", "run"],
        "mocha": ["npx", "mocha"],
        "cargo_test": ["cargo", "test", "--lib"],
        "npm_test": ["npm", "test"],
        "go_test": ["go", "test", "./..."],
    }

    BUILD_COMMANDS: dict[str, list[str]] = {
        "python": ["python3", "-m", "pip", "install", "-e", "."],
        "npm": ["npm", "install"],
        "cargo": ["cargo", "build", "--release"],
        "make": ["make"],
        "go": ["go", "build", "-o", "app"],
    }

    def _detect_language(self, file_path: str) -> str | None:
        """Detect language from file extension."""
        ext = Path(file_path).suffix.lower()
        runners = self.get_language_runners()
        return ext if ext in runners else None

    def _find_test_files(self, repo_path: str) -> dict[str, list[str]]:
        """Auto-discover test files in the repository."""
        repo = Path(repo_path)
        test_info = {
            "pytest": [],
            "jest": [],
            "cargo_test": [],
            "npm_test": [],
            "go_test": [],
        }

        # Python pytest
        if (repo / "pytest.ini").exists() or (repo / "pyproject.toml").exists():
            test_info["pytest"].append("Found pytest config")

        # JS/TS
        if (repo / "package.json").exists():
            test_info["npm_test"].append("package.json")
        if (repo / "jest.config.js").exists() or (repo / "jest.config.ts").exists():
            test_info["jest"].append("jest.config")

        # Rust
        if (repo / "Cargo.toml").exists():
            test_info["cargo_test"].append("Cargo.toml")

        # Go
        if (repo / "go.mod").exists():
            test_info["go_test"].append("go.mod")

        return {k: v for k, v in test_info.items() if v}

    async def _exec_run_code(self, target: str | None) -> ExecutionResult:
        """Smart code runner - auto-detects language and runs .py/.js/etc files."""
        if not target:
            return ExecutionResult(
                ok=False,
                action="run_code",
                error="No file path provided. Usage: run code <file.py>",
            )

        # Resolve path
        path = Path(target).expanduser()
        if not path.is_absolute():
            repo = self._resolve_repo_path()
            if repo:
                path = Path(repo) / path

        if not path.exists():
            return ExecutionResult(ok=False, action="run_code", error=f"File not found: {path}")

        # Auto-detect language
        ext = self._detect_language(str(path))
        if not ext:
            return ExecutionResult(
                ok=False,
                action="run_code",
                error=f"Unsupported file type: {path.suffix}. Supported: {', '.join(self.get_language_runners().keys())}",
            )

        # Build command
        runner = self.get_language_runners().get(ext, "")
        cmd_argv: list[str]
        if ext == ".sh":
            cmd_argv = ["bash", str(path)]
        elif ext in (".c", ".cpp"):
            # Compile first
            compile_result = subprocess.run(
                ["g++", str(path), "-o", "/tmp/a.out"],
                capture_output=True,
                text=True,
            )
            if compile_result.returncode != 0:
                return ExecutionResult(
                    ok=False,
                    action="run_code",
                    error=f"Compilation failed:\n{compile_result.stderr}",
                )
            cmd_argv = ["/tmp/a.out"]
        elif runner:
            cmd_argv = [*shlex.split(runner), str(path)]
        else:
            return ExecutionResult(
                ok=False,
                action="run_code",
                error=f"No runner found for {ext}",
            )

        cwd = str(path.parent)
        execution = await self.tools.execute("run_command", command=cmd_argv, cwd=cwd)
        tokens = 800
        self.budget.spend(tokens)

        if execution.status == ToolExecStatus.SUCCESS and isinstance(execution.result, dict):
            result = execution.result
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            rc = result.get("returncode", -1)

            # Format output like Claude Code
            output_parts = []
            if stdout:
                output_parts.append(f"[stdout]\n{stdout[:5000]}")
            if stderr:
                output_parts.append(f"[stderr]\n{stderr[:2000]}")
            output_parts.append(f"[exit code: {rc}]")

            display = "\n".join(output_parts)
            return ExecutionResult(
                ok=rc == 0,
                action="run_code",
                message=display,
                output={"stdout": stdout, "stderr": stderr, "returncode": rc, "file": str(path)},
                tokens_used=tokens,
            )

        return ExecutionResult(
            ok=False, action="run_code", error=execution.error, tokens_used=tokens
        )

    async def _exec_run_tests(self, target: str | None) -> ExecutionResult:
        """Smart test runner - auto-detects pytest, jest, cargo test, etc."""
        repo_path = self._resolve_repo_path()
        if not repo_path:
            return ExecutionResult(ok=False, action="run_tests", error="No repository set")

        # Find available test frameworks
        test_info = self._find_test_files(repo_path)

        if not test_info:
            return ExecutionResult(
                ok=False,
                action="run_tests",
                error="No test framework detected. Supported: pytest, jest, cargo test, npm test",
            )

        # Determine best test command
        test_cmd = None
        framework = None

        if "pytest" in test_info:
            test_cmd = "pytest -v --tb=short"
            framework = "pytest"
        elif "cargo_test" in test_info:
            test_cmd = "cargo test --lib -- --nocapture"
            framework = "cargo"
        elif "npm_test" in test_info:
            test_cmd = "npm test -- --passWithNoTests"
            framework = "npm"
        elif "jest" in test_info:
            test_cmd = "npx jest --passWithNoTests"
            framework = "jest"
        elif "go_test" in test_info:
            test_cmd = "go test -v ./..."
            framework = "go"

        if not test_cmd:
            return ExecutionResult(
                ok=False, action="run_tests", error="No runnable test framework found"
            )

        # Run tests
        cwd = repo_path
        execution = await self.tools.execute("run_command", command=test_cmd, cwd=cwd)
        tokens = 1000
        self.budget.spend(tokens)

        if execution.status == ToolExecStatus.SUCCESS and isinstance(execution.result, dict):
            result = execution.result
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            rc = result.get("returncode", -1)

            # Parse test results
            passed = failed = 0
            if framework == "pytest":
                # Parse pytest output
                for line in stdout.split("\n"):
                    if "passed" in line.lower():
                        import re

                        m = re.search(r"(\d+) passed", line)
                        if m:
                            passed = int(m.group(1))
                    if "failed" in line.lower():
                        import re

                        m = re.search(r"(\d+) failed", line)
                        if m:
                            failed = int(m.group(1))

            # Format output
            status = "✅ PASSED" if rc == 0 else "❌ FAILED"
            summary = f"[{framework}] {status} — {passed} passed, {failed} failed"

            output_parts = [summary]
            if stdout:
                output_parts.append(f"\n[stdout]\n{stdout[-4000:]}")
            if stderr:
                output_parts.append(f"\n[stderr]\n{stderr[-1000:]}")

            display = "\n".join(output_parts)
            return ExecutionResult(
                ok=rc == 0,
                action="run_tests",
                message=display,
                output={
                    "framework": framework,
                    "passed": passed,
                    "failed": failed,
                    "returncode": rc,
                    "stdout": stdout[-5000:],
                    "stderr": stderr[-2000:],
                },
                tokens_used=tokens,
            )

        return ExecutionResult(
            ok=False, action="run_tests", error=execution.error, tokens_used=tokens
        )

    async def _exec_intelligent_run(self, query: str | None) -> ExecutionResult:
        """Intelligent run - figures out what to execute based on project."""
        repo_path = self._resolve_repo_path()
        if not repo_path:
            return ExecutionResult(ok=False, action="intelligent_run", error="No repository set")

        repo = Path(repo_path)

        # Check for common project files and run appropriate command
        if (repo / "package.json").exists():
            # Node project - run npm install + test
            cmd = "npm install && npm test" if (repo / "jest.config.js").exists() else "npm install"
            framework = "npm"
        elif (repo / "Cargo.toml").exists():
            # Rust project - run cargo test
            cmd = "cargo test --lib -- --nocapture"
            framework = "cargo"
        elif (repo / "go.mod").exists():
            # Go project
            cmd = "go test -v ./..."
            framework = "go"
        elif (repo / "pyproject.toml").exists() or (repo / "setup.py").exists():
            # Python project - run pytest if tests exist
            if (repo / "tests").exists() or (repo / "test").exists():
                cmd = "pytest -v --tb=short"
                framework = "pytest"
            else:
                cmd = "python3 -m pip install -e ."
                framework = "pip"
        elif (repo / "Makefile").exists():
            # Makefile
            cmd = "make"
            framework = "make"
        else:
            # Try to find any .py file and run it
            py_files = list(repo.rglob("*.py"))
            if py_files:
                cmd = f"python3 {py_files[0]}"
                framework = "python"
            else:
                return ExecutionResult(
                    ok=False,
                    action="intelligent_run",
                    error="Cannot determine what to run. No recognized project files found.",
                )

        cwd = str(repo)
        execution = await self.tools.execute("run_command", command=cmd, cwd=cwd)
        tokens = 1500
        self.budget.spend(tokens)

        if execution.status == ToolExecStatus.SUCCESS and isinstance(execution.result, dict):
            result = execution.result
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            rc = result.get("returncode", -1)

            output_parts = [f"[{framework}] Executed: {cmd}"]
            output_parts.append(f"Exit code: {rc}")
            if stdout:
                output_parts.append(f"\n[stdout]\n{stdout[-4000:]}")
            if stderr:
                output_parts.append(f"\n[stderr]\n{stderr[-1000:]}")

            display = "\n".join(output_parts)
            return ExecutionResult(
                ok=rc == 0,
                action="intelligent_run",
                message=display,
                output={"command": cmd, "framework": framework, "returncode": rc},
                tokens_used=tokens,
            )

        return ExecutionResult(
            ok=False, action="intelligent_run", error=execution.error, tokens_used=tokens
        )

    async def _exec_fix_and_test(self, query: str | None) -> ExecutionResult:
        """Auto-fix and test loop: run tests, summarize failure, re-run after fix hints."""
        repo_path = self._resolve_repo_path()
        if not repo_path:
            return ExecutionResult(ok=False, action="fix_and_test", error="No repository set")

        # Step 1: Run tests
        test_result = await self._exec_run_tests(None)
        if test_result.ok:
            return ExecutionResult(
                ok=True,
                action="fix_and_test",
                message="Tests already passing. No fixes needed.",
                output=test_result.output,
                tokens_used=test_result.tokens_used,
            )

        # Step 2: Summarize failure
        failure_summary = test_result.message or test_result.error or "Test failures detected"

        # Step 3: Provide fix guidance (placeholder for future automated fixes)
        guidance = (
            "Auto-fix loop: tests failed. Review the failing tests and logs, "
            "apply fixes, then re-run `test` or `fix tests`."
        )

        return ExecutionResult(
            ok=False,
            action="fix_and_test",
            message=f"{failure_summary}\n\n{guidance}",
            output={"test_result": test_result.output, "guidance": guidance},
            tokens_used=test_result.tokens_used,
        )

    async def _exec_context_probe(self, query: str) -> ExecutionResult:
        """Gather context hints from codebase: repo root, files, and summaries."""
        repo_path = self._resolve_repo_path()
        if not repo_path:
            return ExecutionResult(ok=False, action="context_probe", error="No repository set")

        # Use tool-based commands for quick context
        list_result = await self._exec_tool_list_files(repo_path)
        search_result = await self._exec_tool_search_code(query)

        summary = {
            "repo": repo_path,
            "files": list_result.output.get("files", []) if list_result.output else [],
            "matches": search_result.output.get("matches", 0) if search_result.output else 0,
        }

        return ExecutionResult(
            ok=True,
            action="context_probe",
            message=f"Context probe: {summary['matches']} matches, {len(summary['files'])} files",
            output=summary,
            tokens_used=500,
        )

    # -----------------------------------------------------------------------
    # Memory integration
    # -----------------------------------------------------------------------

    def _retrieve_memory_context(
        self,
        query: str,
        classification: QueryClassification,
    ) -> list[dict]:
        """Retrieve relevant memories for context enrichment."""
        memories = self.memory.retrieve(
            query,
            limit=5,
        )
        self.budget.spend(100)

        context = []
        for m in memories:
            # Only include memories above relevance threshold
            if m.relevance + m.recency + m.importance_boost > 0.5:
                context.append(
                    {
                        "content": m.memory.content[:200],
                        "type": m.memory.memory_type.value,
                        "relevance": round(m.relevance, 2),
                        "tags": m.memory.tags,
                    }
                )
                # Access the memory to update tracking
                self.memory.access(m.memory.id)

        return context

    def _check_memory_cache(self, action: str, target: str | None) -> dict | None:
        """Check if we have recent results for this action+target."""
        cache_query = f"{action} {target or ''}"
        memories = self.memory.retrieve(cache_query, limit=1)

        if memories and memories[0].recency > 0.8:
            return {
                "content": memories[0].memory.content,
                "tags": memories[0].memory.tags,
            }
        return None

    def _store_result_in_memory(
        self,
        action: str,
        target: str | None,
        result: ExecutionResult,
    ) -> None:
        """Store execution result in memory for future context."""
        content = f"{action} on {target or 'unknown'}: {result.message[:200]}"

        self.memory.remember_episode(
            content=content,
            outcome="success" if result.ok else "failure",
            tags=[action, "result", target or "unknown"],
        )

        # Learn facts from successful results
        if result.ok and result.output:
            if action == "analyze" and isinstance(result.output, dict):
                langs = result.output.get("languages", [])
                if langs:
                    self.memory.learn_fact(
                        f"Repository uses languages: {', '.join(langs[:5])}",
                        source=action,
                        tags=["repo", "languages"],
                    )
                frameworks = result.output.get("frameworks", [])
                if frameworks:
                    self.memory.learn_fact(
                        f"Repository uses frameworks: {', '.join(frameworks[:5])}",
                        source=action,
                        tags=["repo", "frameworks"],
                    )

    # -----------------------------------------------------------------------
    # Reflection integration
    # -----------------------------------------------------------------------

    def _reflect_and_score(
        self,
        result: ExecutionResult,
        classification: QueryClassification,
    ) -> QualityRating | None:
        """Reflect on output quality and return rating."""
        if not result.ok or not result.message:
            return None

        reflection = self.reflector.reflect_on_output(
            output=result.message,
            expected=classification.query_text,
        )
        self.budget.spend(200)

        return reflection.quality_rating

    async def _retry_with_context(
        self,
        action: str,
        target: str | None,
        prev_result: ExecutionResult,
        classification: QueryClassification,
    ) -> ExecutionResult | None:
        """Retry an action with enhanced context from the first attempt."""
        if self.max_retries <= 0:
            return None

        # Use reflection to determine what to improve
        self.reflector.analyze_error(
            error=prev_result.error or "Low quality output",
            context=classification.query_text,
        )

        # Re-execute with same parameters
        result = await self._execute_action(action, target, classification.query_text)
        return result if result.ok else None

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _resolve_repo_path(self) -> str | None:
        """Get repo path from session or memory."""
        if self.current_session and self.current_session.repo_path:
            return self.current_session.repo_path

        # Try to find from memory
        memories = self.memory.search_by_tag("repo", limit=1)
        if memories:
            content = memories[0].content
            # Extract path from "working on repository ... at /path"
            if " at " in content:
                return content.split(" at ")[-1].strip()

        return None

    def _extract_spec_from_query(self, query: str) -> str | None:
        """Extract spec name from a query string."""
        for spec in QueryClassifier.SPEC_NAMES:
            if spec in query.lower():
                return spec
        return None

    def _build_greeting(self, memory_context: list[dict]) -> str:
        """Build a personalized greeting using memory context."""
        base = "Hello! I'm ScholarDevClaw — your research-to-code assistant."

        if self.current_session and self.current_session.repo_path:
            repo_name = Path(self.current_session.repo_path).name
            base += f"\n\nCurrently working on: **{repo_name}**"

        if memory_context:
            base += "\n\nFrom our previous sessions, I remember:"
            for ctx in memory_context[:3]:
                base += f"\n  - {ctx['content'][:100]}"

        base += "\n\nWhat would you like to do? Try:"
        base += "\n  - `analyze <path>` — Analyze a repository"
        base += "\n  - `suggest` — Get improvement suggestions"
        base += "\n  - `search <query>` — Find research papers"
        base += "\n  - `integrate <spec>` — Apply a research improvement"
        base += "\n  - `run <command>` — Execute a shell command"
        base += "\n  - `run code <file>` — Run .py/.js/.sh files"
        base += "\n  - `test` — Auto-run tests (pytest, jest, cargo)"
        base += "\n  - `help` — See all commands"

        return base

    def _build_help_text(self) -> str:
        return """**ScholarDevClaw Commands**

**Repository:**
  `analyze <path>` — Analyze repository structure and languages
  `suggest` — Get AI-powered improvement suggestions
  `validate` — Run validation tests
  `security` — Run security scan

**Research:**
  `search <query>` — Search for research papers (arXiv + web)
  `specs` — List available paper specifications

**Integration:**
  `map <spec>` — Map a spec to code locations
  `generate <spec>` — Generate patch artifacts
  `integrate <spec>` — Full pipeline: analyze + research + map + generate + validate

**Recovery:**
  `rollback` — Revert last integration

**Tools:**
  `run <command>` — Execute a shell command
  `cat <path>` — Read a file
  `write file <path> <content>` — Write content to a file
  `grep <pattern>` — Search code for a pattern
  `ls [path]` — List files in a directory
  `git status|log|diff|branch` — Git operations
  `lint [path]` — Run code quality analysis (ruff)

**Advanced (like Claude Code):**
  `run code <file.py>` — Auto-detect language and run (.py, .js, .sh, .ps1, etc.)
  `test` — Auto-detect and run tests (pytest, jest, cargo test, npm test)
  `do it` — Intelligent run: figures out what to build/test based on project
  `terminal` — Enter advanced terminal mode
  `!<cmd>` — Quick terminal command (e.g., `!ls`)

**Slash Commands (REPL only):**
  `/run <cmd>` — Run a terminal command
  `/git <args>` — Git helper
  `/docker <args>` — Docker helper
  `/compose <args>` — Docker compose helper
  `/test` — Run tests
  `/build` — Intelligent build/test

**Terminal Mode** (after `terminal` command):
  `cd <dir>` — Change directory (persists across commands)
  `pwd` — Print working directory
  `export VAR=value` — Set environment variable
  `alias name=cmd` — Create command alias
  `history` — Show command history
  `jobs` — Show background jobs
  `cmd1 | cmd2` — Pipe commands
  `cmd > file` — Redirect output
  `cmd &` — Run in background
  `exit/reset` — Reset terminal session

**Session:**
  `status` — Show current session info (includes OS + shell detection)
  `help` — Show this message
  `status` — Show current session info
  `help` — Show this message

**Available specs:** rmsnorm, swiglu, flashattention, rope, gqa, mixture"""

    def _build_status_text(self) -> str:
        parts = ["**Session Status**"]

        if self.current_session:
            parts.append(f"  Session: {self.current_session.id}")
            parts.append(f"  Repository: {self.current_session.repo_path or 'not set'}")
            parts.append(f"  Messages: {len(self.current_session.messages)}")
        else:
            parts.append("  No active session")

        parts.append("\n**System**")
        parts.append(f"  OS: {self.os.os_name}")
        parts.append(f"  Shell: {self.os.shell} (user: {self.os.user_shell})")

        parts.append("\n**Agent Stats**")
        parts.append(f"  Total queries: {self.total_queries}")
        parts.append(f"  Successful: {self.successful_queries}")
        parts.append(f"  Total tokens used: ~{self.total_tokens_used:,}")

        mem_stats = self.memory.get_stats()
        parts.append("\n**Memory**")
        parts.append(f"  Total memories: {mem_stats['total_memories']}")
        parts.append(f"  Cached: {mem_stats['cached_count']}")

        tool_stats = self.tools.get_statistics()
        parts.append("\n**Tools**")
        parts.append(f"  Executions: {tool_stats['total_executions']}")
        parts.append(f"  Success rate: {tool_stats['success_rate']:.0%}")
        parts.append(f"  Available: {len(self.tools.list_tools())}")

        return "\n".join(parts)

    def _build_tool_summary(self, action: str, result: ExecutionResult) -> str:
        """Build a concise tool orchestration summary."""
        summary = f"[Tools] action={action}"
        if result.output:
            if isinstance(result.output, dict):
                keys = list(result.output.keys())
                if keys:
                    summary += f" | outputs: {', '.join(keys[:5])}"
        if result.duration_ms:
            summary += f" | duration={result.duration_ms}ms"
        return summary

    def _suggest_recovery(self, action: str, error: str) -> str | None:
        """Suggest a recovery action based on error."""
        error_lower = error.lower()

        if "no repository" in error_lower or "not set" in error_lower:
            return "analyze <path> — Set a repository first"
        if "not found" in error_lower:
            return "Check the path exists and try again"
        if "no spec" in error_lower:
            return "specs — List available specifications"
        if "permission" in error_lower:
            return "Check file permissions on the repository"
        if "timeout" in error_lower:
            return "Try a smaller repository or specific subdirectory"
        return None

    def _generate_suggestions(
        self,
        action: str,
        result: ExecutionResult | None,
    ) -> list[str]:
        """Generate contextual next-step suggestions."""
        suggestions = []

        if action == "analyze":
            suggestions.append("suggest — Get improvement suggestions")
            suggestions.append("security — Run security scan")
        elif action == "suggest":
            suggestions.append("integrate <spec> — Apply a suggestion")
            suggestions.append("search <query> — Find more papers")
        elif action == "search":
            suggestions.append("integrate <spec> — Apply a result")
        elif action == "integrate":
            suggestions.append("validate — Test the changes")
            suggestions.append("rollback — Revert if needed")
        elif action == "validate":
            if result and result.ok:
                suggestions.append("Great! Integration validated successfully")
            else:
                suggestions.append("rollback — Revert the changes")
        elif action == "security":
            suggestions.append("suggest — Get improvement suggestions")
        elif action == "run_command":
            suggestions.append("git status — Check repository state")
        elif action in ("read_file", "list_files", "search_code"):
            suggestions.append("analyze <path> — Run full analysis")
        elif action == "git":
            suggestions.append("analyze — Analyze repository")
        elif action == "analyze_code":
            suggestions.append("suggest — Get AI-powered improvement ideas")

        return suggestions

    async def _format_result_details(
        self,
        action: str,
        result: ExecutionResult,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Format detailed output for specific action types."""
        if not result.output:
            return

        output = result.output

        if action == "analyze" and isinstance(output, dict):
            if "languages" in output:
                yield StreamEvent(
                    type=StreamEventType.OUTPUT,
                    message=f"Languages: {', '.join(output['languages'][:10])}",
                )
            if "frameworks" in output:
                yield StreamEvent(
                    type=StreamEventType.OUTPUT,
                    message=f"Frameworks: {', '.join(output['frameworks'][:10])}",
                )
            if "file_count" in output:
                yield StreamEvent(
                    type=StreamEventType.OUTPUT,
                    message=f"Files analyzed: {output['file_count']}",
                )

        elif action == "search" and isinstance(output, dict):
            results = output.get("results", [])
            for r in results[:5]:
                if isinstance(r, dict):
                    title = r.get("title", "Unknown")
                    yield StreamEvent(
                        type=StreamEventType.OUTPUT,
                        message=f"  - {title}",
                    )

        elif action == "suggest" and isinstance(output, dict):
            suggestions = output.get("suggestions", [])
            for s in suggestions[:5]:
                if isinstance(s, dict):
                    paper = s.get("paper", {})
                    conf = s.get("confidence", 0)
                    yield StreamEvent(
                        type=StreamEventType.OUTPUT,
                        message=f"  - {paper.get('title', 'Unknown')} ({conf:.0%} confidence)",
                    )

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive agent statistics."""
        return {
            "agent_id": self.agent_id,
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "success_rate": (
                self.successful_queries / self.total_queries if self.total_queries > 0 else 0
            ),
            "total_tokens_used": self.total_tokens_used,
            "memory_stats": self.memory.get_stats(),
            "reflection_report": self.reflector.generate_report().__dict__,
            "quality_threshold": self.quality_threshold,
            "tool_metrics": self.tools.get_metrics(),
            "tool_statistics": self.tools.get_statistics(),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_smart_engine(
    agent_id: str = "smart-agent",
    max_tokens: int = 50_000,
    quality_threshold: float = 0.6,
) -> SmartAgentEngine:
    """Create a smart agent engine with sensible defaults."""
    return SmartAgentEngine(
        agent_id=agent_id,
        max_tokens=max_tokens,
        quality_threshold=quality_threshold,
    )
