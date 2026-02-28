"""
Agent tool use system for structured action execution.

Provides:
- Tool registry and discovery
- Parameter validation
- Execution with retry logic
- Result caching
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Awaitable


class ToolCategory(Enum):
    """Categories of tools"""

    FILE = "file"
    GIT = "git"
    SEARCH = "search"
    CODE = "code"
    EXECUTION = "execution"
    ANALYSIS = "analysis"
    CUSTOM = "custom"


class ToolStatus(Enum):
    """Tool execution status"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class Tool:
    """A callable tool"""

    name: str
    description: str
    category: ToolCategory
    func: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 60
    retry_count: int = 3
    requires_confirmation: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecution:
    """Record of a tool execution"""

    id: str
    tool_name: str
    parameters: dict[str, Any]
    status: ToolStatus
    result: Any = None
    error: str = ""
    started_at: str = ""
    completed_at: str = ""
    retry_count: int = 0
    cached: bool = False


class ToolRegistry:
    """Registry of available tools"""

    def __init__(self):
        self.tools: dict[str, Tool] = {}
        self._register_builtins()

    def _register_builtins(self):
        """Register built-in tools"""
        pass

    def register(self, tool: Tool):
        """Register a tool"""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name"""
        return self.tools.get(name)

    def list_tools(
        self,
        category: ToolCategory | None = None,
    ) -> list[Tool]:
        """List available tools"""
        tools = list(self.tools.values())

        if category:
            tools = [t for t in tools if t.category == category]

        return tools

    def search(self, query: str) -> list[Tool]:
        """Search tools by name or description"""
        query_lower = query.lower()
        return [
            t
            for t in self.tools.values()
            if query_lower in t.name.lower() or query_lower in t.description.lower()
        ]


class ToolExecutor:
    """Execute tools with retry and caching"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.executions: list[ToolExecution] = []
        self._cache: dict[str, tuple[Any, datetime]] = {}

    async def execute(
        self,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
        use_cache: bool = True,
        max_retries: int | None = None,
    ) -> ToolExecution:
        """Execute a tool"""
        tool = self.registry.get(tool_name)
        if not tool:
            return self._failed_execution(tool_name, parameters, f"Tool {tool_name} not found")

        parameters = parameters or {}

        cache_key = self._get_cache_key(tool_name, parameters)
        if use_cache and cache_key in self._cache:
            result, cached_at = self._cache[cache_key]
            if datetime.now() - cached_at < timedelta(minutes=5):
                execution = ToolExecution(
                    id=str(uuid.uuid4()),
                    tool_name=tool_name,
                    parameters=parameters,
                    status=ToolStatus.SUCCESS,
                    result=result,
                    cached=True,
                    started_at=cached_at.isoformat(),
                    completed_at=datetime.now().isoformat(),
                )
                self.executions.append(execution)
                return execution

        max_retries = max_retries if max_retries is not None else tool.retry_count

        execution = ToolExecution(
            id=str(uuid.uuid4()),
            tool_name=tool_name,
            parameters=parameters,
            status=ToolStatus.RUNNING,
            started_at=datetime.now().isoformat(),
        )

        self.executions.append(execution)

        for attempt in range(max_retries + 1):
            try:
                result = await self._run_tool(tool, parameters)
                execution.result = result
                execution.status = ToolStatus.SUCCESS
                execution.completed_at = datetime.now().isoformat()

                if use_cache:
                    self._cache[cache_key] = (result, datetime.now())

                return execution

            except Exception as e:
                execution.retry_count = attempt + 1
                execution.error = str(e)

                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    execution.status = ToolStatus.FAILED
                    execution.completed_at = datetime.now().isoformat()

        return execution

    async def _run_tool(self, tool: Tool, parameters: dict) -> Any:
        """Run a tool with timeout"""
        if asyncio.iscoroutinefunction(tool.func):
            return await asyncio.wait_for(
                tool.func(**parameters),
                timeout=tool.timeout_seconds,
            )
        else:
            return await asyncio.tool(tool.func, **parameters)

    def _get_cache_key(self, tool_name: str, params: dict) -> str:
        """Generate cache key"""
        return f"{tool_name}:{sorted(params.items())}"

    def _failed_execution(
        self,
        tool_name: str,
        params: dict | None,
        error: str,
    ) -> ToolExecution:
        """Create failed execution record"""
        return ToolExecution(
            id=str(uuid.uuid4()),
            tool_name=tool_name,
            parameters=params or {},
            status=ToolStatus.FAILED,
            error=error,
            completed_at=datetime.now().isoformat(),
        )

    def get_execution_history(
        self,
        tool_name: str | None = None,
        limit: int = 50,
    ) -> list[ToolExecution]:
        """Get execution history"""
        executions = self.executions

        if tool_name:
            executions = [e for e in executions if e.tool_name == tool_name]

        return executions[-limit:]

    def clear_cache(self):
        """Clear execution cache"""
        self._cache.clear()


class ToolManager:
    """High-level tool management"""

    def __init__(self):
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(self.registry)

    def add_tool(
        self,
        name: str,
        description: str,
        category: ToolCategory,
        func: Callable,
        **kwargs,
    ):
        """Add a new tool"""
        tool = Tool(
            name=name,
            description=description,
            category=category,
            func=func,
            **kwargs,
        )
        self.registry.register(tool)

    async def execute(self, tool_name: str, **params) -> ToolExecution:
        """Execute a tool by name"""
        return await self.executor.execute(tool_name, params)

    def list_tools(self, category: ToolCategory | None = None) -> list[Tool]:
        """List tools"""
        return self.registry.list_tools(category)

    def get_tools_by_capability(self, capability: str) -> list[Tool]:
        """Find tools by capability keyword"""
        return self.registry.search(capability)


def create_file_tool(
    name: str,
    read: bool = False,
    write: bool = False,
) -> Callable:
    """Helper to create file operation tools"""

    def read_file(path: str) -> str:
        from pathlib import Path

        return Path(path).read_text()

    def write_file(path: str, content: str):
        from pathlib import Path

        Path(path).write_text(content)

    if read:
        return read_file
    return write_file


def create_shell_tool(name: str) -> Callable:
    """Helper to create shell execution tools"""

    def run_shell(command: str) -> dict:
        import subprocess

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    return run_shell
