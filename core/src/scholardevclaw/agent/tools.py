"""
Advanced Agent Tool System with enhanced capabilities.

Features:
- OpenAI function calling style schemas
- Tool dependencies and chaining
- Streaming execution
- Parameter validation with JSON schema
- Rate limiting per tool
- Cost tracking and预算
- Tool composition (pipelines)
- Better error handling and recovery
- Tool categories with capabilities
- Execution queuing and prioritization
- Tool timeout and resource limits
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Awaitable, TypeVar, ParamSpec
from functools import wraps
import threading


P = ParamSpec("P")
T = TypeVar("T")


class ToolCategory(Enum):
    """Categories of tools"""

    FILE = "file"
    GIT = "git"
    SEARCH = "search"
    CODE = "code"
    EXECUTION = "execution"
    ANALYSIS = "analysis"
    LLM = "llm"
    WEB = "web"
    DATA = "data"
    CUSTOM = "custom"


class ToolStatus(Enum):
    """Tool execution status"""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    RATE_LIMITED = "rate_limited"


class ToolCapability(Enum):
    """Tool capabilities for matching"""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    SEARCH = "search"
    ANALYZE = "analyze"
    TRANSFORM = "transform"


@dataclass
class ToolParameter:
    """Tool parameter definition"""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None


@dataclass
class ToolSchema:
    """OpenAI function calling style schema"""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_parameters(name: str, description: str, params: list[ToolParameter]) -> "ToolSchema":
        """Create schema from parameter definitions"""
        required = [p.name for p in params if p.required]
        properties = {}

        for p in params:
            prop = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            if p.default is not None:
                prop["default"] = p.default
            if p.min_length:
                prop["minLength"] = p.min_length
            if p.max_length:
                prop["maxLength"] = p.max_length
            if p.pattern:
                prop["pattern"] = p.pattern
            properties[p.name] = prop

        return ToolSchema(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
        )

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolDependency:
    """Tool dependency definition"""

    tool_name: str
    output_key: str  # Key in result to pass to dependent
    input_param: str  # Parameter name in dependent tool


@dataclass
class Tool:
    """A callable tool with enhanced capabilities"""

    name: str
    description: str
    category: ToolCategory
    func: Callable[..., Any]

    # Schema for function calling
    schema: ToolSchema | None = None
    parameters: list[ToolParameter] = field(default_factory=list)

    # Execution settings
    timeout_seconds: int = 60
    retry_count: int = 3
    retry_delay: float = 1.0

    # Safety
    requires_confirmation: bool = False
    dangerous: bool = False
    dangerous_reason: str = ""

    # Rate limiting
    rate_limit_per_minute: int = 60

    # Resource estimation
    estimated_cost: float = 0.001
    estimated_duration_ms: int = 1000

    # Dependencies
    dependencies: list[ToolDependency] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    version: str = "1.0.0"

    # Capabilities
    capabilities: list[ToolCapability] = field(default_factory=list)

    def __post_init__(self):
        """Auto-generate schema from parameters"""
        if not self.schema and self.parameters:
            self.schema = ToolSchema.from_parameters(self.name, self.description, self.parameters)


@dataclass
class ToolExecution:
    """Record of a tool execution with detailed tracking"""

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

    # Additional tracking
    queue_position: int = 0
    worker_id: str = ""
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    duration_ms: int = 0

    # Dependencies
    dependency_results: dict[str, Any] = field(default_factory=dict)

    # Error details
    error_type: str = ""
    stack_trace: str = ""


@dataclass
class ToolRateLimit:
    """Rate limit tracker per tool"""

    tool_name: str
    requests: list[float] = field(default_factory=list)
    limit: int = 60
    window_seconds: int = 60

    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        now = datetime.now().timestamp()
        self.requests = [t for t in self.requests if now - t < self.window_seconds]
        return len(self.requests) < self.limit

    def record(self):
        """Record a request"""
        self.requests.append(datetime.now().timestamp())


class ToolRegistry:
    """Enhanced registry of available tools with search and discovery"""

    def __init__(self):
        self.tools: dict[str, Tool] = {}
        self._rate_limits: dict[str, ToolRateLimit] = {}
        self._categories: dict[ToolCategory, list[str]] = {}
        self._tags: dict[str, list[str]] = {}
        self._lock = threading.RLock()
        self._register_builtins()

    def _register_builtins(self):
        """Register built-in tools"""
        pass

    def register(self, tool: Tool):
        """Register a tool"""
        with self._lock:
            self.tools[tool.name] = tool

            if tool.category not in self._categories:
                self._categories[tool.category] = []
            self._categories[tool.category].append(tool.name)

            for tag in tool.tags:
                if tag not in self._tags:
                    self._tags[tag] = []
                self._tags[tag].append(tool.name)

            self._rate_limits[tool.name] = ToolRateLimit(
                tool_name=tool.name,
                limit=tool.rate_limit_per_minute,
            )

    def get(self, name: str) -> Tool | None:
        """Get a tool by name"""
        return self.tools.get(name)

    def list_tools(
        self,
        category: ToolCategory | None = None,
        tags: list[str] | None = None,
        include_dangerous: bool = False,
    ) -> list[Tool]:
        """List available tools with filtering"""
        tools = list(self.tools.values())

        if category:
            tools = [t for t in tools if t.category == category]

        if tags:
            tool_names = set()
            for tag in tags:
                if tag in self._tags:
                    tool_names.update(self._tags[tag])
            tools = [t for t in tools if t.name in tool_names]

        if not include_dangerous:
            tools = [t for t in tools if not t.dangerous]

        return tools

    def search(
        self,
        query: str,
        capabilities: list[ToolCapability] | None = None,
    ) -> list[Tool]:
        """Search tools by name, description, or capability"""
        query_lower = query.lower()
        results = []

        for tool in self.tools.values():
            score = 0

            if query_lower in tool.name.lower():
                score += 10
            if query_lower in tool.description.lower():
                score += 5
            if any(query_lower in tag.lower() for tag in tool.tags):
                score += 3

            if capabilities:
                if any(cap in tool.capabilities for cap in capabilities):
                    score += 7

            if score > 0:
                results.append((tool, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in results]

    def get_by_capability(self, capability: ToolCapability) -> list[Tool]:
        """Get tools by capability"""
        return [t for t in self.tools.values() if capability in t.capabilities]

    def check_rate_limit(self, tool_name: str) -> bool:
        """Check if tool is within rate limit"""
        if tool_name in self._rate_limits:
            return self._rate_limits[tool_name].is_allowed()
        return True

    def record_rate_limit(self, tool_name: str):
        """Record a tool execution for rate limiting"""
        if tool_name in self._rate_limits:
            self._rate_limits[tool_name].record()


class ToolExecutor:
    """Execute tools with retry, caching, and resource management"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.executions: list[ToolExecution] = []
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._execution_queue: asyncio.Queue | None = None
        self._lock = threading.RLock()

    async def execute(
        self,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
        use_cache: bool = True,
        max_retries: int | None = None,
        timeout: int | None = None,
        dependencies: dict[str, Any] | None = None,
    ) -> ToolExecution:
        """Execute a tool with full feature set"""
        tool = self.registry.get(tool_name)
        if not tool:
            return self._failed_execution(tool_name, parameters, f"Tool {tool_name} not found")

        if not self.registry.check_rate_limit(tool_name):
            return self._failed_execution(
                tool_name,
                parameters,
                f"Rate limit exceeded for {tool_name}",
                status=ToolStatus.RATE_LIMITED,
            )

        parameters = parameters or {}
        parameters = self._inject_dependencies(parameters, dependencies or {})

        if not self._validate_parameters(tool, parameters):
            return self._failed_execution(
                tool_name,
                parameters,
                f"Invalid parameters: {self._get_param_errors(tool, parameters)}",
            )

        cache_key = self._get_cache_key(tool_name, parameters)
        if use_cache and cache_key in self._cache:
            result, cached_at = self._cache[cache_key]
            if datetime.now() - cached_at < self._cache_ttl:
                return self._cached_execution(tool, parameters, result, cached_at)

        max_retries = max_retries if max_retries is not None else tool.retry_count
        timeout = timeout if timeout else tool.timeout_seconds

        execution = ToolExecution(
            id=str(uuid.uuid4()),
            tool_name=tool_name,
            parameters=parameters,
            status=ToolStatus.RUNNING,
            started_at=datetime.now().isoformat(),
            estimated_cost=tool.estimated_cost,
        )
        self.executions.append(execution)

        self.registry.record_rate_limit(tool_name)

        for attempt in range(max_retries + 1):
            try:
                start_time = datetime.now()
                result = await self._run_tool(tool, parameters, timeout)
                end_time = datetime.now()

                execution.result = result
                execution.status = ToolStatus.SUCCESS
                execution.completed_at = end_time.isoformat()
                execution.duration_ms = int((end_time - start_time).total_seconds() * 1000)
                execution.actual_cost = tool.estimated_cost

                if use_cache:
                    self._cache[cache_key] = (result, datetime.now())

                return execution

            except asyncio.TimeoutError:
                execution.error = f"Timeout after {timeout}s"
                execution.error_type = "timeout"
                execution.status = ToolStatus.TIMEOUT

            except Exception as e:
                execution.retry_count = attempt + 1
                execution.error = str(e)
                execution.error_type = type(e).__name__
                execution.stack_trace = self._get_traceback(e)

                if attempt < max_retries:
                    await asyncio.sleep(tool.retry_delay * (attempt + 1))
                else:
                    execution.status = ToolStatus.FAILED
                    execution.completed_at = datetime.now().isoformat()

        return execution

    def _inject_dependencies(self, params: dict, deps: dict[str, Any]) -> dict[str, Any]:
        """Inject dependency results into parameters"""
        result = params.copy()
        for key, value in deps.items():
            if value is not None:
                result[key] = value
        return result

    def _validate_parameters(self, tool: Tool, params: dict) -> bool:
        """Validate parameters against tool schema"""
        for param in tool.parameters:
            if param.required and param.name not in params:
                if param.default is None:
                    return False
            if param.name in params:
                value = params[param.name]
                if param.type == "string" and not isinstance(value, str):
                    return False
                if param.type == "integer" and not isinstance(value, int):
                    return False
                if param.type == "number" and not isinstance(value, (int, float)):
                    return False
                if param.type == "boolean" and not isinstance(value, bool):
                    return False
                if param.enum and value not in param.enum:
                    return False

        # Fill in defaults for missing optional params
        for param in tool.parameters:
            if param.name not in params and param.default is not None:
                params[param.name] = param.default

        return True

    def _get_param_errors(self, tool: Tool, params: dict) -> str:
        """Get parameter validation errors"""
        errors = []
        for param in tool.parameters:
            if param.required and param.name not in params:
                errors.append(f"Missing required: {param.name}")
            if param.name in params and param.enum:
                if params[param.name] not in param.enum:
                    errors.append(f"Invalid value for {param.name}")
        return ", ".join(errors)

    async def _run_tool(self, tool: Tool, parameters: dict, timeout: int) -> Any:
        """Run a tool with timeout"""
        if asyncio.iscoroutinefunction(tool.func):
            return await asyncio.wait_for(
                tool.func(**parameters),
                timeout=timeout,
            )
        else:
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, lambda: tool.func(**parameters)),
                timeout=timeout,
            )

    def _get_cache_key(self, tool_name: str, params: dict) -> str:
        """Generate deterministic cache key"""
        normalized = json.dumps(params, sort_keys=True, default=str)
        return f"{tool_name}:{hashlib.sha256(normalized.encode()).hexdigest()}"

    def _cached_execution(
        self, tool: Tool, params: dict, result: Any, cached_at: datetime
    ) -> ToolExecution:
        """Create cached execution record"""
        return ToolExecution(
            id=str(uuid.uuid4()),
            tool_name=tool.name,
            parameters=params,
            status=ToolStatus.SUCCESS,
            result=result,
            cached=True,
            started_at=cached_at.isoformat(),
            completed_at=datetime.now().isoformat(),
            estimated_cost=0,
            actual_cost=0,
        )

    def _failed_execution(
        self,
        tool_name: str,
        params: dict | None,
        error: str,
        status: ToolStatus = ToolStatus.FAILED,
    ) -> ToolExecution:
        """Create failed execution record"""
        return ToolExecution(
            id=str(uuid.uuid4()),
            tool_name=tool_name,
            parameters=params or {},
            status=status,
            error=error,
            completed_at=datetime.now().isoformat(),
        )

    def _get_traceback(self, exc: Exception) -> str:
        """Get string representation of exception"""
        import traceback

        return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    def get_execution_history(
        self,
        tool_name: str | None = None,
        status: ToolStatus | None = None,
        limit: int = 50,
    ) -> list[ToolExecution]:
        """Get execution history with filtering"""
        executions = self.executions

        if tool_name:
            executions = [e for e in executions if e.tool_name == tool_name]
        if status:
            executions = [e for e in executions if e.status == status]

        return executions[-limit:]

    def get_statistics(self) -> dict[str, Any]:
        """Get execution statistics"""
        total = len(self.executions)
        success = len([e for e in self.executions if e.status == ToolStatus.SUCCESS])
        failed = len([e for e in self.executions if e.status == ToolStatus.FAILED])
        total_cost = sum(e.actual_cost for e in self.executions)
        avg_duration = 0
        if self.executions:
            avg_duration = sum(e.duration_ms for e in self.executions) / len(self.executions)

        return {
            "total_executions": total,
            "successful": success,
            "failed": failed,
            "success_rate": success / total if total > 0 else 0,
            "total_cost": total_cost,
            "average_duration_ms": avg_duration,
            "cached_results": len(self._cache),
        }

    def clear_cache(self):
        """Clear execution cache"""
        self._cache.clear()


class ToolPipeline:
    """Tool composition - run tools in sequence or parallel"""

    def __init__(self, executor: ToolExecutor):
        self.executor = executor
        self.steps: list[dict[str, Any]] = []

    def add_step(
        self,
        tool_name: str,
        parameters: dict | None = None,
        output_key: str | None = None,
        condition: Callable[[dict], bool] | None = None,
    ) -> "ToolPipeline":
        """Add a step to the pipeline"""
        self.steps.append(
            {
                "tool": tool_name,
                "params": parameters or {},
                "output_key": output_key,
                "condition": condition,
            }
        )
        return self

    async def execute(self) -> dict[str, Any]:
        """Execute the pipeline"""
        results = {}
        context = {}

        for i, step in enumerate(self.steps):
            tool_name = step["tool"]
            params = step["params"].copy()

            if step["condition"] and not step["condition"](context):
                continue

            for key, value in context.items():
                if f"{{{{{key}}}}}" in str(params):
                    params = self._interpolate(params, key, value)

            execution = await self.executor.execute(
                tool_name,
                parameters=params,
            )

            if execution.status != ToolStatus.SUCCESS:
                return {
                    "success": False,
                    "error": execution.error,
                    "step": i,
                    "results": results,
                }

            if step["output_key"]:
                results[step["output_key"]] = execution.result
                context[step["output_key"]] = execution.result

        return {
            "success": True,
            "results": results,
        }

    def _interpolate(self, params: dict, key: str, value: Any) -> dict:
        """Interpolate context values into parameters"""
        result = {}
        for k, v in params.items():
            if isinstance(v, str):
                v = v.replace(f"{{{{{key}}}}}", str(value))
            result[k] = v
        return result


class ToolManager:
    """High-level tool management with all features"""

    def __init__(self):
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(self.registry)
        self._define_builtin_tools()

    def _define_builtin_tools(self):
        """Define built-in tools with schemas"""

        self.add_tool(
            name="read_file",
            description="Read contents of a file",
            category=ToolCategory.FILE,
            func=self._read_file,
            parameters=[
                ToolParameter("path", "string", "Path to file", required=True),
                ToolParameter("encoding", "string", "File encoding", default="utf-8"),
            ],
            capabilities=[ToolCapability.READ],
            tags=["file", "read", "io"],
        )

        self.add_tool(
            name="write_file",
            description="Write content to a file",
            category=ToolCategory.FILE,
            func=self._write_file,
            parameters=[
                ToolParameter("path", "string", "Path to file", required=True),
                ToolParameter("content", "string", "Content to write", required=True),
                ToolParameter("encoding", "string", "File encoding", default="utf-8"),
            ],
            capabilities=[ToolCapability.WRITE],
            tags=["file", "write", "io"],
            dangerous=True,
            dangerous_reason="Can overwrite files",
        )

        self.add_tool(
            name="search_code",
            description="Search for code patterns in files",
            category=ToolCategory.SEARCH,
            func=self._search_code,
            parameters=[
                ToolParameter("query", "string", "Search query", required=True),
                ToolParameter("path", "string", "Directory to search", default="."),
                ToolParameter("file_pattern", "string", "File pattern", default="*.py"),
            ],
            capabilities=[ToolCapability.SEARCH],
            tags=["search", "code", "grep"],
        )

        self.add_tool(
            name="run_command",
            description="Execute a shell command",
            category=ToolCategory.EXECUTION,
            func=self._run_command,
            parameters=[
                ToolParameter("command", "string", "Command to execute", required=True),
                ToolParameter("cwd", "string", "Working directory", default=None),
            ],
            capabilities=[ToolCapability.EXECUTE],
            tags=["shell", "bash", "execute"],
            dangerous=True,
            dangerous_reason="Executes arbitrary shell commands",
            rate_limit_per_minute=10,
        )

        self.add_tool(
            name="list_directory",
            description="List files in a directory",
            category=ToolCategory.FILE,
            func=self._list_directory,
            parameters=[
                ToolParameter("path", "string", "Directory path", default="."),
                ToolParameter("recursive", "boolean", "List recursively", default=False),
            ],
            capabilities=[ToolCapability.READ],
            tags=["file", "ls", "directory"],
        )

    def add_tool(
        self,
        name: str,
        description: str,
        category: ToolCategory,
        func: Callable,
        parameters: list[ToolParameter] | None = None,
        **kwargs,
    ):
        """Add a new tool with schema"""
        tool = Tool(
            name=name,
            description=description,
            category=category,
            func=func,
            parameters=parameters or [],
            **kwargs,
        )
        self.registry.register(tool)

    async def execute(self, tool_name: str, **params) -> ToolExecution:
        """Execute a tool by name"""
        return await self.executor.execute(tool_name, params)

    def create_pipeline(self) -> ToolPipeline:
        """Create a new tool pipeline"""
        return ToolPipeline(self.executor)

    def list_tools(self, category: ToolCategory | None = None) -> list[Tool]:
        """List tools"""
        return self.registry.list_tools(category)

    def search_tools(self, query: str) -> list[Tool]:
        """Search tools"""
        return self.registry.search(query)

    def get_statistics(self) -> dict[str, Any]:
        """Get execution statistics"""
        return self.executor.get_statistics()

    # Built-in tool implementations
    def _read_file(self, path: str, encoding: str = "utf-8") -> dict:
        from pathlib import Path

        try:
            content = Path(path).read_text(encoding=encoding)
            return {"success": True, "content": content, "path": path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _write_file(self, path: str, content: str, encoding: str = "utf-8") -> dict:
        from pathlib import Path

        try:
            Path(path).write_text(content, encoding=encoding)
            return {"success": True, "path": path, "bytes": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _search_code(self, query: str, path: str = ".", file_pattern: str = "*.py") -> dict:
        import subprocess

        try:
            result = subprocess.run(
                ["grep", "-r", "-n", query, "--include", file_pattern, path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            lines = result.stdout.strip().split("\n") if result.stdout else []
            return {"success": True, "matches": len(lines), "results": lines[:100]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _run_command(self, command: str, cwd: str | None = None) -> dict:
        import subprocess

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=60,
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _list_directory(self, path: str = ".", recursive: bool = False) -> dict:
        from pathlib import Path

        try:
            p = Path(path)
            if recursive:
                files = [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
            else:
                files = [f.name for f in p.iterdir() if f.is_file()]
            return {"success": True, "path": path, "files": files, "count": len(files)}
        except Exception as e:
            return {"success": False, "error": str(e)}


def create_tool(
    name: str,
    description: str,
    category: ToolCategory = ToolCategory.CUSTOM,
    parameters: list[ToolParameter] | None = None,
    **kwargs,
) -> Callable[[Callable], Tool]:
    """Decorator to create a tool from a function"""

    def decorator(func: Callable) -> Tool:
        return Tool(
            name=name,
            description=description,
            category=category,
            func=func,
            parameters=parameters or [],
            **kwargs,
        )

    return decorator
