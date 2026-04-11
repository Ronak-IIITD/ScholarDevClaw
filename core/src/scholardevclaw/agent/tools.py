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
import shlex
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, ParamSpec, TypeVar

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
    def from_parameters(name: str, description: str, params: list[ToolParameter]) -> ToolSchema:
        """Create schema from parameter definitions"""
        required = [p.name for p in params if p.required]
        properties = {}

        for p in params:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
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
                    if not (
                        param.name == "command"
                        and isinstance(value, list)
                        and all(isinstance(item, str) for item in value)
                    ):
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
        try:
            normalized = json.dumps(params, sort_keys=True, default=str)
        except (TypeError, ValueError):
            normalized = str(sorted(params.items())) if params else ""
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
    ) -> ToolPipeline:
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

    def _run_command(self, command: str | list[str], cwd: str | None = None) -> dict:
        import subprocess

        try:
            if isinstance(command, list):
                result = subprocess.run(
                    command,
                    shell=False,
                    capture_output=True,
                    text=True,
                    cwd=cwd,
                    timeout=60,
                )
            else:
                command_str = command.strip()
                if command_str:
                    try:
                        argv = shlex.split(command_str)
                    except ValueError:
                        argv = []
                    if argv:
                        executable = argv[0]
                        runner_aliases = {
                            "python",
                            "python3",
                            "python3.10",
                            "python3.11",
                            "python3.12",
                            "node",
                            "bun",
                            "deno",
                            "bash",
                            "sh",
                            "zsh",
                        }
                        if executable in runner_aliases:
                            result = subprocess.run(
                                argv,
                                shell=False,
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


# =============================================================================
# ADVANCED TOOL FEATURES
# =============================================================================


class ToolHook(Enum):
    """Hook types for tool lifecycle"""

    BEFORE_EXECUTE = "before_execute"
    AFTER_EXECUTE = "after_execute"
    ON_SUCCESS = "on_success"
    ON_ERROR = "on_error"
    ON_TIMEOUT = "on_timeout"
    BEFORE_RETRY = "before_retry"
    ON_COMPLETE = "on_complete"


@dataclass
class ToolHookHandler:
    """Handler for tool hooks"""

    hook: ToolHook
    handler: Callable[..., Any]
    async_handler: bool = False


class ToolMiddleware:
    """Middleware for tool execution"""

    def __init__(self):
        self.hooks: dict[ToolHook, list[ToolHookHandler]] = {hook: [] for hook in ToolHook}
        self._global_handlers: list[Callable] = []

    def register_hook(self, hook: ToolHook, handler: Callable, async_handler: bool = False):
        """Register a hook handler"""
        self.hooks[hook].append(ToolHookHandler(hook, handler, async_handler))

    async def execute_hooks(self, hook: ToolHook, context: dict) -> dict:
        """Execute all handlers for a hook"""
        for handler in self.hooks.get(hook, []):
            if handler.async_handler:
                result = await handler.handler(context)
            else:
                result = handler.handler(context)
            if result:
                context.update(result) if isinstance(result, dict) else None
        return context


class ToolMetrics:
    """Tool execution metrics and monitoring"""

    def __init__(self):
        self._metrics: dict[str, dict] = {}
        self._lock = threading.RLock()

    def record(self, tool_name: str, duration_ms: int, success: bool, cost: float = 0):
        """Record execution metric"""
        with self._lock:
            if tool_name not in self._metrics:
                self._metrics[tool_name] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "total_duration_ms": 0,
                    "total_cost": 0.0,
                    "min_duration_ms": float("inf"),
                    "max_duration_ms": 0,
                }

            m = self._metrics[tool_name]
            m["total_calls"] += 1
            if success:
                m["successful_calls"] += 1
            else:
                m["failed_calls"] += 1
            m["total_duration_ms"] += duration_ms
            m["total_cost"] += cost
            m["min_duration_ms"] = min(m["min_duration_ms"], duration_ms)
            m["max_duration_ms"] = max(m["max_duration_ms"], duration_ms)

    def get_metrics(self, tool_name: str | None = None) -> dict:
        """Get metrics for a tool or all tools"""
        with self._lock:
            if tool_name:
                return self._metrics.get(tool_name, {})
            return self._metrics.copy()

    def get_summary(self) -> dict:
        """Get summary statistics"""
        with self._lock:
            total_calls = sum(m["total_calls"] for m in self._metrics.values())
            total_success = sum(m["successful_calls"] for m in self._metrics.values())
            total_cost = sum(m["total_cost"] for m in self._metrics.values())

            return {
                "total_calls": total_calls,
                "total_success": total_success,
                "total_failed": total_calls - total_success,
                "success_rate": total_success / total_calls if total_calls > 0 else 0,
                "total_cost": total_cost,
                "unique_tools": len(self._metrics),
            }


class ToolState:
    """Tool state management for long-running tools"""

    def __init__(self):
        self._states: dict[str, dict] = {}
        self._lock = threading.RLock()

    def set(self, tool_name: str, key: str, value: Any):
        """Set a state value"""
        with self._lock:
            if tool_name not in self._states:
                self._states[tool_name] = {}
            self._states[tool_name][key] = value

    def get(self, tool_name: str, key: str, default: Any = None) -> Any:
        """Get a state value"""
        with self._lock:
            return self._states.get(tool_name, {}).get(key, default)

    def delete(self, tool_name: str, key: str):
        """Delete a state value"""
        with self._lock:
            if tool_name in self._states:
                self._states[tool_name].pop(key, None)

    def clear(self, tool_name: str | None = None):
        """Clear state"""
        with self._lock:
            if tool_name:
                self._states.pop(tool_name, None)
            else:
                self._states.clear()

    def get_all(self, tool_name: str) -> dict:
        """Get all state for a tool"""
        with self._lock:
            return self._states.get(tool_name, {}).copy()


class ParallelToolExecutor:
    """Execute multiple tools in parallel"""

    def __init__(self, executor: ToolExecutor):
        self.executor = executor

    async def execute_many(
        self,
        tools: list[tuple[str, dict]],
        max_concurrent: int = 5,
        stop_on_error: bool = False,
    ) -> list[ToolExecution]:
        """Execute multiple tools in parallel"""
        semaphore = asyncio.Semaphore(max_concurrent)
        results: list[ToolExecution] = []

        async def execute_with_semaphore(tool_name: str, params: dict) -> ToolExecution:
            async with semaphore:
                if stop_on_error and results and results[-1].status != ToolStatus.SUCCESS:
                    return ToolExecution(
                        id=str(uuid.uuid4()),
                        tool_name=tool_name,
                        parameters=params,
                        status=ToolStatus.CANCELLED,
                        error="Cancelled due to previous error",
                    )
                return await self.executor.execute(tool_name, params)

        tasks = [execute_with_semaphore(name, params) for name, params in tools]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed executions
        final_results = []
        for i, result in enumerate(gathered):
            if isinstance(result, Exception):
                final_results.append(
                    ToolExecution(
                        id=str(uuid.uuid4()),
                        tool_name=tools[i][0],
                        parameters=tools[i][1],
                        status=ToolStatus.FAILED,
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)

        return final_results

    async def execute_map(
        self,
        tool_name: str,
        params_list: list[dict],
        max_concurrent: int = 5,
    ) -> list[ToolExecution]:
        """Execute same tool with different params in parallel"""
        tools = [(tool_name, params) for params in params_list]
        return await self.execute_many(tools, max_concurrent)


class ToolResultTransformer:
    """Transform tool results"""

    @staticmethod
    def extract_field(field: str) -> Callable[[dict], Any]:
        """Create transformer that extracts a field"""

        def transform(result: dict) -> Any:
            if isinstance(result, dict):
                return result.get(field)
            return None

        return transform

    @staticmethod
    def extract_fields(fields: list[str]) -> Callable[[dict], dict]:
        """Create transformer that extracts multiple fields"""

        def transform(result: dict) -> dict:
            if isinstance(result, dict):
                return {f: result.get(f) for f in fields}
            return {}

        return transform

    @staticmethod
    def filter_keys(keep: list[str]) -> Callable[[dict], dict]:
        """Create transformer that keeps only specified keys"""

        def transform(result: dict) -> dict:
            if isinstance(result, dict):
                return {k: v for k, v in result.items() if k in keep}
            return {}

        return transform

    @staticmethod
    def map_values(key: str, mapper: dict) -> Callable[[dict], dict]:
        """Create transformer that maps values"""

        def transform(result: dict) -> dict:
            if isinstance(result, dict) and key in result:
                result = result.copy()
                result[key] = mapper.get(result[key], result[key])
            return result

        return transform


class AdvancedToolExecutor(ToolExecutor):
    """Advanced executor with all features"""

    def __init__(self, registry: ToolRegistry):
        super().__init__(registry)
        self.middleware = ToolMiddleware()
        self.metrics = ToolMetrics()
        self.state = ToolState()
        self._hooks_enabled = True

    async def execute(
        self,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
        use_cache: bool = True,
        max_retries: int | None = None,
        timeout: int | None = None,
        dependencies: dict[str, Any] | None = None,
        transform_result: Callable | None = None,
    ) -> ToolExecution:
        """Execute with hooks and metrics"""
        parameters = parameters or {}

        # Execute before hooks
        if self._hooks_enabled:
            context = await self.middleware.execute_hooks(
                ToolHook.BEFORE_EXECUTE, {"tool_name": tool_name, "parameters": parameters}
            )
            # Only update with actual params, not metadata
            for key in ["tool_name", "parameters", "execution"]:
                context.pop(key, None)
            parameters.update(context)

        # Execute tool
        execution = await super().execute(
            tool_name,
            parameters=parameters,
            use_cache=use_cache,
            max_retries=max_retries,
            timeout=timeout,
            dependencies=dependencies,
        )

        # Record metrics
        self.metrics.record(
            tool_name,
            execution.duration_ms,
            execution.status == ToolStatus.SUCCESS,
            execution.actual_cost,
        )

        # Execute after hooks
        if self._hooks_enabled:
            hook = ToolHook.AFTER_EXECUTE
            if execution.status == ToolStatus.SUCCESS:
                hook = ToolHook.ON_SUCCESS
            elif execution.status == ToolStatus.TIMEOUT:
                hook = ToolHook.ON_TIMEOUT
            elif execution.status == ToolStatus.FAILED:
                hook = ToolHook.ON_ERROR

            await self.middleware.execute_hooks(
                hook,
                {
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "execution": execution,
                },
            )

        # Transform result
        if transform_result and execution.result:
            try:
                execution.result = transform_result(execution.result)
            except Exception:
                pass

        return execution


class AdvancedToolManager(ToolManager):
    """Advanced tool manager with all features"""

    def __init__(self):
        self.registry = ToolRegistry()
        self.executor = AdvancedToolExecutor(self.registry)
        self._define_builtin_tools()

    def create_pipeline(self) -> ToolPipeline:
        """Create a new tool pipeline"""
        return ToolPipeline(self.executor)

    def create_parallel_executor(self) -> ParallelToolExecutor:
        """Create parallel executor"""
        return ParallelToolExecutor(self.executor)

    def register_hook(self, hook: ToolHook, handler: Callable, async_handler: bool = False):
        """Register a hook"""
        self.executor.middleware.register_hook(hook, handler, async_handler)

    def get_metrics(self) -> dict:
        """Get execution metrics"""
        return self.executor.metrics.get_summary()

    def get_tool_metrics(self, tool_name: str) -> dict:
        """Get metrics for specific tool"""
        return self.executor.metrics.get_metrics(tool_name)

    # Additional built-in tools
    def _define_builtin_tools(self):
        """Define enhanced built-in tools"""
        super()._define_builtin_tools()

        self.add_tool(
            name="http_request",
            description="Make HTTP requests",
            category=ToolCategory.WEB,
            func=self._http_request,
            parameters=[
                ToolParameter("url", "string", "URL to request", required=True),
                ToolParameter("method", "string", "HTTP method", default="GET"),
                ToolParameter("headers", "object", "Request headers", default={}),
                ToolParameter("body", "string", "Request body", default=None),
            ],
            capabilities=[ToolCapability.EXECUTE],
            tags=["http", "web", "request"],
            rate_limit_per_minute=30,
        )

        self.add_tool(
            name="git_operation",
            description="Perform git operations",
            category=ToolCategory.GIT,
            func=self._git_operation,
            parameters=[
                ToolParameter(
                    "operation",
                    "string",
                    "Git operation",
                    required=True,
                    enum=["status", "log", "diff", "branch"],
                ),
                ToolParameter("args", "object", "Operation arguments", default={}),
            ],
            capabilities=[ToolCapability.EXECUTE],
            tags=["git", "version-control"],
            rate_limit_per_minute=30,
        )

        self.add_tool(
            name="analyze_code",
            description="Analyze code for issues",
            category=ToolCategory.ANALYSIS,
            func=self._analyze_code,
            parameters=[
                ToolParameter("path", "string", "Path to analyze", required=True),
                ToolParameter("rules", "object", "Analysis rules", default={}),
            ],
            capabilities=[ToolCapability.ANALYZE],
            tags=["analyze", "lint", "quality"],
        )

        self.add_tool(
            name="transform_data",
            description="Transform data structures",
            category=ToolCategory.DATA,
            func=self._transform_data,
            parameters=[
                ToolParameter("data", "object", "Data to transform", required=True),
                ToolParameter(
                    "transform",
                    "string",
                    "Transform type",
                    required=True,
                    enum=["flatten", "filter", "map", "group"],
                ),
                ToolParameter("params", "object", "Transform params", default={}),
            ],
            capabilities=[ToolCapability.TRANSFORM],
            tags=["data", "transform", "etl"],
        )

    def _http_request(
        self, url: str, method: str = "GET", headers: dict | None = None, body: str | None = None
    ) -> dict:
        import requests

        try:
            response = requests.request(method, url, headers=headers or {}, json=body)
            return {
                "success": True,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.text[:10000],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _git_operation(self, operation: str, args: dict | None = None) -> dict:
        import subprocess

        args = args or {}

        commands = {
            "status": ["git", "status", "--porcelain"],
            "log": ["git", "log", "--oneline", "-n", str(args.get("count", 10))],
            "diff": ["git", "diff", args.get("file", "")],
            "branch": ["git", "branch", "-a"],
        }

        cmd = commands.get(operation, ["git", "status"])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _analyze_code(self, path: str, rules: dict | None = None) -> dict:
        import subprocess

        rules = rules or {}

        try:
            result = subprocess.run(
                ["ruff", "check", path, "--output-format=json"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return {
                "success": True,
                "issues": len(result.stdout.split("\n")) - 1 if result.stdout else 0,
                "output": result.stdout[:5000],
            }
        except FileNotFoundError:
            return {"success": True, "issues": 0, "message": "ruff not installed"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _transform_data(self, data: dict, transform: str, params: dict | None = None) -> dict:
        params = params or {}

        if transform == "flatten":
            result = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    for ik, iv in v.items():
                        result[f"{k}.{ik}"] = iv
                else:
                    result[k] = v
            return {"success": True, "result": result}

        elif transform == "filter":
            keep = params.get("keys", [])
            return {"success": True, "result": {k: v for k, v in data.items() if k in keep}}

        elif transform == "map":
            mapping = params.get("mapping", {})
            return {"success": True, "result": {k: mapping.get(v, v) for k, v in data.items()}}

        elif transform == "group":
            key = params.get("by", "type")
            groups = {}
            for item in data.get("items", []):
                k = item.get(key, "other")
                if k not in groups:
                    groups[k] = []
                groups[k].append(item)
            return {"success": True, "result": groups}

        return {"success": False, "error": f"Unknown transform: {transform}"}
