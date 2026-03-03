"""
Sub-Agent System for complex task handling.

Features:
- Task decomposition (break complex tasks into subtasks)
- SubAgent class (specialized agents for specific domains)
- AgentPool (manage multiple sub-agents)
- Agent coordination and communication
- Result aggregation from sub-agents
- Agent spawning and lifecycle management
- Specialized agent types (research, code, analysis, etc.)
- Parallel and sequential subtask execution
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable
from concurrent.futures import ThreadPoolExecutor
import threading


class AgentType(Enum):
    """Types of specialized sub-agents"""

    GENERAL = "general"
    RESEARCH = "research"
    CODE = "code"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    ORCHESTRATOR = "orchestrator"


class TaskStatus(Enum):
    """Status of a subtask"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionMode(Enum):
    """How to execute subtasks"""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"


@dataclass
class SubTask:
    """A subtask for a sub-agent"""

    id: str
    name: str
    description: str
    agent_type: AgentType
    parameters: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: str = ""
    completed_at: str = ""
    duration_ms: int = 0


@dataclass
class SubAgentResult:
    """Result from a sub-agent execution"""

    agent_id: str
    agent_type: AgentType
    task_id: str
    success: bool
    result: Any
    error: str = ""
    duration_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class SubAgent:
    """A specialized sub-agent for specific tasks"""

    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        name: str,
        description: str,
        capabilities: list[str] | None = None,
        tools: list[str] | None = None,
        max_retries: int = 3,
        timeout_seconds: int = 300,
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.name = name
        self.description = description
        self.capabilities = capabilities or []
        self.tools = tools or []
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

        self.is_busy = False
        self.current_task_id: str | None = None
        self.execution_count = 0
        self.total_duration_ms = 0
        self._lock = threading.RLock()

    async def execute(self, task: SubTask) -> SubAgentResult:
        """Execute a task"""
        with self._lock:
            if self.is_busy:
                return SubAgentResult(
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    task_id=task.id,
                    success=False,
                    result=None,
                    error="Agent is busy",
                )
            self.is_busy = True
            self.current_task_id = task.id
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now().isoformat()

        start_time = datetime.now()

        try:
            # Simulate specialized execution
            result = await self._execute_task(task)

            end_time = datetime.now()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = end_time.isoformat()
            task.duration_ms = duration_ms

            with self._lock:
                self.execution_count += 1
                self.total_duration_ms += duration_ms

            return SubAgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                task_id=task.id,
                success=True,
                result=result,
                duration_ms=duration_ms,
            )

        except Exception as e:
            end_time = datetime.now()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = end_time.isoformat()
            task.duration_ms = duration_ms

            return SubAgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                task_id=task.id,
                success=False,
                result=None,
                error=str(e),
                duration_ms=duration_ms,
            )

        finally:
            with self._lock:
                self.is_busy = False
                self.current_task_id = None

    async def _execute_task(self, task: SubTask) -> Any:
        """Execute the actual task based on agent type"""

        if self.agent_type == AgentType.RESEARCH:
            return await self._do_research(task)
        elif self.agent_type == AgentType.CODE:
            return await self._do_code(task)
        elif self.agent_type == AgentType.ANALYSIS:
            return await self._do_analysis(task)
        elif self.agent_type == AgentType.PLANNING:
            return await self._do_planning(task)
        elif self.agent_type == AgentType.EXECUTION:
            return await self._do_execution(task)
        elif self.agent_type == AgentType.VALIDATION:
            return await self._do_validation(task)
        else:
            return await self._do_general(task)

    async def _do_research(self, task: SubTask) -> Any:
        """Research agent — delegates to pipeline run_search."""
        query = task.parameters.get("query", "")
        if not query:
            return {"findings": [], "error": "No query provided"}

        try:
            from scholardevclaw.application.pipeline import run_search

            result = await asyncio.to_thread(run_search, query)
            return {
                "findings": result.payload or {},
                "ok": result.ok,
                "title": result.title,
                "error": result.error,
            }
        except Exception as e:
            return {"findings": [], "error": str(e)}

    async def _do_code(self, task: SubTask) -> Any:
        """Code generation agent — delegates to pipeline run_generate."""
        spec = task.parameters.get("spec", "")
        repo_path = task.parameters.get("repo_path", "")

        if not spec or not repo_path:
            return {"error": "Both 'spec' and 'repo_path' parameters are required"}

        try:
            from scholardevclaw.application.pipeline import run_generate

            result = await asyncio.to_thread(run_generate, repo_path, spec)
            return {
                "ok": result.ok,
                "title": result.title,
                "payload": result.payload or {},
                "error": result.error,
            }
        except Exception as e:
            return {"error": str(e)}

    async def _do_analysis(self, task: SubTask) -> Any:
        """Analysis agent — delegates to pipeline run_analyze."""
        target = task.parameters.get("target", "")
        if not target:
            return {"error": "No target path provided"}

        try:
            from scholardevclaw.application.pipeline import run_analyze

            result = await asyncio.to_thread(run_analyze, target)
            return {
                "ok": result.ok,
                "title": result.title,
                "payload": result.payload or {},
                "error": result.error,
            }
        except Exception as e:
            return {"error": str(e)}

    async def _do_planning(self, task: SubTask) -> Any:
        """Planning agent — delegates to pipeline run_suggest for goal-based planning."""
        goal = task.parameters.get("goal", "")
        repo_path = task.parameters.get("repo_path", "")

        if not repo_path:
            # Planning without a repo is purely heuristic
            return {
                "plan": f"Plan for: {goal}",
                "steps": ["Analyze repo", "Identify improvements", "Generate patches"],
                "note": "No repo_path provided; plan is generic",
            }

        try:
            from scholardevclaw.application.pipeline import run_suggest

            result = await asyncio.to_thread(run_suggest, repo_path)
            suggestions = (result.payload or {}).get("suggestions", [])
            return {
                "ok": result.ok,
                "plan": f"Plan for: {goal}",
                "suggestions": suggestions[:5],
                "error": result.error,
            }
        except Exception as e:
            return {"error": str(e)}

    async def _do_execution(self, task: SubTask) -> Any:
        """Execution agent — delegates to pipeline run_integrate."""
        repo_path = task.parameters.get("repo_path", "")
        spec = task.parameters.get("spec", "")

        if not repo_path or not spec:
            return {"error": "Both 'repo_path' and 'spec' parameters are required"}

        try:
            from scholardevclaw.application.pipeline import run_integrate

            result = await asyncio.to_thread(run_integrate, repo_path, spec)
            return {
                "ok": result.ok,
                "title": result.title,
                "payload": result.payload or {},
                "error": result.error,
            }
        except Exception as e:
            return {"error": str(e)}

    async def _do_validation(self, task: SubTask) -> Any:
        """Validation agent — delegates to pipeline run_validate."""
        target = task.parameters.get("target", "")
        if not target:
            return {"error": "No target path provided"}

        try:
            from scholardevclaw.application.pipeline import run_validate

            result = await asyncio.to_thread(run_validate, target)
            return {
                "ok": result.ok,
                "title": result.title,
                "payload": result.payload or {},
                "error": result.error,
            }
        except Exception as e:
            return {"error": str(e)}

    async def _do_general(self, task: SubTask) -> Any:
        """General agent task execution"""
        await asyncio.sleep(0.1)  # Simulate work
        return {
            "result": f"Completed: {task.name}",
            "agent": self.name,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get agent statistics"""
        avg_duration = (
            self.total_duration_ms / self.execution_count if self.execution_count > 0 else 0
        )
        return {
            "agent_id": self.agent_id,
            "type": self.agent_type.value,
            "busy": self.is_busy,
            "executions": self.execution_count,
            "avg_duration_ms": avg_duration,
        }


class AgentPool:
    """Pool of sub-agents for parallel task execution"""

    def __init__(self, max_agents: int = 10):
        self.max_agents = max_agents
        self.agents: dict[str, SubAgent] = {}
        self._lock = threading.RLock()

    def create_agent(
        self,
        agent_type: AgentType,
        name: str | None = None,
        description: str | None = None,
    ) -> SubAgent:
        """Create a new sub-agent"""
        agent_id = f"agent-{agent_type.value}-{uuid.uuid4().hex[:8]}"

        agent = SubAgent(
            agent_id=agent_id,
            agent_type=agent_type,
            name=name or f"{agent_type.value.title()} Agent",
            description=description or f"Specialized {agent_type.value} agent",
        )

        with self._lock:
            if len(self.agents) < self.max_agents:
                self.agents[agent_id] = agent
            else:
                raise RuntimeError("Agent pool exhausted")

        return agent

    def get_agent(self, agent_id: str) -> SubAgent | None:
        """Get an agent by ID"""
        return self.agents.get(agent_id)

    def get_available_agent(self, agent_type: AgentType | None = None) -> SubAgent | None:
        """Get an available agent, optionally of specific type"""
        with self._lock:
            for agent in self.agents.values():
                if not agent.is_busy:
                    if agent_type is None or agent.agent_type == agent_type:
                        return agent
        return None

    def get_agents_by_type(self, agent_type: AgentType) -> list[SubAgent]:
        """Get all agents of a specific type"""
        with self._lock:
            return [a for a in self.agents.values() if a.agent_type == agent_type]

    def list_agents(self) -> list[dict]:
        """List all agents"""
        with self._lock:
            return [a.get_stats() for a in self.agents.values()]

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent"""
        with self._lock:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                if not agent.is_busy:
                    del self.agents[agent_id]
                    return True
        return False


class TaskDecomposer:
    """Decompose complex tasks into subtasks"""

    def __init__(self):
        self.decomposition_strategies = {
            "research_intensive": self._decompose_research,
            "code_generation": self._decompose_code,
            "analysis": self._decompose_analysis,
            "full_stack": self._decompose_full_stack,
        }

    def decompose(
        self,
        task: str,
        strategy: str = "auto",
    ) -> list[SubTask]:
        """Decompose a complex task into subtasks"""

        if strategy == "auto":
            strategy = self._infer_strategy(task)

        decompose_func = self.decomposition_strategies.get(strategy, self._decompose_general)

        return decompose_func(task)

    def _infer_strategy(self, task: str) -> str:
        """Infer the best decomposition strategy"""
        task_lower = task.lower()

        if "research" in task_lower or "paper" in task_lower:
            return "research_intensive"
        elif "code" in task_lower or "implement" in task_lower or "build" in task_lower:
            return "code_generation"
        elif "analyze" in task_lower or "audit" in task_lower:
            return "analysis"
        else:
            return "full_stack"

    def _decompose_research(self, task: str) -> list[SubTask]:
        """Decompose research-intensive task"""
        return [
            SubTask(
                id=str(uuid.uuid4()),
                name="Search Papers",
                description=f"Search for relevant papers: {task}",
                agent_type=AgentType.RESEARCH,
                parameters={"query": task, "sources": ["arxiv", "github"]},
            ),
            SubTask(
                id=str(uuid.uuid4()),
                name="Extract Specs",
                description="Extract technical specifications from papers",
                agent_type=AgentType.RESEARCH,
                parameters={"source": "papers"},
            ),
            SubTask(
                id=str(uuid.uuid4()),
                name="Analyze Code",
                description="Analyze existing code for improvement opportunities",
                agent_type=AgentType.ANALYSIS,
                parameters={"target": "codebase"},
            ),
        ]

    def _decompose_code(self, task: str) -> list[SubTask]:
        """Decompose code generation task"""
        return [
            SubTask(
                id=str(uuid.uuid4()),
                name="Plan Implementation",
                description=f"Plan code generation: {task}",
                agent_type=AgentType.PLANNING,
                parameters={"goal": task},
            ),
            SubTask(
                id=str(uuid.uuid4()),
                name="Generate Code",
                description="Generate implementation code",
                agent_type=AgentType.CODE,
                parameters={"spec": task},
            ),
            SubTask(
                id=str(uuid.uuid4()),
                name="Validate Code",
                description="Validate generated code",
                agent_type=AgentType.VALIDATION,
                parameters={"target": "generated_code"},
            ),
        ]

    def _decompose_analysis(self, task: str) -> list[SubTask]:
        """Decompose analysis task"""
        return [
            SubTask(
                id=str(uuid.uuid4()),
                name="Collect Data",
                description=f"Collect data for analysis: {task}",
                agent_type=AgentType.RESEARCH,
                parameters={"query": task},
            ),
            SubTask(
                id=str(uuid.uuid4()),
                name="Analyze",
                description="Perform analysis",
                agent_type=AgentType.ANALYSIS,
                parameters={"target": task},
            ),
            SubTask(
                id=str(uuid.uuid4()),
                name="Validate Findings",
                description="Validate analysis findings",
                agent_type=AgentType.VALIDATION,
                parameters={"target": "analysis"},
            ),
        ]

    def _decompose_full_stack(self, task: str) -> list[SubTask]:
        """Decompose full-stack task"""
        return [
            SubTask(
                id=str(uuid.uuid4()),
                name="Research",
                description=f"Research: {task}",
                agent_type=AgentType.RESEARCH,
                parameters={"query": task},
            ),
            SubTask(
                id=str(uuid.uuid4()),
                name="Plan",
                description="Create execution plan",
                agent_type=AgentType.PLANNING,
                parameters={"goal": task},
            ),
            SubTask(
                id=str(uuid.uuid4()),
                name="Generate Code",
                description="Generate code",
                agent_type=AgentType.CODE,
                parameters={"spec": task},
            ),
            SubTask(
                id=str(uuid.uuid4()),
                name="Execute",
                description="Execute code",
                agent_type=AgentType.EXECUTION,
                parameters={"command": "run_tests"},
            ),
            SubTask(
                id=str(uuid.uuid4()),
                name="Validate",
                description="Validate results",
                agent_type=AgentType.VALIDATION,
                parameters={"target": "output"},
            ),
        ]

    def _decompose_general(self, task: str) -> list[SubTask]:
        """General decomposition"""
        return [
            SubTask(
                id=str(uuid.uuid4()),
                name="Process",
                description=task,
                agent_type=AgentType.GENERAL,
                parameters={"task": task},
            )
        ]


class SubAgentOrchestrator:
    """Orchestrate sub-agents for complex tasks"""

    def __init__(
        self,
        max_agents: int = 10,
        execution_mode: ExecutionMode = ExecutionMode.PARALLEL,
    ):
        self.pool = AgentPool(max_agents)
        self.decomposer = TaskDecomposer()
        self.execution_mode = execution_mode
        self.task_history: list[dict] = []

    async def execute_task(
        self,
        task: str,
        strategy: str = "auto",
    ) -> dict[str, Any]:
        """Execute a complex task using sub-agents"""

        # Decompose task
        subtasks = self.decomposer.decompose(task, strategy)

        results = []

        if self.execution_mode == ExecutionMode.PARALLEL:
            results = await self._execute_parallel(subtasks)
        elif self.execution_mode == ExecutionMode.SEQUENTIAL:
            results = await self._execute_sequential(subtasks)
        else:
            results = await self._execute_pipeline(subtasks)

        # Aggregate results
        aggregated = self._aggregate_results(results)

        # Record history
        self.task_history.append(
            {
                "task": task,
                "subtasks": len(subtasks),
                "results": results,
                "aggregated": aggregated,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return aggregated

    async def _execute_parallel(self, subtasks: list[SubTask]) -> list[SubAgentResult]:
        """Execute subtasks in parallel"""

        # Create agents for each subtask type
        agents = {}
        for subtask in subtasks:
            if subtask.agent_type not in agents:
                agents[subtask.agent_type] = self.pool.create_agent(
                    subtask.agent_type,
                    name=f"{subtask.agent_type.value.title()} Worker",
                )

        # Execute in parallel
        tasks = []
        for subtask, agent in zip(subtasks, list(agents.values())):
            tasks.append(agent.execute(subtask))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    SubAgentResult(
                        agent_id=agents[subtasks[i].agent_type].agent_id,
                        agent_type=subtasks[i].agent_type,
                        task_id=subtasks[i].id,
                        success=False,
                        result=None,
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)

        return final_results

    async def _execute_sequential(self, subtasks: list[SubTask]) -> list[SubAgentResult]:
        """Execute subtasks sequentially"""
        results = []

        for subtask in subtasks:
            agent = self.pool.create_agent(
                subtask.agent_type,
                name=f"{subtask.agent_type.value.title()} Worker",
            )
            result = await agent.execute(subtask)
            results.append(result)

            # Stop if failed and not recoverable
            if not result.success and subtask.agent_type != AgentType.VALIDATION:
                break

        return results

    async def _execute_pipeline(self, subtasks: list[SubTask]) -> list[SubAgentResult]:
        """Execute subtasks in pipeline (output feeds input)"""
        results = []
        context = {}

        for subtask in subtasks:
            # Inject context from previous results
            if context:
                subtask.parameters["context"] = context

            agent = self.pool.create_agent(
                subtask.agent_type,
                name=f"{subtask.agent_type.value.title()} Worker",
            )
            result = await agent.execute(subtask)
            results.append(result)

            # Add to context
            if result.success:
                context[subtask.agent_type.value] = result.result

            if not result.success:
                break

        return results

    def _aggregate_results(self, results: list[SubAgentResult]) -> dict[str, Any]:
        """Aggregate results from sub-agents"""

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        total_duration = sum(r.duration_ms for r in results)

        aggregated = {
            "success": len(failed) == 0,
            "total_subtasks": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "total_duration_ms": total_duration,
            "results": [],
        }

        for r in results:
            aggregated["results"].append(
                {
                    "agent_type": r.agent_type.value,
                    "task_id": r.task_id,
                    "success": r.success,
                    "result": r.result,
                    "error": r.error,
                    "duration_ms": r.duration_ms,
                }
            )

        return aggregated

    def get_pool_status(self) -> dict:
        """Get status of agent pool"""
        return {
            "total_agents": len(self.pool.agents),
            "agents": self.pool.list_agents(),
        }

    def get_task_history(self, limit: int = 10) -> list[dict]:
        """Get recent task history"""
        return self.task_history[-limit:]


# Factory function
def create_orchestrator(
    max_agents: int = 10,
    execution_mode: ExecutionMode = ExecutionMode.PARALLEL,
) -> SubAgentOrchestrator:
    """Create a sub-agent orchestrator"""
    return SubAgentOrchestrator(max_agents, execution_mode)


def create_specialized_agent(
    agent_type: AgentType,
    name: str | None = None,
) -> SubAgent:
    """Create a single specialized agent"""
    return SubAgent(
        agent_id=f"agent-{uuid.uuid4().hex[:8]}",
        agent_type=agent_type,
        name=name or f"{agent_type.value.title()} Agent",
        description=f"Specialized {agent_type.value} agent",
    )
