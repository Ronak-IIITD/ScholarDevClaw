"""
Agent planning system for task decomposition and execution.

Provides:
- Goal decomposition into subtasks
- Task dependencies and ordering
- Execution planning
- Plan revision and adaptation
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable


class TaskStatus(Enum):
    """Status of a task"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskPriority(Enum):
    """Task priority levels"""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """A single task in a plan"""

    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    depends_on: list[str] = field(default_factory=list)
    result: Any = None
    error: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: str = ""
    completed_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def can_execute(self, completed: set[str]) -> bool:
        """Check if all dependencies are met"""
        return all(dep_id in completed for dep_id in self.depends_on)


@dataclass
class Plan:
    """A plan containing multiple tasks"""

    id: str
    name: str
    description: str
    tasks: list[Task] = field(default_factory=list)
    status: str = "planning"  # planning, executing, completed, failed
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def get_task(self, task_id: str) -> Task | None:
        """Get task by ID"""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def get_executable_tasks(self, completed: set[str]) -> list[Task]:
        """Get tasks that can be executed now"""
        return [
            t for t in self.tasks if t.status == TaskStatus.PENDING and t.can_execute(completed)
        ]

    def get_completed_tasks(self) -> set[str]:
        """Get set of completed task IDs"""
        return {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}

    def is_complete(self) -> bool:
        """Check if plan is complete"""
        return all(t.status == TaskStatus.COMPLETED for t in self.tasks)

    def success_rate(self) -> float:
        """Calculate success rate"""
        if not self.tasks:
            return 0.0
        completed = [t for t in self.tasks if t.status == TaskStatus.COMPLETED]
        return len(completed) / len(self.tasks)


class Planner:
    """Agent planning system"""

    def __init__(self):
        self.plans: dict[str, Plan] = {}

    def create_plan(
        self,
        name: str,
        description: str = "",
        context: dict | None = None,
    ) -> Plan:
        """Create a new plan"""
        plan = Plan(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            context=context or {},
        )
        self.plans[plan.id] = plan
        return plan

    def add_task(
        self,
        plan_id: str,
        description: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        depends_on: list[str] | None = None,
    ) -> Task | None:
        """Add a task to a plan"""
        plan = self.plans.get(plan_id)
        if not plan:
            return None

        task = Task(
            id=str(uuid.uuid4()),
            description=description,
            priority=priority,
            depends_on=depends_on or [],
        )

        plan.tasks.append(task)
        return task

    def decompose_goal(
        self,
        goal: str,
        context: dict | None = None,
    ) -> Plan:
        """Decompose a high-level goal into tasks"""
        plan = self.create_plan(name=goal, description=f"Plan for: {goal}", context=context)

        goal_lower = goal.lower()

        if "analyze" in goal_lower and "repo" in goal_lower:
            self._decompose_analyze(plan)
        elif "implement" in goal_lower or "integrate" in goal_lower:
            self._decompose_implement(plan, goal)
        elif "research" in goal_lower or "search" in goal_lower:
            self._decompose_research(plan, goal)
        elif "fix" in goal_lower or "bug" in goal_lower:
            self._decompose_fix(plan, goal)
        else:
            self._decompose_generic(plan, goal)

        plan.tasks.sort(key=lambda t: t.priority.value, reverse=True)
        return plan

    def _decompose_analyze(self, plan: Plan):
        """Decompose repository analysis goal"""
        self.add_task(plan.id, "Scan repository structure", TaskPriority.HIGH)
        self.add_task(plan.id, "Identify languages and frameworks", TaskPriority.HIGH)
        self.add_task(plan.id, "Build dependency graph", TaskPriority.MEDIUM)
        self.add_task(plan.id, "Detect key components", TaskPriority.MEDIUM)
        self.add_task(plan.id, "Generate analysis report", TaskPriority.LOW)

    def _decompose_implement(self, plan: Plan, goal: str):
        """Decompose implementation goal"""
        self.add_task(plan.id, "Understand target codebase", TaskPriority.HIGH)
        self.add_task(plan.id, "Research implementation approach", TaskPriority.HIGH)
        self.add_task(plan.id, "Generate code changes", TaskPriority.HIGH)
        self.add_task(plan.id, "Run validation tests", TaskPriority.HIGH)
        self.add_task(plan.id, "Create backup/snapshot", TaskPriority.MEDIUM)
        self.add_task(plan.id, "Apply patch to codebase", TaskPriority.MEDIUM)
        self.add_task(plan.id, "Verify integration", TaskPriority.LOW)

    def _decompose_research(self, plan: Plan, goal: str):
        """Decompose research goal"""
        self.add_task(plan.id, "Search academic papers", TaskPriority.HIGH)
        self.add_task(plan.id, "Filter by relevance", TaskPriority.MEDIUM)
        self.add_task(plan.id, "Extract implementation details", TaskPriority.HIGH)
        self.add_task(plan.id, "Compare alternatives", TaskPriority.MEDIUM)
        self.add_task(plan.id, "Summarize findings", TaskPriority.LOW)

    def _decompose_fix(self, plan: Plan, goal: str):
        """Decompose bug fix goal"""
        self.add_task(plan.id, "Understand the bug", TaskPriority.CRITICAL)
        self.add_task(plan.id, "Find root cause", TaskPriority.CRITICAL)
        self.add_task(plan.id, "Design fix", TaskPriority.HIGH)
        self.add_task(plan.id, "Implement fix", TaskPriority.HIGH)
        self.add_task(plan.id, "Write/update tests", TaskPriority.MEDIUM)
        self.add_task(plan.id, "Verify fix works", TaskPriority.HIGH)

    def _decompose_generic(self, plan: Plan, goal: str):
        """Generic decomposition"""
        self.add_task(plan.id, f"Understand: {goal}", TaskPriority.HIGH)
        self.add_task(plan.id, "Break into subtasks", TaskPriority.MEDIUM)
        self.add_task(plan.id, "Execute subtasks", TaskPriority.HIGH)
        self.add_task(plan.id, "Verify results", TaskPriority.MEDIUM)

    def execute_plan(
        self,
        plan_id: str,
        executor: Callable[[Task], Any],
        max_parallel: int = 3,
    ) -> Plan:
        """Execute a plan with a task executor"""
        plan = self.plans.get(plan_id)
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")

        plan.status = "executing"

        while not plan.is_complete():
            executable = plan.get_executable_tasks(plan.get_completed_tasks())

            if not executable:
                break

            for task in executable[:max_parallel]:
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.now().isoformat()

                try:
                    result = executor(task)
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                except Exception as e:
                    task.error = str(e)
                    task.status = TaskStatus.FAILED

                task.completed_at = datetime.now().isoformat()

        plan.status = "completed" if plan.is_complete() else "failed"
        plan.completed_at = datetime.now().isoformat()

        return plan

    def get_plan(self, plan_id: str) -> Plan | None:
        """Get a plan by ID"""
        return self.plans.get(plan_id)

    def list_plans(self) -> list[Plan]:
        """List all plans"""
        return list(self.plans.values())


class AdaptivePlanner(Planner):
    """Planner that can revise plans based on feedback"""

    def __init__(self):
        super().__init__()
        self.revision_count = 0

    def revise_plan(
        self,
        plan_id: str,
        failed_task_id: str,
        reason: str,
    ) -> Plan | None:
        """Revise a plan after a task fails"""
        plan = self.plans.get(plan_id)
        if not plan:
            return None

        self.revision_count += 1

        failed_task = plan.get_task(failed_task_id)
        if failed_task:
            failed_task.status = TaskStatus.PENDING
            failed_task.error = ""

            new_task = self.add_task(
                plan_id,
                f"Retry/Revise: {failed_task.description}",
                TaskPriority.CRITICAL,
            )

            if new_task:
                new_task.metadata["original_task"] = failed_task_id
                new_task.metadata["revision_reason"] = reason

        return plan
