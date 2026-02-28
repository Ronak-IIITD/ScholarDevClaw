"""
Batch processing for multiple repositories/specs in parallel.

Provides:
- Batch job definitions
- Parallel execution with worker pools
- Progress tracking
- Result aggregation
- Error handling and retry
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Awaitable


class BatchStatus(Enum):
    """Batch job status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(Enum):
    """Individual task status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class BatchTask:
    """A single task in a batch"""

    id: str
    name: str
    task_type: str  # "analyze", "search", "generate", "validate"
    parameters: dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    started_at: str = ""
    completed_at: str = ""
    retry_count: int = 0


@dataclass
class BatchJob:
    """A batch job containing multiple tasks"""

    id: str
    name: str
    description: str = ""
    tasks: list[BatchTask] = field(default_factory=list)
    status: BatchStatus = BatchStatus.PENDING
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BatchProcessor:
    """Process multiple tasks in parallel"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.jobs: dict[str, BatchJob] = {}
        self._running = False

    def create_job(
        self,
        name: str,
        tasks: list[dict[str, Any]],
        description: str = "",
    ) -> BatchJob:
        """Create a new batch job"""
        job = BatchJob(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            created_at=datetime.now().isoformat(),
            total_tasks=len(tasks),
            tasks=[
                BatchTask(
                    id=str(uuid.uuid4()),
                    name=t.get("name", f"task_{i}"),
                    task_type=t.get("type", "analyze"),
                    parameters=t.get("parameters", {}),
                )
                for i, t in enumerate(tasks)
            ],
        )

        self.jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> BatchJob | None:
        """Get a job by ID"""
        return self.jobs.get(job_id)

    def list_jobs(self, status: BatchStatus | None = None) -> list[BatchJob]:
        """List all jobs"""
        jobs = list(self.jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    async def run_job(
        self,
        job_id: str,
        executor: Callable[[BatchTask], Awaitable[dict[str, Any]]],
        on_progress: Callable[[BatchTask], None] | None = None,
    ) -> BatchJob:
        """Run a batch job with async executor"""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        job.status = BatchStatus.RUNNING
        job.started_at = datetime.now().isoformat()

        semaphore = asyncio.Semaphore(self.max_workers)

        async def run_task(task: BatchTask):
            async with semaphore:
                if job.status == BatchStatus.CANCELLED:
                    task.status = TaskStatus.SKIPPED
                    return

                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now().isoformat()

                try:
                    result = await executor(task)
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    job.completed_tasks += 1

                except Exception as e:
                    task.error = str(e)
                    if task.retry_count < 3:
                        task.status = TaskStatus.RETRYING
                        task.retry_count += 1
                    else:
                        task.status = TaskStatus.FAILED
                        job.failed_tasks += 1

                task.completed_at = datetime.now().isoformat()

                if on_progress:
                    on_progress(task)

        await asyncio.gather(*[run_task(task) for task in job.tasks])

        if job.failed_tasks > 0 and job.completed_tasks == 0:
            job.status = BatchStatus.FAILED
        elif job.failed_tasks > 0:
            job.status = BatchStatus.COMPLETED  # Partial success
        else:
            job.status = BatchStatus.COMPLETED

        job.completed_at = datetime.now().isoformat()
        return job

    def run_job_sync(
        self,
        job_id: str,
        executor: Callable[[BatchTask], dict[str, Any]],
    ) -> BatchJob:
        """Run a batch job synchronously"""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        job.status = BatchStatus.RUNNING
        job.started_at = datetime.now().isoformat()

        for task in job.tasks:
            if job.status == BatchStatus.CANCELLED:
                task.status = TaskStatus.SKIPPED
                continue

            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now().isoformat()

            try:
                result = executor(task)
                task.result = result
                task.status = TaskStatus.COMPLETED
                job.completed_tasks += 1

            except Exception as e:
                task.error = str(e)
                task.status = TaskStatus.FAILED
                job.failed_tasks += 1

            task.completed_at = datetime.now().isoformat()

        if job.failed_tasks > 0:
            job.status = BatchStatus.FAILED
        else:
            job.status = BatchStatus.COMPLETED

        job.completed_at = datetime.now().isoformat()
        return job

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status == BatchStatus.RUNNING:
            job.status = BatchStatus.CANCELLED
            return True

        return False

    def get_job_summary(self, job_id: str) -> dict[str, Any]:
        """Get job summary"""
        job = self.jobs.get(job_id)
        if not job:
            return {}

        return {
            "id": job.id,
            "name": job.name,
            "status": job.status.value,
            "total": job.total_tasks,
            "completed": job.completed_tasks,
            "failed": job.failed_tasks,
            "progress": f"{job.completed_tasks}/{job.total_tasks}",
            "duration": self._calculate_duration(job),
        }

    def _calculate_duration(self, job: BatchJob) -> float:
        """Calculate job duration in seconds"""
        if not job.started_at:
            return 0

        end = job.completed_at or datetime.now().isoformat()
        start = datetime.fromisoformat(job.started_at)
        end_dt = datetime.fromisoformat(end)
        return (end_dt - start).total_seconds()


class BatchTemplates:
    """Pre-defined batch job templates"""

    @staticmethod
    def analyze_multiple_repos(repos: list[str]) -> list[dict[str, Any]]:
        """Template for analyzing multiple repos"""
        return [
            {
                "name": f"analyze-{repo}",
                "type": "analyze",
                "parameters": {"repo": repo},
            }
            for repo in repos
        ]

    @staticmethod
    def search_multiple_queries(queries: list[str]) -> list[dict[str, Any]]:
        """Template for searching multiple research queries"""
        return [
            {
                "name": f"search-{query[:20]}",
                "type": "search",
                "parameters": {"query": query},
            }
            for query in queries
        ]

    @staticmethod
    def validate_multiple_branches(
        repo: str,
        branches: list[str],
    ) -> list[dict[str, Any]]:
        """Template for validating multiple branches"""
        return [
            {
                "name": f"validate-{branch}",
                "type": "validate",
                "parameters": {"repo": repo, "branch": branch},
            }
            for branch in branches
        ]

    @staticmethod
    def generate_multiple_specs(
        repo: str,
        specs: list[str],
    ) -> list[dict[str, Any]]:
        """Template for generating patches for multiple specs"""
        return [
            {
                "name": f"generate-{spec}",
                "type": "generate",
                "parameters": {"repo": repo, "spec": spec},
            }
            for spec in specs
        ]


def quick_batch(
    name: str,
    tasks: list[dict[str, Any]],
    workers: int = 4,
) -> BatchJob:
    """Quick batch job creation"""
    processor = BatchProcessor(max_workers=workers)
    return processor.create_job(name, tasks)
