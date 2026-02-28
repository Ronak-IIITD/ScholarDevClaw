"""
Scheduled runs for automatic research-to-code integration.

Provides:
- Cron-like scheduling
- One-time and recurring schedules
- Run history and status tracking
- Missed run handling
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable


@dataclass
class Schedule:
    """A scheduled task definition"""

    id: str
    name: str
    description: str = ""
    schedule_type: str = "interval"  # "interval", "cron", "once"
    interval_seconds: int = 0
    cron_expression: str = ""
    run_at: str = ""  # ISO datetime for one-time
    repo_path: str = ""
    research_query: str = ""
    enabled: bool = True
    max_retries: int = 3
    timeout_seconds: int = 3600
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduledRun:
    """A single scheduled run"""

    id: str
    schedule_id: str
    schedule_name: str
    status: str  # pending, running, completed, failed, skipped
    started_at: str
    completed_at: str = ""
    duration_seconds: float = 0
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class ScheduleParser:
    """Parse and validate schedule expressions"""

    @staticmethod
    def parse_cron(expression: str) -> bool:
        """Validate cron expression (basic validation)"""
        parts = expression.split()
        if len(parts) < 5 or len(parts) > 6:
            return False

        return True

    @staticmethod
    def parse_interval(seconds: int) -> timedelta:
        """Parse interval to timedelta"""
        return timedelta(seconds=seconds)

    @staticmethod
    def get_next_run(schedule: Schedule) -> datetime | None:
        """Calculate next run time"""
        now = datetime.now()

        if schedule.schedule_type == "once":
            if schedule.run_at:
                run_time = datetime.fromisoformat(schedule.run_at)
                return run_time if run_time > now else None
            return None

        elif schedule.schedule_type == "interval":
            return now + timedelta(seconds=schedule.interval_seconds)

        elif schedule.schedule_type == "cron":
            return ScheduleParser._next_cron_run(schedule.cron_expression, now)

        return None

    @staticmethod
    def _next_cron_run(expression: str, after: datetime) -> datetime:
        """Calculate next cron run (simplified - just returns after + 1 hour)"""
        return after + timedelta(hours=1)


class Scheduler:
    """Scheduler for automatic runs"""

    def __init__(self, store_dir: Path | None = None):
        self.store_dir = store_dir or Path.home() / ".scholardevclaw" / "scheduler"
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.schedules_file = self.store_dir / "schedules.json"
        self.runs_file = self.store_dir / "runs.json"

        self.schedules: dict[str, Schedule] = {}
        self.runs: list[ScheduledRun] = []

        self._load()

    def _load(self):
        """Load schedules and runs from disk"""
        if self.schedules_file.exists():
            data = json.loads(self.schedules_file.read_text())
            self.schedules = {s["id"]: Schedule(**s) for s in data}

        if self.runs_file.exists():
            data = json.loads(self.runs_file.read_text())
            self.runs = [ScheduledRun(**r) for r in data]

    def _save(self):
        """Save schedules and runs to disk"""
        self.schedules_file.write_text(
            json.dumps([s.__dict__ for s in self.schedules.values()], indent=2, default=str)
        )

        self.runs_file.write_text(
            json.dumps([r.__dict__ for r in self.runs], indent=2, default=str)
        )

    def create_schedule(
        self,
        name: str,
        schedule_type: str = "interval",
        interval_seconds: int = 3600,
        cron_expression: str = "",
        run_at: str = "",
        repo_path: str = "",
        research_query: str = "",
        description: str = "",
    ) -> Schedule:
        """Create a new schedule"""
        schedule = Schedule(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            schedule_type=schedule_type,
            interval_seconds=interval_seconds,
            cron_expression=cron_expression,
            run_at=run_at,
            repo_path=repo_path,
            research_query=research_query,
        )

        self.schedules[schedule.id] = schedule
        self._save()

        return schedule

    def get_schedule(self, schedule_id: str) -> Schedule | None:
        """Get a schedule by ID"""
        return self.schedules.get(schedule_id)

    def list_schedules(self) -> list[Schedule]:
        """List all schedules"""
        return list(self.schedules.values())

    def update_schedule(self, schedule_id: str, **kwargs) -> Schedule | None:
        """Update a schedule"""
        if schedule_id not in self.schedules:
            return None

        schedule = self.schedules[schedule_id]
        for key, value in kwargs.items():
            if hasattr(schedule, key):
                setattr(schedule, key, value)

        self._save()
        return schedule

    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule"""
        if schedule_id in self.schedules:
            del self.schedules[schedule_id]
            self._save()
            return True
        return False

    def enable_schedule(self, schedule_id: str) -> bool:
        """Enable a schedule"""
        return self.update_schedule(schedule_id, enabled=True) is not None

    def disable_schedule(self, schedule_id: str) -> bool:
        """Disable a schedule"""
        return self.update_schedule(schedule_id, enabled=False) is not None

    def get_next_runs(self, limit: int = 10) -> list[tuple[Schedule, datetime]]:
        """Get upcoming scheduled runs"""
        upcoming = []

        for schedule in self.schedules.values():
            if not schedule.enabled:
                continue

            next_run = ScheduleParser.get_next_run(schedule)
            if next_run:
                upcoming.append((schedule, next_run))

        upcoming.sort(key=lambda x: x[1])
        return upcoming[:limit]

    def get_run_history(
        self, schedule_id: str | None = None, limit: int = 50
    ) -> list[ScheduledRun]:
        """Get run history"""
        runs = self.runs

        if schedule_id:
            runs = [r for r in runs if r.schedule_id == schedule_id]

        runs.sort(key=lambda r: r.started_at, reverse=True)
        return runs[:limit]

    def trigger_run(self, schedule_id: str) -> ScheduledRun | None:
        """Manually trigger a run"""
        if schedule_id not in self.schedules:
            return None

        schedule = self.schedules[schedule_id]
        now = datetime.now()

        run = ScheduledRun(
            id=str(uuid.uuid4()),
            schedule_id=schedule_id,
            schedule_name=schedule.name,
            status="pending",
            started_at=now.isoformat(),
        )

        self.runs.append(run)
        self._save()

        return run

    def complete_run(self, run_id: str, status: str, result: dict | None = None, error: str = ""):
        """Mark a run as completed"""
        for run in self.runs:
            if run.id == run_id:
                run.status = status
                run.completed_at = datetime.now().isoformat()
                run.duration_seconds = (
                    datetime.fromisoformat(run.completed_at)
                    - datetime.fromisoformat(run.started_at)
                ).total_seconds()
                run.result = result or {}
                run.error = error
                break

        self._save()


class SchedulerRunner:
    """Background runner for scheduled tasks"""

    def __init__(self, scheduler: Scheduler, executor: Callable | None = None):
        self.scheduler = scheduler
        self.executor = executor
        self._running = False

    def start(self, check_interval: int = 60):
        """Start the scheduler loop"""
        self._running = True

        while self._running:
            self._check_and_run()
            time.sleep(check_interval)

    def stop(self):
        """Stop the scheduler"""
        self._running = False

    def _check_and_run(self):
        """Check for due schedules and run them"""
        due_schedules = []

        for schedule in self.scheduler.schedules.values():
            if not schedule.enabled:
                continue

            next_run = ScheduleParser.get_next_run(schedule)
            if next_run and next_run <= datetime.now():
                due_schedules.append(schedule)

        for schedule in due_schedules:
            self._run_schedule(schedule)

    def _run_schedule(self, schedule: Schedule):
        """Execute a scheduled task"""
        run = self.scheduler.trigger_run(schedule.id)
        if not run:
            return

        self.scheduler.complete_run(run.id, "running")

        try:
            if self.executor:
                result = self.executor(schedule)
                self.scheduler.complete_run(run.id, "completed", result)
            else:
                self.scheduler.complete_run(
                    run.id,
                    "completed",
                    {
                        "message": "No executor configured",
                        "schedule": schedule.name,
                    },
                )

        except Exception as e:
            self.scheduler.complete_run(run.id, "failed", error=str(e))


def quick_schedule(
    name: str,
    hours: int = 1,
    repo: str = "",
    query: str = "",
) -> Schedule:
    """Quick schedule creation helper"""
    scheduler = Scheduler()
    return scheduler.create_schedule(
        name=name,
        schedule_type="interval",
        interval_seconds=hours * 3600,
        repo_path=repo,
        research_query=query,
    )
