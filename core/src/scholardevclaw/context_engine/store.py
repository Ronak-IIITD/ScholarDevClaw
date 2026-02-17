from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ProjectContext:
    repo_path: str
    languages: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    patterns: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    run_count: int = 0


@dataclass
class IntegrationRecord:
    run_id: str
    spec: str
    status: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    validation_passed: bool | None = None
    confidence: float | None = None
    error: str | None = None


@dataclass
class UserPreferences:
    preferred_specs: list[str] = field(default_factory=list)
    preferred_categories: list[str] = field(default_factory=list)
    validation_threshold: float = 0.7
    auto_approve_confidence: float = 0.9
    require_validation: bool = True
    default_output_dir: str | None = None


@dataclass
class ProjectMemory:
    project_hash: str
    context: ProjectContext
    integrations: list[IntegrationRecord] = field(default_factory=list)
    preferences: UserPreferences = field(default_factory=UserPreferences)
    successful_patterns: list[str] = field(default_factory=list)
    failed_patterns: list[str] = field(default_factory=list)
    total_runs: int = 0
    successful_runs: int = 0


class ContextStore:
    def __init__(self, store_dir: str | None = None):
        if store_dir:
            self.store_dir = Path(store_dir)
        else:
            default_dir = os.environ.get("SCHOLARDEVCLAW_CONTEXT_DIR")
            if default_dir:
                self.store_dir = Path(default_dir)
            else:
                self.store_dir = Path.home() / ".scholardevclaw" / "context"

        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _get_project_hash(self, repo_path: str) -> str:
        return hashlib.sha256(Path(repo_path).resolve().as_posix().encode()).hexdigest()[:16]

    def _get_memory_path(self, project_hash: str) -> Path:
        return self.store_dir / f"{project_hash}.json"

    def load_memory(self, repo_path: str) -> ProjectMemory:
        project_hash = self._get_project_hash(repo_path)
        memory_path = self._get_memory_path(project_hash)

        if not memory_path.exists():
            return ProjectMemory(
                project_hash=project_hash,
                context=ProjectContext(repo_path=repo_path),
            )

        try:
            with open(memory_path) as f:
                data = json.load(f)

            return ProjectMemory(
                project_hash=data.get("project_hash", project_hash),
                context=ProjectContext(**data.get("context", {"repo_path": repo_path})),
                integrations=[IntegrationRecord(**ir) for ir in data.get("integrations", [])],
                preferences=UserPreferences(**data.get("preferences", {})),
                successful_patterns=data.get("successful_patterns", []),
                failed_patterns=data.get("failed_patterns", []),
                total_runs=data.get("total_runs", 0),
                successful_runs=data.get("successful_runs", 0),
            )
        except (json.JSONDecodeError, KeyError):
            return ProjectMemory(
                project_hash=project_hash,
                context=ProjectContext(repo_path=repo_path),
            )

    def save_memory(self, memory: ProjectMemory) -> None:
        memory_path = self._get_memory_path(memory.project_hash)

        data = {
            "project_hash": memory.project_hash,
            "context": asdict(memory.context),
            "integrations": [asdict(ir) for ir in memory.integrations],
            "preferences": asdict(memory.preferences),
            "successful_patterns": memory.successful_patterns,
            "failed_patterns": memory.failed_patterns,
            "total_runs": memory.total_runs,
            "successful_runs": memory.successful_runs,
        }

        with open(memory_path, "w") as f:
            json.dump(data, f, indent=2)

    def update_context(self, repo_path: str, context_data: dict[str, Any]) -> ProjectMemory:
        memory = self.load_memory(repo_path)

        memory.context.languages = context_data.get("languages", memory.context.languages)
        memory.context.frameworks = context_data.get("frameworks", memory.context.frameworks)
        memory.context.entry_points = context_data.get("entry_points", memory.context.entry_points)
        memory.context.patterns = context_data.get("patterns", memory.context.patterns)
        memory.context.last_updated = datetime.now().isoformat()
        memory.context.run_count += 1

        memory.total_runs += 1

        self.save_memory(memory)
        return memory

    def add_integration(
        self,
        repo_path: str,
        spec: str,
        status: str,
        validation_passed: bool | None = None,
        confidence: float | None = None,
        error: str | None = None,
    ) -> ProjectMemory:
        memory = self.load_memory(repo_path)

        record = IntegrationRecord(
            run_id=memory.context.repo_path + f"_{len(memory.integrations)}",
            spec=spec,
            status=status,
            validation_passed=validation_passed,
            confidence=confidence,
            error=error,
        )

        memory.integrations.append(record)

        if validation_passed:
            memory.successful_runs += 1
            if spec not in memory.successful_patterns:
                memory.successful_patterns.append(spec)
        elif status == "failed":
            if spec not in memory.failed_patterns:
                memory.failed_patterns.append(spec)

        self.save_memory(memory)
        return memory

    def get_integration_history(self, repo_path: str) -> list[IntegrationRecord]:
        memory = self.load_memory(repo_path)
        return memory.integrations

    def get_preferences(self, repo_path: str) -> UserPreferences:
        memory = self.load_memory(repo_path)
        return memory.preferences

    def update_preferences(self, repo_path: str, preferences: UserPreferences) -> ProjectMemory:
        memory = self.load_memory(repo_path)
        memory.preferences = preferences
        self.save_memory(memory)
        return memory

    def learn_from_integration(
        self,
        repo_path: str,
        spec: str,
        success: bool,
    ) -> ProjectMemory:
        memory = self.load_memory(repo_path)

        if success:
            if spec not in memory.successful_patterns:
                memory.successful_patterns.append(spec)
        else:
            if spec not in memory.failed_patterns:
                memory.failed_patterns.append(spec)

        self.save_memory(memory)
        return memory

    def get_stats(self, repo_path: str) -> dict[str, Any]:
        memory = self.load_memory(repo_path)

        return {
            "total_runs": memory.total_runs,
            "successful_runs": memory.successful_runs,
            "success_rate": memory.successful_runs / memory.total_runs
            if memory.total_runs > 0
            else 0,
            "unique_specs": len(set(ir.spec for ir in memory.integrations)),
            "successful_specs": memory.successful_patterns,
            "failed_specs": memory.failed_patterns,
        }

    def list_projects(self) -> list[str]:
        projects = []
        for f in self.store_dir.glob("*.json"):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    if "context" in data:
                        projects.append(data["context"].get("repo_path", "unknown"))
            except (json.JSONDecodeError, KeyError):
                continue
        return projects

    def clear_project(self, repo_path: str) -> None:
        project_hash = self._get_project_hash(repo_path)
        memory_path = self._get_memory_path(project_hash)
        if memory_path.exists():
            memory_path.unlink()
