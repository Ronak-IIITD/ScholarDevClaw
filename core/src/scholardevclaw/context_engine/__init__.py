from __future__ import annotations

from typing import Any, Callable

from .store import ContextStore, ProjectContext, IntegrationRecord, UserPreferences, ProjectMemory
from .brain import AgentBrain, BrainResult


def get_context_store(store_dir: str | None = None) -> ContextStore:
    return ContextStore(store_dir)


def get_agent_brain(store: ContextStore | None = None) -> AgentBrain:
    return AgentBrain(store)


class ContextEngine:
    def __init__(self, store_dir: str | None = None):
        self.store = ContextStore(store_dir)
        self.brain = AgentBrain(self.store)

    def initialize_project(self, repo_path: str, context_data: dict[str, Any]) -> None:
        self.store.update_context(repo_path, context_data)

    def record_integration(
        self,
        repo_path: str,
        spec: str,
        status: str,
        validation_passed: bool | None = None,
        confidence: float | None = None,
        error: str | None = None,
    ) -> None:
        self.store.add_integration(repo_path, spec, status, validation_passed, confidence, error)

        if status == "completed" and validation_passed:
            self.store.learn_from_integration(repo_path, spec, True)
        elif status == "failed":
            self.store.learn_from_integration(repo_path, spec, False)

    def get_recommendation(
        self,
        repo_path: str,
        available_specs: list[str],
    ) -> BrainResult:
        return self.brain.analyze_for_integration(repo_path, available_specs)

    def should_auto_approve(
        self,
        repo_path: str,
        spec: str,
        mapping_confidence: float,
    ) -> tuple[bool, str]:
        return self.brain.should_auto_approve(repo_path, spec, mapping_confidence)

    def get_validation_recommendation(self, repo_path: str) -> dict[str, Any]:
        return self.brain.get_validation_recommendation(repo_path)

    def set_preference(
        self,
        repo_path: str,
        preference_type: str,
        value: Any,
    ) -> None:
        self.brain.learn_preference(repo_path, preference_type, value)

    def get_context_summary(self, repo_path: str) -> dict[str, Any]:
        return self.brain.get_context_summary(repo_path)

    def get_integration_history(self, repo_path: str) -> list[dict[str, Any]]:
        records = self.store.get_integration_history(repo_path)
        return [
            {
                "run_id": r.run_id,
                "spec": r.spec,
                "status": r.status,
                "timestamp": r.timestamp,
                "validation_passed": r.validation_passed,
                "confidence": r.confidence,
                "error": r.error,
            }
            for r in records
        ]

    def list_tracked_projects(self) -> list[str]:
        return self.store.list_projects()

    def clear_project_memory(self, repo_path: str) -> None:
        self.store.clear_project(repo_path)
