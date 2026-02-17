from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .store import ContextStore, ProjectMemory, UserPreferences


@dataclass
class BrainResult:
    ok: bool
    recommendation: str
    confidence: float
    reasoning: list[str]
    context_used: dict[str, Any]


class AgentBrain:
    def __init__(self, store: ContextStore | None = None):
        self.store = store or ContextStore()

    def analyze_for_integration(
        self,
        repo_path: str,
        available_specs: list[str],
    ) -> BrainResult:
        memory = self.store.load_memory(repo_path)

        reasoning = []
        recommended_specs = []
        confidence_multiplier = 1.0

        if memory.successful_patterns:
            reasoning.append(
                f"Previously successful specs: {', '.join(memory.successful_patterns[-3:])}"
            )
            for spec in available_specs:
                if spec in memory.successful_patterns:
                    recommended_specs.append((spec, 0.9))
                    confidence_multiplier *= 1.1

        if memory.failed_patterns:
            reasoning.append(f"Previously failed specs: {', '.join(memory.failed_patterns[-3:])}")

        stats = self.store.get_stats(repo_path)
        if stats["total_runs"] > 0:
            reasoning.append(
                f"Project success rate: {stats['success_rate']:.0%} ({stats['successful_runs']}/{stats['total_runs']} runs)"
            )
            if stats["success_rate"] > 0.8:
                confidence_multiplier *= 1.1

        prefs = memory.preferences
        if prefs.preferred_specs:
            reasoning.append(f"User preferred specs: {', '.join(prefs.preferred_specs)}")
            for spec in available_specs:
                if spec in prefs.preferred_specs and spec not in [s[0] for s in recommended_specs]:
                    recommended_specs.append((spec, 0.85))

        if prefs.preferred_categories:
            reasoning.append(f"User preferred categories: {', '.join(prefs.preferred_categories)}")

        recommended_specs.sort(key=lambda x: x[1], reverse=True)

        best_spec = (
            recommended_specs[0][0]
            if recommended_specs
            else available_specs[0]
            if available_specs
            else None
        )
        confidence = (0.7 * confidence_multiplier) if best_spec else 0.0
        confidence = min(confidence, 0.95)

        recommendation = f"Recommended: {best_spec}" if best_spec else "No recommendation available"
        if recommended_specs:
            alt_specs = [s[0] for s in recommended_specs[1:4]]
            if alt_specs:
                recommendation += f". Alternatives: {', '.join(alt_specs)}"

        return BrainResult(
            ok=True,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            context_used={
                "total_runs": stats["total_runs"],
                "success_rate": stats["success_rate"],
                "successful_specs": stats["successful_specs"],
                "failed_specs": stats["failed_specs"],
            },
        )

    def should_auto_approve(
        self,
        repo_path: str,
        spec: str,
        mapping_confidence: float,
    ) -> tuple[bool, str]:
        memory = self.store.load_memory(repo_path)
        prefs = memory.preferences

        if mapping_confidence >= prefs.auto_approve_confidence:
            return (
                True,
                f"Confidence {mapping_confidence:.0%} >= auto-approve threshold {prefs.auto_approve_confidence:.0%}",
            )

        if spec in memory.successful_patterns:
            return True, f"Spec '{spec}' was previously successful"

        stats = self.store.get_stats(repo_path)
        if stats["success_rate"] >= 0.9 and mapping_confidence >= 0.7:
            return True, f"High project success rate ({stats['success_rate']:.0%})"

        return False, "Manual approval recommended"

    def get_validation_recommendation(
        self,
        repo_path: str,
    ) -> dict[str, Any]:
        memory = self.store.load_memory(repo_path)
        prefs = memory.preferences

        if not prefs.require_validation:
            return {
                "should_validate": False,
                "reason": "User preference: validation disabled",
            }

        stats = self.store.get_stats(repo_path)

        if stats["total_runs"] == 0:
            return {
                "should_validate": True,
                "reason": "First run - validation recommended",
            }

        if stats["success_rate"] >= 0.9:
            return {
                "should_validate": False,
                "reason": f"High success rate ({stats['success_rate']:.0%}) - can skip validation",
            }

        return {
            "should_validate": True,
            "reason": f"Standard validation (success rate: {stats['success_rate']:.0%})",
        }

    def suggest_output_directory(self, repo_path: str) -> str | None:
        memory = self.store.load_memory(repo_path)
        prefs = memory.preferences

        return prefs.default_output_dir

    def get_context_summary(self, repo_path: str) -> dict[str, Any]:
        memory = self.store.load_memory(repo_path)
        stats = self.store.get_stats(repo_path)

        return {
            "repo_path": repo_path,
            "languages": memory.context.languages,
            "frameworks": memory.context.frameworks,
            "stats": stats,
            "preferences": {
                "preferred_specs": memory.preferences.preferred_specs,
                "preferred_categories": memory.preferences.preferred_categories,
                "require_validation": memory.preferences.require_validation,
            },
        }

    def learn_preference(
        self,
        repo_path: str,
        preference_type: str,
        value: Any,
    ) -> None:
        memory = self.store.load_memory(repo_path)

        if preference_type == "preferred_specs":
            if value not in memory.preferences.preferred_specs:
                memory.preferences.preferred_specs.append(value)

        elif preference_type == "preferred_categories":
            if value not in memory.preferences.preferred_categories:
                memory.preferences.preferred_categories.append(value)

        elif preference_type == "require_validation":
            memory.preferences.require_validation = value

        elif preference_type == "validation_threshold":
            memory.preferences.validation_threshold = float(value)

        elif preference_type == "auto_approve_confidence":
            memory.preferences.auto_approve_confidence = float(value)

        elif preference_type == "default_output_dir":
            memory.preferences.default_output_dir = str(value)

        self.store.save_memory(memory)
