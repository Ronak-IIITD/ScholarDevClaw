from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NANOGPT_REPO = ROOT / "test_repos" / "nanogpt"


def get_nanogpt_path() -> Path:
    if not NANOGPT_REPO.exists():
        raise RuntimeError(
            f"nanoGPT not found at {NANOGPT_REPO}. "
            "Run: git clone https://github.com/karpathy/nanoGPT.git test_repos/nanogpt"
        )
    return NANOGPT_REPO


import pytest

from scholardevclaw.context_engine import ContextEngine, get_agent_brain


class TestE2EContext:
    def test_context_init_stores_project_info(self):
        engine = ContextEngine()
        repo_path = str(get_nanogpt_path())

        engine.initialize_project(
            repo_path,
            {
                "languages": ["python"],
                "frameworks": ["pytorch"],
                "entry_points": ["train.py"],
                "patterns": {},
            },
        )

        summary = engine.get_context_summary(repo_path)

        assert summary["languages"] == ["python"]
        assert summary["frameworks"] == ["pytorch"]

    def test_context_records_integration(self):
        engine = ContextEngine()
        repo_path = str(get_nanogpt_path())

        engine.record_integration(
            repo_path,
            "rmsnorm",
            "completed",
            validation_passed=True,
            confidence=0.9,
        )

        history = engine.get_integration_history(repo_path)

        assert len(history) >= 1
        assert history[-1]["spec"] == "rmsnorm"
        assert history[-1]["validation_passed"] is True

    def test_brain_recommends_successful_specs(self):
        engine = ContextEngine()
        repo_path = str(get_nanogpt_path())

        engine.record_integration(
            repo_path,
            "rmsnorm",
            "completed",
            validation_passed=True,
        )

        result = engine.get_recommendation(repo_path, ["rmsnorm", "swiglu", "flashattention"])

        assert "rmsnorm" in result.recommendation.lower()

    def test_brain_learns_from_failures(self):
        engine = ContextEngine()
        repo_path = str(get_nanogpt_path()) + "_fail_test"

        engine.record_integration(
            repo_path,
            "nonexistent_spec",
            "failed",
            validation_passed=False,
        )

        summary = engine.get_context_summary(repo_path)

        assert "nonexistent_spec" in summary["stats"]["failed_specs"]

    def test_should_auto_approve_high_confidence(self):
        engine = ContextEngine()
        repo_path = str(get_nanogpt_path())

        should_approve, reason = engine.should_auto_approve(repo_path, "rmsnorm", 0.95)

        assert isinstance(should_approve, bool)

    def test_get_validation_recommendation(self):
        engine = ContextEngine()
        repo_path = str(get_nanogpt_path())

        recommendation = engine.get_validation_recommendation(repo_path)

        assert "should_validate" in recommendation
        assert "reason" in recommendation

    def test_set_preference(self):
        engine = ContextEngine()
        repo_path = str(get_nanogpt_path()) + "_prefs_test"

        engine.set_preference(repo_path, "preferred_specs", "rmsnorm")

        summary = engine.get_context_summary(repo_path)

        assert "rmsnorm" in summary["preferences"]["preferred_specs"]

    def test_list_tracked_projects(self):
        engine = ContextEngine()
        repo_path = str(get_nanogpt_path()) + "_list_test"

        engine.initialize_project(repo_path, {"languages": ["python"]})

        projects = engine.list_tracked_projects()

        assert isinstance(projects, list)

    def test_clear_project_memory(self):
        engine = ContextEngine()
        repo_path = str(get_nanogpt_path()) + "_clear_test"

        engine.initialize_project(repo_path, {"languages": ["python"]})
        engine.clear_project_memory(repo_path)

        summary = engine.get_context_summary(repo_path)

        assert summary["stats"]["total_runs"] == 0

    def test_brain_provides_reasoning(self):
        brain = get_agent_brain()

        result = brain.analyze_for_integration(
            str(get_nanogpt_path()),
            ["rmsnorm", "swiglu"],
        )

        assert isinstance(result.reasoning, list)
        assert result.confidence >= 0
        assert result.confidence <= 1
