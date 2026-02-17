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

from scholardevclaw.planner import run_planner


class TestE2EPlanner:
    def test_planner_nanogpt_returns_plan(self):
        repo_path = get_nanogpt_path()
        result = run_planner(str(repo_path))

        assert result.ok is True
        assert "selected_specs" in result.payload
        assert len(result.payload["selected_specs"]) > 0

    def test_planner_returns_ordered_specs(self):
        repo_path = get_nanogpt_path()
        result = run_planner(str(repo_path))

        assert result.ok is True
        assert "dependency_order" in result.payload
        assert len(result.payload["dependency_order"]) > 0

    def test_planner_has_dependency_reasoning(self):
        repo_path = get_nanogpt_path()
        result = run_planner(str(repo_path))

        assert result.ok is True
        assert "dependency_reasoning" in result.payload
        assert len(result.payload["dependency_reasoning"]) > 0

    def test_planner_estimates_combined_impact(self):
        repo_path = get_nanogpt_path()
        result = run_planner(str(repo_path))

        assert result.ok is True
        assert "combined_benefits" in result.payload
        assert "speedup_estimate" in result.payload["combined_benefits"]
        assert "benefit_categories" in result.payload["combined_benefits"]

    def test_planner_has_total_improvement_summary(self):
        repo_path = get_nanogpt_path()
        result = run_planner(str(repo_path))

        assert result.ok is True
        assert "total_expected_improvement" in result.payload

    def test_planner_respects_max_specs(self):
        repo_path = get_nanogpt_path()
        result = run_planner(str(repo_path), max_specs=2)

        assert result.ok is True
        assert len(result.payload["selected_specs"]) <= 2

    def test_planner_filters_by_category(self):
        repo_path = get_nanogpt_path()
        result = run_planner(str(repo_path), target_categories=["normalization"])

        assert result.ok is True
        specs = result.payload["selected_specs"]
        if specs:
            for spec in specs:
                assert spec.get("category") == "normalization"

    def test_planner_invalid_path_returns_error(self):
        result = run_planner("/nonexistent/path/to/repo")

        assert result.ok is False
        assert result.error is not None
