"""Comprehensive tests for the planner module (planner/__init__.py).

Covers:
  - PlannerResult dataclass
  - run_planner() — success with suggestions, fallback to available specs,
    target_categories filter, repo not found, exception handling
  - _order_specs_by_dependency() — single spec, multiple specs, unknown categories
  - _estimate_combined_impact() — known specs, unknown specs, cap at max
  - _summarize_improvement() — various inputs
  - _log() helper
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scholardevclaw.planner import (
    PlannerResult,
    run_planner,
    _order_specs_by_dependency,
    _estimate_combined_impact,
    _summarize_improvement,
    _log,
)


# =========================================================================
# PlannerResult dataclass
# =========================================================================


class TestPlannerResult:
    def test_construction(self):
        pr = PlannerResult(ok=True, title="Planner", payload={"x": 1}, logs=["log1"])
        assert pr.ok is True
        assert pr.title == "Planner"
        assert pr.payload["x"] == 1
        assert len(pr.logs) == 1
        assert pr.error is None

    def test_with_error(self):
        pr = PlannerResult(ok=False, title="Planner", payload={}, logs=[], error="boom")
        assert pr.ok is False
        assert pr.error == "boom"


# =========================================================================
# _log helper
# =========================================================================


class TestLog:
    def test_appends_to_list(self):
        logs: list[str] = []
        _log(logs, "hello")
        assert logs == ["hello"]

    def test_calls_callback(self):
        logs: list[str] = []
        captured: list[str] = []
        _log(logs, "hello", log_callback=captured.append)
        assert logs == ["hello"]
        assert captured == ["hello"]

    def test_no_callback(self):
        logs: list[str] = []
        _log(logs, "hello", log_callback=None)
        assert logs == ["hello"]


# =========================================================================
# _order_specs_by_dependency
# =========================================================================


class TestOrderSpecsByDependency:
    def test_single_spec(self):
        specs = [{"name": "rmsnorm", "category": "normalization"}]
        ordered, reasoning = _order_specs_by_dependency(specs)
        assert len(ordered) == 1
        assert ordered[0]["name"] == "rmsnorm"
        assert len(reasoning) == 0  # no reasoning for single spec

    def test_multiple_specs_sorted_by_category(self):
        specs = [
            {"name": "flashattention", "category": "attention"},
            {"name": "rmsnorm", "category": "normalization"},
            {"name": "swiglu", "category": "activation"},
        ]
        ordered, reasoning = _order_specs_by_dependency(specs)
        assert ordered[0]["name"] == "rmsnorm"  # normalization first
        assert ordered[1]["name"] == "swiglu"  # activation second
        assert ordered[2]["name"] == "flashattention"  # attention third
        assert len(reasoning) >= 3

    def test_unknown_categories_sorted_last(self):
        specs = [
            {"name": "unknown", "category": "exotic"},
            {"name": "rmsnorm", "category": "normalization"},
        ]
        ordered, _ = _order_specs_by_dependency(specs)
        assert ordered[0]["name"] == "rmsnorm"
        assert ordered[1]["name"] == "unknown"

    def test_empty_specs(self):
        ordered, reasoning = _order_specs_by_dependency([])
        assert ordered == []
        assert reasoning == []

    def test_two_specs_gives_reasoning(self):
        specs = [
            {"name": "a", "category": "normalization"},
            {"name": "b", "category": "attention"},
        ]
        ordered, reasoning = _order_specs_by_dependency(specs)
        assert len(reasoning) >= 1
        assert "normalization" in reasoning[0].lower() or "a" in reasoning[0]

    def test_position_encoding_comes_after_attention(self):
        specs = [
            {"name": "rope", "category": "position_encoding"},
            {"name": "flash", "category": "attention"},
        ]
        ordered, _ = _order_specs_by_dependency(specs)
        assert ordered[0]["name"] == "flash"
        assert ordered[1]["name"] == "rope"


# =========================================================================
# _estimate_combined_impact
# =========================================================================


class TestEstimateCombinedImpact:
    def test_known_specs(self):
        specs = [
            {"name": "rmsnorm", "expected_benefits": ["faster training"]},
            {"name": "flashattention", "expected_benefits": ["lower memory"]},
        ]
        result = _estimate_combined_impact(specs)
        assert result["speedup_estimate"] > 0
        assert result["memory_estimate"] > 0
        assert "faster training" in result["benefit_categories"]
        assert "lower memory" in result["benefit_categories"]

    def test_unknown_spec_uses_default(self):
        specs = [{"name": "totally_novel", "expected_benefits": []}]
        result = _estimate_combined_impact(specs)
        assert result["speedup_estimate"] == 0.05  # default
        assert result["memory_estimate"] == 0.02  # default

    def test_speedup_capped_at_half(self):
        specs = [
            {"name": "rmsnorm", "expected_benefits": []},
            {"name": "swiglu", "expected_benefits": []},
            {"name": "flashattention", "expected_benefits": []},
            {"name": "rope", "expected_benefits": []},
        ] + [{"name": f"extra{i}", "expected_benefits": []} for i in range(20)]
        result = _estimate_combined_impact(specs)
        assert result["speedup_estimate"] <= 0.5

    def test_memory_capped_at_0_4(self):
        specs = [{"name": f"spec{i}", "expected_benefits": []} for i in range(50)]
        result = _estimate_combined_impact(specs)
        assert result["memory_estimate"] <= 0.4

    def test_empty_specs(self):
        result = _estimate_combined_impact([])
        assert result["speedup_estimate"] == 0.0
        assert result["memory_estimate"] == 0.0
        assert result["benefit_categories"] == []

    def test_benefit_categories_deduplicated(self):
        specs = [
            {"name": "a", "expected_benefits": ["faster"]},
            {"name": "b", "expected_benefits": ["faster", "smaller"]},
        ]
        result = _estimate_combined_impact(specs)
        # set ensures dedup, but result is a list
        assert len(result["benefit_categories"]) == 2


# =========================================================================
# _summarize_improvement
# =========================================================================


class TestSummarizeImprovement:
    def test_basic_summary(self):
        combined = {
            "speedup_estimate": 0.15,
            "memory_estimate": 0.10,
            "benefit_categories": ["a", "b", "c"],
        }
        summary = _summarize_improvement(combined)
        assert "15%" in summary
        assert "10%" in summary
        assert "3 benefit areas" in summary

    def test_zero_values(self):
        combined = {
            "speedup_estimate": 0.0,
            "memory_estimate": 0.0,
            "benefit_categories": [],
        }
        summary = _summarize_improvement(combined)
        assert "0%" in summary
        assert "0 benefit areas" in summary


# =========================================================================
# run_planner() — integration (with monkeypatched dependencies)
# =========================================================================


def _install_fake_analyzer(monkeypatch):
    module = ModuleType("scholardevclaw.repo_intelligence.tree_sitter_analyzer")

    class FakeAnalysis:
        def __init__(self):
            self.languages = ["python"]
            self.elements = [SimpleNamespace(name="x")]
            self.frameworks = ["pytorch"]
            self.entry_points = ["train.py"]
            self.patterns = {"normalization": ["model.py:10"]}

    class FakeAnalyzer:
        def __init__(self, repo_path):
            self.repo_path = repo_path

        def analyze(self):
            return FakeAnalysis()

        def suggest_research_papers(self):
            return [
                {"paper": {"name": "rmsnorm"}, "confidence": 92.0},
                {"paper": {"name": "swiglu"}, "confidence": 85.0},
            ]

    module.TreeSitterAnalyzer = FakeAnalyzer
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _install_fake_extractor(monkeypatch):
    module = ModuleType("scholardevclaw.research_intelligence.extractor")

    class FakeExtractor:
        def list_available_specs(self):
            return ["rmsnorm", "swiglu", "flashattention"]

        def get_categories(self):
            return {"normalization": ["rmsnorm"], "activation": ["swiglu"]}

        def get_spec(self, name):
            specs = {
                "rmsnorm": {
                    "algorithm": {
                        "name": "RMSNorm",
                        "category": "normalization",
                        "replaces": "LayerNorm",
                    },
                    "changes": {"expected_benefits": ["faster training"]},
                },
                "swiglu": {
                    "algorithm": {"name": "SwiGLU", "category": "activation", "replaces": "GELU"},
                    "changes": {"expected_benefits": ["better FFN"]},
                },
                "flashattention": {
                    "algorithm": {
                        "name": "FlashAttention",
                        "category": "attention",
                        "replaces": "vanilla attention",
                    },
                    "changes": {"expected_benefits": ["lower memory"]},
                },
            }
            return specs.get(name)

    module.ResearchExtractor = FakeExtractor
    monkeypatch.setitem(sys.modules, module.__name__, module)


class TestRunPlanner:
    def test_success_with_suggestions(self, monkeypatch, tmp_path):
        _install_fake_analyzer(monkeypatch)
        _install_fake_extractor(monkeypatch)

        result = run_planner(str(tmp_path))
        assert result.ok is True
        assert result.title == "Migration Planner"
        assert len(result.payload["selected_specs"]) >= 1
        assert result.payload["dependency_order"]
        assert result.payload["combined_benefits"]

    def test_repo_not_found(self, monkeypatch):
        _install_fake_analyzer(monkeypatch)
        _install_fake_extractor(monkeypatch)

        result = run_planner("/nonexistent/path/that/does/not/exist")
        assert result.ok is False
        assert "not found" in (result.error or "").lower()

    def test_max_specs_limit(self, monkeypatch, tmp_path):
        _install_fake_analyzer(monkeypatch)
        _install_fake_extractor(monkeypatch)

        result = run_planner(str(tmp_path), max_specs=1)
        assert result.ok is True
        assert len(result.payload["selected_specs"]) <= 1

    def test_target_categories_filter(self, monkeypatch, tmp_path):
        _install_fake_analyzer(monkeypatch)
        _install_fake_extractor(monkeypatch)

        result = run_planner(str(tmp_path), target_categories=["normalization"])
        assert result.ok is True
        for spec in result.payload["selected_specs"]:
            assert spec["category"] == "normalization"

    def test_log_callback(self, monkeypatch, tmp_path):
        _install_fake_analyzer(monkeypatch)
        _install_fake_extractor(monkeypatch)

        captured: list[str] = []
        result = run_planner(str(tmp_path), log_callback=captured.append)
        assert result.ok is True
        assert len(captured) > 0

    def test_exception_handling(self, monkeypatch, tmp_path):
        """If analyzer raises, planner catches and returns error."""
        module = ModuleType("scholardevclaw.repo_intelligence.tree_sitter_analyzer")

        class BrokenAnalyzer:
            def __init__(self, repo_path):
                raise RuntimeError("broken analyzer")

        module.TreeSitterAnalyzer = BrokenAnalyzer
        monkeypatch.setitem(sys.modules, module.__name__, module)
        _install_fake_extractor(monkeypatch)

        result = run_planner(str(tmp_path))
        assert result.ok is False
        assert "broken analyzer" in (result.error or "")
