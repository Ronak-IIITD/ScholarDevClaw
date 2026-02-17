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

from scholardevclaw.experiment import (
    generate_variants,
    _rank_results,
    _calculate_variant_score,
    ExperimentVariant,
)


class TestE2EExperiment:
    def test_generate_variants_returns_list(self):
        variants = generate_variants("rmsnorm", str(get_nanogpt_path()), variant_count=2)

        assert isinstance(variants, list)
        assert len(variants) >= 1

    def test_generate_variants_have_names(self):
        variants = generate_variants("rmsnorm", str(get_nanogpt_path()), variant_count=3)

        for v in variants:
            assert hasattr(v, "name")
            assert "variant" in v.name

    def test_generate_variants_have_parameters(self):
        variants = generate_variants("rmsnorm", str(get_nanogpt_path()), variant_count=3)

        for v in variants:
            assert hasattr(v, "parameters")
            assert isinstance(v.parameters, dict)

    def test_calculate_variant_score(self):
        metrics = {"speedup": 1.5, "loss_change": 2.0}
        score = _calculate_variant_score(metrics, 0.8)

        assert score > 0
        assert score <= 120

    def test_rank_results_sorts_by_score(self):
        results = [
            {"variant_name": "a", "score": 50, "status": "completed"},
            {"variant_name": "b", "score": 80, "status": "completed"},
            {"variant_name": "c", "score": 60, "status": "completed"},
        ]

        ranked = _rank_results(results)

        assert ranked[0]["variant_name"] == "b"
        assert ranked[0]["rank"] == 1
        assert ranked[1]["variant_name"] == "c"
        assert ranked[2]["variant_name"] == "a"

    def test_rank_results_handles_failures(self):
        results = [
            {"variant_name": "a", "score": 80, "status": "completed"},
            {"variant_name": "b", "score": 0, "status": "failed"},
        ]

        ranked = _rank_results(results)

        assert ranked[0]["variant_name"] == "a"
        assert ranked[0]["rank"] == 1
        assert ranked[1]["rank"] == 2

    def test_variants_have_confidence(self):
        variants = generate_variants("rmsnorm", str(get_nanogpt_path()), variant_count=3)

        for v in variants:
            assert hasattr(v, "confidence")
            assert 0 <= v.confidence <= 1

    def test_variants_have_expected_benefits(self):
        variants = generate_variants("rmsnorm", str(get_nanogpt_path()), variant_count=3)

        for v in variants:
            assert hasattr(v, "expected_benefits")
            assert isinstance(v.expected_benefits, list)
