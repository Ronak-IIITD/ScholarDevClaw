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

from scholardevclaw.application.pipeline import run_map


class TestE2EMap:
    def test_map_rmsnorm_to_nanogpt(self):
        repo_path = get_nanogpt_path()
        result = run_map(str(repo_path), "rmsnorm")

        assert result.ok is True
        assert result.payload["spec"] == "rmsnorm"
        assert result.payload["algorithm"] == "RMSNorm"
        assert result.payload["target_count"] >= 0

    def test_map_rmsnorm_finds_targets(self):
        repo_path = get_nanogpt_path()
        result = run_map(str(repo_path), "rmsnorm")

        assert result.ok is True
        if result.payload["target_count"] > 0:
            targets = result.payload["targets"]
            assert len(targets) > 0
            assert all("file" in t for t in targets)

    def test_map_rmsnorm_has_confidence(self):
        repo_path = get_nanogpt_path()
        result = run_map(str(repo_path), "rmsnorm")

        assert result.ok is True
        assert "confidence" in result.payload
        assert isinstance(result.payload["confidence"], (int, float))

    def test_map_rmsnorm_has_strategy(self):
        repo_path = get_nanogpt_path()
        result = run_map(str(repo_path), "rmsnorm")

        assert result.ok is True
        assert "strategy" in result.payload

    def test_map_invalid_spec_returns_error(self):
        repo_path = get_nanogpt_path()
        result = run_map(str(repo_path), "nonexistent_spec")

        assert result.ok is False
        assert result.error is not None
