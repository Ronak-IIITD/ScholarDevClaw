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

from scholardevclaw.application.pipeline import run_suggest


class TestE2ESuggest:
    def test_suggest_nanogpt_returns_suggestions(self):
        repo_path = get_nanogpt_path()
        result = run_suggest(str(repo_path))

        assert result.ok is True
        assert "suggestions" in result.payload

    def test_suggest_nanogpt_finds_rmsnorm_opportunity(self):
        repo_path = get_nanogpt_path()
        result = run_suggest(str(repo_path))

        assert result.ok is True
        suggestions = result.payload.get("suggestions", [])
        if len(suggestions) > 0:
            paper_names = [s.get("paper", {}).get("name", "") for s in suggestions]
            assert any("rmsnorm" in name.lower() for name in paper_names)

    def test_suggest_invalid_path_returns_error(self):
        result = run_suggest("/nonexistent/path/to/repo")

        assert result.ok is False
        assert result.error is not None
