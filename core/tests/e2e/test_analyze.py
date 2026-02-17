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

from scholardevclaw.application.pipeline import run_analyze


class TestE2EAnalyze:
    def test_analyze_nanogpt_detects_python(self):
        repo_path = get_nanogpt_path()
        result = run_analyze(str(repo_path))

        assert result.ok is True
        assert result.title == "Repository Analysis"
        assert "python" in result.payload["languages"]

    def test_analyze_nanogpt_finds_pytorch(self):
        repo_path = get_nanogpt_path()
        result = run_analyze(str(repo_path))

        assert result.ok is True
        assert "python" in result.payload["languages"]

    def test_analyze_nanogpt_finds_entry_points(self):
        repo_path = get_nanogpt_path()
        result = run_analyze(str(repo_path))

        assert result.ok is True
        assert isinstance(result.payload["entry_points"], list)

    def test_analyze_nanogpt_finds_test_files(self):
        repo_path = get_nanogpt_path()
        result = run_analyze(str(repo_path))

        assert result.ok is True
        assert isinstance(result.payload["test_files"], list)

    def test_analyze_nanogpt_finds_patterns(self):
        repo_path = get_nanogpt_path()
        result = run_analyze(str(repo_path))

        assert result.ok is True
        assert isinstance(result.payload.get("patterns"), dict)

    def test_analyze_invalid_path_returns_error(self):
        result = run_analyze("/nonexistent/path/to/repo")

        assert result.ok is False
        assert result.error is not None
        assert "not found" in result.error.lower()
