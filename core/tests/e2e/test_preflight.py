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

from scholardevclaw.application.pipeline import run_preflight


class TestE2EPreflight:
    def test_preflight_nanogpt_passes(self):
        repo_path = get_nanogpt_path()
        result = run_preflight(str(repo_path))

        assert result.ok is True
        assert result.payload["repo_exists"] is True
        assert result.payload["repo_is_writable"] is True
        assert result.payload["python_file_count"] > 0

    def test_preflight_detects_git(self):
        repo_path = get_nanogpt_path()
        result = run_preflight(str(repo_path))

        assert result.ok is True
        assert result.payload["has_git_dir"] is True

    def test_preflight_require_clean_passes_for_clean_repo(self):
        repo_path = get_nanogpt_path()
        result = run_preflight(str(repo_path), require_clean=True)

        if result.payload.get("is_clean") is True:
            assert result.ok is True
        else:
            assert result.ok is False

    def test_preflight_invalid_path_returns_error(self):
        result = run_preflight("/nonexistent/path/to/repo")

        assert result.ok is False
        assert result.error is not None
