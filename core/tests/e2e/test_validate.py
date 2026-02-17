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

from scholardevclaw.application.pipeline import run_validate


class TestE2EValidate:
    def test_validate_nanogpt_returns_result(self):
        repo_path = get_nanogpt_path()
        result = run_validate(str(repo_path))

        assert result.ok is True
        assert "passed" in result.payload
        assert "stage" in result.payload
        assert "scorecard" in result.payload

    def test_validate_has_scorecard(self):
        repo_path = get_nanogpt_path()
        result = run_validate(str(repo_path))

        assert result.ok is True
        scorecard = result.payload["scorecard"]
        assert isinstance(scorecard, dict)
        assert "summary" in scorecard
        assert "checks" in scorecard
        assert "highlights" in scorecard

    def test_validate_scorecard_has_version(self):
        repo_path = get_nanogpt_path()
        result = run_validate(str(repo_path))

        scorecard = result.payload["scorecard"]
        assert "version" in scorecard

    def test_validate_invalid_path_returns_error(self):
        result = run_validate("/nonexistent/path/to/repo")

        assert result.ok is False
        assert result.error is not None

    def test_validate_has_schema_metadata(self):
        repo_path = get_nanogpt_path()
        result = run_validate(str(repo_path))

        assert "_meta" in result.payload
        assert result.payload["_meta"]["payload_type"] == "validation"
        assert "schema_version" in result.payload["_meta"]
