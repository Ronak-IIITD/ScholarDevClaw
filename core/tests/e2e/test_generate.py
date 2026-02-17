from __future__ import annotations

import sys
from pathlib import Path
import tempfile

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

from scholardevclaw.application.pipeline import run_generate


class TestE2EGenerate:
    def test_generate_rmsnorm_for_nanogpt(self):
        repo_path = get_nanogpt_path()
        result = run_generate(str(repo_path), "rmsnorm")

        assert result.ok is True
        assert result.payload["spec"] == "rmsnorm"
        assert result.payload["algorithm"] == "RMSNorm"

    def test_generate_creates_branch_name(self):
        repo_path = get_nanogpt_path()
        result = run_generate(str(repo_path), "rmsnorm")

        assert result.ok is True
        assert "branch_name" in result.payload
        assert result.payload["branch_name"].startswith("integration/")

    def test_generate_returns_new_files(self):
        repo_path = get_nanogpt_path()
        result = run_generate(str(repo_path), "rmsnorm")

        assert result.ok is True
        assert "new_files" in result.payload
        assert isinstance(result.payload["new_files"], list)

    def test_generate_returns_transformations(self):
        repo_path = get_nanogpt_path()
        result = run_generate(str(repo_path), "rmsnorm")

        assert result.ok is True
        assert "transformations" in result.payload
        assert isinstance(result.payload["transformations"], list)

    def test_generate_writes_to_output_dir(self, tmp_path):
        repo_path = get_nanogpt_path()
        output_dir = tmp_path / "patches"
        result = run_generate(str(repo_path), "rmsnorm", output_dir=str(output_dir))

        assert result.ok is True
        assert result.payload.get("output_dir") is not None
        assert len(result.payload.get("written_files", [])) > 0

    def test_generate_invalid_spec_returns_error(self):
        repo_path = get_nanogpt_path()
        result = run_generate(str(repo_path), "nonexistent_spec")

        assert result.ok is False
        assert result.error is not None
