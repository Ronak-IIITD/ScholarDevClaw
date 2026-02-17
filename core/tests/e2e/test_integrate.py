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

from scholardevclaw.application.pipeline import run_integrate


class TestE2EIntegrate:
    def test_integrate_dry_run_returns_plan(self):
        repo_path = get_nanogpt_path()
        result = run_integrate(str(repo_path), "rmsnorm", dry_run=True)

        assert result.ok is True
        assert result.payload["dry_run"] is True
        assert result.payload["generation"] is None
        assert result.payload["validation"] is None

    def test_integrate_dry_run_includes_analysis(self):
        repo_path = get_nanogpt_path()
        result = run_integrate(str(repo_path), "rmsnorm", dry_run=True)

        assert result.ok is True
        analysis = result.payload.get("analysis", {})
        assert "languages" in analysis
        assert "frameworks" in analysis

    def test_integrate_dry_run_includes_mapping(self):
        repo_path = get_nanogpt_path()
        result = run_integrate(str(repo_path), "rmsnorm", dry_run=True)

        assert result.ok is True
        mapping = result.payload.get("mapping", {})
        assert "targets" in mapping
        assert "strategy" in mapping

    def test_integrate_dry_run_includes_preflight(self):
        repo_path = get_nanogpt_path()
        result = run_integrate(str(repo_path), "rmsnorm", dry_run=True)

        assert result.ok is True
        assert "preflight" in result.payload
        assert result.payload["preflight"]["repo_exists"] is True

    def test_integrate_with_output_dir(self, tmp_path):
        repo_path = get_nanogpt_path()
        output_dir = tmp_path / "output"
        result = run_integrate(str(repo_path), "rmsnorm", dry_run=True, output_dir=str(output_dir))

        assert result.ok is True
        assert result.payload.get("output_dir") == str(output_dir)

    def test_integrate_require_clean_blocks_dirty_repo(self, tmp_path):
        repo_path = get_nanogpt_path()
        result = run_integrate(str(repo_path), "rmsnorm", require_clean=True)

        assert result.ok is False or result.payload.get("preflight", {}).get("is_clean") is True

    def test_integrate_full_workflow(self):
        repo_path = get_nanogpt_path()
        result = run_integrate(str(repo_path), "rmsnorm", dry_run=False)

        assert "validation" in result.payload or result.payload.get("generation") is not None

    def test_integrate_has_schema_metadata(self):
        repo_path = get_nanogpt_path()
        result = run_integrate(str(repo_path), "rmsnorm", dry_run=True)

        assert "_meta" in result.payload
        assert result.payload["_meta"]["payload_type"] == "integration"
        assert "schema_version" in result.payload["_meta"]

    def test_integrate_invalid_spec_returns_error(self):
        repo_path = get_nanogpt_path()
        result = run_integrate(str(repo_path), "nonexistent_spec", dry_run=True)

        assert result.ok is False
        assert result.error is not None
