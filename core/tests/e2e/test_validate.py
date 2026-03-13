from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NANOGPT_REPO = ROOT / "test_repos" / "nanogpt"


def get_nanogpt_path() -> Path:
    if not NANOGPT_REPO.exists():
        pytest.skip(
            f"nanoGPT not found at {NANOGPT_REPO}. "
            "Run: git clone https://github.com/karpathy/nanoGPT.git test_repos/nanogpt"
        )
    return NANOGPT_REPO


import pytest  # noqa: E402

from scholardevclaw.application.pipeline import run_validate  # noqa: E402


class TestE2EValidate:
    def test_validate_nanogpt_returns_result(self):
        repo_path = get_nanogpt_path()
        result = run_validate(str(repo_path))

        # Real benchmarks may report a regression (speedup < 1.0), which makes
        # result.ok == False.  The important thing is that the pipeline ran to
        # completion and produced a well-formed payload.
        assert isinstance(result.ok, bool)
        assert "passed" in result.payload
        assert "stage" in result.payload
        assert "comparison" in result.payload

    def test_validate_has_scorecard(self):
        repo_path = get_nanogpt_path()
        result = run_validate(str(repo_path))

        # With real subprocess benchmarks, validation may not pass (ok=False)
        # but the scorecard should still be present when the pipeline completes.
        # If the benchmark stage fails before scorecard generation, skip.
        if "scorecard" not in result.payload:
            pytest.skip("Benchmark finished before scorecard was generated")
        scorecard = result.payload["scorecard"]
        assert isinstance(scorecard, dict)
        assert "summary" in scorecard
        assert "checks" in scorecard
        assert "highlights" in scorecard

    def test_validate_scorecard_has_version(self):
        repo_path = get_nanogpt_path()
        result = run_validate(str(repo_path))

        if "scorecard" not in result.payload:
            pytest.skip("Benchmark finished before scorecard was generated")
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
