from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[2]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from benchmarks.acceptance import (
    AcceptanceCase,
    AcceptanceResult,
    AcceptanceThresholds,
    apply_patch_payload,
    build_markdown_summary,
    build_report,
    load_manifest,
    run_case,
)


def _case(repo: Path, **overrides) -> AcceptanceCase:
    values = {
        "id": "demo-rmsnorm",
        "repository": "example/repo",
        "spec": "rmsnorm",
        "commit": "abc123",
        "local_path": repo,
        "test_command": [sys.executable, "-m", "compileall", "-q", "."],
    }
    values.update(overrides)
    return AcceptanceCase(**values)


def test_manifest_contains_five_v1_canaries():
    thresholds, cases = load_manifest()

    assert thresholds.patch_apply_rate == 0.9
    assert {case.spec for case in cases} == {
        "rmsnorm",
        "rope",
        "swiglu",
        "lora",
        "flashattention",
    }
    assert len(cases) == 5
    assert len({case.commit for case in cases}) == 1


def test_apply_patch_payload_rejects_stale_transformation(tmp_path: Path):
    target = tmp_path / "model.py"
    target.write_text("current = True\n")

    try:
        apply_patch_payload(
            tmp_path,
            {
                "transformations": [
                    {
                        "file": "model.py",
                        "original": "old = True\n",
                        "modified": "new = True\n",
                    }
                ]
            },
        )
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("Expected stale transformation to be rejected")


def test_run_case_applies_patch_compiles_and_runs_tests(tmp_path: Path):
    source_repo = tmp_path / "source"
    source_repo.mkdir()
    (source_repo / "model.py").write_text("value = 1\n")

    def generate(repo_path: Path, spec: str):
        assert spec == "rmsnorm"
        assert (repo_path / "model.py").read_text() == "value = 1\n"
        return (
            {"target_count": 1, "confidence": 91},
            {
                "new_files": [{"path": "rmsnorm.py", "content": "class RMSNorm:\n    pass\n"}],
                "transformations": [
                    {
                        "file": "model.py",
                        "original": "value = 1\n",
                        "modified": "value = 2\n",
                    }
                ],
            },
        )

    result = run_case(
        _case(source_repo),
        human_reviews={"demo-rmsnorm": "accept"},
        generate=generate,
    )

    assert result.mapping_ok
    assert result.patch_generated
    assert result.patch_applied
    assert result.compile_ok
    assert result.tests_status == "passed"
    assert result.human_review == "accepted"
    assert result.failure_stage is None
    assert (source_repo / "model.py").read_text() == "value = 1\n"


def test_run_case_records_compile_failure(tmp_path: Path):
    source_repo = tmp_path / "source"
    source_repo.mkdir()

    result = run_case(
        _case(source_repo, test_command=None),
        generate=lambda _repo, _spec: (
            {"target_count": 1, "confidence": 80},
            {"new_files": [{"path": "broken.py", "content": "def broken(:\n"}]},
        ),
    )

    assert result.patch_applied
    assert not result.compile_ok
    assert result.failure_stage == "compile"


def test_report_requires_test_and_human_review_coverage():
    thresholds = AcceptanceThresholds()
    results = [
        AcceptanceResult(
            id="a",
            repository="repo",
            spec="rmsnorm",
            commit="abc",
            patch_applied=True,
            tests_status="passed",
            human_review="accepted",
        ),
        AcceptanceResult(
            id="b",
            repository="repo",
            spec="rope",
            commit="abc",
            patch_applied=True,
        ),
    ]

    report = build_report(results, thresholds)
    summary = build_markdown_summary(report)

    assert report.patch_apply_rate == 1.0
    assert report.test_pass_rate == 1.0
    assert report.test_coverage_rate == 0.5
    assert report.human_acceptance_rate == 1.0
    assert report.human_review_coverage_rate == 0.5
    assert report.gate_status == "incomplete"
    assert "INCOMPLETE" in summary


def test_report_passes_when_all_thresholds_and_coverage_are_met():
    results = [
        AcceptanceResult(
            id=str(index),
            repository="repo",
            spec="rmsnorm",
            commit="abc",
            patch_applied=True,
            tests_status="passed",
            human_review="accepted" if index < 4 else "rejected",
        )
        for index in range(5)
    ]

    report = build_report(results, AcceptanceThresholds())
    payload = asdict(report)

    assert report.gate_status == "passed"
    assert report.human_acceptance_rate == 0.8
    assert payload["total_cases"] == 5
