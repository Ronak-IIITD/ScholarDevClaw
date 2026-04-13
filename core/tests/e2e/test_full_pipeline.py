from __future__ import annotations

from pathlib import Path

from scholardevclaw.application.pipeline import (
    run_analyze,
    run_generate,
    run_integrate,
    run_map,
    run_suggest,
    run_validate,
)


def test_full_pipeline_nanogpt_rmsnorm(nanogpt_repo_path: Path, tmp_path: Path) -> None:
    repo_path = str(nanogpt_repo_path)

    analyze_result = run_analyze(repo_path)
    assert analyze_result.ok is True
    assert "python" in analyze_result.payload.get("languages", [])

    suggest_result = run_suggest(repo_path)
    assert suggest_result.ok is True
    assert "suggestions" in suggest_result.payload
    assert isinstance(suggest_result.payload["suggestions"], list)

    map_result = run_map(repo_path, "rmsnorm")
    assert map_result.ok is True
    assert map_result.payload.get("spec") == "rmsnorm"
    assert map_result.payload.get("algorithm") == "RMSNorm"
    assert map_result.payload.get("target_count", 0) >= 1
    assert "confidence" in map_result.payload

    output_dir = tmp_path / "pipeline-generate-output"
    generate_result = run_generate(repo_path, "rmsnorm", output_dir=str(output_dir))
    assert generate_result.ok is True
    assert str(generate_result.payload.get("branch_name", "")).startswith("integration/")
    assert generate_result.payload.get("output_dir") is not None
    assert "transformations" in generate_result.payload

    validate_result = run_validate(repo_path, generate_result.payload)
    assert isinstance(validate_result.ok, bool)
    assert "passed" in validate_result.payload
    assert "stage" in validate_result.payload
    assert "comparison" in validate_result.payload
    assert "_meta" in validate_result.payload
    assert validate_result.payload["_meta"].get("payload_type") == "validation"

    integrate_result = run_integrate(
        repo_path,
        "rmsnorm",
        dry_run=False,
        create_rollback=False,
    )
    assert "_meta" in integrate_result.payload
    assert integrate_result.payload["_meta"].get("payload_type") == "integration"

    full_validation_keys = {
        "generation",
        "validation",
        "quality_gates",
        "diff_evidence",
        "validation_provenance",
        "hunk_review",
    }
    has_full_validation_path = full_validation_keys.issubset(integrate_result.payload)
    has_actionable_failure_path = integrate_result.payload.get("step") in {
        "patch_apply",
        "quality_gate",
        "approval_gate",
        "generate",
        "preflight",
    }
    assert has_full_validation_path or has_actionable_failure_path
