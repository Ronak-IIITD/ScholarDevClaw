from __future__ import annotations

import json
from pathlib import Path

from scholardevclaw.execution.sandbox import ExecutionReport
from scholardevclaw.execution.scorer import ReproducibilityReport
from scholardevclaw.generation.models import GenerationResult, ModuleResult
from scholardevclaw.ingestion.models import Equation, PaperDocument, Section
from scholardevclaw.planning.models import CodeModule, ImplementationPlan
from scholardevclaw.product.traceability import TraceabilityReport
from scholardevclaw.product.trust_report import build_trust_report, write_paper_workflow_reports
from scholardevclaw.understanding.models import PaperUnderstanding


def test_write_paper_workflow_reports_exports_trust_and_traceability_artifacts(tmp_path: Path):
    project_dir = tmp_path / "demo_project"
    source_dir = project_dir / "src"
    tests_dir = project_dir / "tests"
    source_dir.mkdir(parents=True)
    tests_dir.mkdir()

    source_file = source_dir / "attention.py"
    source_file.write_text(
        "# Equation 1: Attention score\ndef attention_score(q, k):\n    return q @ k\n",
        encoding="utf-8",
    )
    (tests_dir / "test_attention.py").write_text(
        "def test_attention_score():\n    assert True\n",
        encoding="utf-8",
    )

    document = PaperDocument(
        title="Attention Demo",
        authors=["Ada"],
        arxiv_id="2401.00001",
        doi=None,
        year=2024,
        abstract="demo",
        sections=[Section(title="Method", level=1, content="body", page_start=1)],
        equations=[Equation(latex="qk", description="Attention score", page=1)],
        algorithms=[],
        figures=[],
    )
    understanding = PaperUnderstanding(
        paper_title="Attention Demo",
        one_line_summary="Implements a tiny attention scoring kernel.",
        complexity="medium",
        estimated_impl_hours=4,
        confidence=0.9,
    )
    plan = ImplementationPlan(
        project_name="demo_project",
        target_language="python",
        tech_stack="pytorch",
        modules=[
            CodeModule(
                id="attention",
                name="Attention",
                description="Attention score kernel",
                file_path="src/attention.py",
                test_file_path="tests/test_attention.py",
                estimated_lines=12,
            )
        ],
        estimated_total_lines=12,
    )
    generation_result = GenerationResult(
        plan=plan,
        module_results=[
            ModuleResult(
                module_id="attention",
                file_path="src/attention.py",
                test_file_path="tests/test_attention.py",
                code=source_file.read_text(encoding="utf-8"),
                test_code=(tests_dir / "test_attention.py").read_text(encoding="utf-8"),
                generation_attempts=1,
                final_errors=[],
                tokens_used=321,
            )
        ],
        output_dir=project_dir,
        success_rate=1.0,
        total_tokens_used=321,
        duration_seconds=1.25,
    )
    execution_report = ExecutionReport(
        exit_code=0,
        stdout="tests/test_attention.py::test_attention_score PASSED",
        stderr="",
        duration_seconds=0.8,
        peak_memory_mb=64.0,
        tests_passed=1,
        tests_failed=0,
        tests_errors=0,
        success=True,
    )
    reproducibility_report = ReproducibilityReport(
        paper_title="Attention Demo",
        claimed_metrics={"accuracy": 0.95},
        achieved_metrics={"accuracy": 0.95},
        delta={"accuracy": 0.0},
        score=0.95,
        verdict="reproduced",
    )

    artifacts = write_paper_workflow_reports(
        source="arxiv:2401.00001",
        document=document,
        understanding=understanding,
        plan=plan,
        generation_result=generation_result,
        execution_report=execution_report,
        reproducibility_report=reproducibility_report,
        project_dir=project_dir,
    )

    assert artifacts.trust_report.status == "trusted"
    assert artifacts.traceability_report.coverage_score == 1.0
    assert artifacts.trust_report_markdown_path.exists()
    assert artifacts.traceability_markdown_path.exists()

    trust_payload = json.loads(artifacts.trust_report_json_path.read_text(encoding="utf-8"))
    assert trust_payload["status"] == "trusted"
    assert trust_payload["traceability_coverage"] == 1.0
    assert trust_payload["artifacts"]["trust_report_markdown"] == "TRUST_REPORT.md"

    trust_markdown = artifacts.trust_report_markdown_path.read_text(encoding="utf-8")
    assert "# ScholarDevClaw Trust Report" in trust_markdown
    assert "`trusted`" in trust_markdown


def test_build_trust_report_adds_review_notes_for_low_reproducibility():
    document = PaperDocument(
        title="Sparse Demo",
        authors=["Ada"],
        arxiv_id=None,
        doi=None,
        year=2024,
        abstract="demo",
        sections=[],
        equations=[],
        algorithms=[],
        figures=[],
    )
    understanding = PaperUnderstanding(
        paper_title="Sparse Demo",
        one_line_summary="Sparse demo",
        complexity="high",
        estimated_impl_hours=10,
        confidence=0.4,
    )
    plan = ImplementationPlan(
        project_name="sparse_demo",
        target_language="python",
        tech_stack="pytorch",
        modules=[
            CodeModule(
                id="kernel",
                name="Kernel",
                description="kernel",
                file_path="src/kernel.py",
                test_file_path="tests/test_kernel.py",
                estimated_lines=50,
            )
        ],
        estimated_total_lines=50,
    )
    generation_result = GenerationResult(
        plan=plan,
        module_results=[],
        output_dir=Path("/tmp/sparse_demo"),
        success_rate=0.8,
        total_tokens_used=100,
        duration_seconds=2.0,
    )
    execution_report = ExecutionReport(
        exit_code=0,
        stdout="",
        stderr="",
        duration_seconds=1.0,
        peak_memory_mb=32.0,
        tests_passed=2,
        tests_failed=0,
        tests_errors=0,
        success=True,
    )
    reproducibility_report = ReproducibilityReport(
        paper_title="Sparse Demo",
        claimed_metrics={"accuracy": 0.9},
        achieved_metrics={"accuracy": 0.5},
        delta={"accuracy": -0.4},
        score=0.4,
        verdict="failed",
    )
    traceability_report = TraceabilityReport(
        paper_title="Sparse Demo",
        paper_id="manual",
        implementation_dir="/tmp/sparse_demo",
        total_equations=0,
        mapped_equations=0,
        unmapped_equations=[],
        mappings=[],
        coverage_score=0.0,
        generated_at="2026-01-01T00:00:00+00:00",
    )

    report = build_trust_report(
        source="manual",
        document=document,
        understanding=understanding,
        plan=plan,
        generation_result=generation_result,
        execution_report=execution_report,
        reproducibility_report=reproducibility_report,
        traceability_report=traceability_report,
        artifacts={"trust_report_markdown": "TRUST_REPORT.md"},
    )

    assert report.status == "needs_review"
    assert any("Reproducibility is 40%" in note for note in report.notes)
    assert any("No equations were extracted" in note for note in report.notes)
