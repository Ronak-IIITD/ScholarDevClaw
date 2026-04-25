from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scholardevclaw.execution.sandbox import ExecutionReport
from scholardevclaw.execution.scorer import ReproducibilityReport
from scholardevclaw.generation.models import GenerationResult, ModuleResult
from scholardevclaw.ingestion.models import PaperDocument
from scholardevclaw.planning.models import CodeModule, ImplementationPlan
from scholardevclaw.product.traceability import (
    TraceabilityBuilder,
    TraceabilityReport,
    export_traceability_markdown,
)
from scholardevclaw.understanding.models import PaperUnderstanding

TRUSTED_REPRODUCIBILITY_THRESHOLD = 0.75
REVIEW_TRACEABILITY_THRESHOLD = 0.50


@dataclass(slots=True)
class ModuleTrustSummary:
    module_id: str
    file_path: str
    test_file_path: str
    estimated_lines: int
    generation_attempts: int
    tokens_used: int
    status: str
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "file_path": self.file_path,
            "test_file_path": self.test_file_path,
            "estimated_lines": self.estimated_lines,
            "generation_attempts": self.generation_attempts,
            "tokens_used": self.tokens_used,
            "status": self.status,
            "errors": list(self.errors),
        }


@dataclass(slots=True)
class PaperWorkflowTrustReport:
    source: str
    paper_title: str
    paper_id: str
    project_name: str
    generated_at: str
    status: str
    one_line_summary: str
    tech_stack: str
    target_language: str
    understanding_confidence: float
    complexity: str
    estimated_impl_hours: int
    equation_count: int
    algorithm_count: int
    module_count: int
    generated_module_count: int
    failed_module_count: int
    generation_success_rate: float
    generation_duration_seconds: float
    total_tokens_used: int
    execution_success: bool
    execution_duration_seconds: float
    tests_passed: int
    tests_failed: int
    tests_errors: int
    peak_memory_mb: float
    reproducibility_score: float
    reproducibility_verdict: str
    claimed_metrics: dict[str, float]
    achieved_metrics: dict[str, float]
    traceability_coverage: float
    mapped_equations: int
    total_equations: int
    artifacts: dict[str, str]
    modules: list[ModuleTrustSummary]
    notes: list[str] = field(default_factory=list)
    healing: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["modules"] = [module.to_dict() for module in self.modules]
        return payload


@dataclass(slots=True)
class PaperWorkflowReportArtifacts:
    trust_report: PaperWorkflowTrustReport
    traceability_report: TraceabilityReport
    trust_report_json_path: Path
    trust_report_markdown_path: Path
    traceability_json_path: Path
    traceability_markdown_path: Path


def build_traceability_report(
    *,
    source: str,
    document: PaperDocument,
    project_dir: Path,
) -> TraceabilityReport:
    project_root = project_dir.expanduser().resolve()
    builder = TraceabilityBuilder(
        paper_title=document.title or "Unknown paper",
        paper_id=_paper_id(document, source),
        equations=[
            {
                "id": f"eq_{index + 1}",
                "latex": equation.latex,
                "description": equation.description or f"Equation {index + 1}",
                "section": _section_for_page(document, equation.page),
                "page": equation.page,
            }
            for index, equation in enumerate(document.equations)
        ],
    )
    builder.scan_code_for_references(project_root)
    return builder.build_report(implementation_dir=str(project_root))


def build_trust_report(
    *,
    source: str,
    document: PaperDocument,
    understanding: PaperUnderstanding,
    plan: ImplementationPlan,
    generation_result: GenerationResult,
    execution_report: ExecutionReport,
    reproducibility_report: ReproducibilityReport,
    traceability_report: TraceabilityReport,
    artifacts: dict[str, str],
    healing_payload: dict[str, Any] | None = None,
) -> PaperWorkflowTrustReport:
    module_summaries = _build_module_summaries(plan, generation_result)
    generated_module_count = sum(1 for module in module_summaries if module.status != "missing")
    failed_module_count = sum(1 for module in module_summaries if module.status != "generated")
    notes = _build_review_notes(
        plan=plan,
        generated_module_count=generated_module_count,
        generation_result=generation_result,
        execution_report=execution_report,
        reproducibility_report=reproducibility_report,
        traceability_report=traceability_report,
        healing_payload=healing_payload,
    )

    return PaperWorkflowTrustReport(
        source=source,
        paper_title=understanding.paper_title or document.title or "Unknown paper",
        paper_id=_paper_id(document, source),
        project_name=plan.project_name or "generated_project",
        generated_at=datetime.now(timezone.utc).isoformat(),
        status=_determine_trust_status(
            generation_result=generation_result,
            execution_report=execution_report,
            reproducibility_report=reproducibility_report,
            traceability_report=traceability_report,
        ),
        one_line_summary=understanding.one_line_summary,
        tech_stack=plan.tech_stack,
        target_language=plan.target_language,
        understanding_confidence=understanding.confidence,
        complexity=understanding.complexity,
        estimated_impl_hours=understanding.estimated_impl_hours,
        equation_count=len(document.equations),
        algorithm_count=len(document.algorithms),
        module_count=len(plan.modules),
        generated_module_count=generated_module_count,
        failed_module_count=failed_module_count,
        generation_success_rate=generation_result.success_rate,
        generation_duration_seconds=generation_result.duration_seconds,
        total_tokens_used=generation_result.total_tokens_used,
        execution_success=execution_report.success,
        execution_duration_seconds=execution_report.duration_seconds,
        tests_passed=execution_report.tests_passed,
        tests_failed=execution_report.tests_failed,
        tests_errors=execution_report.tests_errors,
        peak_memory_mb=execution_report.peak_memory_mb,
        reproducibility_score=reproducibility_report.score,
        reproducibility_verdict=reproducibility_report.verdict,
        claimed_metrics=dict(reproducibility_report.claimed_metrics),
        achieved_metrics=dict(reproducibility_report.achieved_metrics),
        traceability_coverage=traceability_report.coverage_score,
        mapped_equations=traceability_report.mapped_equations,
        total_equations=traceability_report.total_equations,
        artifacts=dict(artifacts),
        modules=module_summaries,
        notes=notes,
        healing=dict(healing_payload) if healing_payload else None,
    )


def export_trust_report_markdown(
    report: PaperWorkflowTrustReport,
    output_path: Path | None = None,
) -> str:
    lines = [
        "# ScholarDevClaw Trust Report",
        "",
        f"- Status: `{report.status}`",
        f"- Paper: {report.paper_title}",
        f"- Paper ID: `{report.paper_id}`",
        f"- Source: `{report.source}`",
        f"- Project: `{report.project_name}`",
        f"- Generated: `{report.generated_at}`",
        "",
        "## Snapshot",
        "",
        f"- Summary: {report.one_line_summary or 'Unavailable'}",
        f"- Stack: `{report.tech_stack or 'unspecified'}` on `{report.target_language}`",
        f"- Understanding confidence: {report.understanding_confidence:.0%}",
        f"- Complexity: `{report.complexity}`",
        f"- Estimated implementation time: {report.estimated_impl_hours} hours",
        f"- Equations extracted: {report.equation_count}",
        f"- Algorithms extracted: {report.algorithm_count}",
        "",
        "## Pipeline Scorecard",
        "",
        f"- Modules planned: {report.module_count}",
        f"- Modules generated: {report.generated_module_count}",
        f"- Module failures or gaps: {report.failed_module_count}",
        f"- Generation success rate: {report.generation_success_rate:.0%}",
        f"- Generation duration: {report.generation_duration_seconds:.2f}s",
        f"- Tokens used: {report.total_tokens_used}",
        f"- Tests passed: {report.tests_passed}",
        f"- Tests failed: {report.tests_failed}",
        f"- Test errors: {report.tests_errors}",
        f"- Execution success: {'yes' if report.execution_success else 'no'}",
        f"- Peak memory: {report.peak_memory_mb:.1f} MB",
        f"- Reproducibility: {report.reproducibility_score:.0%} ({report.reproducibility_verdict})",
        (
            f"- Traceability coverage: {report.traceability_coverage:.0%} "
            f"({report.mapped_equations}/{report.total_equations} equations mapped)"
        ),
        "",
        "## Module Outcomes",
        "",
        "| Module | File | Status | Attempts | Tokens |",
        "| --- | --- | --- | ---: | ---: |",
    ]

    for module in report.modules:
        lines.append(
            f"| `{module.module_id}` | `{module.file_path}` | `{module.status}` | "
            f"{module.generation_attempts} | {module.tokens_used} |"
        )
        if module.errors:
            lines.append(f"|  |  | errors: {'; '.join(module.errors[:2])} |  |  |")

    lines.extend(["", "## Review Notes", ""])
    if report.notes:
        lines.extend([f"- {note}" for note in report.notes])
    else:
        lines.append("- No immediate review warnings were raised.")

    if report.claimed_metrics or report.achieved_metrics:
        lines.extend(["", "## Metrics", ""])
        if report.claimed_metrics:
            lines.append("- Claimed metrics:")
            lines.extend(
                [
                    f"  - `{name}`: {value:.4f}"
                    for name, value in sorted(report.claimed_metrics.items())
                ]
            )
        if report.achieved_metrics:
            lines.append("- Achieved metrics:")
            lines.extend(
                [
                    f"  - `{name}`: {value:.4f}"
                    for name, value in sorted(report.achieved_metrics.items())
                ]
            )

    lines.extend(["", "## Artifacts", ""])
    for name, relative_path in sorted(report.artifacts.items()):
        lines.append(f"- `{name}`: `{relative_path}`")

    content = "\n".join(lines)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
    return content


def write_paper_workflow_reports(
    *,
    source: str,
    document: PaperDocument,
    understanding: PaperUnderstanding,
    plan: ImplementationPlan,
    generation_result: GenerationResult,
    execution_report: ExecutionReport,
    reproducibility_report: ReproducibilityReport,
    project_dir: Path,
    healing_payload: dict[str, Any] | None = None,
) -> PaperWorkflowReportArtifacts:
    project_root = project_dir.expanduser().resolve()
    project_root.mkdir(parents=True, exist_ok=True)

    trust_report_json_path = project_root / "trust_report.json"
    trust_report_markdown_path = project_root / "TRUST_REPORT.md"
    traceability_json_path = project_root / "traceability_report.json"
    traceability_markdown_path = project_root / "TRACEABILITY_REPORT.md"
    artifact_paths = {
        "trust_report_json": trust_report_json_path.name,
        "trust_report_markdown": trust_report_markdown_path.name,
        "traceability_report_json": traceability_json_path.name,
        "traceability_report_markdown": traceability_markdown_path.name,
    }
    generation_report_path = project_root / "generation_report.json"
    if generation_report_path.exists():
        artifact_paths["generation_report_json"] = generation_report_path.name

    traceability_report = build_traceability_report(
        source=source,
        document=document,
        project_dir=project_root,
    )
    traceability_json_path.write_text(
        json.dumps(traceability_report.to_dict(), indent=2),
        encoding="utf-8",
    )
    export_traceability_markdown(
        traceability_report,
        output_path=traceability_markdown_path,
    )

    trust_report = build_trust_report(
        source=source,
        document=document,
        understanding=understanding,
        plan=plan,
        generation_result=generation_result,
        execution_report=execution_report,
        reproducibility_report=reproducibility_report,
        traceability_report=traceability_report,
        artifacts=artifact_paths,
        healing_payload=healing_payload,
    )
    trust_report_json_path.write_text(
        json.dumps(trust_report.to_dict(), indent=2),
        encoding="utf-8",
    )
    export_trust_report_markdown(
        trust_report,
        output_path=trust_report_markdown_path,
    )

    return PaperWorkflowReportArtifacts(
        trust_report=trust_report,
        traceability_report=traceability_report,
        trust_report_json_path=trust_report_json_path,
        trust_report_markdown_path=trust_report_markdown_path,
        traceability_json_path=traceability_json_path,
        traceability_markdown_path=traceability_markdown_path,
    )


def _build_module_summaries(
    plan: ImplementationPlan,
    generation_result: GenerationResult,
) -> list[ModuleTrustSummary]:
    results_by_id = {
        str(result.module_id).strip(): result
        for result in generation_result.module_results
        if str(result.module_id).strip()
    }
    summaries: list[ModuleTrustSummary] = []

    for module in plan.modules:
        result = results_by_id.get(module.id)
        summaries.append(_module_summary(module, result))

    known_ids = {module.id for module in plan.modules if module.id}
    for module_id, result in sorted(results_by_id.items()):
        if module_id in known_ids:
            continue
        summaries.append(
            ModuleTrustSummary(
                module_id=module_id,
                file_path=result.file_path,
                test_file_path=result.test_file_path,
                estimated_lines=0,
                generation_attempts=result.generation_attempts,
                tokens_used=result.tokens_used,
                status="generated" if not result.final_errors else "failed",
                errors=list(result.final_errors),
            )
        )

    return summaries


def _module_summary(module: CodeModule, result: ModuleResult | None) -> ModuleTrustSummary:
    if result is None:
        return ModuleTrustSummary(
            module_id=module.id or module.name,
            file_path=module.file_path,
            test_file_path=module.test_file_path,
            estimated_lines=module.estimated_lines,
            generation_attempts=0,
            tokens_used=0,
            status="missing",
            errors=[],
        )

    return ModuleTrustSummary(
        module_id=module.id or module.name,
        file_path=result.file_path or module.file_path,
        test_file_path=result.test_file_path or module.test_file_path,
        estimated_lines=module.estimated_lines,
        generation_attempts=result.generation_attempts,
        tokens_used=result.tokens_used,
        status="generated" if not result.final_errors else "failed",
        errors=list(result.final_errors),
    )


def _build_review_notes(
    *,
    plan: ImplementationPlan,
    generated_module_count: int,
    generation_result: GenerationResult,
    execution_report: ExecutionReport,
    reproducibility_report: ReproducibilityReport,
    traceability_report: TraceabilityReport,
    healing_payload: dict[str, Any] | None,
) -> list[str]:
    notes: list[str] = []
    if generated_module_count < len(plan.modules):
        notes.append(
            f"Only {generated_module_count}/{len(plan.modules)} planned modules produced results."
        )
    if generation_result.success_rate < 1.0:
        notes.append(
            f"Generation success rate is {generation_result.success_rate:.0%}, so manual review is still required."
        )
    if not execution_report.success:
        notes.append(
            "Validation did not pass cleanly. Review test failures before trusting the generated project."
        )
    if reproducibility_report.score < TRUSTED_REPRODUCIBILITY_THRESHOLD:
        notes.append(
            f"Reproducibility is {reproducibility_report.score:.0%}, below the trusted threshold of "
            f"{TRUSTED_REPRODUCIBILITY_THRESHOLD:.0%}."
        )
    if traceability_report.total_equations == 0:
        notes.append(
            "No equations were extracted from the paper, so equation-to-code traceability is unavailable."
        )
    elif traceability_report.coverage_score < REVIEW_TRACEABILITY_THRESHOLD:
        notes.append(
            f"Traceability coverage is {traceability_report.coverage_score:.0%}; inspect generated code against the paper manually."
        )
    if healing_payload:
        round_count = int(healing_payload.get("round_count", 0) or 0)
        if round_count > 0:
            notes.append(
                f"Self-healing ran for {round_count} rounds to improve the generated project."
            )
    return notes


def _determine_trust_status(
    *,
    generation_result: GenerationResult,
    execution_report: ExecutionReport,
    reproducibility_report: ReproducibilityReport,
    traceability_report: TraceabilityReport,
) -> str:
    traceability_ok = (
        traceability_report.total_equations == 0
        or traceability_report.coverage_score >= REVIEW_TRACEABILITY_THRESHOLD
    )
    if (
        not execution_report.success
        or generation_result.success_rate < REVIEW_TRACEABILITY_THRESHOLD
    ):
        return "failed"
    if (
        generation_result.success_rate >= 0.90
        and reproducibility_report.score >= TRUSTED_REPRODUCIBILITY_THRESHOLD
        and traceability_ok
    ):
        return "trusted"
    return "needs_review"


def _paper_id(document: PaperDocument, source: str) -> str:
    return (
        str(document.arxiv_id or "").strip()
        or str(document.doi or "").strip()
        or str(document.source_url or "").strip()
        or source
    )


def _section_for_page(document: PaperDocument, page: int) -> str:
    if not document.sections:
        return ""
    ordered_sections = sorted(document.sections, key=lambda section: section.page_start)
    current_section = ""
    for section in ordered_sections:
        if section.page_start <= page:
            current_section = section.title
        else:
            break
    return current_section
