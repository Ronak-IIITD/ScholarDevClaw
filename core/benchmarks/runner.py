from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCHMARK_ROOT = Path(__file__).resolve().parent
CORE_ROOT = BENCHMARK_ROOT.parent
SRC_ROOT = CORE_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_CATALOG_PATH = BENCHMARK_ROOT / "papers" / "catalog.json"
DEFAULT_REPORT_PATH = BENCHMARK_ROOT / "benchmark_report.json"


class UnsupportedSpecError(RuntimeError):
    """Raised when a benchmark case has no mapped runtime spec."""


@dataclass(slots=True)
class BenchmarkCase:
    id: str
    title: str
    arxiv_id: str
    pipeline_spec: str | None
    target_repo: Path
    expected_file: Path
    candidate_hints: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(slots=True)
class CandidateArtifact:
    sources: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkResult:
    id: str
    title: str
    arxiv_id: str
    pipeline_spec: str | None
    status: str
    score: float
    candidate_file: str | None = None
    ast_match: bool = False
    symbol_overlap: float = 0.0
    expected_symbols: list[str] = field(default_factory=list)
    candidate_symbols: list[str] = field(default_factory=list)
    import_ok: bool = False
    smoke_ok: bool | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkSuiteReport:
    generated_at: str
    total_cases: int
    supported_cases: int
    unsupported_cases: int
    aggregate_score: float
    supported_score: float
    results: list[BenchmarkResult]


def load_cases(catalog_path: Path = DEFAULT_CATALOG_PATH) -> list[BenchmarkCase]:
    payload = json.loads(catalog_path.read_text())
    cases: list[BenchmarkCase] = []
    for item in payload.get("cases", []):
        cases.append(
            BenchmarkCase(
                id=str(item["id"]),
                title=str(item["title"]),
                arxiv_id=str(item["arxiv_id"]),
                pipeline_spec=item.get("pipeline_spec"),
                target_repo=(BENCHMARK_ROOT / str(item["target_repo"])).resolve(),
                expected_file=(BENCHMARK_ROOT / str(item["expected_file"])).resolve(),
                candidate_hints=[str(value) for value in item.get("candidate_hints", [])],
                notes=str(item.get("notes", "")),
            )
        )
    return cases


def _ast_dump(source: str) -> str:
    return ast.dump(ast.parse(source), include_attributes=False)


def _top_level_symbols(source: str) -> list[str]:
    tree = ast.parse(source)
    symbols: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(node.name)
    return sorted(symbols)


def _load_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_mapping_payload(mapping: Any, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "targets": [
            {
                "file": target.file,
                "line": target.line,
                "current_code": target.current_code,
                "replacement_required": target.replacement_required,
                "context": target.context,
            }
            for target in getattr(mapping, "targets", [])
        ],
        "strategy": getattr(mapping, "strategy", "unknown"),
        "confidence": getattr(mapping, "confidence", 0),
        "research_spec": spec,
    }


def generate_candidate_artifact(case: BenchmarkCase) -> CandidateArtifact:
    if not case.pipeline_spec:
        raise UnsupportedSpecError(
            f"Benchmark case '{case.id}' has no mapped runtime spec in the current codebase."
        )

    from scholardevclaw.mapping.engine import MappingEngine
    from scholardevclaw.patch_generation.generator import PatchGenerator
    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer
    from scholardevclaw.research_intelligence.extractor import ResearchExtractor

    analyzer = TreeSitterAnalyzer(case.target_repo)
    analysis = analyzer.analyze()

    extractor = ResearchExtractor()
    spec = extractor.get_spec(case.pipeline_spec)
    if spec is None:
        raise UnsupportedSpecError(
            f"Benchmark case '{case.id}' references unknown spec '{case.pipeline_spec}'."
        )

    mapping = MappingEngine(analysis.__dict__, spec).map()
    patch = PatchGenerator(case.target_repo).generate(_build_mapping_payload(mapping, spec))

    sources: dict[str, str] = {}
    for new_file in getattr(patch, "new_files", []):
        sources[str(new_file.path)] = str(new_file.content)
    for transformation in getattr(patch, "transformations", []):
        sources[str(transformation.file)] = str(transformation.modified)

    metadata = {
        "branch_name": getattr(patch, "branch_name", ""),
        "mapping_confidence": getattr(mapping, "confidence", 0),
        "mapping_strategy": getattr(mapping, "strategy", "unknown"),
        "mapping_target_count": len(getattr(mapping, "targets", [])),
        "generated_file_count": len(getattr(patch, "new_files", [])),
        "transformation_count": len(getattr(patch, "transformations", [])),
    }
    return CandidateArtifact(sources=sources, metadata=metadata)


def _choose_candidate_source(case: BenchmarkCase, sources: dict[str, str]) -> tuple[str | None, str | None]:
    if not sources:
        return None, None

    normalized = {
        path.replace("\\", "/"): content
        for path, content in sources.items()
        if str(path).endswith(".py")
    }
    if not normalized:
        return None, None

    hint_order = list(case.candidate_hints)
    hint_order.append(case.expected_file.name)

    for hint in hint_order:
        for path, content in normalized.items():
            if path.endswith(hint):
                return path, content

    first_path = sorted(normalized)[0]
    return first_path, normalized[first_path]


def evaluate_candidate_artifact(case: BenchmarkCase, artifact: CandidateArtifact) -> BenchmarkResult:
    expected_source = case.expected_file.read_text()
    expected_symbols = _top_level_symbols(expected_source)
    candidate_path, candidate_source = _choose_candidate_source(case, artifact.sources)

    if not candidate_source:
        return BenchmarkResult(
            id=case.id,
            title=case.title,
            arxiv_id=case.arxiv_id,
            pipeline_spec=case.pipeline_spec,
            status="missing_candidate",
            score=0.0,
            expected_symbols=expected_symbols,
            error="No Python candidate file was generated for this benchmark case.",
            metadata=dict(artifact.metadata),
        )

    candidate_symbols = _top_level_symbols(candidate_source)
    expected_set = set(expected_symbols)
    candidate_set = set(candidate_symbols)
    overlap = len(expected_set & candidate_set) / max(len(expected_set), 1)

    result = BenchmarkResult(
        id=case.id,
        title=case.title,
        arxiv_id=case.arxiv_id,
        pipeline_spec=case.pipeline_spec,
        status="mismatch",
        score=0.0,
        candidate_file=candidate_path,
        ast_match=_ast_dump(expected_source) == _ast_dump(candidate_source),
        symbol_overlap=round(overlap, 3),
        expected_symbols=expected_symbols,
        candidate_symbols=candidate_symbols,
        metadata=dict(artifact.metadata),
    )

    try:
        with tempfile.TemporaryDirectory(prefix=f"sdc-bench-{case.id}-") as tmpdir:
            temp_candidate = Path(tmpdir) / Path(candidate_path or "candidate.py").name
            temp_candidate.write_text(candidate_source)
            candidate_module = _load_module(temp_candidate, f"candidate_{case.id}")
            expected_module = _load_module(case.expected_file, f"expected_{case.id}")
            result.import_ok = True

            smoke_test = getattr(expected_module, "smoke_test", None)
            if callable(smoke_test):
                result.smoke_ok = bool(smoke_test(candidate_module))
    except Exception as exc:
        result.import_ok = False
        result.error = f"Candidate import failed: {exc}"

    if result.ast_match and result.smoke_ok is not False:
        result.status = "matched"
        result.score = 1.0
        return result

    if result.symbol_overlap >= 0.5 or result.smoke_ok:
        result.status = "partial"
        result.score = 0.5
        return result

    return result


def run_case(
    case: BenchmarkCase,
    candidate_factory: Callable[[BenchmarkCase], CandidateArtifact] = generate_candidate_artifact,
) -> BenchmarkResult:
    if not case.pipeline_spec:
        return BenchmarkResult(
            id=case.id,
            title=case.title,
            arxiv_id=case.arxiv_id,
            pipeline_spec=case.pipeline_spec,
            status="unsupported_spec",
            score=0.0,
            error="This benchmark case is tracked in the hardening doc but has no matching runtime spec yet.",
            metadata={"notes": case.notes},
        )

    try:
        artifact = candidate_factory(case)
    except UnsupportedSpecError as exc:
        return BenchmarkResult(
            id=case.id,
            title=case.title,
            arxiv_id=case.arxiv_id,
            pipeline_spec=case.pipeline_spec,
            status="unsupported_spec",
            score=0.0,
            error=str(exc),
            metadata={"notes": case.notes},
        )
    except Exception as exc:
        return BenchmarkResult(
            id=case.id,
            title=case.title,
            arxiv_id=case.arxiv_id,
            pipeline_spec=case.pipeline_spec,
            status="generation_failed",
            score=0.0,
            error=f"{exc}\n{traceback.format_exc()}",
            metadata={"notes": case.notes},
        )

    result = evaluate_candidate_artifact(case, artifact)
    result.metadata.setdefault("notes", case.notes)
    return result


def _serialize_report(report: BenchmarkSuiteReport) -> dict[str, Any]:
    return {
        "generated_at": report.generated_at,
        "aggregate": {
            "total_cases": report.total_cases,
            "supported_cases": report.supported_cases,
            "unsupported_cases": report.unsupported_cases,
            "aggregate_score": report.aggregate_score,
            "supported_score": report.supported_score,
        },
        "results": [asdict(result) for result in report.results],
    }


def write_report(report: BenchmarkSuiteReport, output_path: Path = DEFAULT_REPORT_PATH) -> Path:
    output_path.write_text(json.dumps(_serialize_report(report), indent=2))
    return output_path


def run_benchmarks(
    cases: list[BenchmarkCase] | None = None,
    candidate_factory: Callable[[BenchmarkCase], CandidateArtifact] = generate_candidate_artifact,
    output_path: Path = DEFAULT_REPORT_PATH,
) -> BenchmarkSuiteReport:
    benchmark_cases = load_cases() if cases is None else cases
    results = [run_case(case, candidate_factory=candidate_factory) for case in benchmark_cases]
    supported = [result for result in results if result.status != "unsupported_spec"]

    aggregate_score = round(sum(result.score for result in results) / max(len(results), 1), 3)
    supported_score = round(
        sum(result.score for result in supported) / max(len(supported), 1),
        3,
    )

    report = BenchmarkSuiteReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_cases=len(results),
        supported_cases=len(supported),
        unsupported_cases=len(results) - len(supported),
        aggregate_score=aggregate_score,
        supported_score=supported_score,
        results=results,
    )
    write_report(report, output_path=output_path)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ScholarDevClaw benchmark harness.")
    parser.add_argument("--case", action="append", default=[], help="Run only the named benchmark id.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_REPORT_PATH),
        help="Path to write benchmark_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cases = load_cases()
    if args.case:
        selected = set(args.case)
        cases = [case for case in cases if case.id in selected]

    report = run_benchmarks(cases=cases, output_path=Path(args.output).resolve())

    print("ScholarDevClaw benchmark harness")
    print(f"  Cases           : {report.total_cases}")
    print(f"  Supported specs : {report.supported_cases}")
    print(f"  Unsupported     : {report.unsupported_cases}")
    print(f"  Aggregate score : {report.aggregate_score:.3f}")
    print(f"  Supported score : {report.supported_score:.3f}")
    print(f"  Report          : {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
