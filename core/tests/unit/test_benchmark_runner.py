from __future__ import annotations

import json
import sys
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[2]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from benchmarks.report import build_markdown_summary
from benchmarks.runner import BenchmarkCase, CandidateArtifact, load_cases, run_benchmarks, run_case


def test_catalog_contains_hardening_doc_cases():
    cases = load_cases()
    case_ids = {case.id for case in cases}
    assert {
        "rmsnorm",
        "rope",
        "swiglu",
        "flashattention",
        "lora",
        "layernorm",
        "gelu",
        "grouped_query_attention",
        "alibi",
        "cosine_lr_schedule",
    } <= case_ids


def test_run_case_scores_identical_ast_as_full_match(tmp_path: Path):
    expected_file = tmp_path / "expected.py"
    expected_source = "class RMSNorm:\n    pass\n"
    expected_file.write_text(expected_source)

    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    case = BenchmarkCase(
        id="demo",
        title="Demo",
        arxiv_id="0000.00000",
        pipeline_spec="rmsnorm",
        target_repo=repo_path,
        expected_file=expected_file,
        candidate_hints=["rmsnorm.py"],
    )

    result = run_case(
        case, candidate_factory=lambda _: CandidateArtifact({"rmsnorm.py": expected_source})
    )

    assert result.status == "matched"
    assert result.score == 1.0
    assert result.ast_match is True


def test_run_case_scores_symbol_overlap_as_partial(tmp_path: Path):
    expected_file = tmp_path / "expected.py"
    expected_file.write_text("class SwiGLU:\n    pass\n\ndef swiglu(x):\n    return x\n")

    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    case = BenchmarkCase(
        id="demo",
        title="Demo",
        arxiv_id="0000.00000",
        pipeline_spec="swiglu",
        target_repo=repo_path,
        expected_file=expected_file,
        candidate_hints=["swiglu.py"],
    )

    candidate_source = "class SwiGLU:\n    def __init__(self):\n        self.enabled = True\n"
    result = run_case(
        case, candidate_factory=lambda _: CandidateArtifact({"swiglu.py": candidate_source})
    )

    assert result.status == "partial"
    assert result.score == 0.5
    assert result.symbol_overlap == 0.5


def test_run_case_ignores_smoke_test_for_ast_and_symbol_matching(tmp_path: Path):
    expected_file = tmp_path / "expected.py"
    expected_file.write_text(
        "class FlashCausalSelfAttention:\n"
        "    pass\n\n"
        "def smoke_test(candidate_module):\n"
        "    return hasattr(candidate_module, 'FlashCausalSelfAttention')\n"
    )

    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    case = BenchmarkCase(
        id="flash-demo",
        title="Flash Demo",
        arxiv_id="0000.00000",
        pipeline_spec="flashattention",
        target_repo=repo_path,
        expected_file=expected_file,
        candidate_hints=["flash_attention.py"],
    )

    candidate_source = "class FlashCausalSelfAttention:\n    pass\n"
    result = run_case(
        case,
        candidate_factory=lambda _: CandidateArtifact({"flash_attention.py": candidate_source}),
    )

    assert result.status == "matched"
    assert result.score == 1.0
    assert result.ast_match is True
    assert result.expected_symbols == ["FlashCausalSelfAttention"]
    assert result.candidate_symbols == ["FlashCausalSelfAttention"]


def test_run_benchmarks_writes_json_and_markdown_summary(tmp_path: Path):
    expected_file = tmp_path / "expected.py"
    expected_source = "def gelu(value):\n    return value\n"
    expected_file.write_text(expected_source)

    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    case = BenchmarkCase(
        id="gelu-demo",
        title="GELU Demo",
        arxiv_id="0000.00000",
        pipeline_spec="gelu",
        target_repo=repo_path,
        expected_file=expected_file,
        candidate_hints=["gelu.py"],
    )

    report_path = tmp_path / "benchmark_report.json"
    report = run_benchmarks(
        cases=[case],
        candidate_factory=lambda _: CandidateArtifact({"gelu.py": expected_source}),
        output_path=report_path,
    )

    payload = json.loads(report_path.read_text())
    summary = build_markdown_summary(payload)

    assert report.aggregate_score == 1.0
    assert payload["aggregate"]["total_cases"] == 1
    assert "gelu-demo" in summary
    assert "Aggregate score" in summary
