from __future__ import annotations

import json
import sys
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[2]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from benchmarks.regression import (
    DEFAULT_REGRESSION_THRESHOLD,
    CaseDelta,
    detect_regressions,
    format_regression_summary,
    load_report_payload,
    write_regression_report,
)


def _payload(results: list[dict], aggregate: float = 0.5) -> dict:
    return {
        "generated_at": "2026-06-05T00:00:00Z",
        "aggregate": {
            "total_cases": len(results),
            "supported_cases": len(results),
            "unsupported_cases": 0,
            "aggregate_score": aggregate,
            "supported_score": aggregate,
        },
        "results": results,
    }


def _result(case_id: str, score: float, status: str = "matched") -> dict:
    return {
        "id": case_id,
        "title": case_id,
        "arxiv_id": "0000.00000",
        "pipeline_spec": case_id,
        "status": status,
        "score": score,
        "expected_symbols": [],
        "candidate_symbols": [],
    }


def test_detect_regressions_flags_large_score_drop():
    baseline = _payload([_result("a", 1.0), _result("b", 1.0)], aggregate=1.0)
    current = _payload([_result("a", 1.0), _result("b", 0.4)], aggregate=0.7)

    report = detect_regressions(baseline, current, threshold=0.1)

    assert report.has_regressions
    by_id = {d.case_id: d for d in report.deltas}
    assert by_id["a"].direction == "unchanged"
    assert by_id["b"].direction == "regression"
    assert by_id["b"].is_regression
    assert by_id["b"].delta == pytest_approx(-0.6)


def test_detect_regressions_treats_small_drift_as_noise():
    baseline = _payload([_result("a", 0.95)], aggregate=0.95)
    current = _payload([_result("a", 0.93)], aggregate=0.93)

    report = detect_regressions(baseline, current, threshold=0.1)

    assert not report.has_regressions
    assert report.deltas[0].direction == "unchanged"


def test_detect_regressions_flags_status_downgrade_even_within_threshold():
    baseline = _payload([_result("a", 0.5, status="matched")], aggregate=0.5)
    current = _payload([_result("a", 0.45, status="partial")], aggregate=0.45)

    report = detect_regressions(baseline, current, threshold=0.5)

    assert report.has_regressions
    assert report.deltas[0].direction == "regression"


def test_detect_regressions_marks_improvements():
    baseline = _payload([_result("a", 0.5, status="partial")], aggregate=0.5)
    current = _payload([_result("a", 1.0, status="matched")], aggregate=1.0)

    report = detect_regressions(baseline, current, threshold=0.1)

    assert not report.has_regressions
    assert report.improvements and report.improvements[0].case_id == "a"
    assert report.deltas[0].direction == "improvement"


def test_detect_regressions_handles_added_and_removed_cases():
    baseline = _payload([_result("a", 1.0), _result("b", 1.0)], aggregate=1.0)
    current = _payload([_result("a", 1.0), _result("c", 0.8)], aggregate=0.9)

    report = detect_regressions(baseline, current, threshold=0.1)

    by_id = {d.case_id: d for d in report.deltas}
    assert by_id["a"].direction == "unchanged"
    assert by_id["b"].direction == "removed"
    assert by_id["c"].direction == "added"
    assert not report.has_regressions


def test_detect_regressions_aggregates_scores():
    baseline = _payload([_result("a", 1.0), _result("b", 0.0)], aggregate=0.5)
    current = _payload([_result("a", 1.0), _result("b", 1.0)], aggregate=1.0)

    report = detect_regressions(baseline, current, threshold=0.1)

    assert report.baseline_aggregate == 0.5
    assert report.current_aggregate == 1.0
    assert report.aggregate_delta == pytest_approx(0.5)


def test_write_and_read_regression_report(tmp_path: Path):
    baseline = _payload([_result("a", 1.0)], aggregate=1.0)
    current = _payload([_result("a", 0.5)], aggregate=0.5)

    report = detect_regressions(baseline, current, baseline_path=tmp_path / "base.json")
    out = tmp_path / "regression.json"
    write_regression_report(report, out)

    payload = json.loads(out.read_text())
    assert payload["regression_count"] >= 1
    assert payload["deltas"][0]["case_id"] == "a"


def test_format_regression_summary_mentions_regressions():
    baseline = _payload([_result("a", 1.0), _result("b", 1.0)], aggregate=1.0)
    current = _payload([_result("a", 1.0), _result("b", 0.0)], aggregate=0.5)

    report = detect_regressions(baseline, current, threshold=0.1)
    text = format_regression_summary(report)

    assert "Regressions" in text
    assert "b" in text
    assert "+0.000" in text or "-0.500" in text or "-0.5" in text


def test_format_regression_summary_clean_run():
    baseline = _payload([_result("a", 1.0)], aggregate=1.0)
    current = _payload([_result("a", 1.0)], aggregate=1.0)

    report = detect_regressions(baseline, current, threshold=0.1)
    text = format_regression_summary(report)

    assert "No per-case changes" in text


def test_load_report_payload_roundtrip(tmp_path: Path):
    payload = _payload([_result("a", 1.0)], aggregate=1.0)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(payload))

    loaded = load_report_payload(path)
    assert loaded["results"][0]["id"] == "a"


def test_default_threshold_is_sane():
    assert 0.0 < DEFAULT_REGRESSION_THRESHOLD <= 0.5


def test_classify_handles_missing_both_sides():
    # Should never happen in practice (we iterate over union of keys),
    # but if it did, we want a clear failure, not a crash.
    delta = CaseDelta(
        case_id="x",
        baseline_score=None,
        current_score=None,
        baseline_status=None,
        current_status=None,
        delta=None,
        direction="unchanged",
    )
    assert not delta.is_regression


# Tiny shim so we don't have to import pytest for one helper.
class _Approx:
    def __init__(self, value: float):
        self.value = value

    def __eq__(self, other):
        return abs(self.value - float(other)) < 1e-6


def pytest_approx(value: float):
    return _Approx(value)
