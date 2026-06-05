"""Regression detection for ScholarDevClaw benchmark reports.

A regression is defined as a per-case score drop that exceeds a configured
threshold, or a previously supported case becoming unsupported. The detector
also surfaces improvements (score increases) so they show up in CI output.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Default drop considered a regression. Anything less than this is treated
# as noise (e.g. floating point drift or non-deterministic ordering).
DEFAULT_REGRESSION_THRESHOLD = 0.1


@dataclass(slots=True)
class CaseDelta:
    case_id: str
    baseline_score: float | None
    current_score: float | None
    baseline_status: str | None
    current_status: str | None
    delta: float | None
    """Signed score change. None when one side is missing."""
    direction: str
    """'regression', 'improvement', 'unchanged', 'added', or 'removed'."""

    @property
    def is_regression(self) -> bool:
        return self.direction == "regression"


@dataclass(slots=True)
class RegressionReport:
    baseline_path: str
    current_path: str | None
    threshold: float
    baseline_aggregate: float | None
    current_aggregate: float | None
    aggregate_delta: float | None
    deltas: list[CaseDelta] = field(default_factory=list)

    @property
    def regressions(self) -> list[CaseDelta]:
        return [d for d in self.deltas if d.is_regression]

    @property
    def improvements(self) -> list[CaseDelta]:
        return [d for d in self.deltas if d.direction == "improvement"]

    @property
    def has_regressions(self) -> bool:
        return any(d.is_regression for d in self.deltas)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_path": self.baseline_path,
            "current_path": self.current_path,
            "threshold": self.threshold,
            "baseline_aggregate": self.baseline_aggregate,
            "current_aggregate": self.current_aggregate,
            "aggregate_delta": self.aggregate_delta,
            "regression_count": len(self.regressions),
            "improvement_count": len(self.improvements),
            "deltas": [asdict(d) for d in self.deltas],
        }


def _result_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(r.get("id", "")): r for r in payload.get("results", [])}


def _aggregate_score(payload: dict[str, Any]) -> float | None:
    agg = payload.get("aggregate", {})
    value = agg.get("aggregate_score")
    if value is None:
        return None
    return float(value)


def _classify(baseline: dict[str, Any] | None, current: dict[str, Any] | None) -> CaseDelta:
    case_id = (current or baseline or {}).get("id", "")
    if baseline is None and current is not None:
        return CaseDelta(
            case_id=case_id,
            baseline_score=None,
            current_score=float(current.get("score", 0.0)),
            baseline_status=None,
            current_status=str(current.get("status", "")),
            delta=None,
            direction="added",
        )
    if baseline is not None and current is None:
        return CaseDelta(
            case_id=case_id,
            baseline_score=float(baseline.get("score", 0.0)),
            current_score=None,
            baseline_status=str(baseline.get("status", "")),
            current_status=None,
            delta=None,
            direction="removed",
        )

    assert baseline is not None and current is not None
    base_score = float(baseline.get("score", 0.0))
    cur_score = float(current.get("score", 0.0))
    delta = round(cur_score - base_score, 4)
    if abs(delta) < 1e-9:
        direction = "unchanged"
    elif delta < 0:
        direction = "regression"
    else:
        direction = "improvement"

    # Status downgrades also count as regressions even if the score happens
    # to land within the threshold. For example, a previously matched case
    # becoming "partial" is a real regression regardless of numeric delta.
    base_status = str(baseline.get("status", ""))
    cur_status = str(current.get("status", ""))
    if _status_rank(cur_status) < _status_rank(base_status):
        direction = "regression"

    return CaseDelta(
        case_id=case_id,
        baseline_score=base_score,
        current_score=cur_score,
        baseline_status=base_status,
        current_status=cur_status,
        delta=delta,
        direction=direction,
    )


# Status ordering from "best" to "worst" used to detect downgrades.
_STATUS_RANK = {
    "matched": 3,
    "partial": 2,
    "missing_candidate": 1,
    "generation_failed": 1,
    "unsupported_spec": 0,
    "mismatch": 0,
}


def _status_rank(status: str) -> int:
    return _STATUS_RANK.get(status, -1)


def detect_regressions(
    baseline_payload: dict[str, Any],
    current_payload: dict[str, Any],
    baseline_path: str | Path = "<memory>",
    current_path: str | Path | None = None,
    threshold: float = DEFAULT_REGRESSION_THRESHOLD,
) -> RegressionReport:
    """Compare two benchmark payloads and return a structured report.

    A drop larger than `threshold` is reported as a regression. The function
    is symmetric: if the current run has cases the baseline lacked, those are
    reported as `added` rather than as regressions.
    """
    base_index = _result_index(baseline_payload)
    cur_index = _result_index(current_payload)
    all_ids = sorted(set(base_index) | set(cur_index))

    deltas: list[CaseDelta] = []
    for case_id in all_ids:
        base = base_index.get(case_id)
        cur = cur_index.get(case_id)
        delta = _classify(base, cur)
        # Only mark numeric regressions that exceed the threshold; the
        # status-based downgrade path is independent of threshold.
        if delta.direction == "regression":
            if delta.delta is not None and abs(delta.delta) < threshold:
                # Numeric change is below threshold, so do not flag purely
                # on score change. But we still keep the downgrade logic.
                if not _is_status_downgrade(delta):
                    delta.direction = "unchanged"
        deltas.append(delta)

    base_agg = _aggregate_score(baseline_payload)
    cur_agg = _aggregate_score(current_payload)
    agg_delta = None
    if base_agg is not None and cur_agg is not None:
        agg_delta = round(cur_agg - base_agg, 4)

    return RegressionReport(
        baseline_path=str(baseline_path),
        current_path=str(current_path) if current_path is not None else None,
        threshold=threshold,
        baseline_aggregate=base_agg,
        current_aggregate=cur_agg,
        aggregate_delta=agg_delta,
        deltas=deltas,
    )


def _is_status_downgrade(delta: CaseDelta) -> bool:
    if delta.baseline_status is None or delta.current_status is None:
        return False
    return _status_rank(delta.current_status) < _status_rank(delta.baseline_status)


def load_report_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_regression_report(report: RegressionReport, path: str | Path) -> Path:
    output = Path(path)
    output.write_text(json.dumps(report.to_dict(), indent=2))
    return output


def format_regression_summary(report: RegressionReport) -> str:
    """Format a short, human-readable summary suitable for CI logs."""
    lines: list[str] = []
    agg = report.aggregate_delta
    agg_str = f"{agg:+.3f}" if agg is not None else "n/a"
    base_agg = (
        f"{report.baseline_aggregate:.3f}" if report.baseline_aggregate is not None else "n/a"
    )
    cur_agg = f"{report.current_aggregate:.3f}" if report.current_aggregate is not None else "n/a"
    lines.append(
        f"Benchmark regression check (threshold={report.threshold:.2f}): "
        f"aggregate {base_agg} -> {cur_agg} ({agg_str})"
    )
    if report.regressions:
        lines.append(f"  Regressions ({len(report.regressions)}):")
        for d in report.regressions:
            delta_str = f"{d.delta:+.3f}" if d.delta is not None else "n/a"
            lines.append(
                f"    - {d.case_id}: {d.baseline_status} -> {d.current_status} "
                f"({d.baseline_score} -> {d.current_score}, {delta_str})"
            )
    if report.improvements:
        lines.append(f"  Improvements ({len(report.improvements)}):")
        for d in report.improvements:
            delta_str = f"{d.delta:+.3f}" if d.delta is not None else "n/a"
            lines.append(
                f"    - {d.case_id}: {d.baseline_status} -> {d.current_status} "
                f"({d.baseline_score} -> {d.current_score}, {delta_str})"
            )
    if not report.regressions and not report.improvements:
        lines.append("  No per-case changes detected.")
    return "\n".join(lines) + "\n"
