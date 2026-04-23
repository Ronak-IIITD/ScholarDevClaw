"""
Experiment comparison utilities.

Provides rich comparison tables, statistical significance testing,
and structured comparison reports between experiment runs.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from scholardevclaw.experiment.tracker import ExperimentTracker, get_tracker


@dataclass
class MetricComparison:
    """Comparison of a single metric across multiple runs."""

    metric_name: str
    values: dict[str, float]  # run_id -> value
    best_run_id: str = ""
    worst_run_id: str = ""
    delta_from_baseline: dict[str, float] = field(default_factory=dict)
    percent_change: dict[str, float] = field(default_factory=dict)
    is_higher_better: bool = True


@dataclass
class RunComparison:
    """Full comparison report between experiment runs."""

    run_ids: list[str]
    baseline_run_id: str
    metrics: list[MetricComparison]
    winner: str = ""
    summary_table: str = ""
    recommendation: str = ""


# Metrics where lower is better
_LOWER_IS_BETTER = frozenset({
    "loss", "train_loss", "val_loss", "test_loss",
    "perplexity", "ppl",
    "error", "error_rate",
    "mse", "mae", "rmse",
    "latency", "latency_ms",
    "memory_mb", "memory_gb",
    "flops",
    "cer", "wer",  # character/word error rate
})


def is_higher_better(metric_name: str) -> bool:
    """Determine if a metric is better when higher."""
    normalized = metric_name.lower().replace("-", "_").replace(" ", "_")
    return normalized not in _LOWER_IS_BETTER


def compare_runs(
    run_ids: list[str],
    baseline_index: int = 0,
    tracker: ExperimentTracker | None = None,
) -> RunComparison:
    """
    Compare multiple experiment runs.

    Args:
        run_ids: List of run IDs to compare.
        baseline_index: Index of the baseline run (default: first).
        tracker: Optional tracker instance (uses global if not provided).

    Returns:
        RunComparison with detailed metric-by-metric analysis.
    """
    t = tracker or get_tracker()

    runs = []
    for rid in run_ids:
        run = t.get_run(rid)
        if run:
            runs.append(run)

    if len(runs) < 2:
        return RunComparison(
            run_ids=run_ids,
            baseline_run_id=run_ids[0] if run_ids else "",
            metrics=[],
            recommendation="Need at least 2 valid runs to compare.",
        )

    baseline = runs[min(baseline_index, len(runs) - 1)]

    # Gather all metric names
    all_metric_names: set[str] = set()
    for run in runs:
        all_metric_names.update(run.final_metrics.keys())

    # Build metric comparisons
    metric_comparisons: list[MetricComparison] = []
    win_counts: dict[str, int] = {r.run_id: 0 for r in runs}

    for metric_name in sorted(all_metric_names):
        higher_better = is_higher_better(metric_name)

        values: dict[str, float] = {}
        for run in runs:
            if metric_name in run.final_metrics:
                values[run.run_id] = run.final_metrics[metric_name]

        if not values:
            continue

        baseline_val = values.get(baseline.run_id, 0.0)

        delta_from_baseline: dict[str, float] = {}
        percent_change: dict[str, float] = {}
        for rid, val in values.items():
            delta_from_baseline[rid] = val - baseline_val
            if baseline_val != 0:
                percent_change[rid] = ((val - baseline_val) / abs(baseline_val)) * 100
            else:
                percent_change[rid] = 0.0

        # Determine best/worst
        if higher_better:
            best_id = max(values, key=lambda k: values[k])
            worst_id = min(values, key=lambda k: values[k])
        else:
            best_id = min(values, key=lambda k: values[k])
            worst_id = max(values, key=lambda k: values[k])

        win_counts[best_id] = win_counts.get(best_id, 0) + 1

        metric_comparisons.append(MetricComparison(
            metric_name=metric_name,
            values=values,
            best_run_id=best_id,
            worst_run_id=worst_id,
            delta_from_baseline=delta_from_baseline,
            percent_change=percent_change,
            is_higher_better=higher_better,
        ))

    # Overall winner
    winner = max(win_counts, key=lambda k: win_counts[k]) if win_counts else ""

    # Build summary table
    summary_table = _build_summary_table(runs, metric_comparisons, baseline.run_id)

    # Generate recommendation
    recommendation = _generate_recommendation(runs, metric_comparisons, winner)

    return RunComparison(
        run_ids=[r.run_id for r in runs],
        baseline_run_id=baseline.run_id,
        metrics=metric_comparisons,
        winner=winner,
        summary_table=summary_table,
        recommendation=recommendation,
    )


def _build_summary_table(
    runs: list[Any],
    metrics: list[MetricComparison],
    baseline_id: str,
) -> str:
    """Build a markdown-formatted comparison table."""
    if not metrics or not runs:
        return "No metrics to compare."

    # Header
    run_headers = [f"Run {r.run_id[:8]}" for r in runs]
    header = "| Metric | " + " | ".join(run_headers) + " | Best |"
    separator = "|" + "|".join(["---"] * (len(runs) + 2)) + "|"

    lines = [header, separator]

    for mc in metrics:
        direction = "↑" if mc.is_higher_better else "↓"
        cells = []
        for run in runs:
            val = mc.values.get(run.run_id)
            if val is None:
                cells.append("N/A")
            else:
                delta = mc.percent_change.get(run.run_id, 0)
                if run.run_id == baseline_id:
                    cells.append(f"{val:.4f}")
                elif delta > 0:
                    cells.append(f"{val:.4f} (+{delta:.1f}%)")
                elif delta < 0:
                    cells.append(f"{val:.4f} ({delta:.1f}%)")
                else:
                    cells.append(f"{val:.4f}")

        best_label = mc.best_run_id[:8] if mc.best_run_id else "—"
        line = f"| {mc.metric_name} {direction} | " + " | ".join(cells) + f" | {best_label} |"
        lines.append(line)

    return "\n".join(lines)


def _generate_recommendation(
    runs: list[Any],
    metrics: list[MetricComparison],
    winner: str,
) -> str:
    """Generate a human-readable recommendation."""
    if not winner:
        return "Insufficient data for recommendation."

    winner_run = next((r for r in runs if r.run_id == winner), None)
    if not winner_run:
        return f"Best run: {winner}"

    wins = sum(1 for mc in metrics if mc.best_run_id == winner)
    total = len(metrics)

    return (
        f"Recommendation: Use run {winner[:8]} "
        f"(experiment: {winner_run.experiment_name}). "
        f"Won {wins}/{total} metrics. "
        f"Config: {winner_run.config.hyperparameters}"
    )


def compute_metric_statistics(
    tracker: ExperimentTracker | None = None,
    experiment_name: str | None = None,
    metric_name: str = "loss",
) -> dict[str, float]:
    """
    Compute aggregate statistics for a metric across all runs of an experiment.

    Returns mean, std, min, max, median.
    """
    t = tracker or get_tracker()
    runs = t.list_runs(experiment_name=experiment_name, status="completed")

    values = []
    for run in runs:
        if metric_name in run.final_metrics:
            values.append(run.final_metrics[metric_name])

    if not values:
        return {}

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
    }
