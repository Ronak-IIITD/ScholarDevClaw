"""
Experiment visualization utilities.

Generates matplotlib plots for experiment metrics:
- Training curves (loss, accuracy over epochs)
- Multi-run comparison bar charts
- Metric correlation heatmaps
- Hyperparameter impact scatter plots

All plots can be saved as PNG or returned as matplotlib figures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy matplotlib import to avoid startup cost
_MPL_AVAILABLE: bool | None = None


def _check_matplotlib() -> bool:
    """Check if matplotlib is available."""
    global _MPL_AVAILABLE
    if _MPL_AVAILABLE is None:
        try:
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend
            _MPL_AVAILABLE = True
        except ImportError:
            _MPL_AVAILABLE = False
            logger.warning("matplotlib not available — visualization disabled")
    return _MPL_AVAILABLE


def plot_training_curves(
    run_id: str,
    metric_names: list[str] | None = None,
    save_path: Path | None = None,
    title: str | None = None,
    tracker: Any | None = None,
) -> Any | None:
    """
    Plot training curves (metric values over epochs) for a single run.

    Args:
        run_id: The experiment run ID.
        metric_names: Metrics to plot (default: all).
        save_path: Optional path to save the figure as PNG.
        title: Optional plot title.
        tracker: Optional ExperimentTracker instance.

    Returns:
        matplotlib Figure object or None if matplotlib unavailable.
    """
    if not _check_matplotlib():
        return None

    import matplotlib.pyplot as plt

    from scholardevclaw.experiment.tracker import get_tracker

    t = tracker or get_tracker()

    if not metric_names:
        metric_names = t.get_metric_names(run_id)

    if not metric_names:
        logger.warning("No metrics found for run %s", run_id)
        return None

    fig, axes = plt.subplots(
        len(metric_names),
        1,
        figsize=(10, 4 * len(metric_names)),
        squeeze=False,
    )

    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#EECA3B"]

    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx][0]
        entries = t.get_metrics(run_id, metric_name)

        if not entries:
            ax.text(0.5, 0.5, f"No data for {metric_name}", ha="center", va="center")
            continue

        epochs = [e.epoch for e in entries]
        values = [e.metric_value for e in entries]
        color = colors[idx % len(colors)]

        ax.plot(epochs, values, color=color, linewidth=2, marker="o", markersize=4)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f"{metric_name} over training", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        title or f"Training Curves — Run {run_id[:8]}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Saved training curves to %s", save_path)

    return fig


def plot_comparison_bars(
    run_ids: list[str],
    metric_name: str,
    save_path: Path | None = None,
    title: str | None = None,
    labels: list[str] | None = None,
    tracker: Any | None = None,
) -> Any | None:
    """
    Plot a bar chart comparing a single metric across multiple runs.

    Args:
        run_ids: List of run IDs to compare.
        metric_name: The metric to compare.
        save_path: Optional path to save the figure.
        title: Optional plot title.
        labels: Optional custom labels for each run.
        tracker: Optional ExperimentTracker instance.

    Returns:
        matplotlib Figure object or None.
    """
    if not _check_matplotlib():
        return None

    import matplotlib.pyplot as plt

    from scholardevclaw.experiment.tracker import get_tracker

    t = tracker or get_tracker()

    values = []
    run_labels = labels or []

    for i, rid in enumerate(run_ids):
        run = t.get_run(rid)
        if run:
            val = run.final_metrics.get(metric_name, 0.0)
            values.append(val)
            if not labels:
                run_labels.append(f"{run.experiment_name}\n({rid[:8]})")
        else:
            values.append(0.0)
            if not labels:
                run_labels.append(f"Unknown\n({rid[:8]})")

    fig, ax = plt.subplots(figsize=(max(8, len(run_ids) * 2), 6))

    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#EECA3B"]
    bar_colors = [colors[i % len(colors)] for i in range(len(values))]

    # Highlight the best
    from scholardevclaw.experiment.compare import is_higher_better

    best_idx = (
        values.index(max(values)) if is_higher_better(metric_name) else values.index(min(values))
    )
    bar_colors[best_idx] = "#2CA02C"  # Green for best

    bars = ax.bar(range(len(values)), values, color=bar_colors, width=0.6, edgecolor="white")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(range(len(run_labels)))
    ax.set_xticklabels(run_labels, fontsize=10)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(
        title or f"{metric_name} — Run Comparison",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Saved comparison chart to %s", save_path)

    return fig


def plot_multi_run_curves(
    run_ids: list[str],
    metric_name: str,
    save_path: Path | None = None,
    title: str | None = None,
    labels: list[str] | None = None,
    tracker: Any | None = None,
) -> Any | None:
    """
    Overlay training curves from multiple runs on a single plot.

    Args:
        run_ids: List of run IDs.
        metric_name: The metric to plot.
        save_path: Optional path to save the figure.
        title: Optional plot title.
        labels: Optional custom labels for each run.
        tracker: Optional ExperimentTracker instance.

    Returns:
        matplotlib Figure object or None.
    """
    if not _check_matplotlib():
        return None

    import matplotlib.pyplot as plt

    from scholardevclaw.experiment.tracker import get_tracker

    t = tracker or get_tracker()

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [
        "#4C78A8",
        "#F58518",
        "#E45756",
        "#72B7B2",
        "#54A24B",
        "#EECA3B",
        "#B07AA1",
        "#FF9DA7",
    ]

    for idx, rid in enumerate(run_ids):
        entries = t.get_metrics(rid, metric_name)
        if not entries:
            continue

        epochs = [e.epoch for e in entries]
        values = [e.metric_value for e in entries]
        label = labels[idx] if labels and idx < len(labels) else f"Run {rid[:8]}"
        color = colors[idx % len(colors)]

        ax.plot(
            epochs,
            values,
            color=color,
            linewidth=2,
            marker="o",
            markersize=3,
            label=label,
            alpha=0.85,
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(
        title or f"{metric_name} — Multi-Run Training Curves",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Saved multi-run curves to %s", save_path)

    return fig


def plot_metric_heatmap(
    run_ids: list[str],
    save_path: Path | None = None,
    title: str | None = None,
    tracker: Any | None = None,
) -> Any | None:
    """
    Plot a heatmap of all metrics across all runs.

    Args:
        run_ids: List of run IDs.
        save_path: Optional path to save the figure.
        title: Optional plot title.
        tracker: Optional ExperimentTracker instance.

    Returns:
        matplotlib Figure object or None.
    """
    if not _check_matplotlib():
        return None

    import matplotlib.pyplot as plt
    import numpy as np

    from scholardevclaw.experiment.tracker import get_tracker

    t = tracker or get_tracker()

    runs = [t.get_run(rid) for rid in run_ids]
    runs = [r for r in runs if r is not None]

    if not runs:
        return None

    # Collect all metrics
    all_metrics: set[str] = set()
    for run in runs:
        all_metrics.update(run.final_metrics.keys())

    metric_names = sorted(all_metrics)
    if not metric_names:
        return None

    # Build data matrix
    data = np.zeros((len(runs), len(metric_names)))
    for i, run in enumerate(runs):
        for j, metric in enumerate(metric_names):
            data[i, j] = run.final_metrics.get(metric, float("nan"))

    fig, ax = plt.subplots(figsize=(max(8, len(metric_names) * 1.5), max(4, len(runs) * 0.8)))

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([f"{r.run_id[:8]}" for r in runs], fontsize=10)

    # Add text annotations
    for i in range(len(runs)):
        for j in range(len(metric_names)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    ax.set_title(
        title or "Metric Heatmap Across Runs",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        logger.info("Saved metric heatmap to %s", save_path)

    return fig
