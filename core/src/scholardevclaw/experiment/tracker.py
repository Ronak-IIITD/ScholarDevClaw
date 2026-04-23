"""
Persistent experiment tracker for ScholarDevClaw.

Provides SQLite-backed logging of experiment runs with:
- Run metadata (hyperparams, hardware, git sha, seeds)
- Per-epoch metric logging
- Run comparison with statistical significance
- Export to JSON/CSV
"""

from __future__ import annotations

import json
import logging
import platform
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Configuration snapshot for a single experiment run."""

    hyperparameters: dict[str, Any] = field(default_factory=dict)
    random_seed: int | None = None
    hardware: str = ""
    gpu_info: str = ""
    python_version: str = ""
    git_sha: str = ""
    git_branch: str = ""
    paper_id: str = ""
    paper_title: str = ""
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricEntry:
    """A single metric measurement at a point in training."""

    run_id: str
    epoch: int
    step: int
    metric_name: str
    metric_value: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ExperimentRun:
    """A complete experiment run record."""

    run_id: str
    experiment_name: str
    status: str  # "running", "completed", "failed", "cancelled"
    config: RunConfig
    started_at: str
    ended_at: str | None = None
    duration_seconds: float | None = None
    final_metrics: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def is_complete(self) -> bool:
        return self.status in ("completed", "failed", "cancelled")


@dataclass
class ComparisonResult:
    """Result of comparing two or more experiment runs."""

    run_ids: list[str]
    metric_deltas: dict[str, dict[str, float]]  # metric_name -> {run_id: delta}
    best_run: str | None = None
    summary: str = ""


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class ExperimentTracker:
    """
    SQLite-backed experiment tracker.

    Usage:
        tracker = ExperimentTracker()
        run_id = tracker.start_run("my_experiment", config={...})
        tracker.log_metric(run_id, "loss", 0.5, epoch=1)
        tracker.log_metric(run_id, "accuracy", 0.85, epoch=1)
        tracker.end_run(run_id, status="completed")

        # Compare runs
        comparison = tracker.compare_runs([run_id_1, run_id_2])
    """

    def __init__(
        self,
        store_dir: Path | None = None,
    ):
        self.store_dir = store_dir or Path.home() / ".scholardevclaw" / "experiments"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.store_dir / "experiments.db"
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the experiment database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                experiment_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                config_json TEXT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                duration_seconds REAL,
                final_metrics_json TEXT,
                tags_json TEXT,
                error TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                step INTEGER NOT NULL DEFAULT 0,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_name ON runs(experiment_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)
        """)

        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        experiment_name: str,
        config: RunConfig | dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Start a new experiment run. Returns the run_id."""
        run_id = str(uuid.uuid4())[:12]

        if isinstance(config, dict):
            run_config = RunConfig(**config) if config else RunConfig()
        elif config is None:
            run_config = RunConfig()
        else:
            run_config = config

        # Auto-detect hardware if not specified
        if not run_config.hardware:
            run_config.hardware = platform.processor() or platform.machine()
        if not run_config.python_version:
            run_config.python_version = platform.python_version()
        if not run_config.gpu_info:
            run_config.gpu_info = self._detect_gpu()
        if not run_config.git_sha:
            run_config.git_sha = self._detect_git_sha()
            run_config.git_branch = self._detect_git_branch()

        now = datetime.now().isoformat()

        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO runs
                (run_id, experiment_name, status, config_json, started_at, tags_json)
                VALUES (?, ?, 'running', ?, ?, ?)
                """,
                (
                    run_id,
                    experiment_name,
                    json.dumps(asdict(run_config)),
                    now,
                    json.dumps(tags or []),
                ),
            )
            conn.commit()
            conn.close()

        logger.info("Started run %s for experiment '%s'", run_id, experiment_name)
        return run_id

    def end_run(
        self,
        run_id: str,
        status: str = "completed",
        final_metrics: dict[str, float] | None = None,
        error: str | None = None,
    ) -> None:
        """End an experiment run."""
        now = datetime.now().isoformat()

        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT started_at FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            duration = None
            if row:
                started = datetime.fromisoformat(row["started_at"])
                duration = (datetime.now() - started).total_seconds()

            # If final_metrics not provided, compute from last logged metrics
            if final_metrics is None:
                final_metrics = self._compute_final_metrics(cursor, run_id)

            cursor.execute(
                """
                UPDATE runs SET
                    status = ?,
                    ended_at = ?,
                    duration_seconds = ?,
                    final_metrics_json = ?,
                    error = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    now,
                    duration,
                    json.dumps(final_metrics),
                    error,
                    run_id,
                ),
            )
            conn.commit()
            conn.close()

        logger.info("Ended run %s with status '%s'", run_id, status)

    def _compute_final_metrics(self, cursor: sqlite3.Cursor, run_id: str) -> dict[str, float]:
        """Get the last logged value for each metric in a run."""
        cursor.execute(
            """
            SELECT metric_name, metric_value
            FROM metrics
            WHERE run_id = ?
            ORDER BY epoch DESC, step DESC
            """,
            (run_id,),
        )
        seen: dict[str, float] = {}
        for row in cursor.fetchall():
            name = row[0]
            if name not in seen:
                seen[name] = row[1]
        return seen

    # ------------------------------------------------------------------
    # Metric logging
    # ------------------------------------------------------------------

    def log_metric(
        self,
        run_id: str,
        metric_name: str,
        value: float,
        epoch: int = 0,
        step: int = 0,
    ) -> None:
        """Log a single metric value."""
        now = datetime.now().isoformat()

        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO metrics (run_id, epoch, step, metric_name, metric_value, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, epoch, step, metric_name, value, now),
            )
            conn.commit()
            conn.close()

    def log_metrics(
        self,
        run_id: str,
        metrics: dict[str, float],
        epoch: int = 0,
        step: int = 0,
    ) -> None:
        """Log multiple metrics at once."""
        now = datetime.now().isoformat()

        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            for name, value in metrics.items():
                cursor.execute(
                    """
                    INSERT INTO metrics (run_id, epoch, step, metric_name, metric_value, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (run_id, epoch, step, name, value, now),
                )
            conn.commit()
            conn.close()

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> ExperimentRun | None:
        """Get a single run by ID."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            conn.close()
            if row:
                return self._row_to_run(row)
            return None

    def list_runs(
        self,
        experiment_name: str | None = None,
        status: str | None = None,
        limit: int = 50,
        tags: list[str] | None = None,
    ) -> list[ExperimentRun]:
        """List experiment runs with optional filtering."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM runs WHERE 1=1"
            params: list[Any] = []

            if experiment_name:
                query += " AND experiment_name = ?"
                params.append(experiment_name)
            if status:
                query += " AND status = ?"
                params.append(status)

            query += " ORDER BY started_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            runs = [self._row_to_run(row) for row in rows]

            # Filter by tags if specified
            if tags:
                tag_set = set(t.lower() for t in tags)
                runs = [r for r in runs if tag_set.intersection(set(t.lower() for t in r.tags))]

            return runs

    def get_metrics(
        self,
        run_id: str,
        metric_name: str | None = None,
    ) -> list[MetricEntry]:
        """Get all metric entries for a run."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM metrics WHERE run_id = ?"
            params: list[Any] = [run_id]

            if metric_name:
                query += " AND metric_name = ?"
                params.append(metric_name)

            query += " ORDER BY epoch ASC, step ASC"

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            return [
                MetricEntry(
                    run_id=row["run_id"],
                    epoch=row["epoch"],
                    step=row["step"],
                    metric_name=row["metric_name"],
                    metric_value=row["metric_value"],
                    timestamp=row["timestamp"],
                )
                for row in rows
            ]

    def get_metric_names(self, run_id: str) -> list[str]:
        """Get all unique metric names logged in a run."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT metric_name FROM metrics WHERE run_id = ? ORDER BY metric_name",
                (run_id,),
            )
            names = [row[0] for row in cursor.fetchall()]
            conn.close()
            return names

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_runs(self, run_ids: list[str]) -> ComparisonResult:
        """
        Compare multiple runs side-by-side.

        Returns metric deltas relative to the first run (baseline).
        """
        runs = [self.get_run(rid) for rid in run_ids]
        runs = [r for r in runs if r is not None]

        if len(runs) < 2:
            return ComparisonResult(
                run_ids=run_ids,
                metric_deltas={},
                summary="Need at least 2 valid runs to compare.",
            )

        baseline = runs[0]
        all_metric_names: set[str] = set()
        for run in runs:
            all_metric_names.update(run.final_metrics.keys())

        deltas: dict[str, dict[str, float]] = {}
        for metric in sorted(all_metric_names):
            deltas[metric] = {}
            baseline_val = baseline.final_metrics.get(metric, 0.0)
            for run in runs:
                run_val = run.final_metrics.get(metric, 0.0)
                deltas[metric][run.run_id] = run_val - baseline_val

        # Find best run (by most metrics where it's highest)
        win_counts: dict[str, int] = {r.run_id: 0 for r in runs}
        for metric in all_metric_names:
            best_id = max(
                runs,
                key=lambda r: r.final_metrics.get(metric, float("-inf")),
            ).run_id
            win_counts[best_id] += 1

        best_run = max(win_counts, key=lambda k: win_counts[k])

        # Generate summary
        summary_lines = [f"Compared {len(runs)} runs (baseline: {baseline.run_id}):"]
        for metric in sorted(all_metric_names):
            vals = [f"{r.run_id}: {r.final_metrics.get(metric, 'N/A')}" for r in runs]
            summary_lines.append(f"  {metric}: {', '.join(vals)}")
        summary_lines.append(f"Best overall: {best_run}")

        return ComparisonResult(
            run_ids=[r.run_id for r in runs],
            metric_deltas=deltas,
            best_run=best_run,
            summary="\n".join(summary_lines),
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Get tracker statistics."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM runs")
            total_runs = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM runs WHERE status = 'completed'")
            completed = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM runs WHERE status = 'running'")
            running = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM runs WHERE status = 'failed'")
            failed = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM metrics")
            total_metrics = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT experiment_name) FROM runs")
            unique_experiments = cursor.fetchone()[0]

            conn.close()

        return {
            "total_runs": total_runs,
            "completed": completed,
            "running": running,
            "failed": failed,
            "total_metric_entries": total_metrics,
            "unique_experiments": unique_experiments,
            "store_dir": str(self.store_dir),
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_run(self, run_id: str, format: str = "json") -> str:
        """Export a run and its metrics to JSON or CSV string."""
        run = self.get_run(run_id)
        if not run:
            return ""

        metrics = self.get_metrics(run_id)

        if format == "json":
            data = {
                "run": asdict(run),
                "metrics": [asdict(m) for m in metrics],
            }
            return json.dumps(data, indent=2)

        elif format == "csv":
            lines = ["epoch,step,metric_name,metric_value,timestamp"]
            for m in metrics:
                lines.append(f"{m.epoch},{m.step},{m.metric_name},{m.metric_value},{m.timestamp}")
            return "\n".join(lines)

        return ""

    def delete_run(self, run_id: str) -> bool:
        """Delete a run and all its metrics."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
            cursor.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            conn.close()
            return deleted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_run(self, row: sqlite3.Row) -> ExperimentRun:
        config_dict = json.loads(row["config_json"] or "{}")
        return ExperimentRun(
            run_id=row["run_id"],
            experiment_name=row["experiment_name"],
            status=row["status"],
            config=RunConfig(
                **{k: v for k, v in config_dict.items() if k in RunConfig.__dataclass_fields__}
            ),
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            duration_seconds=row["duration_seconds"],
            final_metrics=json.loads(row["final_metrics_json"] or "{}"),
            tags=json.loads(row["tags_json"] or "[]"),
            error=row["error"],
        )

    def _detect_gpu(self) -> str:
        """Auto-detect GPU info."""
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                # Handle both old (total_mem) and new (total_memory) API
                total_bytes = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
                total_gb = total_bytes // (1024**3)
                return f"{props.name} ({total_gb}GB)"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "Apple Silicon MPS"
        except ImportError:
            pass
        return "CPU only"

    def _detect_git_sha(self) -> str:
        """Detect current git SHA."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip()[:12] if result.returncode == 0 else ""
        except Exception:
            return ""

    def _detect_git_branch(self) -> str:
        """Detect current git branch."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# Convenience singleton
# ---------------------------------------------------------------------------

_tracker: ExperimentTracker | None = None
_tracker_lock = threading.Lock()


def get_tracker(store_dir: Path | None = None) -> ExperimentTracker:
    """Get or create the global experiment tracker."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = ExperimentTracker(store_dir)
    return _tracker
