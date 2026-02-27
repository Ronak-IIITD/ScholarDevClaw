"""
Benchmark suite for standardized performance testing.

Provides:
- Pre-built benchmark tasks
- Result collection and comparison
- Historical tracking
- Performance regression detection
"""

from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class BenchmarkTask:
    """A single benchmark task"""

    name: str
    description: str
    func: Callable
    inputs: list[Any]
    iterations: int = 10
    warmup: int = 2


@dataclass
class BenchmarkMetrics:
    """Performance metrics for a single run"""

    duration_seconds: float
    memory_mb: float | None = None
    cpu_percent: float | None = None
    peak_memory_mb: float | None = None


@dataclass
class BenchmarkRun:
    """Result of a single benchmark run"""

    task_name: str
    timestamp: str
    iterations: int
    metrics: BenchmarkMetrics
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuiteReport:
    """Complete benchmark suite results"""

    suite_name: str
    timestamp: str
    tasks: list[BenchmarkRun]
    total_duration_seconds: float
    baseline: dict[str, float] | None = None
    comparison: dict[str, dict] = field(default_factory=dict)


class BenchmarkSuite:
    """Run standardized benchmarks"""

    TASKS: dict[str, BenchmarkTask] = {}

    def __init__(self, name: str = "default"):
        self.name = name
        self.results: list[BenchmarkRun] = []
        self._baseline: dict[str, float] | None = None

    def register_task(self, task: BenchmarkTask):
        """Register a benchmark task"""
        self.TASKS[task.name] = task

    def run_task(
        self,
        task: BenchmarkTask,
        iterations: int | None = None,
        warmup: int | None = None,
    ) -> BenchmarkRun:
        """Run a single benchmark task"""
        iterations = iterations or task.iterations
        warmup = warmup or task.warmup

        for _ in range(warmup):
            for inp in task.inputs:
                task.func(inp)

        durations = []
        memory_usages = []

        for inp in task.inputs:
            if HAS_PSUTIL:
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024

            tracemalloc.start()
            start = time.perf_counter()

            for _ in range(iterations):
                task.func(inp)

            end = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            if HAS_PSUTIL:
                mem_after = process.memory_info().rss / 1024 / 1024
                memory_usages.append(mem_after - mem_before)

            durations.append((end - start) / iterations)

        avg_duration = sum(durations) / len(durations)
        avg_memory = sum(memory_usages) / len(memory_usages) if memory_usages else None

        metrics = BenchmarkMetrics(
            duration_seconds=avg_duration,
            memory_mb=avg_memory,
            peak_memory_mb=peak / 1024 / 1024,
        )

        run = BenchmarkRun(
            task_name=task.name,
            timestamp=datetime.now().isoformat(),
            iterations=iterations,
            metrics=metrics,
        )

        self.results.append(run)
        return run

    def run_all(
        self,
        task_names: list[str] | None = None,
    ) -> BenchmarkSuiteReport:
        """Run all registered tasks"""
        import time

        start_time = time.time()

        tasks_to_run = []
        if task_names:
            for name in task_names:
                if name in self.TASKS:
                    tasks_to_run.append(self.TASKS[name])
        else:
            tasks_to_run = list(self.TASKS.values())

        runs = []
        for task in tasks_to_run:
            run = self.run_task(task)
            runs.append(run)

        comparison = {}
        if self._baseline:
            for run in runs:
                baseline_duration = self._baseline.get(run.task_name)
                if baseline_duration:
                    change = (
                        (run.metrics.duration_seconds - baseline_duration) / baseline_duration
                    ) * 100
                    comparison[run.task_name] = {
                        "baseline": baseline_duration,
                        "current": run.metrics.duration_seconds,
                        "change_percent": change,
                        "status": "regression"
                        if change > 10
                        else "improvement"
                        if change < -10
                        else "stable",
                    }

        return BenchmarkSuiteReport(
            suite_name=self.name,
            timestamp=datetime.now().isoformat(),
            tasks=runs,
            total_duration_seconds=time.time() - start_time,
            baseline=self._baseline,
            comparison=comparison,
        )

    def set_baseline(self):
        """Set current results as baseline"""
        self._baseline = {}
        for run in self.results:
            self._baseline[run.task_name] = run.metrics.duration_seconds

    def save_results(self, path: Path):
        """Save results to file"""
        data = {
            "suite_name": self.name,
            "results": [
                {
                    "task_name": r.task_name,
                    "timestamp": r.timestamp,
                    "iterations": r.iterations,
                    "duration_seconds": r.metrics.duration_seconds,
                    "memory_mb": r.metrics.memory_mb,
                    "peak_memory_mb": r.metrics.peak_memory_mb,
                }
                for r in self.results
            ],
        }
        path.write_text(json.dumps(data, indent=2))

    def load_baseline(self, path: Path):
        """Load baseline from file"""
        data = json.loads(path.read_text())
        self._baseline = {r["task_name"]: r["duration_seconds"] for r in data.get("results", [])}


class PrebuiltBenchmarks:
    """Pre-built benchmark tasks for common operations"""

    @staticmethod
    def string_operations() -> list[BenchmarkTask]:
        """Benchmarks for string operations"""
        return [
            BenchmarkTask(
                name="string_concat",
                description="String concatenation in loop",
                func=lambda n: "".join(f"item{i}" for i in range(n)),
                inputs=[100, 1000, 10000],
                iterations=100,
            ),
            BenchmarkTask(
                name="string_format",
                description="String formatting",
                func=lambda n: "hello {} {}".format(n, n * 2),
                inputs=[100, 1000, 10000],
                iterations=100,
            ),
            BenchmarkTask(
                name="regex_match",
                description="Regular expression matching",
                func=lambda s: __import__("re").match(r"\w+@\w+\.\w+", s),
                inputs=["test@example.com" for _ in range(100)],
                iterations=100,
            ),
        ]

    @staticmethod
    def list_operations() -> list[BenchmarkTask]:
        """Benchmarks for list operations"""
        return [
            BenchmarkTask(
                name="list_comprehension",
                description="List comprehension",
                func=lambda n: [i * 2 for i in range(n)],
                inputs=[1000, 10000, 100000],
                iterations=50,
            ),
            BenchmarkTask(
                name="list_filter",
                description="List filtering",
                func=lambda n: list(filter(lambda x: x % 2 == 0, range(n))),
                inputs=[1000, 10000, 100000],
                iterations=50,
            ),
            BenchmarkTask(
                name="list_sort",
                description="List sorting",
                func=lambda n: sorted(range(n), reverse=True),
                inputs=[1000, 10000, 100000],
                iterations=20,
            ),
        ]

    @staticmethod
    def dict_operations() -> list[BenchmarkTask]:
        """Benchmarks for dict operations"""
        return [
            BenchmarkTask(
                name="dict_lookup",
                description="Dictionary lookup",
                func=lambda d: all(k in d for k in list(d.keys())[: len(d) // 2]),
                inputs=[{i: i for i in range(n)} for n in [100, 1000, 10000]],
                iterations=100,
            ),
            BenchmarkTask(
                name="dict_comprehension",
                description="Dict comprehension",
                func=lambda n: {k: v * 2 for k, v in {i: i for i in range(n)}.items()},
                inputs=[100, 1000, 10000],
                iterations=50,
            ),
        ]

    @staticmethod
    def json_operations() -> list[BenchmarkTask]:
        """Benchmarks for JSON operations"""
        data = {"key": "value", "nested": {"a": 1, "b": 2}, "list": list(range(100))}

        return [
            BenchmarkTask(
                name="json_dumps",
                description="JSON serialization",
                func=lambda d: json.dumps(d),
                inputs=[data for _ in range(100)],
                iterations=50,
            ),
            BenchmarkTask(
                name="json_loads",
                description="JSON deserialization",
                func=lambda s: json.loads(s),
                inputs=[json.dumps({"key": "value", "list": list(range(100))}) for _ in range(100)],
                iterations=50,
            ),
        ]

    @staticmethod
    def file_operations(tmp_path: Path) -> list[BenchmarkTask]:
        """Benchmarks for file I/O"""
        test_file = tmp_path / "test.txt"

        return [
            BenchmarkTask(
                name="file_write",
                description="File writing",
                func=lambda f, n: f.write("x" * n),
                inputs=[(test_file, n) for n in [1000, 10000, 100000]],
                iterations=20,
            ),
            BenchmarkTask(
                name="file_read",
                description="File reading",
                func=lambda f: f.read(),
                inputs=[test_file for _ in range(20)],
                iterations=20,
            ),
        ]

    @staticmethod
    def create_default_suite() -> BenchmarkSuite:
        """Create suite with common benchmarks"""
        suite = BenchmarkSuite("standard")

        for task in PrebuiltBenchmarks.string_operations():
            suite.register_task(task)

        for task in PrebuiltBenchmarks.list_operations():
            suite.register_task(task)

        for task in PrebuiltBenchmarks.dict_operations():
            suite.register_task(task)

        for task in PrebuiltBenchmarks.json_operations():
            suite.register_task(task)

        return suite


class PerformanceComparator:
    """Compare performance across runs"""

    def __init__(self):
        self.historical: list[BenchmarkSuiteReport] = []

    def add_result(self, report: BenchmarkSuiteReport):
        """Add a result to history"""
        self.historical.append(report)

    def detect_regressions(
        self,
        current: BenchmarkSuiteReport,
        threshold: float = 10.0,
    ) -> list[dict]:
        """Detect performance regressions"""
        if not self.historical:
            return []

        previous = self.historical[-1]
        regressions = []

        current_by_task = {r.task_name: r for r in current.tasks}
        previous_by_task = {r.task_name: r for r in previous.tasks}

        for task_name, current_run in current_by_task.items():
            if task_name in previous_by_task:
                previous_run = previous_by_task[task_name]
                change = (
                    (current_run.metrics.duration_seconds - previous_run.metrics.duration_seconds)
                    / previous_run.metrics.duration_seconds
                ) * 100

                if change > threshold:
                    regressions.append(
                        {
                            "task": task_name,
                            "previous_duration": previous_run.metrics.duration_seconds,
                            "current_duration": current_run.metrics.duration_seconds,
                            "change_percent": change,
                            "severity": "critical"
                            if change > 50
                            else "major"
                            if change > 25
                            else "minor",
                        }
                    )

        return regressions


def quick_benchmark(func: Callable, inputs: list[Any], iterations: int = 10) -> dict:
    """Quick benchmark of a function"""
    durations = []

    for inp in inputs:
        start = time.perf_counter()
        for _ in range(iterations):
            func(inp)
        durations.append((time.perf_counter() - start) / iterations)

    return {
        "avg_duration": sum(durations) / len(durations),
        "min_duration": min(durations),
        "max_duration": max(durations),
        "iterations": iterations,
        "inputs": len(inputs),
    }
