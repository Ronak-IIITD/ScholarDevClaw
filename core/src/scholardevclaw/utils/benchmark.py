from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable
from pathlib import Path
import subprocess
import psutil


@dataclass
class BenchmarkResult:
    operation: str
    duration_seconds: float
    memory_mb: float | None = None
    cpu_percent: float | None = None
    iterations: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


class Benchmark:
    """Benchmark utilities for performance measurement."""

    @staticmethod
    def time_operation(
        func: Callable,
        *args,
        iterations: int = 1,
        **kwargs,
    ) -> BenchmarkResult:
        """Time an operation."""
        durations = []

        for _ in range(iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            durations.append(end - start)

        avg_duration = sum(durations) / len(durations)

        return BenchmarkResult(
            operation=func.__name__,
            duration_seconds=avg_duration,
            iterations=iterations,
        )

    @staticmethod
    def time_with_memory(
        func: Callable,
        *args,
        **kwargs,
    ) -> tuple[Any, BenchmarkResult]:
        """Time operation with memory tracking."""
        tracemalloc.start()

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return result, BenchmarkResult(
            operation=func.__name__,
            duration_seconds=end_time - start_time,
            memory_mb=peak / 1024 / 1024,
            iterations=1,
        )

    @staticmethod
    def compare_operations(
        operations: dict[str, Callable],
        *args,
        iterations: int = 10,
        **kwargs,
    ) -> dict[str, BenchmarkResult]:
        """Compare multiple operations."""
        results = {}

        for name, func in operations.items():
            result = Benchmark.time_operation(func, *args, iterations=iterations, **kwargs)
            results[name] = result

        return results

    @staticmethod
    def get_system_info() -> dict[str, Any]:
        """Get system information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "memory_available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,
            "memory_percent": psutil.virtual_memory().percent,
        }


class Profiler:
    """Simple profiler for code sections."""

    def __init__(self):
        self.sections: dict[str, list[float]] = {}

    def start_section(self, name: str) -> None:
        """Start timing a section."""
        if name not in self.sections:
            self.sections[name] = []
        self.sections[name].append(time.perf_counter())

    def end_section(self, name: str) -> None:
        """End timing a section."""
        if name in self.sections and self.sections[name]:
            start_time = self.sections[name][-1]
            duration = time.perf_counter() - start_time
            self.sections[name][-1] = duration

    def get_results(self) -> dict[str, dict[str, float]]:
        """Get profiling results."""
        results = {}

        for name, timings in self.sections.items():
            durations = [t for t in timings if isinstance(t, (int, float))]
            if durations:
                results[name] = {
                    "count": len(durations),
                    "total": sum(durations),
                    "average": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations),
                }

        return results

    def reset(self) -> None:
        """Reset profiler."""
        self.sections.clear()


def profile_block(name: str, profiler: Profiler):
    """Context manager for profiling a code block."""

    class ProfileContext:
        def __enter__(ctx):
            profiler.start_section(name)
            return ctx

        def __exit__(ctx, *args):
            profiler.end_section(name)

    return ProfileContext()


class OperationTimer:
    """Context manager for timing operations with logging."""

    def __init__(self, operation_name: str, logger=None):
        self.operation_name = operation_name
        self.logger = logger
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        if self.logger:
            self.logger.info(f"Starting: {self.operation_name}")
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time

        if self.logger:
            self.logger.info(f"Completed: {self.operation_name} in {duration:.3f}s")
        else:
            print(f"[TIMER] {self.operation_name}: {duration:.3f}s")

    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
