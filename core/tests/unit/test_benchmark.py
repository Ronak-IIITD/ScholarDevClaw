"""Tests for utils/benchmark.py"""

from unittest.mock import MagicMock

from scholardevclaw.utils.benchmark import (
    Benchmark,
    BenchmarkResult,
    OperationTimer,
    Profiler,
    profile_block,
)


class TestBenchmarkResult:
    def test_defaults(self):
        r = BenchmarkResult(operation="test", duration_seconds=0.5)
        assert r.operation == "test"
        assert r.duration_seconds == 0.5
        assert r.memory_mb is None
        assert r.cpu_percent is None
        assert r.iterations == 1
        assert r.metadata == {}

    def test_with_all_fields(self):
        r = BenchmarkResult(
            operation="test",
            duration_seconds=1.0,
            memory_mb=256.0,
            cpu_percent=50.0,
            iterations=5,
            metadata={"key": "val"},
        )
        assert r.memory_mb == 256.0
        assert r.cpu_percent == 50.0
        assert r.iterations == 5
        assert r.metadata == {"key": "val"}


class TestBenchmark:
    def test_time_operation_single(self):
        def add(a, b):
            return a + b

        result = Benchmark.time_operation(add, 1, 2)
        assert result.operation == "add"
        assert result.duration_seconds >= 0
        assert result.iterations == 1

    def test_time_operation_multiple_iterations(self):
        def return_arg(x):
            return x

        result = Benchmark.time_operation(return_arg, 42, iterations=5)
        assert result.iterations == 5
        assert result.duration_seconds >= 0

    def test_time_with_memory(self, monkeypatch):
        monkeypatch.setattr(
            "scholardevclaw.utils.benchmark.tracemalloc",
            MagicMock(get_traced_memory=lambda: (1024, 2048)),
        )

        def identity(x):
            return x

        result_val, bench = Benchmark.time_with_memory(identity, 42)
        assert result_val == 42
        assert bench.operation == "identity"
        assert bench.duration_seconds >= 0
        assert bench.memory_mb is not None

    def test_compare_operations(self):
        def add_one(x):
            return x + 1

        def double(x):
            return x * 2

        results = Benchmark.compare_operations(
            {"add_one": add_one, "double": double}, 5, iterations=3
        )
        assert "add_one" in results
        assert "double" in results
        assert results["add_one"].iterations == 3
        assert results["double"].iterations == 3

    def test_get_system_info(self, monkeypatch):
        mock_vm = MagicMock(total=8589934592, available=4294967296, percent=50.0)
        monkeypatch.setattr(
            "scholardevclaw.utils.benchmark.psutil",
            MagicMock(
                cpu_count=lambda: 8,
                cpu_percent=lambda interval=0.1: 25.0,
                virtual_memory=lambda: mock_vm,
            ),
        )

        info = Benchmark.get_system_info()
        assert info["cpu_count"] == 8
        assert info["cpu_percent"] == 25.0
        assert info["memory_total_gb"] == 8.0
        assert info["memory_available_gb"] == 4.0
        assert info["memory_percent"] == 50.0


class TestProfiler:
    def test_start_and_end_section(self):
        p = Profiler()
        p.start_section("load")
        p.end_section("load")
        results = p.get_results()
        assert "load" in results
        assert results["load"]["count"] == 1
        assert results["load"]["total"] >= 0

    def test_multiple_sections(self):
        p = Profiler()
        p.start_section("a")
        p.end_section("a")
        p.start_section("b")
        p.end_section("b")
        results = p.get_results()
        assert "a" in results
        assert "b" in results

    def test_multiple_timings_same_section(self):
        p = Profiler()
        p.start_section("x")
        p.end_section("x")
        p.start_section("x")
        p.end_section("x")
        results = p.get_results()
        assert results["x"]["count"] == 2

    def test_end_without_start(self):
        p = Profiler()
        p.end_section("never_started")
        results = p.get_results()
        assert "never_started" not in results

    def test_reset(self):
        p = Profiler()
        p.start_section("a")
        p.end_section("a")
        p.reset()
        assert p.get_results() == {}

    def test_get_results_empty(self):
        p = Profiler()
        assert p.get_results() == {}


class TestProfileBlock:
    def test_context_manager(self):
        p = Profiler()
        with profile_block("test_block", p):
            pass
        results = p.get_results()
        assert "test_block" in results
        assert results["test_block"]["count"] == 1


class TestOperationTimer:
    def test_duration_property(self):
        with OperationTimer("op") as timer:
            pass
        assert timer.duration >= 0

    def test_operation_name(self):
        timer = OperationTimer("my_op")
        assert timer.operation_name == "my_op"

    def test_duration_zero_before_exit(self):
        timer = OperationTimer("test")
        assert timer.duration == 0.0

    def test_unknown_duration_before_start(self):
        timer = OperationTimer("test")
        timer.end_time = 10.0
        assert timer.duration == 0.0

    def test_start_and_end_time_set(self):
        timer = OperationTimer("test")
        timer.__enter__()
        assert timer.start_time is not None
        timer.__exit__(None, None, None)
        assert timer.end_time is not None
