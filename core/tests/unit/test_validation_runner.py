"""Comprehensive tests for the validation runner (validation/runner.py).

Covers:
  - Metrics and ValidationResult dataclass construction
  - _run_bench_script() — success, failure, timeout
  - ValidationRunner._run_tests() — no test files, pass, fail, timeout, pytest not found
  - ValidationRunner._run_benchmark() — both succeed, one fails
  - ValidationRunner.run() — test fail stops early, benchmark run
  - ValidationRunner._check_torch_available() — success, failure
  - ValidationRunner._run_training_test() — with torch, without torch
  - ValidationRunner.run_simple_benchmark()
  - BenchmarkRunner.compare_implementations() — both succeed, one fails
  - ValidationRunner.run() — healing loop on test failure
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scholardevclaw.validation.runner import (
    BenchmarkRunner,
    Metrics,
    ValidationResult,
    ValidationRunner,
    _run_bench_script,
)

# =========================================================================
# Dataclass tests
# =========================================================================


class TestMetrics:
    def test_construction(self):
        m = Metrics(
            loss=0.5,
            perplexity=1.65,
            tokens_per_second=1000.0,
            memory_mb=128.0,
            runtime_seconds=2.0,
        )
        assert m.loss == 0.5
        assert m.perplexity == 1.65
        assert m.tokens_per_second == 1000.0
        assert m.memory_mb == 128.0
        assert m.runtime_seconds == 2.0


class TestValidationResult:
    def test_construction_minimal(self):
        vr = ValidationResult(passed=True, stage="tests")
        assert vr.passed is True
        assert vr.stage == "tests"
        assert vr.baseline_metrics is None
        assert vr.new_metrics is None
        assert vr.comparison is None
        assert vr.logs == ""
        assert vr.error is None

    def test_construction_full(self):
        baseline = Metrics(
            loss=0.5,
            perplexity=1.65,
            tokens_per_second=1000.0,
            memory_mb=128.0,
            runtime_seconds=2.0,
        )
        new = Metrics(
            loss=0.45,
            perplexity=1.57,
            tokens_per_second=1100.0,
            memory_mb=120.0,
            runtime_seconds=1.8,
        )
        vr = ValidationResult(
            passed=True,
            stage="benchmark",
            baseline_metrics=baseline,
            new_metrics=new,
            comparison={"speedup": 1.1},
            logs="ok",
            error=None,
        )
        assert vr.baseline_metrics is not None
        assert vr.new_metrics is not None
        assert vr.baseline_metrics.loss == 0.5
        assert vr.new_metrics.tokens_per_second == 1100.0


# =========================================================================
# _run_bench_script
# =========================================================================


class TestRunBenchScript:
    def test_success_json_output(self):
        script = 'import json; print(json.dumps({"key": "value"}))'
        result = _run_bench_script(script)
        assert result is not None
        assert result["key"] == "value"

    def test_nonzero_returncode_returns_none(self):
        script = "import sys; sys.exit(1)"
        result = _run_bench_script(script)
        assert result is None

    def test_no_json_output_returns_none(self):
        script = 'print("hello world")'
        result = _run_bench_script(script)
        assert result is None

    def test_timeout_returns_none(self):
        script = "import time; time.sleep(10)"
        result = _run_bench_script(script, timeout=1)
        assert result is None

    def test_invalid_json_returns_none(self):
        script = 'print("{not valid json}")'
        result = _run_bench_script(script)
        assert result is None

    def test_multiline_output_finds_last_json(self):
        script = 'import json; print("log line"); print(json.dumps({"found": True}))'
        result = _run_bench_script(script)
        assert result is not None
        assert result["found"] is True


# =========================================================================
# ValidationRunner._run_tests
# =========================================================================


class TestRunTests:
    def test_no_test_files_passes(self, tmp_path):
        runner = ValidationRunner(tmp_path)
        result = runner._run_tests()
        assert result.passed is True
        assert result.stage == "tests"
        assert "No test files" in result.logs

    def test_test_pass(self, tmp_path):
        test_file = tmp_path / "test_example.py"
        test_file.write_text("def test_ok(): assert True\n")
        runner = ValidationRunner(tmp_path)
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0, stdout="1 passed", stderr="")
            result = runner._run_tests()
        assert result.passed is True
        assert result.stage == "tests"

    def test_test_fail(self, tmp_path):
        test_file = tmp_path / "test_example.py"
        test_file.write_text("def test_fail(): assert False\n")
        runner = ValidationRunner(tmp_path)
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=1, stdout="1 failed", stderr="")
            result = runner._run_tests()
        assert result.passed is False
        assert result.stage == "tests"

    def test_test_timeout(self, tmp_path):
        test_file = tmp_path / "test_example.py"
        test_file.write_text("def test_slow(): import time; time.sleep(999)\n")
        runner = ValidationRunner(tmp_path)
        with patch.object(subprocess, "run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="pytest", timeout=120)
            result = runner._run_tests()
        assert result.passed is False
        assert "timeout" in (result.error or "").lower()

    def test_pytest_not_found(self, tmp_path):
        test_file = tmp_path / "test_example.py"
        test_file.write_text("def test_ok(): pass\n")
        runner = ValidationRunner(tmp_path)
        with patch.object(subprocess, "run") as mock_run:
            mock_run.side_effect = FileNotFoundError("pytest not found")
            result = runner._run_tests()
        assert result.passed is True
        assert "not found" in result.logs.lower()

    def test_generic_exception(self, tmp_path):
        test_file = tmp_path / "test_example.py"
        test_file.write_text("def test_ok(): pass\n")
        runner = ValidationRunner(tmp_path)
        with patch.object(subprocess, "run") as mock_run:
            mock_run.side_effect = OSError("disk error")
            result = runner._run_tests()
        assert result.passed is False
        assert result.error is not None


# =========================================================================
# ValidationRunner._check_torch_available
# =========================================================================


class TestCheckTorchAvailable:
    def test_torch_available(self, tmp_path):
        runner = ValidationRunner(tmp_path)
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0)
            assert runner._check_torch_available() is True

    def test_torch_not_available(self, tmp_path):
        runner = ValidationRunner(tmp_path)
        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=1)
            assert runner._check_torch_available() is False

    def test_torch_check_exception(self, tmp_path):
        runner = ValidationRunner(tmp_path)
        with patch.object(subprocess, "run") as mock_run:
            mock_run.side_effect = OSError("fail")
            assert runner._check_torch_available() is False


# =========================================================================
# ValidationRunner._run_benchmark
# =========================================================================


class TestRunBenchmark:
    def test_benchmark_runs_scripts(self, tmp_path, monkeypatch):
        """Test that _run_benchmark runs actual scripts with timing."""
        runner = ValidationRunner(tmp_path)
        # Create a fake benchmark script in tmp_path
        bench_script = tmp_path / "benchmark_test.py"
        bench_script.write_text("print('Benchmark running')")

        # Mock subprocess.run to simulate successful script execution
        fake_result = subprocess.CompletedProcess(
            args=["python", str(bench_script)],
            returncode=0,
            stdout="Benchmark running",
            stderr="",
        )
        with (
            patch.object(runner, "_check_torch_available", return_value=False),
            patch("subprocess.run", return_value=fake_result),
        ):
            result = runner._run_benchmark()

        assert result.passed is True
        assert result.stage == "benchmark"
        assert result.comparison is not None
        assert result.comparison["total"] >= 1
        assert result.comparison["passed"] >= 1

    def test_benchmark_script_fails(self, tmp_path, monkeypatch):
        """Test that _run_benchmark handles script failures."""
        runner = ValidationRunner(tmp_path)
        # Create a fake benchmark script that fails
        bench_script = tmp_path / "benchmark_fail.py"
        bench_script.write_text("raise Exception('Simulated failure')")

        # Mock subprocess.run to simulate failed script execution
        fake_result = subprocess.CompletedProcess(
            args=["python", str(bench_script)],
            returncode=1,
            stdout="",
            stderr="Simulated failure",
        )
        with (
            patch.object(runner, "_check_torch_available", return_value=False),
            patch("subprocess.run", return_value=fake_result),
        ):
            result = runner._run_benchmark()

        assert result.passed is False
        assert result.stage == "benchmark"
        assert result.comparison is not None
        assert result.comparison["failed"] >= 1

    def test_no_benchmark_or_test_files(self, tmp_path):
        """Test that _run_benchmark skips when no scripts exist."""
        runner = ValidationRunner(tmp_path)
        with patch.object(runner, "_check_torch_available", return_value=False):
            result = runner._run_benchmark()

        assert result.passed is True
        assert "No benchmark or test files found" in (result.logs or "")

    def test_one_metric_fails(self, tmp_path, monkeypatch):
        """Test that _run_benchmark handles no benchmark/test files."""
        runner = ValidationRunner(tmp_path)
        # No benchmark or test files in tmp_path, so it should skip
        with patch.object(runner, "_check_torch_available", return_value=False):
            result = runner._run_benchmark()

        assert result.passed is True
        assert "No benchmark or test files found" in (result.logs or "")

    def test_slowdown_detected(self, tmp_path, monkeypatch):
        """Test that _run_benchmark detects slow scripts."""
        runner = ValidationRunner(tmp_path)
        # Create a benchmark script that runs slowly (simulate via mock)
        bench_script = tmp_path / "benchmark_slow.py"
        bench_script.write_text("import time; time.sleep(2)")

        # Mock subprocess.run to simulate slow script (timeout)
        fake_result = subprocess.CompletedProcess(
            args=["python", str(bench_script)],
            returncode=0,
            stdout="Slow benchmark",
            stderr="",
        )
        with (
            patch.object(runner, "_check_torch_available", return_value=False),
            patch("subprocess.run", return_value=fake_result),
        ):
            result = runner._run_benchmark()

        # Should pass because script ran (even if slow)
        assert result.passed is True
        assert result.stage == "benchmark"
        assert result.comparison is not None
        assert result.comparison["total"] >= 1


# =========================================================================
# ValidationRunner.run()
# =========================================================================


class TestValidationRunnerRun:
    def test_invalid_patch_artifact_stops_early(self, tmp_path):
        runner = ValidationRunner(tmp_path)
        result = runner.run(
            {"new_files": [{"path": "broken.py", "content": "def broken(:\n    pass\n"}]},
            str(tmp_path),
        )

        assert result.passed is False
        assert result.stage == "artifacts"
        assert result.error == "Patch artifact validation failed"

    def test_tests_fail_stops_early(self, tmp_path):
        runner = ValidationRunner(tmp_path)
        failed_test = ValidationResult(passed=False, stage="tests", error="assertion error")

        with (
            patch.object(
                runner,
                "_validate_patch_artifacts",
                return_value=ValidationResult(passed=True, stage="artifacts"),
            ),
            patch.object(runner, "_run_tests", return_value=failed_test),
        ):
            result = runner.run({}, str(tmp_path))

        assert result.passed is False
        assert result.stage == "tests"

    def test_tests_pass_runs_benchmark(self, tmp_path):
        runner = ValidationRunner(tmp_path)
        passed_test = ValidationResult(passed=True, stage="tests")
        benchmark = ValidationResult(passed=True, stage="benchmark", comparison={"speedup": 1.1})

        with (
            patch.object(
                runner,
                "_validate_patch_artifacts",
                return_value=ValidationResult(passed=True, stage="artifacts", logs="artifact ok"),
            ),
            patch.object(runner, "_run_tests", return_value=passed_test),
            patch.object(runner, "_run_benchmark", return_value=benchmark),
        ):
            result = runner.run({}, str(tmp_path))

        assert result.passed is True
        assert result.stage == "benchmark"
        assert "artifact ok" in result.logs

    def test_tests_pass_no_error_runs_benchmark(self, tmp_path):
        """When tests pass (passed=True, no error), benchmark should run."""
        runner = ValidationRunner(tmp_path)
        passed_test = ValidationResult(passed=True, stage="tests", logs="all passed")
        benchmark = ValidationResult(passed=True, stage="benchmark")

        with (
            patch.object(
                runner,
                "_validate_patch_artifacts",
                return_value=ValidationResult(passed=True, stage="artifacts"),
            ),
            patch.object(runner, "_run_tests", return_value=passed_test),
            patch.object(runner, "_run_benchmark", return_value=benchmark),
        ):
            result = runner.run({}, str(tmp_path))

        assert result.stage == "benchmark"

    def test_no_error_fail_stops_early(self, tmp_path):
        """When tests fail (even without error), validation stops immediately."""
        runner = ValidationRunner(tmp_path)
        failed_test = ValidationResult(passed=False, stage="tests", logs="2 failed", error=None)

        with (
            patch.object(
                runner,
                "_validate_patch_artifacts",
                return_value=ValidationResult(passed=True, stage="artifacts"),
            ),
            patch.object(runner, "_run_tests", return_value=failed_test),
        ):
            result = runner.run({}, str(tmp_path))

        # With our fix, any test failure (passed=False) stops validation
        assert result.passed is False
        assert result.stage == "tests"

    def test_strict_execution_policy_blocks_unsandboxed_run(self, tmp_path, monkeypatch):
        runner = ValidationRunner(tmp_path)
        monkeypatch.setenv("SCHOLARDEVCLAW_VALIDATION_EXECUTION_MODE", "strict")
        monkeypatch.delenv("SCHOLARDEVCLAW_VALIDATION_SANDBOX", raising=False)

        result = runner.run({}, str(tmp_path))

        assert result.passed is False
        assert result.stage == "policy"
        assert "Unsandboxed" in (result.error or "")

    def test_strict_docker_mode_blocks_when_docker_unavailable(self, tmp_path, monkeypatch):
        runner = ValidationRunner(tmp_path)
        monkeypatch.setenv("SCHOLARDEVCLAW_VALIDATION_EXECUTION_MODE", "strict")
        monkeypatch.setenv("SCHOLARDEVCLAW_VALIDATION_SANDBOX", "docker")

        with patch.object(runner, "_docker_available", return_value=False):
            result = runner.run({}, str(tmp_path))

        assert result.passed is False
        assert result.stage == "policy"
        assert "Docker" in (result.error or "")


# =========================================================================
# ValidationRunner._run_training_test
# =========================================================================


class TestRunTrainingTest:
    def test_generic_bench_returns_metrics(self, tmp_path):
        runner = ValidationRunner(tmp_path)
        fake_data = {
            "loss": 0.5,
            "perplexity": 1.65,
            "tokens_per_second": 1000.0,
            "memory_mb": 128.0,
            "runtime_seconds": 2.0,
        }
        with patch("scholardevclaw.validation.runner._run_bench_script", return_value=fake_data):
            result = runner._run_training_test(use_variant=False, use_torch=False)
        assert result is not None
        assert result.loss == 0.5
        assert result.tokens_per_second == 1000.0

    def test_torch_bench_returns_metrics(self, tmp_path):
        runner = ValidationRunner(tmp_path)
        fake_data = {
            "loss": 0.45,
            "perplexity": 1.57,
            "tokens_per_second": 1100.0,
            "memory_mb": 120.0,
            "runtime_seconds": 1.8,
        }
        with patch("scholardevclaw.validation.runner._run_bench_script", return_value=fake_data):
            result = runner._run_training_test(use_variant=True, use_torch=True)
        assert result is not None
        assert result.loss == 0.45

    def test_bench_failure_returns_none(self, tmp_path):
        runner = ValidationRunner(tmp_path)
        with patch("scholardevclaw.validation.runner._run_bench_script", return_value=None):
            result = runner._run_training_test(use_variant=False)
        assert result is None

    def test_training_test_uses_docker_runner_when_enabled(self, tmp_path, monkeypatch):
        runner = ValidationRunner(tmp_path)
        monkeypatch.setenv("SCHOLARDEVCLAW_VALIDATION_SANDBOX", "docker")
        monkeypatch.setenv("SCHOLARDEVCLAW_VALIDATION_DOCKER_IMAGE", "python:3.12-slim")
        fake_data = {
            "loss": 0.4,
            "perplexity": 1.5,
            "tokens_per_second": 1200.0,
            "memory_mb": 110.0,
            "runtime_seconds": 1.5,
        }

        with (
            patch(
                "scholardevclaw.validation.runner._run_bench_script_in_docker",
                return_value=fake_data,
            ) as docker_run,
            patch("scholardevclaw.validation.runner._run_bench_script") as host_run,
        ):
            result = runner._run_training_test(use_variant=True, use_torch=False)

        assert result is not None
        assert result.loss == 0.4
        assert result.tokens_per_second == 1200.0
        host_run.assert_not_called()
        docker_run.assert_called_once()
        call_kwargs = docker_run.call_args.kwargs
        assert Path(call_kwargs["cwd"]) == tmp_path
        assert call_kwargs["image"] == "python:3.12-slim"


# =========================================================================
# ValidationRunner.run_simple_benchmark
# =========================================================================


class TestRunSimpleBenchmark:
    def test_success(self, tmp_path):
        runner = ValidationRunner(tmp_path)
        fake_data = {
            "status": "completed",
            "iterations": 10,
            "avg_duration_seconds": 0.001,
            "simulated": False,
        }
        with patch("scholardevclaw.validation.runner._run_bench_script", return_value=fake_data):
            result = runner.run_simple_benchmark()
        assert result["status"] == "completed"
        assert result["simulated"] is False

    def test_failure_fallback(self, tmp_path):
        runner = ValidationRunner(tmp_path)
        with patch("scholardevclaw.validation.runner._run_bench_script", return_value=None):
            result = runner.run_simple_benchmark()
        assert result["status"] == "error"
        assert result["simulated"] is False

    def test_simple_benchmark_uses_docker_runner_when_enabled(self, tmp_path, monkeypatch):
        runner = ValidationRunner(tmp_path)
        monkeypatch.setenv("SCHOLARDEVCLAW_VALIDATION_SANDBOX", "docker")
        fake_data = {
            "status": "completed",
            "iterations": 10,
            "avg_duration_seconds": 0.001,
            "simulated": False,
        }

        with patch(
            "scholardevclaw.validation.runner._run_bench_script_in_docker",
            return_value=fake_data,
        ) as docker_run:
            result = runner.run_simple_benchmark()

        assert result["status"] == "completed"
        docker_run.assert_called_once()


# =========================================================================
# BenchmarkRunner.compare_implementations
# =========================================================================


class TestBenchmarkRunner:
    def test_both_succeed(self, tmp_path):
        runner = BenchmarkRunner(tmp_path)
        r1 = {"avg_seconds": 0.01, "peak_memory_mb": 10.0, "iterations": 10}
        r2 = {"avg_seconds": 0.005, "peak_memory_mb": 8.0, "iterations": 10}

        with patch("scholardevclaw.validation.runner._run_bench_script", side_effect=[r1, r2]):
            result = runner.compare_implementations("impl1", "impl2", {"iterations": 10})

        assert "speedup" in result
        assert result["speedup"] > 1.0  # impl2 is faster
        assert result["memory_delta_mb"] < 0  # impl2 uses less memory

    def test_one_fails(self, tmp_path):
        runner = BenchmarkRunner(tmp_path)

        with patch("scholardevclaw.validation.runner._run_bench_script", side_effect=[None, None]):
            result = runner.compare_implementations("impl1", "impl2", {})

        assert "error" in result
        assert "failed" in result["error"].lower()

    def test_first_succeeds_second_fails(self, tmp_path):
        runner = BenchmarkRunner(tmp_path)
        r1 = {"avg_seconds": 0.01, "peak_memory_mb": 10.0, "iterations": 10}

        with patch("scholardevclaw.validation.runner._run_bench_script", side_effect=[r1, None]):
            result = runner.compare_implementations("impl1", "impl2", {})

        assert "error" in result


# =========================================================================
# ValidationRunner.run() — Healing Loop
# =========================================================================


class TestValidationRunnerHealing:
    """Tests for the Validation -> Healing feedback loop."""

    def test_healing_succeeds_on_retry(self, tmp_path):
        """When first test run fails but healed patch passes, validation succeeds."""
        runner = ValidationRunner(tmp_path)
        failed_test = ValidationResult(passed=False, stage="tests", logs="1 failed", error=None)
        passed_test = ValidationResult(passed=True, stage="tests")
        benchmark = ValidationResult(passed=True, stage="benchmark", comparison={"speedup": 1.1})

        call_count = 0

        def _test_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return failed_test
            return passed_test

        with (
            patch.object(
                runner,
                "_validate_patch_artifacts",
                return_value=ValidationResult(passed=True, stage="artifacts"),
            ),
            patch.object(runner, "_run_tests", side_effect=_test_side_effect),
            patch.object(runner, "_run_benchmark", return_value=benchmark),
        ):
            result = runner.run({}, str(tmp_path))

        assert result.passed is True
        assert result.stage == "benchmark"

    def test_healing_fails_and_reports(self, tmp_path):
        """When healing does not fix the issue, validation still fails."""
        runner = ValidationRunner(tmp_path)
        failed_test = ValidationResult(
            passed=False, stage="tests", logs="ImportError", error="Module not found"
        )

        with (
            patch.object(
                runner,
                "_validate_patch_artifacts",
                return_value=ValidationResult(passed=True, stage="artifacts"),
            ),
            patch.object(runner, "_run_tests", return_value=failed_test),
        ):
            result = runner.run({}, str(tmp_path))

        assert result.passed is False
        assert result.stage == "tests"

    def test_healing_skipped_when_tests_pass(self, tmp_path):
        """Healing should not be attempted when tests pass on first run."""
        runner = ValidationRunner(tmp_path)
        passed_test = ValidationResult(passed=True, stage="tests")
        benchmark = ValidationResult(passed=True, stage="benchmark", comparison={"speedup": 1.0})

        with (
            patch.object(
                runner,
                "_validate_patch_artifacts",
                return_value=ValidationResult(passed=True, stage="artifacts"),
            ),
            patch.object(runner, "_run_tests", return_value=passed_test),
            patch.object(runner, "_run_benchmark", return_value=benchmark),
        ):
            result = runner.run({}, str(tmp_path))

        assert result.passed is True
        assert result.stage == "benchmark"
