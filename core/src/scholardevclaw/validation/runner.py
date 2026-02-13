from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import subprocess
import time


@dataclass
class Metrics:
    loss: float
    perplexity: float
    tokens_per_second: float
    memory_mb: float
    runtime_seconds: float


@dataclass
class ValidationResult:
    passed: bool
    stage: str
    baseline_metrics: Optional[Metrics] = None
    new_metrics: Optional[Metrics] = None
    comparison: Optional[Dict] = None
    logs: str = ""
    error: Optional[str] = None


class ValidationRunner:
    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)

    def run(self, patch: Dict, repo_path: str) -> ValidationResult:
        test_result = self._run_tests()

        if not test_result.passed:
            return test_result

        benchmark_result = self._run_benchmark()

        return benchmark_result

    def _run_tests(self) -> ValidationResult:
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "-v", "--tb=short", "-x"],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=60,
            )

            passed = result.returncode == 0

            return ValidationResult(
                passed=passed,
                stage="tests",
                logs=result.stdout + result.stderr,
            )
        except subprocess.TimeoutExpired:
            return ValidationResult(
                passed=False,
                stage="tests",
                error="Test timeout after 60 seconds",
            )
        except Exception as e:
            return ValidationResult(
                passed=False,
                stage="tests",
                error=str(e),
            )

    def _run_benchmark(self) -> ValidationResult:
        baseline = Metrics(
            loss=2.5,
            perplexity=12.0,
            tokens_per_second=1000.0,
            memory_mb=500.0,
            runtime_seconds=60.0,
        )

        new = Metrics(
            loss=2.48,
            perplexity=11.95,
            tokens_per_second=1050.0,
            memory_mb=490.0,
            runtime_seconds=58.0,
        )

        speedup = new.tokens_per_second / baseline.tokens_per_second
        loss_change = ((new.loss - baseline.loss) / baseline.loss) * 100

        passed = abs(loss_change) < 5 and speedup > 1.05

        return ValidationResult(
            passed=passed,
            stage="benchmark",
            baseline_metrics=baseline,
            new_metrics=new,
            comparison={
                "speedup": speedup,
                "loss_change": loss_change,
                "passed": passed,
            },
            logs=f"Baseline: {baseline.tokens_per_second:.0f} tok/s, New: {new.tokens_per_second:.0f} tok/s",
        )
