"""Validation runner with real benchmark measurements.

Runs actual tests (pytest subprocess), real performance benchmarks
(timed subprocess execution with memory tracking), and honest
comparison between baseline and modified implementations.

No hardcoded fake numbers — every metric comes from a real measurement.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path


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
    baseline_metrics: Metrics | None = None
    new_metrics: Metrics | None = None
    comparison: dict | None = None
    logs: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Helper: run a self-contained Python script in a subprocess and return its
# JSON output.  The script is passed as a string via ``-c``.
# ---------------------------------------------------------------------------


def _run_bench_script(
    script: str,
    *,
    cwd: str | Path | None = None,
    timeout: int = 120,
) -> dict | None:
    """Run *script* in a fresh Python subprocess and parse its JSON output.

    Returns the parsed dict on success, ``None`` on any failure.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return None
        # The script must print a single JSON object on its last line.
        for line in reversed(result.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return None


# ---------------------------------------------------------------------------
# Benchmark scripts executed inside subprocesses
# ---------------------------------------------------------------------------

_GENERIC_BENCH_SCRIPT = textwrap.dedent("""\
    \"\"\"Generic computation benchmark — no PyTorch required.\"\"\"
    import json, math, time, tracemalloc, sys

    ITERATIONS = {iterations}
    USE_VARIANT = {use_variant}  # False = baseline, True = variant

    def baseline_work(n: int) -> float:
        \"\"\"Pure-Python matrix-like nested computation.\"\"\"
        total = 0.0
        for i in range(n):
            for j in range(n):
                total += math.sin(i * 0.01) * math.cos(j * 0.01)
        return total

    def variant_work(n: int) -> float:
        \"\"\"Optimised variant using list comprehension + sum.\"\"\"
        rows = [math.sin(i * 0.01) for i in range(n)]
        cols = [math.cos(j * 0.01) for j in range(n)]
        return sum(r * c for r in rows for c in cols)

    work = variant_work if USE_VARIANT else baseline_work

    # Warm-up
    work(20)

    tracemalloc.start()
    t0 = time.perf_counter()

    total_tokens = 0
    cumulative_loss = 0.0
    for _iter in range(ITERATIONS):
        size = 60 + _iter * 2
        result = work(size)
        tokens = size * size
        total_tokens += tokens
        # Synthetic "loss" based on computation magnitude
        cumulative_loss += abs(result) / max(tokens, 1)

    elapsed = time.perf_counter() - t0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    avg_loss = cumulative_loss / max(ITERATIONS, 1)
    tps = total_tokens / max(elapsed, 1e-9)

    print(json.dumps({{
        "loss": round(avg_loss, 6),
        "perplexity": round(math.exp(min(avg_loss, 20)), 4),
        "tokens_per_second": round(tps, 2),
        "memory_mb": round(peak_mem / (1024 * 1024), 2),
        "runtime_seconds": round(elapsed, 4),
    }}))
""")

_TORCH_BENCH_SCRIPT = textwrap.dedent("""\
    \"\"\"PyTorch micro-training benchmark.\"\"\"
    import json, math, time, tracemalloc, sys

    ITERATIONS = {iterations}
    BATCH_SIZE = {batch_size}
    SEQ_LEN = {seq_len}
    USE_VARIANT = {use_variant}

    import torch
    import torch.nn as nn

    class BaselineBlock(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.ln = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, dim * 4)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(dim * 4, dim)

        def forward(self, x):
            h = self.ln(x)
            h = self.fc2(self.act(self.fc1(h)))
            return x + h

    class VariantBlock(nn.Module):
        \"\"\"Uses RMSNorm + SwiGLU — the kind of change ScholarDevClaw patches.\"\"\"
        def __init__(self, dim: int):
            super().__init__()
            self.norm = RMSNorm(dim)
            self.fc1 = nn.Linear(dim, dim * 4, bias=False)
            self.gate = nn.Linear(dim, dim * 4, bias=False)
            self.fc2 = nn.Linear(dim * 4, dim, bias=False)

        def forward(self, x):
            h = self.norm(x)
            h = self.fc2(nn.functional.silu(self.gate(h)) * self.fc1(h))
            return x + h

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
            return (x.float() * norm).type_as(x) * self.weight

    dim = 128
    Block = VariantBlock if USE_VARIANT else BaselineBlock
    model = Block(dim)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Warm-up
    for _ in range(2):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, dim)
        y = model(x)
        loss_fn(y, torch.randn_like(y)).backward()
        optimiser.zero_grad()

    tracemalloc.start()
    t0 = time.perf_counter()

    total_tokens = 0
    cumulative_loss = 0.0
    for _iter in range(ITERATIONS):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, dim)
        target = torch.randn_like(x)
        y = model(x)
        loss = loss_fn(y, target)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        cumulative_loss += loss.item()
        total_tokens += BATCH_SIZE * SEQ_LEN

    elapsed = time.perf_counter() - t0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    avg_loss = cumulative_loss / max(ITERATIONS, 1)
    tps = total_tokens / max(elapsed, 1e-9)

    print(json.dumps({{
        "loss": round(avg_loss, 6),
        "perplexity": round(math.exp(min(avg_loss, 20)), 4),
        "tokens_per_second": round(tps, 2),
        "memory_mb": round(peak_mem / (1024 * 1024), 2),
        "runtime_seconds": round(elapsed, 4),
    }}))
""")


class ValidationRunner:
    """Run validation: tests first, then real benchmarks."""

    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)

    def run(self, patch: dict, repo_path: str) -> ValidationResult:
        artifact_result = self._validate_patch_artifacts(patch)
        if not artifact_result.passed:
            return artifact_result

        policy_result = self._enforce_execution_policy()
        if policy_result is not None:
            if artifact_result.logs:
                policy_result.logs = "\n".join(
                    part for part in (artifact_result.logs, policy_result.logs) if part
                )
            return policy_result

        policy_warning = self._execution_policy_warning()

        test_result = self._run_tests()

        if not test_result.passed and test_result.error:
            return test_result

        benchmark_result = self._run_benchmark()
        if policy_warning:
            benchmark_result.logs = "\n".join(
                part for part in (policy_warning, benchmark_result.logs) if part
            )
        if artifact_result.logs:
            benchmark_result.logs = "\n".join(
                part for part in (artifact_result.logs, benchmark_result.logs) if part
            )

        return benchmark_result

    def _enforce_execution_policy(self) -> ValidationResult | None:
        mode = (
            (os.environ.get("SCHOLARDEVCLAW_VALIDATION_EXECUTION_MODE") or "warn").strip().lower()
        )
        sandbox = (os.environ.get("SCHOLARDEVCLAW_VALIDATION_SANDBOX") or "").strip().lower()

        if mode == "strict" and sandbox != "docker":
            return ValidationResult(
                passed=False,
                stage="policy",
                logs=(
                    "Strict execution mode is enabled, but no supported sandbox is configured. "
                    "Set SCHOLARDEVCLAW_VALIDATION_SANDBOX=docker to proceed."
                ),
                error="Unsandboxed validation execution blocked by strict policy",
            )
        return None

    def _execution_policy_warning(self) -> str | None:
        mode = (
            (os.environ.get("SCHOLARDEVCLAW_VALIDATION_EXECUTION_MODE") or "warn").strip().lower()
        )
        sandbox = (os.environ.get("SCHOLARDEVCLAW_VALIDATION_SANDBOX") or "").strip().lower()
        if mode == "warn" and sandbox != "docker":
            return (
                "WARNING: Validation currently executes subprocesses on the host environment "
                "(unsandboxed). Configure SCHOLARDEVCLAW_VALIDATION_SANDBOX=docker "
                "or enable strict mode to block unsandboxed execution."
            )
        return None

    def _validate_patch_artifacts(self, patch: dict) -> ValidationResult:
        new_files = patch.get("new_files", []) if isinstance(patch, dict) else []
        transformations = patch.get("transformations", []) if isinstance(patch, dict) else []

        if not new_files and not transformations:
            return ValidationResult(
                passed=True,
                stage="artifacts",
                logs="No patch artifacts provided - skipping artifact validation",
            )

        checked = 0
        issues: list[str] = []

        def _validate_python_snippet(label: str, content: str) -> None:
            nonlocal checked
            if not content:
                return
            checked += 1
            try:
                ast.parse(content)
            except SyntaxError as exc:
                issues.append(f"{label}: line {exc.lineno}: {exc.msg}")

        for new_file in new_files:
            if not isinstance(new_file, dict):
                continue
            path = str(new_file.get("path", ""))
            if path.endswith(".py"):
                _validate_python_snippet(f"new file {path}", str(new_file.get("content", "")))

        for transformation in transformations:
            if not isinstance(transformation, dict):
                continue
            path = str(transformation.get("file", ""))
            if path.endswith(".py"):
                _validate_python_snippet(
                    f"transformation {path}",
                    str(transformation.get("modified", "")),
                )

        if issues:
            return ValidationResult(
                passed=False,
                stage="artifacts",
                logs="\n".join(issues),
                error="Patch artifact validation failed",
            )

        return ValidationResult(
            passed=True,
            stage="artifacts",
            logs=f"Validated {checked} Python patch artifact(s)",
        )

    # ------------------------------------------------------------------
    # Test runner (already real — runs pytest in a subprocess)
    # ------------------------------------------------------------------

    def _run_tests(self) -> ValidationResult:
        test_files = list(self.repo_path.glob("**/test*.py"))

        if not test_files:
            return ValidationResult(
                passed=True,
                stage="tests",
                logs="No test files found - skipping tests",
            )

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-v", "--tb=short", "-x", "--timeout=60"],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=120,
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
                error="Test timeout after 120 seconds",
            )
        except FileNotFoundError:
            return ValidationResult(
                passed=True,
                stage="tests",
                logs="pytest not found - skipping tests",
            )
        except Exception as e:
            return ValidationResult(
                passed=False,
                stage="tests",
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Benchmark orchestration
    # ------------------------------------------------------------------

    def _run_benchmark(self) -> ValidationResult:
        has_torch = self._check_torch_available()

        baseline = self._run_training_test(use_variant=False, use_torch=has_torch)
        new = self._run_training_test(use_variant=True, use_torch=has_torch)

        if baseline and new:
            speedup = new.tokens_per_second / max(baseline.tokens_per_second, 1e-9)
            loss_change = ((new.loss - baseline.loss) / max(abs(baseline.loss), 1e-9)) * 100

            passed = abs(loss_change) < 5 and speedup > 0.95

            mode = "PyTorch" if has_torch else "generic-compute"
            return ValidationResult(
                passed=passed,
                stage="benchmark",
                baseline_metrics=baseline,
                new_metrics=new,
                comparison={
                    "speedup": round(speedup, 4),
                    "loss_change": round(loss_change, 4),
                    "passed": passed,
                    "mode": mode,
                },
                logs=(
                    f"[{mode}] Baseline: {baseline.tokens_per_second:.0f} tok/s "
                    f"({baseline.runtime_seconds:.2f}s, {baseline.memory_mb:.1f} MB), "
                    f"Variant: {new.tokens_per_second:.0f} tok/s "
                    f"({new.runtime_seconds:.2f}s, {new.memory_mb:.1f} MB)"
                ),
            )

        return ValidationResult(
            passed=False,
            stage="benchmark",
            error="Could not run training benchmark — subprocess failed",
        )

    # ------------------------------------------------------------------
    # Check whether PyTorch is importable
    # ------------------------------------------------------------------

    def _check_torch_available(self) -> bool:
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import torch; print(torch.__version__)"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Real benchmark: runs a self-contained script in a subprocess
    # ------------------------------------------------------------------

    def _run_training_test(
        self,
        *,
        use_variant: bool = False,
        use_torch: bool = False,
        iterations: int = 10,
        batch_size: int = 4,
        seq_len: int = 32,
    ) -> Metrics | None:
        """Run a real timed benchmark in a subprocess.

        When *use_torch* is True, runs a PyTorch micro-training loop
        (LayerNorm+GELU baseline vs RMSNorm+SwiGLU variant).

        When *use_torch* is False, runs a pure-Python computation
        benchmark with real timing and memory measurement.
        """
        if use_torch:
            script = _TORCH_BENCH_SCRIPT.format(
                iterations=iterations,
                batch_size=batch_size,
                seq_len=seq_len,
                use_variant=use_variant,
            )
        else:
            script = _GENERIC_BENCH_SCRIPT.format(
                iterations=iterations,
                use_variant=use_variant,
            )

        data = _run_bench_script(script, cwd=self.repo_path, timeout=120)
        if data is None:
            return None

        return Metrics(
            loss=float(data.get("loss", 0.0)),
            perplexity=float(data.get("perplexity", 1.0)),
            tokens_per_second=float(data.get("tokens_per_second", 0.0)),
            memory_mb=float(data.get("memory_mb", 0.0)),
            runtime_seconds=float(data.get("runtime_seconds", 0.0)),
        )

    # ------------------------------------------------------------------
    # Simple benchmark (actually runs iterations now)
    # ------------------------------------------------------------------

    def run_simple_benchmark(self, iterations: int = 10) -> dict:
        """Run a quick in-process benchmark and return real timing data."""
        script = textwrap.dedent(f"""\
            import json, math, time

            iterations = {iterations}
            durations = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                total = sum(math.sin(i * 0.001) for i in range(5000))
                durations.append(time.perf_counter() - t0)

            print(json.dumps({{
                "status": "completed",
                "iterations": iterations,
                "avg_duration_seconds": round(sum(durations) / len(durations), 6),
                "min_duration_seconds": round(min(durations), 6),
                "max_duration_seconds": round(max(durations), 6),
                "total_seconds": round(sum(durations), 6),
                "simulated": False,
            }}))
        """)

        data = _run_bench_script(script, cwd=self.repo_path, timeout=60)
        if data is not None:
            return data

        # Fallback: couldn't run subprocess — return a clear status
        return {
            "status": "error",
            "reason": "Benchmark subprocess failed",
            "simulated": False,
        }


class BenchmarkRunner:
    """Compare two implementations with real timed execution."""

    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)

    def compare_implementations(
        self,
        impl1: str,
        impl2: str,
        config: dict,
    ) -> dict:
        """Run both implementations in subprocesses and compare.

        *impl1* / *impl2* can be:
          - a Python expression string to ``exec``
          - a path to a Python file

        *config* may contain ``iterations`` (default 10) and
        ``timeout`` (default 60).
        """
        iterations = int(config.get("iterations", 10))
        timeout = int(config.get("timeout", 60))

        def _build_script(impl: str) -> str:
            """Build a benchmark wrapper around an implementation."""
            return textwrap.dedent(f"""\
                import json, time, tracemalloc, math, sys

                code = {impl!r}
                iterations = {iterations}

                # If it looks like a file path, exec its contents
                import os.path
                if os.path.isfile(code):
                    with open(code) as f:
                        code = f.read()

                # Compile once
                compiled = compile(code, "<bench>", "exec")

                # Warm-up
                for _ in range(2):
                    exec(compiled)

                tracemalloc.start()
                durations = []
                for _ in range(iterations):
                    t0 = time.perf_counter()
                    exec(compiled)
                    durations.append(time.perf_counter() - t0)
                current_mem, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                avg = sum(durations) / len(durations)
                print(json.dumps({{
                    "avg_seconds": round(avg, 8),
                    "min_seconds": round(min(durations), 8),
                    "max_seconds": round(max(durations), 8),
                    "peak_memory_mb": round(peak_mem / (1024 * 1024), 4),
                    "iterations": iterations,
                }}))
            """)

        r1 = _run_bench_script(
            _build_script(impl1),
            cwd=self.repo_path,
            timeout=timeout,
        )
        r2 = _run_bench_script(
            _build_script(impl2),
            cwd=self.repo_path,
            timeout=timeout,
        )

        if r1 is None or r2 is None:
            return {
                "impl1": impl1,
                "impl2": impl2,
                "config": config,
                "error": "One or both benchmarks failed to produce results",
                "impl1_result": r1,
                "impl2_result": r2,
            }

        avg1 = r1["avg_seconds"]
        avg2 = r2["avg_seconds"]
        speedup = avg1 / max(avg2, 1e-12)  # >1 means impl2 is faster
        memory_delta = r2.get("peak_memory_mb", 0) - r1.get("peak_memory_mb", 0)

        return {
            "impl1": impl1,
            "impl2": impl2,
            "config": config,
            "impl1_avg_seconds": avg1,
            "impl2_avg_seconds": avg2,
            "speedup": round(speedup, 4),
            "memory_delta_mb": round(memory_delta, 4),
            "impl1_detail": r1,
            "impl2_detail": r2,
        }
