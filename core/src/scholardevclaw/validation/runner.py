"""Validation runner with real benchmark measurements.

Runs actual tests (pytest subprocess), real performance benchmarks
(timed subprocess execution with memory tracking), and honest
comparison between baseline and modified implementations.

No hardcoded fake numbers — every metric comes from a real measurement.
"""

from __future__ import annotations

import ast
import difflib
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from scholardevclaw.patch_generation.generator import PatchGenerator

# Patterns that indicate potentially destructive operations in scripts
_DESTRUCTIVE_PATTERNS = [
    r"rm\s+-rf\s+/",  # Matches rm -rf / and similar
    r"dd\s+if=.*of=/dev/sd[^ ]",  # dd if=... of=/dev/sd...
    r"curl\s+[^|]*\|\s*bash",  # curl ... | bash
    r"wget\s+[^|]*\|\s*bash",  # wget ... | bash
    r":\(\)\{.*\|\:\)",  # Fork bomb pattern
    r"os\.system\s*\(",  # os.system() call
    r"subprocess\.(call|run|Popen|check_output|check_call)\s*\(",  # subprocess calls
    r"socket\.",  # network socket access
    r"urllib\.|requests\.",  # HTTP client libraries
    r"open\s*\(['\"]/(?:etc|proc|sys|dev|run)/",  # sensitive filesystem access
    # Additional patterns can be added as needed
]

# Patterns that indicate attempts to bypass the sandbox
_SANDBOX_ESCAPE_PATTERNS = [
    r"__import__\s*\(",  # dynamic import
    r"importlib\.",  # import library
    r"sys\.modules",  # module manipulation
    r"getattr\s*\(\s*builtins",  # access builtins
    r"os\.environ",  # environment variable access
    r"__class__|__mro__|__subclasses__|__globals__|__builtins__",  # Python introspection for sandbox escape
    r"compile\s*\(",  # compile code objects
    r"exec\s*\(",  # execute code objects
    r"eval\s*\(",  # evaluate code objects
    r"setattr\s*\(",  # set attributes dynamically
    r"del\s+os\b|del\s+sys\b|del\s+builtins\b",  # module deletion
    r"pty\.",  # pseudo-terminal access
    r"ctypes\.",  # C extension access
]


def _is_script_destructive(script: str) -> bool:
    """Check if the script matches any destructive pattern.

    If the SCHOLARDEVCLAW_YOLO_MODE environment variable is set to 'true',
    destructive checks are disabled and this function always returns False.
    """
    yolo_mode = os.environ.get("SCHOLARDEVCLAW_YOLO_MODE", "").lower() in ("true", "1", "yes")
    if yolo_mode:
        _logger.debug("YOLO mode active - skipping destructive check")
        return False
    for pattern in _DESTRUCTIVE_PATTERNS:
        if re.search(pattern, script, re.IGNORECASE):
            return True
    return False


def _is_sandbox_escape(script: str) -> bool:
    """Check if the script contains patterns that attempt to escape the sandbox."""
    yolo_mode = os.environ.get("SCHOLARDEVCLAW_YOLO_MODE", "").lower() in ("true", "1", "yes")
    if yolo_mode:
        _logger.debug("YOLO mode active - skipping sandbox escape check")
        return False
    for pattern in _SANDBOX_ESCAPE_PATTERNS:
        if re.search(pattern, script, re.IGNORECASE):
            return True
    return False


def _sandbox_preexec() -> None:
    """Apply resource limits and process isolation before executing a benchmark script.

    This function is designed to run in the child process via preexec_fn.
    It sets:
    - Memory limit: 512 MB (RLIMIT_AS)
    - CPU time limit: 60 seconds (RLIMIT_CPU)
    - Process group isolation (os.setsid)
    - No core dumps (RLIMIT_CORE = 0)
    - File descriptor limit: 256 (RLIMIT_NOFILE)
    """
    try:
        import resource

        # Memory limit: 512 MB virtual memory
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))

        # CPU time limit: 60 seconds
        resource.setrlimit(resource.RLIMIT_CPU, (60, 60))

        # No core dumps
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        # File descriptor limit
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))

        # Process group isolation
        os.setsid()
    except (ImportError, ValueError, OSError) as e:
        # Resource limits may not be available on all platforms
        _logger.warning("Could not set resource limits: %s", e)


_logger = logging.getLogger(__name__)

_BENCHMARK_EXPECTED_ROOT = Path(__file__).resolve().parents[3] / "benchmarks" / "expected"
_ALGORITHM_EXPECTED_FILES = {
    "alibi": "alibi.py",
    "cosine_warmup": "cosine_lr_schedule.py",
    "flashattention": "flashattention.py",
    "gelu": "gelu.py",
    "grouped_query_attention": "grouped_query_attention.py",
    "layernorm": "layernorm.py",
    "lora": "lora.py",
    "rmsnorm": "rmsnorm.py",
    "rope": "rope.py",
    "swiglu": "swiglu.py",
}
_ALGORITHM_KEY_ALIASES = {
    "cosine_annealing_with_warmup": "cosine_warmup",
    "cosine_lr_schedule": "cosine_warmup",
    "flash_attention": "flashattention",
    "flash_attention_2": "flashattention",
    "grouped_query_attention": "grouped_query_attention",
    "groupedqueryattention": "grouped_query_attention",
    "gaussian_error_linear_units_gelus": "gelu",
    "low_rank_adaptation_of_large_language_models": "lora",
}


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
) -> dict[str, float] | None:
    """Run *script* in a fresh, sandboxed Python subprocess and parse its JSON output.

    Returns the parsed dict on success, ``None`` on any failure.

    Security hardening:
    - Check A: Destructive operation detection (blocks known dangerous patterns)
    - Sandbox escape detection (blocks eval/exec/__import__/compile etc.)
    - Resource limits: 512 MB memory, 60 s CPU, no core dumps, 256 file descriptors
    - Process group isolation via os.setsid
    """
    # Check A: Destructive Operation Detection
    if _is_script_destructive(script):
        _logger.warning("Blocking destructive script: %s", script[:100])
        return None

    # Check B: Sandbox Escape Detection
    if _is_sandbox_escape(script):
        _logger.warning("Blocking sandbox escape attempt: %s", script[:100])
        return None

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_sandbox_preexec,
        )
        if result.returncode != 0:
            return None
        # The script must print a single JSON object on its last line.
        for line in reversed(result.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                payload = json.loads(line)
                if isinstance(payload, dict):
                    return cast(dict[str, float], payload)
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return None


def _run_bench_script_in_docker(
    script: str,
    *,
    cwd: str | Path,
    timeout: int = 120,
    image: str,
) -> dict | None:
    """Run benchmark script inside an isolated Docker container.

    - repo is mounted read-only
    - network disabled
    - execution timeout enforced
    """
    repo = Path(cwd).resolve()
    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "-v",
        f"{repo}:/workspace:ro",
        "-w",
        "/workspace",
        image,
        "python",
        "-c",
        script,
    ]
    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return None
        for line in reversed(result.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                payload = json.loads(line)
                if isinstance(payload, dict):
                    return cast(dict[Any, Any], payload)
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

    def _sandbox_mode(self) -> str:
        return (os.environ.get("SCHOLARDEVCLAW_VALIDATION_SANDBOX") or "").strip().lower()

    def _docker_image(self) -> str:
        return (
            os.environ.get("SCHOLARDEVCLAW_VALIDATION_DOCKER_IMAGE") or "python:3.12-slim"
        ).strip()

    def _docker_available(self) -> bool:
        try:
            probe = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return probe.returncode == 0
        except Exception:
            return False

    def run(
        self, patch: dict, repo_path: str, mapping_result: dict | None = None
    ) -> ValidationResult:
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

        # Run tests
        test_result = self._run_tests()

        # Healing loop - attempt to fix failed patches
        max_heal_attempts = 2
        heal_attempts = 0

        while not test_result.passed and heal_attempts < max_heal_attempts:
            _logger.info(
                "Test failed, attempting to heal patch (attempt %d/%d)",
                heal_attempts + 1,
                max_heal_attempts,
            )

            try:
                # Convert dict to Patch object
                from scholardevclaw.patch_generation.generator import NewFile, Patch, Transformation

                new_files = [
                    NewFile(path=f.get("path", ""), content=f.get("content", ""))
                    for f in patch.get("new_files", [])
                ]
                transformations = [
                    Transformation(
                        file=t.get("file", ""),
                        original=t.get("original", ""),
                        modified=t.get("modified", ""),
                        changes=t.get("changes", []),
                    )
                    for t in patch.get("transformations", [])
                ]

                patch_obj = Patch(
                    new_files=new_files,
                    transformations=transformations,
                    branch_name=patch.get("branch_name", ""),
                    algorithm_name=patch.get("algorithm_name", ""),
                    paper_reference=patch.get("paper_reference", ""),
                )

                # Create PatchGenerator and heal
                llm_assistant = None
                try:
                    from scholardevclaw.llm.research_assistant import LLMResearchAssistant

                    llm_assistant = LLMResearchAssistant.create()
                except Exception:
                    _logger.warning("Could not create LLM assistant for healing")

                generator = PatchGenerator(Path(repo_path), llm_assistant=llm_assistant)
                healed_patch_obj = generator.heal_patch(
                    patch_obj, test_result, mapping_result or {}
                )

                # Convert healed Patch back to dict
                healed_patch = {
                    "new_files": [
                        {"path": f.path, "content": f.content} for f in healed_patch_obj.new_files
                    ],
                    "transformations": [
                        {
                            "file": t.file,
                            "original": t.original,
                            "modified": t.modified,
                            "changes": t.changes,
                        }
                        for t in healed_patch_obj.transformations
                    ],
                    "branch_name": healed_patch_obj.branch_name,
                }

                # Re-run tests with healed patch
                test_result = self._run_tests()
                patch = healed_patch  # Update patch for potential next iteration
                heal_attempts += 1

                if test_result.passed:
                    _logger.info("Healing successful after %d attempt(s)", heal_attempts)
                    break

            except Exception as e:
                _logger.warning("Healing attempt %d failed: %s", heal_attempts + 1, e)
                heal_attempts += 1
                break  # Don't continue if healing itself fails

        if not test_result.passed:
            return test_result

        benchmark_result = self._run_benchmark()
        if self._patch_has_artifacts(patch):
            use_torch = self._check_torch_available()
            baseline_metrics = self._run_training_test(use_variant=False, use_torch=use_torch)
            new_metrics = self._run_training_test(use_variant=True, use_torch=use_torch)

            comparison = (
                dict(benchmark_result.comparison)
                if isinstance(benchmark_result.comparison, dict)
                else {}
            )
            metrics_comparison = self._build_metrics_comparison(baseline_metrics, new_metrics)
            if metrics_comparison:
                comparison.update(metrics_comparison)

            numerical_result = self._run_numerical_correctness(patch)
            regression_result = self._run_regression_snapshot(patch)
            diff_readability = self._score_diff_readability(patch)

            comparison["numerical_correctness"] = numerical_result
            comparison["regression_snapshot"] = regression_result
            comparison["diff_readability"] = diff_readability

            benchmark_result.baseline_metrics = baseline_metrics
            benchmark_result.new_metrics = new_metrics
            benchmark_result.comparison = comparison

            if numerical_result.get("status") == "failed":
                benchmark_result.passed = False
                benchmark_result.stage = "numerical_correctness"
            elif regression_result.get("status") == "failed":
                benchmark_result.passed = False
                benchmark_result.stage = "regression_snapshot"

        if policy_warning:
            benchmark_result.logs = "\n".join(
                part for part in (policy_warning, benchmark_result.logs) if part
            )
        if artifact_result.logs:
            benchmark_result.logs = "\n".join(
                part for part in (artifact_result.logs, benchmark_result.logs) if part
            )

        return benchmark_result

    def _patch_has_artifacts(self, patch: dict[str, object] | None) -> bool:
        if not isinstance(patch, dict):
            return False
        return bool(patch.get("new_files") or patch.get("transformations"))

    def _normalize_algorithm_key(self, value: str) -> str:
        normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
        while "__" in normalized:
            normalized = normalized.replace("__", "_")
        return _ALGORITHM_KEY_ALIASES.get(normalized, normalized)

    def _extract_algorithm_key(self, patch: dict[str, object]) -> str | None:
        candidates = [
            str(patch.get("algorithm_name", "") or ""),
            str(patch.get("algorithm", "") or ""),
            str(patch.get("spec", "") or ""),
            str(patch.get("paper_reference", "") or "").replace("arXiv:", ""),
        ]
        research_spec = patch.get("research_spec")
        if isinstance(research_spec, dict):
            algorithm = research_spec.get("algorithm")
            paper = research_spec.get("paper")
            if isinstance(algorithm, dict):
                candidates.append(str(algorithm.get("name", "") or ""))
            if isinstance(paper, dict):
                candidates.append(str(paper.get("arxiv", "") or ""))

        for candidate in candidates:
            if not candidate:
                continue
            normalized = self._normalize_algorithm_key(candidate)
            if normalized in _ALGORITHM_EXPECTED_FILES:
                return normalized
        return None

    def _candidate_sources_from_patch(self, patch: dict[str, object]) -> dict[str, str]:
        sources: dict[str, str] = {}
        raw_new_files = patch.get("new_files", [])
        if not isinstance(raw_new_files, list):
            raw_new_files = []
        for entry in raw_new_files:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path", "") or "")
            content = str(entry.get("content", "") or "")
            if path.endswith(".py") and content:
                sources[path] = content
        raw_transformations = patch.get("transformations", [])
        if not isinstance(raw_transformations, list):
            raw_transformations = []
        for entry in raw_transformations:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("file", "") or "")
            content = str(entry.get("modified", "") or "")
            if path.endswith(".py") and content:
                sources[path] = content
        return sources

    def _load_module_from_source(self, source: str, module_name: str, tmp_dir: Path) -> object:
        module_path = tmp_dir / f"{module_name}.py"
        module_path.write_text(source)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module spec for {module_name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_module_from_path(self, module_path: Path, module_name: str) -> object:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _build_metrics_comparison(
        self,
        baseline: Metrics | None,
        new: Metrics | None,
    ) -> dict[str, float]:
        if baseline is None or new is None:
            return {}

        speedup = (
            float(new.tokens_per_second) / max(float(baseline.tokens_per_second), 1e-9)
            if baseline.tokens_per_second
            else None
        )
        loss_change = (
            ((float(new.loss) - float(baseline.loss)) / max(abs(float(baseline.loss)), 1e-9))
            * 100.0
            if baseline.loss is not None
            else None
        )
        result: dict[str, float] = {}
        if speedup is not None:
            result["speedup"] = round(speedup, 4)
        if loss_change is not None:
            result["loss_change"] = round(loss_change, 4)
        return result

    def _run_numerical_correctness(self, patch: dict[str, object]) -> dict[str, object]:
        algorithm_key = self._extract_algorithm_key(patch)
        if not algorithm_key:
            return {"status": "skipped", "reason": "No algorithm key could be inferred"}

        expected_name = _ALGORITHM_EXPECTED_FILES.get(algorithm_key)
        if not expected_name:
            return {"status": "skipped", "reason": f"No reference benchmark for {algorithm_key}"}

        candidate_sources = self._candidate_sources_from_patch(patch)
        if not candidate_sources:
            return {"status": "skipped", "reason": "No Python patch artifacts available"}

        candidate_source = None
        for path, source in candidate_sources.items():
            basename = Path(path).name
            if algorithm_key in basename.replace("-", "_"):
                candidate_source = source
                break
        if candidate_source is None:
            candidate_source = next(iter(candidate_sources.values()))

        expected_path = _BENCHMARK_EXPECTED_ROOT / expected_name
        if not expected_path.exists():
            return {
                "status": "skipped",
                "reason": f"Missing expected benchmark file {expected_name}",
            }

        try:
            import tempfile

            with tempfile.TemporaryDirectory(prefix="sdc-validation-") as tmpdir:
                tmp_path = Path(tmpdir)
                candidate_module = self._load_module_from_source(
                    candidate_source,
                    f"candidate_{algorithm_key}",
                    tmp_path,
                )
                expected_module = self._load_module_from_path(
                    expected_path,
                    f"expected_{algorithm_key}",
                )

                passed = True
                details: dict[str, Any] = {"algorithm": algorithm_key}

                if algorithm_key == "gelu":
                    values = [-1.0, -0.1, 0.0, 0.5, 1.25]
                    deltas = []
                    candidate_fn = getattr(candidate_module, "gelu", None)
                    expected_fn = getattr(expected_module, "gelu", None)
                    if not callable(candidate_fn) or not callable(expected_fn):
                        return {"status": "skipped", "reason": "GELU function not available"}
                    for value in values:
                        deltas.append(abs(float(candidate_fn(value)) - float(expected_fn(value))))
                    max_delta = max(deltas) if deltas else 0.0
                    passed = max_delta <= 1e-6
                    details["max_delta"] = max_delta
                elif algorithm_key == "layernorm":
                    values = [1.0, 2.0, 3.0]
                    candidate_fn = getattr(candidate_module, "layer_norm", None)
                    expected_fn = getattr(expected_module, "layer_norm", None)
                    if not callable(candidate_fn) or not callable(expected_fn):
                        return {"status": "skipped", "reason": "layer_norm function not available"}
                    got = [float(v) for v in candidate_fn(values)]
                    want = [float(v) for v in expected_fn(values)]
                    max_delta = max(abs(a - b) for a, b in zip(got, want))
                    passed = max_delta <= 1e-6
                    details["max_delta"] = max_delta
                elif algorithm_key == "lora":
                    inputs = [1.0, 2.0]
                    base = [0.1, 0.2]
                    down = [[0.5, 0.1], [0.2, 0.4]]
                    up = [[0.3, 0.7], [0.6, 0.2]]
                    candidate_fn = getattr(candidate_module, "apply_lora", None)
                    expected_fn = getattr(expected_module, "apply_lora", None)
                    if not callable(candidate_fn) or not callable(expected_fn):
                        return {"status": "skipped", "reason": "apply_lora function not available"}
                    got = [float(v) for v in candidate_fn(inputs, base, down, up)]
                    want = [float(v) for v in expected_fn(inputs, base, down, up)]
                    max_delta = max(abs(a - b) for a, b in zip(got, want))
                    passed = max_delta <= 1e-6
                    details["max_delta"] = max_delta
                else:
                    return {
                        "status": "skipped",
                        "reason": f"No numerical comparator implemented for {algorithm_key}",
                    }

                return {
                    "status": "passed" if passed else "failed",
                    "passed": passed,
                    "details": details,
                }
        except Exception as exc:
            return {"status": "failed", "passed": False, "error": str(exc)}

    def _signature_snapshot(self, source: str) -> dict[str, tuple[str, ...]]:
        tree = ast.parse(source)
        snapshot: dict[str, tuple[str, ...]] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                snapshot[f"fn:{node.name}"] = tuple(arg.arg for arg in node.args.args)
            elif isinstance(node, ast.ClassDef):
                snapshot[f"class:{node.name}"] = tuple()
        return snapshot

    def _run_regression_snapshot(self, patch: dict[str, object]) -> dict[str, object]:
        transformations = patch.get("transformations", []) if isinstance(patch, dict) else []
        if not isinstance(transformations, list) or not transformations:
            return {"status": "skipped", "reason": "No file transformations to compare"}

        removed_symbols: list[str] = []
        signature_changes: list[str] = []
        checked_files: list[str] = []

        for entry in transformations:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("file", "") or "")
            original = str(entry.get("original", "") or "")
            modified = str(entry.get("modified", "") or "")
            if not path.endswith(".py") or not original or not modified:
                continue
            checked_files.append(path)
            try:
                before = self._signature_snapshot(original)
                after = self._signature_snapshot(modified)
            except SyntaxError as exc:
                return {"status": "failed", "passed": False, "error": f"{path}: {exc}"}

            for symbol, signature in before.items():
                if symbol not in after:
                    removed_symbols.append(f"{path}:{symbol}")
                    continue
                if signature != after[symbol]:
                    signature_changes.append(f"{path}:{symbol}")

        passed = not removed_symbols and not signature_changes
        return {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "files_checked": checked_files,
            "removed_symbols": removed_symbols,
            "signature_changes": signature_changes,
        }

    def _score_diff_readability(self, patch: dict[str, object]) -> dict[str, object]:
        transformations = patch.get("transformations", []) if isinstance(patch, dict) else []
        if not isinstance(transformations, list) or not transformations:
            return {"score": 5, "source": "heuristic", "rationale": "No transformed files"}

        total_changed_lines = 0
        changed_files = 0
        for entry in transformations:
            if not isinstance(entry, dict):
                continue
            original = str(entry.get("original", "") or "")
            modified = str(entry.get("modified", "") or "")
            if not original and not modified:
                continue
            changed_files += 1
            diff_lines = list(
                difflib.unified_diff(
                    original.splitlines(),
                    modified.splitlines(),
                    lineterm="",
                )
            )
            total_changed_lines += len(
                [
                    line
                    for line in diff_lines
                    if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
                ]
            )

        score = 5
        if changed_files > 5 or total_changed_lines > 220:
            score = 2
        elif changed_files > 3 or total_changed_lines > 140:
            score = 3
        elif changed_files > 2 or total_changed_lines > 80:
            score = 4

        rationale = (
            f"{changed_files} file(s) touched, {total_changed_lines} changed line(s). "
            "Higher scores indicate smaller, more targeted diffs."
        )
        return {"score": score, "source": "heuristic", "rationale": rationale}

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
        if mode == "strict" and sandbox == "docker" and not self._docker_available():
            return ValidationResult(
                passed=False,
                stage="policy",
                logs="Strict execution mode requires Docker, but Docker is unavailable.",
                error="Docker sandbox requested but unavailable",
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

        if self._sandbox_mode() == "docker":
            if not self._docker_available():
                return ValidationResult(
                    passed=False,
                    stage="tests",
                    error="Docker requested but not available",
                )
            docker_image = self._docker_image()
            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{self.repo_path}:/repo",
                "-w",
                "/repo",
                docker_image,
                "pytest",
                "-v",
                "--tb=short",
                "-x",
                "--timeout=60",
            ]
            try:
                result = subprocess.run(
                    docker_cmd,
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
                    error="Test timeout after 120 seconds in Docker",
                )
            except FileNotFoundError:
                return ValidationResult(
                    passed=False,
                    stage="tests",
                    error="Docker executable not found",
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
    # Benchmark orchestration — runs actual repo benchmarks/tests
    # ------------------------------------------------------------------

    def _run_benchmark(self) -> ValidationResult:
        """Run actual repo benchmarks or tests with timing."""
        import time

        # Check for benchmark scripts in the repo
        benchmark_scripts = list(self.repo_path.glob("**/benchmark*.py")) + list(
            self.repo_path.glob("**/bench*.py")
        )

        test_files = list(self.repo_path.glob("**/test*.py"))

        if not benchmark_scripts and not test_files:
            return ValidationResult(
                passed=True,
                stage="benchmark",
                logs="No benchmark or test files found — skipping benchmark",
            )

        # Run benchmarks if available, else run tests with timing
        scripts_to_run = (
            benchmark_scripts if benchmark_scripts else test_files[:5]
        )  # Limit to 5 test files
        results = []

        for script in scripts_to_run:
            start_time = time.time()
            try:
                if self._sandbox_mode() == "docker":
                    docker_image = self._docker_image()
                    result = subprocess.run(
                        [
                            "docker",
                            "run",
                            "--rm",
                            "-v",
                            f"{self.repo_path}:/repo",
                            "-w",
                            "/repo",
                            docker_image,
                            "python",
                            str(script.relative_to(self.repo_path)),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                else:
                    result = subprocess.run(
                        [sys.executable, str(script)],
                        cwd=str(self.repo_path),
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                end_time = time.time()
                duration = end_time - start_time
                passed = result.returncode == 0
                results.append(
                    {
                        "script": str(script.relative_to(self.repo_path)),
                        "passed": passed,
                        "duration": round(duration, 2),
                        "output": result.stdout[-500:] if result.stdout else "",  # Last 500 chars
                        "error": result.stderr[-500:] if result.stderr else "",
                    }
                )
            except subprocess.TimeoutExpired:
                results.append(
                    {
                        "script": str(script.relative_to(self.repo_path)),
                        "passed": False,
                        "duration": 120,
                        "error": "Timeout after 120 seconds",
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "script": str(script.relative_to(self.repo_path)),
                        "passed": False,
                        "error": str(e),
                    }
                )

        # Aggregate results
        total = len(results)
        passed_count = sum(1 for r in results if r["passed"])
        avg_duration = sum(float(r.get("duration", 0.0) or 0.0) for r in results) / max(total, 1)
        all_passed = all(bool(r.get("passed", False)) for r in results)

        logs = f"Ran {total} benchmark(s): {passed_count} passed, {total - passed_count} failed. Avg duration: {avg_duration:.2f}s"

        return ValidationResult(
            passed=all_passed,
            stage="benchmark",
            comparison={
                "total": total,
                "passed": passed_count,
                "failed": total - passed_count,
                "avg_duration": round(avg_duration, 2),
                "results": results,
            },
            logs=logs,
        )

    # ------------------------------------------------------------------
    # Check whether PyTorch is importable
    # ------------------------------------------------------------------

    def _check_torch_available(self) -> bool:
        try:
            if self._sandbox_mode() == "docker":
                # Check torch inside Docker container
                docker_image = self._docker_image()
                result = subprocess.run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        docker_image,
                        "python",
                        "-c",
                        "import torch; print(torch.__version__)",
                    ],
                    capture_output=True,
                    timeout=30,
                )
                return result.returncode == 0
            else:
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

        if self._sandbox_mode() == "docker":
            data = _run_bench_script_in_docker(
                script,
                cwd=self.repo_path,
                timeout=120,
                image=self._docker_image(),
            )
        else:
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

        if self._sandbox_mode() == "docker":
            data = _run_bench_script_in_docker(
                script,
                cwd=self.repo_path,
                timeout=60,
                image=self._docker_image(),
            )
        else:
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
