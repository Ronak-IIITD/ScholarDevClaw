"""
Fuzzing integration for robustness testing.

Supports:
- Python fuzzing with custom fuzzing strategies
- libFuzzer wrapper (when available)
- AFL wrapper (when available)
- Coverage-guided fuzzing
- Corpus management
- Parallel fuzzing via ProcessPoolExecutor
- Content-hash deduplication for crash tracking
"""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FuzzTarget:
    """A function to fuzz"""

    name: str
    module_path: str
    function_name: str
    input_types: list[str]  # "bytes", "string", "int", "float"
    output_check: str | None = None  # "none", "exception", "return"


@dataclass
class FuzzResult:
    """Result of a fuzzing run"""

    target: str
    runs: int
    unique_crashes: int
    unique_hangs: int
    coverage_percent: float
    duration_seconds: float
    corpus_size: int
    findings: list[dict] = field(default_factory=list)


class PythonFuzzer:
    """Pure Python fuzzing without external tools.

    Supports parallel fuzzing via :meth:`fuzz_parallel` and content-hash
    deduplication for crash/hang tracking.
    """

    def __init__(self):
        self.runs = 0
        self.crashes: list[tuple[Any, str]] = []
        self.hangs: list[tuple[Any, float]] = []
        self.seeds: list[bytes] = []
        # Content-hash deduplication sets (SHA-256 of serialised input)
        self._crash_hashes: set[str] = set()
        self._hang_hashes: set[str] = set()

    def add_seed(self, data: bytes):
        """Add a seed input"""
        self.seeds.append(data)

    @staticmethod
    def _content_hash(data: Any) -> str:
        """Return a SHA-256 hex digest for deduplication."""
        return hashlib.sha256(repr(data).encode()).hexdigest()

    def _is_new_crash(self, inputs: Any) -> bool:
        """Check if this crash is unique via content-hash deduplication."""
        h = self._content_hash(inputs)
        if h in self._crash_hashes:
            return False
        self._crash_hashes.add(h)
        return True

    def _is_new_hang(self, inputs: Any) -> bool:
        """Check if this hang is unique via content-hash deduplication."""
        h = self._content_hash(inputs)
        if h in self._hang_hashes:
            return False
        self._hang_hashes.add(h)
        return True

    def _generate_input(self, input_type: str) -> Any:
        """Generate random input based on type.

        Uses optimised C-implemented functions (os.urandom, random.choices)
        instead of Python-level loops for orders-of-magnitude speedup.
        """
        if input_type == "bytes":
            length = random.randint(0, 10000)
            return os.urandom(length)

        elif input_type == "string":
            length = random.randint(0, 1000)
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-.,;:!?()[]{}"
            return "".join(random.choices(chars, k=length))

        elif input_type == "int":
            return random.randint(-(2**31), 2**31 - 1)

        elif input_type == "float":
            return random.uniform(-1e10, 1e10)

        elif input_type == "list":
            length = random.randint(0, 100)
            return random.choices(range(256), k=length)

        elif input_type == "dict":
            length = random.randint(0, 50)
            return {f"key_{i}": f"value_{i}" for i in range(length)}

        return b""

    def fuzz(
        self,
        target: FuzzTarget,
        runs: int = 10000,
        timeout: float = 5.0,
    ) -> FuzzResult:
        """Run fuzzing on a target (single-threaded)."""
        import importlib.util
        import time

        start_time = time.time()

        try:
            spec = importlib.util.spec_from_file_location("target_module", target.module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module spec for {target.module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            target_func = getattr(module, target.function_name)
        except Exception as e:
            return FuzzResult(
                target=target.name,
                runs=0,
                unique_crashes=0,
                unique_hangs=0,
                coverage_percent=0.0,
                duration_seconds=time.time() - start_time,
                corpus_size=len(self.seeds),
                findings=[{"type": "load_error", "error": str(e)}],
            )

        for i in range(runs):
            inputs = []
            for input_type in target.input_types:
                if self.seeds and random.random() < 0.1:
                    inputs.append(random.choice(self.seeds))
                else:
                    inputs.append(self._generate_input(input_type))

            self.runs = i + 1

            try:
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError()

                if timeout > 0:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(timeout))

                start = time.time()
                target_func(*inputs)
                elapsed = time.time() - start

                if timeout > 0:
                    signal.alarm(0)

                if elapsed > timeout * 0.8:
                    if self._is_new_hang(inputs):
                        self.hangs.append((inputs, elapsed))

                if target.output_check == "exception":
                    pass

            except TimeoutError:
                if self._is_new_hang(inputs):
                    self.hangs.append((inputs, timeout))

            except Exception as e:
                if self._is_new_crash(inputs):
                    self.crashes.append((inputs, str(e)))

            if i % 1000 == 0 and i > 0:
                if self.seeds:
                    self.seeds.extend([c[0] for c in self.crashes[-10:]])
                self.seeds = self.seeds[-1000:]

        duration = time.time() - start_time

        findings = []
        for crash in self.crashes:
            findings.append(
                {
                    "type": "crash",
                    "input": str(crash[0])[:200],
                    "error": crash[1],
                }
            )

        for hang in self.hangs:
            findings.append(
                {
                    "type": "hang",
                    "input": str(hang[0])[:200],
                    "duration": hang[1],
                }
            )

        return FuzzResult(
            target=target.name,
            runs=runs,
            unique_crashes=len(self._crash_hashes),
            unique_hangs=len(self._hang_hashes),
            coverage_percent=0.0,
            duration_seconds=duration,
            corpus_size=len(self.seeds),
            findings=findings[:50],
        )

    # ------------------------------------------------------------------
    # Parallel fuzzing
    # ------------------------------------------------------------------

    @staticmethod
    def _fuzz_chunk(
        target_name: str,
        module_path: str,
        function_name: str,
        input_types: list[str],
        chunk_size: int,
        timeout: float,
        seeds: list[bytes],
    ) -> tuple[list[tuple[Any, str]], list[tuple[Any, float]], int]:
        """Worker function executed in a subprocess for parallel fuzzing.

        Returns (crashes, hangs, total_runs) for the given chunk.
        """
        import importlib.util
        import time

        try:
            spec = importlib.util.spec_from_file_location("target_module", module_path)
            if spec is None or spec.loader is None:
                return ([], [], 0)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            target_func = getattr(module, function_name)
        except Exception:
            return ([], [], 0)

        crashes: list[tuple[Any, str]] = []
        hangs: list[tuple[Any, float]] = []

        for _ in range(chunk_size):
            inputs = []
            for input_type in input_types:
                if seeds and random.random() < 0.1:
                    inputs.append(random.choice(seeds))
                else:
                    if input_type == "bytes":
                        inputs.append(os.urandom(random.randint(0, 10000)))
                    elif input_type == "string":
                        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-.,;:!?()[]{}"
                        inputs.append("".join(random.choices(chars, k=random.randint(0, 1000))))
                    elif input_type == "int":
                        inputs.append(random.randint(-(2**31), 2**31 - 1))
                    elif input_type == "float":
                        inputs.append(random.uniform(-1e10, 1e10))
                    elif input_type == "list":
                        inputs.append(random.choices(range(256), k=random.randint(0, 100)))
                    elif input_type == "dict":
                        length = random.randint(0, 50)
                        inputs.append({f"key_{i}": f"value_{i}" for i in range(length)})
                    else:
                        inputs.append(b"")

            try:
                start = time.time()
                target_func(*inputs)
                elapsed = time.time() - start
                if elapsed > timeout * 0.8:
                    hangs.append((inputs, elapsed))
            except Exception as e:
                crashes.append((inputs, str(e)))

        return (crashes, hangs, chunk_size)

    def fuzz_parallel(
        self,
        target: FuzzTarget,
        runs: int = 10000,
        timeout: float = 5.0,
        max_workers: int | None = None,
    ) -> FuzzResult:
        """Run fuzzing across multiple CPU cores using ProcessPoolExecutor.

        Splits *runs* into equal chunks and distributes them across workers.
        Each worker loads the target module independently and runs its chunk.
        """
        import importlib.util
        import time

        start_time = time.time()

        # Validate the target module is loadable
        try:
            spec = importlib.util.spec_from_file_location("target_module", target.module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module spec for {target.module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            return FuzzResult(
                target=target.name,
                runs=0,
                unique_crashes=0,
                unique_hangs=0,
                coverage_percent=0.0,
                duration_seconds=time.time() - start_time,
                corpus_size=len(self.seeds),
                findings=[{"type": "load_error", "error": str(e)}],
            )

        num_workers = max_workers or min(8, os.cpu_count() or 1)
        chunk_size = max(1, runs // num_workers)
        remaining = runs - chunk_size * num_workers

        all_crashes: list[tuple[Any, str]] = []
        all_hangs: list[tuple[Any, float]] = []
        total_runs = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for _ in range(num_workers):
                futures.append(
                    executor.submit(
                        self._fuzz_chunk,
                        target.name,
                        target.module_path,
                        target.function_name,
                        target.input_types,
                        chunk_size,
                        timeout,
                        self.seeds,
                    )
                )
            if remaining > 0:
                futures.append(
                    executor.submit(
                        self._fuzz_chunk,
                        target.name,
                        target.module_path,
                        target.function_name,
                        target.input_types,
                        remaining,
                        timeout,
                        self.seeds,
                    )
                )

            for future in as_completed(futures):
                try:
                    crashes, hangs, runs_done = future.result()
                    all_crashes.extend(crashes)
                    all_hangs.extend(hangs)
                    total_runs += runs_done
                except Exception:
                    pass

        # Deduplicate using content-hash
        for inputs, err_msg in all_crashes:
            if self._is_new_crash(inputs):
                self.crashes.append((inputs, err_msg))
        for inputs, dur in all_hangs:
            if self._is_new_hang(inputs):
                self.hangs.append((inputs, dur))

        duration = time.time() - start_time
        self.runs = total_runs

        findings: list[dict] = []
        for crash in self.crashes:
            findings.append({"type": "crash", "input": str(crash[0])[:200], "error": crash[1]})
        for hang in self.hangs:
            findings.append({"type": "hang", "input": str(hang[0])[:200], "duration": hang[1]})

        return FuzzResult(
            target=target.name,
            runs=total_runs,
            unique_crashes=len(self._crash_hashes),
            unique_hangs=len(self._hang_hashes),
            coverage_percent=0.0,
            duration_seconds=duration,
            corpus_size=len(self.seeds),
            findings=findings[:50],
        )


class LibFuzzerRunner:
    """Wrapper for libFuzzer-based fuzzing"""

    def __init__(self, compiler: str = "clang"):
        self.compiler = compiler

    def create_fuzz_target(self, source_file: Path, function_name: str, output_file: Path):
        """Create a libFuzzer-compatible C/C++ target"""
        fuzz_code = f"""
#include <stdint.h>
#include <stddef.h>
#include <string.h>

extern "C" int {function_name}(const uint8_t* data, size_t size);

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {{
    {function_name}(data, size);
    return 0;
}}
"""
        output_file.write_text(fuzz_code)

    def compile(self, source: Path, output: Path, libraries: list[str] | None = None) -> bool:
        """Compile a fuzzing target"""
        libs = libraries or []

        cmd = [
            self.compiler,
            "-fsanitize=fuzzer,address,undefined",
            "-g",
            "-O1",
            str(source),
            "-o",
            str(output),
        ] + libs

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False

    def run(self, binary: Path, corpus_dir: Path, timeout: int = 60) -> FuzzResult:
        """Run libFuzzer"""
        import time

        start_time = time.time()

        cmd = [str(binary), corpus_dir, f"-max_total_time={timeout}"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 10,
            )
            output = result.stdout + result.stderr

            runs = 0
            crashes = 0

            import re

            match = re.search(r"(\d+) execs", output)
            if match:
                runs = int(match.group(1))

            match = re.search(r"(\d+) crashes", output)
            if match:
                crashes = int(match.group(1))

            return FuzzResult(
                target=str(binary),
                runs=runs,
                unique_crashes=crashes,
                unique_hangs=0,
                coverage_percent=0.0,
                duration_seconds=time.time() - start_time,
                corpus_size=len(list(corpus_dir.glob("*"))) if corpus_dir.exists() else 0,
                findings=[{"raw": output[-1000:]}],
            )

        except subprocess.TimeoutExpired:
            return FuzzResult(
                target=str(binary),
                runs=runs,
                unique_crashes=0,
                unique_hangs=0,
                coverage_percent=0.0,
                duration_seconds=timeout,
                corpus_size=0,
                findings=[{"type": "timeout"}],
            )


class AFLRunner:
    """Wrapper for AFL (American Fuzzy Lop) fuzzing"""

    def __init__(self, afl_path: str = "afl-fuzz"):
        self.afl_path = afl_path

    def check_installed(self) -> bool:
        """Check if AFL is installed"""
        try:
            result = subprocess.run(
                [self.afl_path, "-V"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception:
            return False

    def run(
        self,
        binary: Path,
        input_dir: Path,
        output_dir: Path,
        timeout: int = 60,
    ) -> FuzzResult:
        """Run AFL fuzzing"""
        import time

        start_time = time.time()

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.afl_path,
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "-t",
            str(timeout * 1000),
            "-f",
            "input.txt",
            str(binary),
            "@@",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 10,
            )

            crashes = len(list(output_dir.glob("crashes/*"))) - 1
            hangs = len(list(output_dir.glob("hangs/*"))) - 1

            return FuzzResult(
                target=str(binary),
                runs=0,
                unique_crashes=max(0, crashes),
                unique_hangs=max(0, hangs),
                coverage_percent=0.0,
                duration_seconds=time.time() - start_time,
                corpus_size=len(list(input_dir.glob("*"))) if input_dir.exists() else 0,
                findings=[{"raw": result.stdout[-500:]}],
            )

        except subprocess.TimeoutExpired:
            return FuzzResult(
                target=str(binary),
                runs=0,
                unique_crashes=0,
                unique_hangs=0,
                coverage_percent=0.0,
                duration_seconds=timeout,
                corpus_size=0,
                findings=[{"type": "timeout"}],
            )


class FuzzerManager:
    """Unified fuzzing interface"""

    def __init__(self):
        self.python_fuzzer = PythonFuzzer()
        self.libfuzzer = LibFuzzerRunner()
        self.afl = AFLRunner()

    def fuzz_python_function(
        self,
        module_path: str,
        function_name: str,
        input_types: list[str],
        runs: int = 10000,
        parallel: bool = False,
        max_workers: int | None = None,
    ) -> FuzzResult:
        """Fuzz a Python function.

        When *parallel* is True, distributes work across CPU cores via
        :class:`ProcessPoolExecutor`.
        """
        target = FuzzTarget(
            name=function_name,
            module_path=module_path,
            function_name=function_name,
            input_types=input_types,
        )
        if parallel:
            return self.python_fuzzer.fuzz_parallel(target, runs=runs, max_workers=max_workers)
        return self.python_fuzzer.fuzz(target, runs=runs)

    def fuzz_with_corpus(
        self,
        target: Callable,
        corpus: list[bytes],
        mutations: int = 1000,
    ) -> FuzzResult:
        """Fuzz using a corpus of inputs with optimised mutation generation."""
        import time

        start_time = time.time()
        crashes = []
        crash_hashes: set[str] = set()

        for _ in range(mutations):
            seed = random.choice(corpus) if corpus else b""

            if random.random() < 0.3:
                # XOR mutation — vectorised with bytearray
                seed_arr = bytearray(seed)
                xor_vals = os.urandom(len(seed_arr))
                mutated = bytes(a ^ b for a, b in zip(seed_arr, xor_vals))
            elif random.random() < 0.5:
                mutated = seed + os.urandom(random.randint(1, 100))
            else:
                mutated = os.urandom(random.randint(0, len(seed) + 100))

            try:
                if isinstance(target, bytes):
                    # SECURITY: Never eval() arbitrary bytes.
                    # If you need to serialize function references, use
                    # module_path + function_name and import them safely.
                    raise TypeError(
                        "fuzz_with_corpus: target must be a callable, not bytes. "
                        "Pass the actual function object instead of serialized code."
                    )
                target(mutated)

            except Exception as e:
                h = hashlib.sha256(mutated).hexdigest()
                if h not in crash_hashes:
                    crash_hashes.add(h)
                    crashes.append({"input": mutated.hex()[:100], "error": str(e)})

        return FuzzResult(
            target="corpus_fuzz",
            runs=mutations,
            unique_crashes=len(crashes),
            unique_hangs=0,
            coverage_percent=0.0,
            duration_seconds=time.time() - start_time,
            corpus_size=len(corpus),
            findings=crashes[:50],
        )


def fuzz_function(func: Callable, inputs: list[Any]) -> list[dict]:
    """Quick fuzzing of a function with given inputs"""
    results = []

    for inp in inputs:
        try:
            result = func(inp)
            results.append({"input": str(inp)[:100], "status": "ok", "result": str(result)[:100]})
        except Exception as e:
            results.append({"input": str(inp)[:100], "status": "crash", "error": str(e)})

    return results
