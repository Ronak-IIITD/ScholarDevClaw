"""
Fuzzing integration for robustness testing.

Supports:
- Python fuzzing with custom fuzzing strategies
- libFuzzer wrapper (when available)
- AFL wrapper (when available)
- Coverage-guided fuzzing
- Corpus management
"""

from __future__ import annotations

import ast
import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


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
    """Pure Python fuzzing without external tools"""

    def __init__(self):
        self.runs = 0
        self.crashes: list[tuple[Any, str]] = []
        self.hangs: list[tuple[Any, float]] = []
        self.seeds: list[bytes] = []

    def add_seed(self, data: bytes):
        """Add a seed input"""
        self.seeds.append(data)

    def _generate_input(self, input_type: str) -> Any:
        """Generate random input based on type"""
        if input_type == "bytes":
            length = random.randint(0, 10000)
            return bytes(random.getrandbits(8) for _ in range(length))

        elif input_type == "string":
            length = random.randint(0, 1000)
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-.,;:!?()[]{}"
            return "".join(random.choice(chars) for _ in range(length))

        elif input_type == "int":
            return random.randint(-(2**31), 2**31 - 1)

        elif input_type == "float":
            return random.uniform(-1e10, 1e10)

        elif input_type == "list":
            length = random.randint(0, 100)
            return [random.randint(0, 255) for _ in range(length)]

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
        """Run fuzzing on a target"""
        import time
        import importlib.util

        start_time = time.time()

        try:
            spec = importlib.util.spec_from_file_location("target_module", target.module_path)
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

        crash_signatures = set()
        hang_signatures = set()

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
                result = target_func(*inputs)
                elapsed = time.time() - start

                if timeout > 0:
                    signal.alarm(0)

                if elapsed > timeout * 0.8:
                    signature = (str(inputs)[:100], "hang")
                    if signature not in hang_signatures:
                        hang_signatures.add(signature)
                        self.hangs.append((inputs, elapsed))

                if target.output_check == "exception":
                    pass

            except TimeoutError:
                signature = (str(inputs)[:100], "hang")
                if signature not in hang_signatures:
                    hang_signatures.add(signature)
                    self.hangs.append((inputs, timeout))

            except Exception as e:
                signature = (type(e).__name__, str(e)[:100])
                if signature not in crash_signatures:
                    crash_signatures.add(signature)
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
            unique_crashes=len(crash_signatures),
            unique_hangs=len(hang_signatures),
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
    ) -> FuzzResult:
        """Fuzz a Python function"""
        target = FuzzTarget(
            name=function_name,
            module_path=module_path,
            function_name=function_name,
            input_types=input_types,
        )
        return self.python_fuzzer.fuzz(target, runs=runs)

    def fuzz_with_corpus(
        self,
        target: Callable,
        corpus: list[bytes],
        mutations: int = 1000,
    ) -> FuzzResult:
        """Fuzz using a corpus of inputs"""
        import time

        start_time = time.time()
        crashes = []

        for _ in range(mutations):
            seed = random.choice(corpus) if corpus else b""

            if random.random() < 0.3:
                mutated = bytes(b ^ random.randint(0, 255) for b in seed)
            elif random.random() < 0.5:
                mutated = seed + bytes(random.getrandbits(8) for _ in range(random.randint(1, 100)))
            else:
                length = random.randint(0, len(seed) + 100)
                mutated = bytes(random.getrandbits(8) for _ in range(length))

            try:
                if isinstance(target, bytes):
                    target_func = eval(target)
                    target_func(mutated)
                else:
                    target(mutated)

            except Exception as e:
                crashes.append({"input": mutated.hex()[:100], "error": str(e)})

        return FuzzResult(
            target="corpus_fuzz",
            runs=mutations,
            unique_crashes=len(set(c["input"] for c in crashes)),
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
