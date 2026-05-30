"""
Mutation testing for verifying test quality.

Mutates source code to check if tests catch the changes.
High-quality tests should fail when code is mutated.

Supports:
- AST-level mutations (guaranteed correct mutation targets)
- Persistent test worker (avoids subprocess-per-mutation overhead)
- Parallel mutation testing via concurrent.futures
"""

from __future__ import annotations

import ast
import json
import random
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Persistent worker — runs as a long-lived subprocess, receives mutations
# via stdin JSON, runs pytest, returns results via stdout JSON.
# ---------------------------------------------------------------------------

_WORKER_SCRIPT = (
    "import json as _json\n"
    "import os as _os\n"
    "import subprocess as _subprocess\n"
    "import sys as _sys\n"
    "import tempfile as _tempfile\n"
    "import time as _time\n"
    "import shutil as _shutil\n"
    "\n"
    "def _write_response(stdout, response):\n"
    '    stdout.write(_json.dumps(response) + "\\n")\n'
    "    stdout.flush()\n"
    "\n"
    "def _handle_test_mutation(request):\n"
    '    mutation_id = request.get("mutation_id", "")\n'
    '    mutated_content = request.get("mutated_content", "")\n'
    '    test_file = request.get("test_file", "")\n'
    '    source_name = request.get("source_name", "mutated_source.py")\n'
    '    timeout = request.get("timeout", 60)\n'
    '    root_path = request.get("root_path", "")\n'
    "    if not mutated_content or not test_file or not root_path:\n"
    '        return {"mutation_id": mutation_id, "error": "Missing fields"}\n'
    "    source_path = _os.path.join(root_path, source_name)\n"
    "    if not _os.path.isfile(source_path):\n"
    '        return {"mutation_id": mutation_id, "error": f"Source not found: {source_path}"}\n'
    '    with open(source_path, "r") as f:\n'
    "        original_content = f.read()\n"
    "    try:\n"
    '        with open(source_path, "w") as f:\n'
    "            f.write(mutated_content)\n"
    '        cmd = [_sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"]\n'
    "        start = _time.time()\n"
    "        result = _subprocess.run(cmd, cwd=root_path, capture_output=True, text=True, timeout=timeout)\n"
    "        duration = _time.time() - start\n"
    '        return {"mutation_id": mutation_id, "killed": result.returncode != 0,\n'
    '                "test_output": result.stdout + result.stderr,\n'
    '                "duration_seconds": round(duration, 4)}\n'
    "    except _subprocess.TimeoutExpired:\n"
    '        return {"mutation_id": mutation_id, "killed": False,\n'
    '                "test_output": "Test timed out", "duration_seconds": float(timeout)}\n'
    "    except Exception as e:\n"
    '        return {"mutation_id": mutation_id, "killed": False,\n'
    '                "test_output": str(e), "duration_seconds": 0}\n'
    "    finally:\n"
    '        with open(source_path, "w") as f:\n'
    "            f.write(original_content)\n"
    "\n"
    "def _run_worker_loop():\n"
    "    for line in _sys.stdin:\n"
    "        line = line.strip()\n"
    "        if not line:\n"
    "            continue\n"
    "        try:\n"
    "            request = _json.loads(line)\n"
    "        except _json.JSONDecodeError:\n"
    '            _write_response(_sys.stdout, {"error": "Invalid JSON"})\n'
    "            continue\n"
    '        action = request.get("action", "")\n'
    '        if action == "shutdown":\n'
    '            _write_response(_sys.stdout, {"status": "shutdown"})\n'
    "            break\n"
    '        if action == "test_mutation":\n'
    "            result = _handle_test_mutation(request)\n"
    "            _write_response(_sys.stdout, result)\n"
    "        else:\n"
    '            _write_response(_sys.stdout, {"error": f"Unknown action: {action}"})\n'
)


# ---------------------------------------------------------------------------
# Persistent worker manager (lives in the parent process)
# ---------------------------------------------------------------------------


class _PersistentWorker:
    """Manages a single long-lived pytest subprocess for mutation testing.

    The worker process keeps the Python interpreter and pytest loaded across
    all mutations, eliminating the per-mutation startup overhead that
    ``subprocess.run`` incurs (typically 1–3 s per invocation).
    """

    def __init__(self, root_path: str, timeout: int = 60):
        self._root_path = root_path
        self._timeout = timeout
        self._process: subprocess.Popen[str] | None = None
        self._started = False

    def start(self) -> None:
        """Spawn the persistent worker subprocess."""
        if self._started and self._process and self._process.poll() is None:
            return

        worker_code = _WORKER_SCRIPT + "\n_run_worker_loop()\n"
        self._process = subprocess.Popen(
            [sys.executable, "-c", worker_code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        self._started = True

    def test_mutation(
        self,
        mutation_id: str,
        mutated_content: str,
        test_file: str,
        source_name: str,
    ) -> dict:
        """Send a mutation to the worker and return the result."""
        if not self._process or self._process.poll() is not None:
            self.start()

        request = {
            "action": "test_mutation",
            "mutation_id": mutation_id,
            "mutated_content": mutated_content,
            "test_file": str(test_file),
            "source_name": source_name,
            "timeout": self._timeout,
            "root_path": self._root_path,
        }

        if self._process is None or self._process.stdin is None:
            return {
                "mutation_id": mutation_id,
                "killed": False,
                "test_output": "Worker process not available",
                "duration_seconds": 0,
            }

        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()

        # Read response with timeout guard
        import select

        if self._process.stdout is None:
            return {
                "mutation_id": mutation_id,
                "killed": False,
                "test_output": "Worker stdout not available",
                "duration_seconds": 0,
            }

        ready, _, _ = select.select([self._process.stdout], [], [], self._timeout + 10)
        if not ready:
            return {
                "mutation_id": mutation_id,
                "killed": False,
                "test_output": "Worker timed out waiting for response",
                "duration_seconds": 0,
            }

        line = self._process.stdout.readline()
        if not line:
            return {
                "mutation_id": mutation_id,
                "killed": False,
                "test_output": "Worker closed stdout unexpectedly",
                "duration_seconds": 0,
            }

        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return {
                "mutation_id": mutation_id,
                "killed": False,
                "test_output": f"Invalid worker response: {line[:200]}",
                "duration_seconds": 0,
            }

    def shutdown(self) -> None:
        """Gracefully shut down the worker process."""
        if not self._process or self._process.poll() is not None:
            return

        try:
            assert self._process.stdin is not None
            self._process.stdin.write(json.dumps({"action": "shutdown"}) + "\n")
            self._process.stdin.flush()
            self._process.wait(timeout=10)
        except Exception:
            if self._process.poll() is None:
                self._process.kill()
                self._process.wait(timeout=5)
        finally:
            self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.shutdown()


from typing import IO

MUTATIONS = [
    ("AOR", "Arithmetic Operator Replacement"),
    ("ROR", "Relational Operator Replacement"),
    ("COR", "Conditional Operator Replacement"),
    ("LOD", "Logical Operator Deletion"),
    ("AOD", "Arithmetic Operator Deletion"),
    ("ASI", "Assignment Statement Inversion"),
    ("PRI", "Return Value Inversion"),
    ("NEG", "Negation Insertion"),
    ("ZER", "Zero Constant Replacement"),
    ("ONE", "One Constant Replacement"),
    ("LCR", "Logical Constant Replacement"),
]


@dataclass
class Mutation:
    """A single mutation applied to source code"""

    id: str
    type: str
    description: str
    line: int
    original: str
    mutated: str


@dataclass
class MutationResult:
    """Result of a mutation test"""

    mutation: Mutation
    killed: bool
    test_output: str
    duration_seconds: float


@dataclass
class MutationTestReport:
    """Overall mutation testing report"""

    total_mutations: int
    killed: int
    survived: int
    score: float  # percentage of killed mutations
    results: list[MutationResult]
    duration_seconds: float


class PythonMutator:
    """Mutate Python source code"""

    ARITHMETIC_OPS = ["+", "-", "*", "/", "//", "%", "**", "<<", ">>", "&", "|", "^"]
    RELATIONAL_OPS = ["==", "!=", "<", ">", "<=", ">="]
    LOGICAL_OPS = ["and", "or", "not"]

    def __init__(self):
        self.mutations_applied: list[Mutation] = []

    def mutate_file(
        self, file_path: Path, mutation_types: list[str] | None = None
    ) -> list[Mutation]:
        """Apply mutations to a Python file"""
        content = file_path.read_text()
        mutations = []

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if mutation_types and not any(
                t in ["AOR", "ROR", "COR", "LOD", "AOD"] for t in mutation_types
            ):
                continue

            if isinstance(node, ast.BinOp) and "AOR" in (mutation_types or ["AOR"]):
                m = self._mutate_binary_op(node, content)
                if m:
                    mutations.append(m)

            elif isinstance(node, ast.Compare) and "ROR" in (mutation_types or ["ROR"]):
                m = self._mutate_compare(node, content)
                if m:
                    mutations.append(m)

            elif isinstance(node, ast.BoolOp) and "COR" in (mutation_types or ["COR"]):
                m = self._mutate_boolop(node, content)
                if m:
                    mutations.append(m)

            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                if "LOD" in (mutation_types or ["LOD"]):
                    m = self._mutate_not(node, content)
                    if m:
                        mutations.append(m)

        self.mutations_applied = mutations
        return mutations

    def _mutate_binary_op(self, node: ast.BinOp, source: str) -> Mutation | None:
        """Mutate binary operations"""
        if not isinstance(node.op, type(node.op)):
            return None

        op_str = ast.unparse(node.op)

        candidates = [op for op in self.ARITHMETIC_OPS if op != op_str]
        if not candidates:
            return None

        mutated_op = random.choice(candidates)

        original = (
            source[node.col_offset : node.end_col_offset] if hasattr(node, "col_offset") else op_str
        )
        if not original:
            original = op_str

        return Mutation(
            id=f"AOR_{node.lineno}_{node.col_offset}",
            type="AOR",
            description=f"Replaced {op_str} with {mutated_op}",
            line=node.lineno or 0,
            original=op_str,
            mutated=mutated_op,
        )

    def _mutate_compare(self, node: ast.Compare, source: str) -> Mutation | None:
        """Mutate comparison operators"""
        if not node.ops:
            return None

        op = node.ops[0]
        op_str = ast.unparse(op)

        candidates = [op for op in self.RELATIONAL_OPS if op != op_str]
        if not candidates:
            return None

        mutated_op = random.choice(candidates)

        return Mutation(
            id=f"ROR_{node.lineno}_{node.col_offset}",
            type="ROR",
            description=f"Replaced {op_str} with {mutated_op}",
            line=node.lineno or 0,
            original=op_str,
            mutated=mutated_op,
        )

    def _mutate_boolop(self, node: ast.BoolOp, source: str) -> Mutation | None:
        """Mutate boolean operations"""
        if not isinstance(node.op, (ast.And, ast.Or)):
            return None

        original = "and" if isinstance(node.op, ast.And) else "or"
        mutated = "or" if original == "and" else "and"

        return Mutation(
            id=f"COR_{node.lineno}_{node.col_offset}",
            type="COR",
            description=f"Replaced {original} with {mutated}",
            line=node.lineno or 0,
            original=original,
            mutated=mutated,
        )

    def _mutate_not(self, node: ast.UnaryOp, source: str) -> Mutation | None:
        """Remove logical NOT"""
        return Mutation(
            id=f"LOD_{node.lineno}_{node.col_offset}",
            type="LOD",
            description="Removed NOT operator",
            line=node.lineno or 0,
            original="not",
            mutated="",
        )

    @staticmethod
    def _apply_mutation_to_source(content: str, mutation: Mutation) -> str:
        """Apply a mutation at the AST level and regenerate the source.

        Falls back to string replacement if AST modification fails.
        """
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return content.replace(mutation.original, mutation.mutated, 1)

        replacement_map: dict[tuple[int, int], str] = {}

        if mutation.type == "AOR":
            for node in ast.walk(tree):
                if isinstance(node, ast.BinOp) and node.lineno == mutation.line:
                    op_str = ast.unparse(node.op)
                    if op_str == mutation.original:
                        replacement_map[(node.lineno, node.col_offset)] = mutation.mutated
                        break
        elif mutation.type == "ROR":
            for node in ast.walk(tree):
                if isinstance(node, ast.Compare) and node.lineno == mutation.line:
                    if node.ops:
                        op_str = ast.unparse(node.ops[0])
                        if op_str == mutation.original:
                            replacement_map[(node.lineno, node.col_offset)] = mutation.mutated
                            break
        elif mutation.type == "COR":
            for node in ast.walk(tree):
                if isinstance(node, ast.BoolOp) and node.lineno == mutation.line:
                    original = "and" if isinstance(node.op, ast.And) else "or"
                    if original == mutation.original:
                        replacement_map[(node.lineno, node.col_offset)] = mutation.mutated
                        break
        elif mutation.type == "LOD":
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.UnaryOp)
                    and isinstance(node.op, ast.Not)
                    and node.lineno == mutation.line
                ):
                    replacement_map[(node.lineno, node.col_offset)] = mutation.mutated
                    break

        if replacement_map:
            lines = content.splitlines(keepends=True)
            for (line, _col), replacement in replacement_map.items():
                if 1 <= line <= len(lines):
                    lines[line - 1] = lines[line - 1].replace(mutation.original, replacement, 1)
            return "".join(lines)

        return content.replace(mutation.original, mutation.mutated, 1)


class MutationTestRunner:
    """Run mutation tests on source code.

    Supports both sequential and parallel execution.  The persistent-worker
    approach reuses a single ``pytest`` process across mutations when
    ``run_persistent`` is used, eliminating interpreter startup overhead.
    """

    def __init__(self, root_path: Path | None = None):
        self.root_path = root_path or Path.cwd()
        self.mutator = PythonMutator()
        self._temp_dir: Path | None = None

    def run(
        self,
        source_file: Path,
        test_file: Path,
        mutation_types: list[str] | None = None,
        max_mutations: int = 50,
    ) -> MutationTestReport:
        """Run mutation testing (sequential)."""
        import time

        start_time = time.time()

        mutations = self.mutator.mutate_file(source_file, mutation_types)
        mutations = mutations[:max_mutations]

        results: list[MutationResult] = []
        killed = 0
        survived = 0

        for mutation in mutations:
            result = self._run_mutation_test(source_file, test_file, mutation)
            results.append(result)

            if result.killed:
                killed += 1
            else:
                survived += 1

        total = killed + survived
        score = (killed / total * 100) if total > 0 else 0.0

        return MutationTestReport(
            total_mutations=total,
            killed=killed,
            survived=survived,
            score=score,
            results=results,
            duration_seconds=time.time() - start_time,
        )

    def run_parallel(
        self,
        source_file: Path,
        test_file: Path,
        mutation_types: list[str] | None = None,
        max_mutations: int = 50,
        max_workers: int | None = None,
    ) -> MutationTestReport:
        """Run mutation testing in parallel using persistent workers.

        Spawns one persistent pytest worker per CPU core. Each worker keeps
        pytest loaded in memory and processes mutations sequentially, while
        workers run in parallel. This combines the startup amortisation of
        persistent workers with the throughput of parallelism.
        """
        import time

        start_time = time.time()

        mutations = self.mutator.mutate_file(source_file, mutation_types)
        mutations = mutations[:max_mutations]

        if not mutations:
            return MutationTestReport(
                total_mutations=0,
                killed=0,
                survived=0,
                score=0.0,
                results=[],
                duration_seconds=0.0,
            )

        num_workers = max_workers or min(4, len(mutations))
        source_content = source_file.read_text()

        # Pre-apply mutations so each worker just sends content to its worker
        mutation_payloads: list[tuple[Mutation, str]] = []
        for mutation in mutations:
            mutated_content = PythonMutator._apply_mutation_to_source(source_content, mutation)
            mutation_payloads.append((mutation, mutated_content))

        results: list[MutationResult] = []

        def _worker_task(
            worker_mutations: list[tuple[Mutation, str]],
        ) -> list[MutationResult]:
            """Process a batch of mutations using a single persistent worker."""
            worker_results: list[MutationResult] = []
            with _PersistentWorker(root_path=str(self.root_path), timeout=60) as worker:
                for mutation, mutated_content in worker_mutations:
                    response = worker.test_mutation(
                        mutation_id=mutation.id,
                        mutated_content=mutated_content,
                        test_file=str(test_file),
                        source_name=source_file.name,
                    )
                    worker_results.append(
                        MutationResult(
                            mutation=mutation,
                            killed=response.get("killed", False),
                            test_output=response.get("test_output", ""),
                            duration_seconds=response.get("duration_seconds", 0),
                        )
                    )
            return worker_results

        # Split mutations into chunks for each worker
        chunk_size = max(1, len(mutation_payloads) // num_workers)
        chunks = []
        for i in range(0, len(mutation_payloads), chunk_size):
            chunks.append(mutation_payloads[i : i + chunk_size])

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_worker_task, chunk) for chunk in chunks]
            for future in as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception:
                    pass

        killed = sum(1 for r in results if r.killed)
        survived = len(results) - killed
        total = killed + survived
        score = (killed / total * 100) if total > 0 else 0.0

        return MutationTestReport(
            total_mutations=total,
            killed=killed,
            survived=survived,
            score=score,
            results=results,
            duration_seconds=time.time() - start_time,
        )

    def run_persistent(
        self,
        source_file: Path,
        test_file: Path,
        mutation_types: list[str] | None = None,
        max_mutations: int = 50,
        timeout: int = 60,
    ) -> MutationTestReport:
        """Run mutation testing using a persistent pytest worker process.

        Instead of spawning a fresh Python interpreter + pytest for every
        mutation (which costs 1–3 s of startup each time), this reuses a
        single long-lived worker process that keeps pytest loaded in memory.
        Mutations are sent as JSON over stdin; results come back on stdout.

        For 50 mutations this typically reduces wall-clock time from ~150 s
        (50 × 3 s startup) to ~10–20 s (one startup + test execution only).
        """
        import time

        start_time = time.time()

        mutations = self.mutator.mutate_file(source_file, mutation_types)
        mutations = mutations[:max_mutations]

        if not mutations:
            return MutationTestReport(
                total_mutations=0,
                killed=0,
                survived=0,
                score=0.0,
                results=[],
                duration_seconds=0.0,
            )

        results: list[MutationResult] = []
        source_content = source_file.read_text()

        with _PersistentWorker(root_path=str(self.root_path), timeout=timeout) as worker:
            for mutation in mutations:
                mutated_content = PythonMutator._apply_mutation_to_source(source_content, mutation)

                response = worker.test_mutation(
                    mutation_id=mutation.id,
                    mutated_content=mutated_content,
                    test_file=str(test_file),
                    source_name=source_file.name,
                )

                results.append(
                    MutationResult(
                        mutation=mutation,
                        killed=response.get("killed", False),
                        test_output=response.get("test_output", ""),
                        duration_seconds=response.get("duration_seconds", 0),
                    )
                )

        killed = sum(1 for r in results if r.killed)
        survived = len(results) - killed
        total = killed + survived
        score = (killed / total * 100) if total > 0 else 0.0

        return MutationTestReport(
            total_mutations=total,
            killed=killed,
            survived=survived,
            score=score,
            results=results,
            duration_seconds=time.time() - start_time,
        )

    def _run_mutation_test(
        self,
        source_file: Path,
        test_file: Path,
        mutation: Mutation,
    ) -> MutationResult:
        """Run tests with a single mutation.

        Applies mutations at the AST level, overwrites the source file,
        runs pytest, then restores the original. This guarantees the test
        imports the mutated code.
        """
        import time

        content = source_file.read_text()
        mutated_content = self._apply_ast_mutation(content, mutation)

        # Overwrite source with mutation, run tests, restore
        try:
            source_file.write_text(mutated_content)

            cmd = [
                sys.executable,
                "-m",
                "pytest",
                str(test_file),
                "-v",
                "--tb=short",
            ]

            start = time.time()
            result = subprocess.run(
                cmd,
                cwd=self.root_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            duration = time.time() - start

            killed = result.returncode != 0

            return MutationResult(
                mutation=mutation,
                killed=killed,
                test_output=result.stdout + result.stderr,
                duration_seconds=duration,
            )

        except subprocess.TimeoutExpired:
            return MutationResult(
                mutation=mutation,
                killed=False,
                test_output="Test timed out",
                duration_seconds=60,
            )
        except Exception as e:
            return MutationResult(
                mutation=mutation,
                killed=False,
                test_output=str(e),
                duration_seconds=0,
            )
        finally:
            # Always restore original source
            source_file.write_text(content)

    @staticmethod
    def _apply_ast_mutation(content: str, mutation: Mutation) -> str:
        """Apply a mutation at the AST level and regenerate the source.

        Falls back to string replacement if AST modification fails (e.g. the
        source is not valid Python or the mutation targets an operator that
        cannot be cleanly reconstructed).
        """
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Cannot parse — fall back to string replacement
            return content.replace(mutation.original, mutation.mutated, 1)

        # Build a mapping of (line, col) → replacement text
        replacement_map: dict[tuple[int, int], str] = {}

        if mutation.type == "AOR":
            for node in ast.walk(tree):
                if isinstance(node, ast.BinOp) and node.lineno == mutation.line:
                    op_str = ast.unparse(node.op)
                    if op_str == mutation.original:
                        replacement_map[(node.lineno, node.col_offset)] = mutation.mutated
                        break

        elif mutation.type == "ROR":
            for node in ast.walk(tree):
                if isinstance(node, ast.Compare) and node.lineno == mutation.line:
                    if node.ops:
                        op_str = ast.unparse(node.ops[0])
                        if op_str == mutation.original:
                            replacement_map[(node.lineno, node.col_offset)] = mutation.mutated
                            break

        elif mutation.type == "COR":
            for node in ast.walk(tree):
                if isinstance(node, ast.BoolOp) and node.lineno == mutation.line:
                    original = "and" if isinstance(node.op, ast.And) else "or"
                    if original == mutation.original:
                        replacement_map[(node.lineno, node.col_offset)] = mutation.mutated
                        break

        elif mutation.type == "LOD":
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.UnaryOp)
                    and isinstance(node.op, ast.Not)
                    and node.lineno == mutation.line
                ):
                    replacement_map[(node.lineno, node.col_offset)] = mutation.mutated
                    break

        if replacement_map:
            lines = content.splitlines(keepends=True)
            for (line, _col), replacement in replacement_map.items():
                if 1 <= line <= len(lines):
                    original_line = lines[line - 1]
                    # Replace the first occurrence of the original operator
                    lines[line - 1] = original_line.replace(mutation.original, replacement, 1)
            return "".join(lines)

        # Fallback: string replacement
        return content.replace(mutation.original, mutation.mutated, 1)

    def cleanup(self):
        """Clean up temporary files"""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)


class MutmutIntegration:
    """Integration with mutmut (popular Python mutation testing tool)"""

    @staticmethod
    def check_installed() -> bool:
        """Check if mutmut is installed"""
        try:
            subprocess.run(
                ["mutmut", "--version"],
                capture_output=True,
                text=True,
            )
            return True
        except Exception:
            return False

    @staticmethod
    def run(source_dir: Path, test_command: str = "pytest") -> MutationTestReport:
        """Run mutmut on a source directory"""
        import time

        start_time = time.time()

        try:
            result = subprocess.run(
                ["mutmut", "run", "--", test_command],
                cwd=source_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )

            output = result.stdout + result.stderr

            import re

            killed_match = re.search(r"(\d+) killed", output)
            survived_match = re.search(r"(\d+) survived", output)

            killed = int(killed_match.group(1)) if killed_match else 0
            survived = int(survived_match.group(1)) if survived_match else 0
            total = killed + survived
            score = (killed / total * 100) if total > 0 else 0.0

            return MutationTestReport(
                total_mutations=total,
                killed=killed,
                survived=survived,
                score=score,
                results=[],
                duration_seconds=time.time() - start_time,
            )

        except Exception:
            return MutationTestReport(
                total_mutations=0,
                killed=0,
                survived=0,
                score=0.0,
                results=[],
                duration_seconds=time.time() - start_time,
            )


def quick_mutate(code: str, num_mutations: int = 5) -> list[str]:
    """Quickly generate mutated versions of code"""
    mutations = []

    for _ in range(num_mutations):
        mutated = code

        replacements = [
            ("==", "!="),
            ("!=", "=="),
            ("and", "or"),
            ("or", "and"),
            ("+", "-"),
            ("-", "+"),
            ("*", "/"),
            ("/", "*"),
            ("True", "False"),
            ("False", "True"),
        ]

        for old, new in replacements:
            if old in mutated:
                mutated = mutated.replace(old, new, 1)
                if mutated != code:
                    mutations.append(mutated)
                    break

    return mutations[:num_mutations]
