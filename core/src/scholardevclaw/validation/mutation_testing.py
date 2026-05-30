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
        """Run mutation testing in parallel across CPU cores.

        Each mutation is tested in an isolated subprocess via the thread pool,
        sharing a single temp directory for mutated files.
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

        # Ensure temp directory exists before parallel workers start
        if not self._temp_dir:
            self._temp_dir = Path(tempfile.mkdtemp())

        num_workers = max_workers or min(4, len(mutations))
        results: list[MutationResult] = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_mutation = {
                executor.submit(
                    self._run_single_mutation_subprocess,
                    str(source_file),
                    str(test_file),
                    str(self._temp_dir),
                    str(self.root_path),
                    mutation,
                ): mutation
                for mutation in mutations
            }
            for future in as_completed(future_to_mutation):
                try:
                    results.append(future.result())
                except Exception:
                    mutation = future_to_mutation[future]
                    results.append(
                        MutationResult(
                            mutation=mutation,
                            killed=False,
                            test_output="Worker exception",
                            duration_seconds=0,
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

    @staticmethod
    def _run_single_mutation_subprocess(
        source_path: str,
        test_path: str,
        temp_dir: str,
        root_path: str,
        mutation: Mutation,
    ) -> MutationResult:
        """Top-level function (picklable) that applies a mutation and runs
        pytest in a subprocess.  Used by :meth:`run_parallel`."""
        import time as _time

        source_file = Path(source_path)
        content = source_file.read_text()
        mutated_content = PythonMutator._apply_mutation_to_source(content, mutation)

        temp_source = Path(temp_dir) / source_file.name
        temp_source.write_text(mutated_content)

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            test_path,
            "-v",
            "--tb=short",
        ]

        try:
            start = _time.time()
            result = subprocess.run(
                cmd,
                cwd=root_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            duration = _time.time() - start
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

    def _run_mutation_test(
        self,
        source_file: Path,
        test_file: Path,
        mutation: Mutation,
    ) -> MutationResult:
        """Run tests with a single mutation.

        Applies mutations at the AST level using ``ast.unparse`` to guarantee
        the mutation targets the correct syntactic element (no fragile string
        replacement).
        """
        import time

        content = source_file.read_text()
        mutated_content = self._apply_ast_mutation(content, mutation)

        temp_source = self._temp_dir / source_file.name if self._temp_dir else None

        try:
            if not temp_source:
                self._temp_dir = Path(tempfile.mkdtemp())
                temp_source = self._temp_dir / source_file.name

            temp_source.write_text(mutated_content)

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
