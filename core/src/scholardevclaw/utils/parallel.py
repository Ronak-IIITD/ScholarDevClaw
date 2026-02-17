from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, TypeVar, Iterable, Generator

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    func: Callable[[T], R],
    items: list[T],
    *,
    max_workers: int | None = None,
    use_processes: bool = False,
) -> list[R]:
    """Parallel map using thread or process pool."""
    if len(items) <= 1:
        return [func(item) for item in items]

    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    with executor_class(max_workers=max_workers) as executor:
        results = list(executor.map(func, items))
    return results


class LazyFileScanner:
    """Lazy file scanner for large repositories."""

    def __init__(self, root: Path, pattern: str = "*.py"):
        self.root = root
        self.pattern = pattern
        self._cache: list[Path] | None = None

    def _scan(self) -> Generator[Path, None, None]:
        """Generator-based scan to avoid loading all files into memory."""
        for item in self.root.rglob(self.pattern):
            if item.is_file():
                yield item

    def files(self) -> list[Path]:
        if self._cache is None:
            self._cache = list(self._scan())
        return self._cache

    def count(self) -> int:
        return len(self.files())

    def iter_chunks(self, chunk_size: int = 100) -> Generator[list[Path], None, None]:
        """Iterate in chunks for batch processing."""
        files = self.files()
        for i in range(0, len(files), chunk_size):
            yield files[i : i + chunk_size]


class ParallelGit:
    """Parallel git operations."""

    @staticmethod
    def check_status(repo_path: Path) -> dict[str, Any]:
        """Check git status quickly."""
        try:
            import subprocess

            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if status.returncode == 0:
                lines = [l for l in status.stdout.splitlines() if l.strip()]
                return {
                    "available": True,
                    "is_clean": len(lines) == 0,
                    "changed_files": lines,
                }
            return {"available": False, "error": status.stderr}
        except Exception as e:
            return {"available": False, "error": str(e)}

    @staticmethod
    def get_branches(repo_path: Path) -> list[str]:
        """Get all branches."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "branch", "-a"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                return [b.strip().replace("* ", "") for b in result.stdout.splitlines()]
            return []
        except Exception:
            return []


def count_files_fast(root: Path, pattern: str = "*.py") -> int:
    """Fast file counting using generator."""
    return sum(1 for _ in root.rglob(pattern) if _.is_file())


def find_files_by_ext(root: Path, extensions: list[str]) -> dict[str, list[Path]]:
    """Find files by multiple extensions."""
    results: dict[str, list[Path]] = {ext: [] for ext in extensions}

    for ext in extensions:
        if not ext.startswith("*."):
            ext = f"*.{ext}"
        results[ext] = [f for f in root.rglob(ext) if f.is_file()]

    return results
