"""Git context helpers for the ScholarDevClaw TUI.

Lightweight wrappers around the ``git`` CLI that return small
dataclasses suitable for display in :class:`StatusBar`. Each helper
is a pure function with no UI dependencies so it can be unit-tested
with a temporary git repository.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GitContext:
    """The git state of a working directory."""

    branch: str
    dirty: bool
    ahead: int = 0
    behind: int = 0
    is_repo: bool = True

    def short(self) -> str:
        """A compact one-line representation suitable for a status bar."""
        if not self.is_repo:
            return "[dim]not a git repo[/dim]"
        marker = "*" if self.dirty else ""
        ahead_behind = ""
        if self.ahead or self.behind:
            parts: list[str] = []
            if self.ahead:
                parts.append(f"↑{self.ahead}")
            if self.behind:
                parts.append(f"↓{self.behind}")
            ahead_behind = f" {' '.join(parts)}"
        return f"⎇ {self.branch}{marker}{ahead_behind}"

    @classmethod
    def empty(cls) -> GitContext:
        """A no-info context for paths where git is unavailable."""
        return cls(branch="", dirty=False, is_repo=False)


def _run_git(args: list[str], cwd: str, timeout: float = 1.0) -> str | None:
    """Run a git subcommand and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def get_git_context(path: str | Path) -> GitContext:
    """Return the git context for a working directory.

    Returns a :class:`GitContext` with ``is_repo=False`` if the path
    is not inside a git repository or if git is not installed.
    """
    path_str = str(path or ".") or "."
    # Top-level path of the current repo
    top = _run_git(["rev-parse", "--show-toplevel"], path_str)
    if not top:
        return GitContext.empty()
    # Current branch (or detached HEAD short SHA)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], path_str)
    if not branch or branch == "HEAD":
        # Detached HEAD — fall back to short SHA
        sha = _run_git(["rev-parse", "--short", "HEAD"], path_str)
        branch = sha or "detached"
    # Dirty state
    porcelain = _run_git(["status", "--porcelain"], top)
    dirty = bool(porcelain)
    # Ahead/behind tracking branch
    ahead = 0
    behind = 0
    upstream = _run_git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], top)
    if upstream:
        counts = _run_git(["rev-list", "--left-right", "--count", f"HEAD...{upstream}"], top)
        if counts:
            # Format: "<ahead>\t<behind>"
            parts = counts.split()
            if len(parts) == 2:
                try:
                    ahead = int(parts[0])
                    behind = int(parts[1])
                except ValueError:
                    pass
    return GitContext(
        branch=branch,
        dirty=dirty,
        ahead=ahead,
        behind=behind,
        is_repo=True,
    )


def is_git_available() -> bool:
    """Return True if the ``git`` executable is on PATH."""
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def short_path(path: str | Path, max_length: int = 40) -> str:
    """Return a path shortened for display in the status bar.

    Keeps the trailing portion (which usually contains the project
    name) and adds an ellipsis prefix when the path is too long.
    """
    text = str(path or ".") or "."
    if len(text) <= max_length:
        return text
    return f"…{text[-(max_length - 1) :]}"
