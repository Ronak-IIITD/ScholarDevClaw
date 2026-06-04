"""Tests for the git context helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scholardevclaw.tui.git_status import (
    GitContext,
    get_git_context,
    is_git_available,
    short_path,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _init_git_repo(path: Path) -> None:
    """Initialize a git repository at ``path`` with user config + initial commit."""
    subprocess.run(
        ["git", "init", str(path)],
        capture_output=True,
        check=True,
        timeout=5,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        capture_output=True,
        check=True,
        timeout=5,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test User"],
        capture_output=True,
        check=True,
        timeout=5,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "commit.gpgsign", "false"],
        capture_output=True,
        check=True,
        timeout=5,
    )
    # Create an initial commit so the repo has a branch (otherwise HEAD
    # is detached with no refs)
    (path / ".gitkeep").write_text("")
    subprocess.run(
        ["git", "-C", str(path), "add", ".gitkeep"],
        capture_output=True,
        check=True,
        timeout=5,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True,
        check=True,
        timeout=5,
    )


def _make_commit(path: Path, filename: str, content: str, msg: str) -> None:
    """Stage a new file and create a commit."""
    (path / filename).write_text(content)
    subprocess.run(
        ["git", "-C", str(path), "add", filename],
        capture_output=True,
        check=True,
        timeout=5,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", msg],
        capture_output=True,
        check=True,
        timeout=5,
    )


# -----------------------------------------------------------------------------
# is_git_available
# -----------------------------------------------------------------------------


class TestIsGitAvailable:
    def test_returns_bool(self) -> None:
        result = is_git_available()
        assert isinstance(result, bool)


# -----------------------------------------------------------------------------
# get_git_context
# -----------------------------------------------------------------------------


@pytest.mark.skipif(not is_git_available(), reason="git not installed in test env")
class TestGetGitContext:
    """Integration tests using a real temporary git repo."""

    def test_non_repo_returns_empty(self, tmp_path: Path) -> None:
        ctx = get_git_context(tmp_path)
        assert ctx.is_repo is False
        assert ctx.branch == ""

    def test_fresh_repo(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        ctx = get_git_context(tmp_path)
        assert ctx.is_repo is True
        # Default branch can be 'main' or 'master' depending on git config
        assert ctx.branch in ("main", "master")
        assert ctx.dirty is False
        assert ctx.ahead == 0
        assert ctx.behind == 0

    def test_dirty_state(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        _make_commit(tmp_path, "README.md", "hello", "initial")
        # Untracked file
        (tmp_path / "new.txt").write_text("dirty")
        ctx = get_git_context(tmp_path)
        assert ctx.dirty is True

    def test_clean_after_commit(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        _make_commit(tmp_path, "README.md", "hello", "initial")
        ctx = get_git_context(tmp_path)
        assert ctx.dirty is False

    def test_staged_changes_are_dirty(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        _make_commit(tmp_path, "README.md", "hello", "initial")
        (tmp_path / "staged.txt").write_text("content")
        subprocess.run(
            ["git", "-C", str(tmp_path), "add", "staged.txt"],
            capture_output=True,
            check=True,
            timeout=5,
        )
        ctx = get_git_context(tmp_path)
        assert ctx.dirty is True

    def test_modified_tracked_file_is_dirty(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        _make_commit(tmp_path, "README.md", "hello", "initial")
        (tmp_path / "README.md").write_text("changed")
        ctx = get_git_context(tmp_path)
        assert ctx.dirty is True

    def test_subdir_returns_top_level_branch(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        _make_commit(tmp_path, "README.md", "hello", "initial")
        sub = tmp_path / "sub"
        sub.mkdir()
        ctx = get_git_context(sub)
        assert ctx.is_repo is True
        assert ctx.branch in ("main", "master")

    def test_string_path_works(self, tmp_path: Path) -> None:
        _init_git_repo(tmp_path)
        _make_commit(tmp_path, "README.md", "hello", "initial")
        ctx = get_git_context(str(tmp_path))
        assert ctx.is_repo is True


# -----------------------------------------------------------------------------
# GitContext
# -----------------------------------------------------------------------------


class TestGitContext:
    """Dataclass behaviour and short() formatting."""

    def test_short_clean(self) -> None:
        ctx = GitContext(branch="main", dirty=False)
        out = ctx.short()
        assert "main" in out
        assert "*" not in out
        assert "↑" not in out

    def test_short_dirty(self) -> None:
        ctx = GitContext(branch="dev", dirty=True)
        out = ctx.short()
        assert "dev" in out
        assert "*" in out

    def test_short_ahead(self) -> None:
        ctx = GitContext(branch="main", dirty=False, ahead=3)
        out = ctx.short()
        assert "↑3" in out

    def test_short_behind(self) -> None:
        ctx = GitContext(branch="main", dirty=False, behind=2)
        out = ctx.short()
        assert "↓2" in out

    def test_short_ahead_and_behind(self) -> None:
        ctx = GitContext(branch="main", dirty=False, ahead=1, behind=4)
        out = ctx.short()
        assert "↑1" in out
        assert "↓4" in out

    def test_short_not_a_repo(self) -> None:
        ctx = GitContext.empty()
        out = ctx.short()
        assert "not a git repo" in out

    def test_short_includes_branch_icon(self) -> None:
        ctx = GitContext(branch="feat/foo", dirty=False)
        out = ctx.short()
        assert "⎇" in out

    def test_frozen(self) -> None:
        ctx = GitContext(branch="x", dirty=False)
        with pytest.raises((AttributeError, Exception)):
            ctx.branch = "y"  # type: ignore[misc]

    def test_empty_factory(self) -> None:
        ctx = GitContext.empty()
        assert ctx.is_repo is False
        assert ctx.branch == ""
        assert ctx.dirty is False


# -----------------------------------------------------------------------------
# short_path
# -----------------------------------------------------------------------------


class TestShortPath:
    def test_short_path_unchanged(self) -> None:
        assert short_path("/a/b") == "/a/b"

    def test_short_path_truncates(self) -> None:
        long_path = "/" + "x" * 50
        out = short_path(long_path, max_length=20)
        assert len(out) <= 20
        assert out.startswith("…")

    def test_short_path_empty(self) -> None:
        assert short_path("") == "."
        assert short_path(None) == "."  # type: ignore[arg-type]

    def test_short_path_custom_max(self) -> None:
        out = short_path("/abcdefg", max_length=5)
        assert len(out) <= 5
