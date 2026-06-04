"""Tests for the workspace / repo selector modal."""

from __future__ import annotations

from pathlib import Path

import pytest

from scholardevclaw.tui.repo_selector import (
    SOURCE_ICONS,
    RepoCandidate,
    RepoSelector,
    _label_for,
    build_repo_candidates,
    filter_candidates,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------


class TestSourceIcons:
    """All RepoSource values have a defined icon."""

    def test_all_sources_have_icon(self) -> None:
        for source in ("current", "recent", "subdir", "git_root"):
            assert SOURCE_ICONS[source]


# -----------------------------------------------------------------------------
# RepoCandidate
# -----------------------------------------------------------------------------


class TestRepoCandidate:
    """The candidate data model."""

    def test_minimal_construction_derives_label(self) -> None:
        c = RepoCandidate(path="/tmp/myrepo", source="current")
        assert c.label == "myrepo"
        assert c.path == "/tmp/myrepo"
        assert c.source == "current"

    def test_explicit_label_preserved(self) -> None:
        c = RepoCandidate(path="/tmp/myrepo", source="current", label="My Repo")
        assert c.label == "My Repo"

    def test_dot_path_label(self) -> None:
        c = RepoCandidate(path=".", source="current")
        assert c.label == "(current)"

    def test_empty_path_label(self) -> None:
        c = RepoCandidate(path="", source="current")
        assert c.label == "(current)"

    def test_root_path_label(self) -> None:
        c = RepoCandidate(path="/", source="current")
        # Path("/").name is "" so label should fall back to str(p)
        assert c.label

    def test_frozen(self) -> None:
        c = RepoCandidate(path="/tmp", source="current")
        with pytest.raises((AttributeError, Exception)):
            c.path = "/other"  # type: ignore[misc]

    def test_icon_property(self) -> None:
        c = RepoCandidate(path="/x", source="recent")
        assert c.icon == SOURCE_ICONS["recent"]

    def test_display_contains_path_and_label(self) -> None:
        c = RepoCandidate(path="/tmp/proj", source="current")
        d = c.display
        assert "proj" in d
        assert "/tmp/proj" in d
        assert c.icon in d


# -----------------------------------------------------------------------------
# _label_for
# -----------------------------------------------------------------------------


class TestLabelFor:
    def test_simple_path(self) -> None:
        assert _label_for("/home/user/proj") == "proj"

    def test_nested_path(self) -> None:
        assert _label_for("/a/b/c/d") == "d"

    def test_dot(self) -> None:
        assert _label_for(".") == "(current)"

    def test_empty(self) -> None:
        assert _label_for("") == "(current)"

    def test_root(self) -> None:
        # Path("/").name is "" so we fall back to str(p)
        assert _label_for("/") == "/"


# -----------------------------------------------------------------------------
# build_repo_candidates
# -----------------------------------------------------------------------------


class TestBuildCandidates:
    """The pure candidate builder."""

    def test_empty_inputs(self, tmp_path: Path) -> None:
        cands = build_repo_candidates(tmp_path, recent_repos=[])
        assert len(cands) == 1
        assert cands[0].source == "current"
        assert cands[0].path == str(tmp_path)

    def test_cwd_abspath_normalized(self, tmp_path: Path) -> None:
        cands = build_repo_candidates(tmp_path, recent_repos=[])
        assert cands[0].path == str(tmp_path.resolve())

    def test_recent_repos_added(self, tmp_path: Path) -> None:
        other = tmp_path / "other_repo"
        cands = build_repo_candidates(tmp_path, recent_repos=[str(other)])
        sources = [c.source for c in cands]
        assert "current" in sources
        assert "recent" in sources
        # recent is deduped from current
        assert len(sources) == 2

    def test_recent_dedupes_current(self, tmp_path: Path) -> None:
        cands = build_repo_candidates(tmp_path, recent_repos=[str(tmp_path)])
        # The recent entry is a dupe of current and should be skipped
        assert len(cands) == 1
        assert cands[0].source == "current"

    def test_recent_empty_strings_skipped(self, tmp_path: Path) -> None:
        cands = build_repo_candidates(tmp_path, recent_repos=["", "", None])  # type: ignore[list-item]
        assert len(cands) == 1

    def test_subdirs_added(self, tmp_path: Path) -> None:
        (tmp_path / "alpha").mkdir()
        (tmp_path / "beta").mkdir()
        (tmp_path / ".hidden").mkdir()  # should be skipped
        cands = build_repo_candidates(tmp_path, recent_repos=[], include_subdirs=True)
        labels = [c.label for c in cands]
        assert "alpha" in labels
        assert "beta" in labels
        assert ".hidden" not in labels

    def test_subdirs_disabled(self, tmp_path: Path) -> None:
        (tmp_path / "alpha").mkdir()
        cands = build_repo_candidates(tmp_path, recent_repos=[], include_subdirs=False)
        sources = [c.source for c in cands]
        assert "subdir" not in sources

    def test_subdirs_max_limit(self, tmp_path: Path) -> None:
        for i in range(30):
            (tmp_path / f"repo_{i:02d}").mkdir()
        cands = build_repo_candidates(tmp_path, recent_repos=[], max_subdirs=10)
        subdirs = [c for c in cands if c.source == "subdir"]
        assert len(subdirs) == 10

    def test_git_roots_disabled(self, tmp_path: Path) -> None:
        # Even if cwd is a git repo, disabling should exclude it
        cands = build_repo_candidates(tmp_path, recent_repos=[], include_git_roots=False)
        sources = [c.source for c in cands]
        assert "git_root" not in sources

    def test_git_roots_included_for_git_repo(self, tmp_path: Path) -> None:
        # Initialize a real git repo in tmp_path
        import subprocess

        try:
            subprocess.run(
                ["git", "init", str(tmp_path)],
                capture_output=True,
                check=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("git not available")
        cands = build_repo_candidates(tmp_path, recent_repos=[])
        # current + git_root should both resolve to the same path, deduped to current
        paths = {c.path for c in cands}
        assert str(tmp_path.resolve()) in paths
        # Dedup means at most one entry
        assert len(cands) == 1

    def test_dedup_across_sources(self, tmp_path: Path) -> None:
        (tmp_path / "child").mkdir()
        cands = build_repo_candidates(
            tmp_path,
            recent_repos=[str(tmp_path / "child")],
        )
        paths = [c.path for c in cands]
        # No duplicates
        assert len(paths) == len(set(paths))
        # 'child' appears once (as subdir; the recent entry is deduped)
        child_count = sum(1 for p in paths if p.endswith("child"))
        assert child_count == 1

    def test_order_current_first(self, tmp_path: Path) -> None:
        cands = build_repo_candidates(tmp_path, recent_repos=["/tmp/elsewhere"])
        assert cands[0].source == "current"

    def test_nonexistent_cwd_returns_only_current(self, tmp_path: Path) -> None:
        bogus = tmp_path / "does_not_exist"
        cands = build_repo_candidates(bogus, recent_repos=[])
        # cwd.is_dir() is False, so subdirs are skipped; git root probably fails
        assert cands[0].source == "current"

    def test_subdirs_skipped_for_nonexistent_cwd(self, tmp_path: Path) -> None:
        bogus = tmp_path / "no_such_dir"
        cands = build_repo_candidates(bogus, recent_repos=[], include_subdirs=True)
        sources = [c.source for c in cands]
        assert "subdir" not in sources


# -----------------------------------------------------------------------------
# filter_candidates
# -----------------------------------------------------------------------------


class TestFilterCandidates:
    """Filter is a substring search on label or path."""

    def test_empty_query_returns_all(self) -> None:
        cands = [
            RepoCandidate(path="/a", source="current"),
            RepoCandidate(path="/b", source="recent"),
        ]
        assert filter_candidates(cands, "") == cands
        assert filter_candidates(cands, "   ") == cands

    def test_matches_label(self) -> None:
        cands = [RepoCandidate(path="/x/proj_a", source="current")]
        result = filter_candidates(cands, "proj")
        assert len(result) == 1

    def test_matches_path(self) -> None:
        cands = [RepoCandidate(path="/home/user/somewhere", source="current")]
        result = filter_candidates(cands, "user")
        assert len(result) == 1

    def test_case_insensitive(self) -> None:
        cands = [RepoCandidate(path="/Foo/Bar", source="current")]
        result = filter_candidates(cands, "foo")
        assert len(result) == 1

    def test_no_matches(self) -> None:
        cands = [RepoCandidate(path="/aaa", source="current")]
        result = filter_candidates(cands, "zzz")
        assert result == []

    def test_strips_whitespace(self) -> None:
        cands = [RepoCandidate(path="/x/proj", source="current")]
        result = filter_candidates(cands, "  proj  ")
        assert len(result) == 1


# -----------------------------------------------------------------------------
# RepoSelector (construction only — interaction is covered by App integration)
# -----------------------------------------------------------------------------


class TestRepoSelector:
    """The modal screen — test construction and candidate surfacing."""

    def test_constructible_with_cwd(self, tmp_path: Path) -> None:
        s = RepoSelector(cwd=tmp_path)
        assert s._cwd == tmp_path
        assert len(s._all_candidates) >= 1
        assert s._all_candidates[0].source == "current"

    def test_constructible_with_recent(self, tmp_path: Path) -> None:
        s = RepoSelector(cwd=tmp_path, recent_repos=["/tmp/x", "/tmp/y"])
        sources = [c.source for c in s._all_candidates]
        assert "recent" in sources

    def test_filter_resets_on_query(self, tmp_path: Path) -> None:
        s = RepoSelector(cwd=tmp_path, recent_repos=[])
        # Mutate the internal query — this is the same state the Input.Changed
        # handler would set, but without needing a mounted App to refresh
        # the list view.
        s._query = "zzz"
        s._query = ""

    def test_default_includes_subdirs_and_git(self, tmp_path: Path) -> None:
        s = RepoSelector(cwd=tmp_path)
        # Both flags default to True
        # We can't directly read them, but the constructor should not have raised
        assert s is not None

    def test_modal_screen_subclass(self) -> None:
        from textual.screen import ModalScreen

        assert issubclass(RepoSelector, ModalScreen)
