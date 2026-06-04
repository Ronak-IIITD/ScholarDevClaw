"""Workspace / repo selector modal for the ScholarDevClaw TUI.

Provides a keyboard-navigable picker for switching the active repository.
Candidates come from four sources (deduplicated, in priority order):

1. **current**   — the active working directory (``app._directory``)
2. **recent**    — unique repo paths from the last N run artifacts
3. **subdir**    — direct subdirectories of the current directory
4. **git_root**  — git repository roots discovered by walking up from cwd

A filter input at the top of the modal allows fuzzy substring filtering
by path or label. ``Enter`` activates the highlighted candidate,
``Esc`` dismisses without changing the active repo.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, ListItem, ListView, Static

# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

RepoSource = Literal["current", "recent", "subdir", "git_root"]

SOURCE_ICONS: dict[RepoSource, str] = {
    "current": "◉",
    "recent": "↻",
    "subdir": "▸",
    "git_root": "⎇",
}


@dataclass(frozen=True)
class RepoCandidate:
    """A single repository candidate in the selector."""

    path: str
    source: RepoSource
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            object.__setattr__(self, "label", _label_for(self.path))

    @property
    def icon(self) -> str:
        return SOURCE_ICONS.get(self.source, "•")

    @property
    def display(self) -> str:
        """The single-line representation shown in the list view."""
        return f"{self.icon}  {self.label:<32}  {self.path}"


def _label_for(path: str) -> str:
    """Derive a human label from a filesystem path."""
    if not path or path == ".":
        return "(current)"
    p = Path(path)
    if p.name:
        return p.name
    return str(p)


# -----------------------------------------------------------------------------
# Candidate building (pure functions — fully testable)
# -----------------------------------------------------------------------------


def build_repo_candidates(
    cwd: Path,
    recent_repos: list[str] | None = None,
    *,
    include_subdirs: bool = True,
    include_git_roots: bool = True,
    max_subdirs: int = 20,
) -> list[RepoCandidate]:
    """Build a deduplicated list of repo candidates.

    Order: current → recent → subdirs → git roots. Each path is normalized
    via ``os.path.abspath`` and added at most once. The first occurrence
    wins (priority order is preserved).
    """
    seen: set[str] = set()
    out: list[RepoCandidate] = []

    def add(path: str, source: RepoSource) -> None:
        norm = os.path.abspath(path) if path else ""
        if not norm or norm in seen:
            return
        seen.add(norm)
        out.append(RepoCandidate(path=norm, source=source))

    add(str(cwd), "current")

    for r in recent_repos or []:
        if r:
            add(r, "recent")

    if include_subdirs and cwd.is_dir():
        try:
            children = sorted(
                (c for c in cwd.iterdir() if c.is_dir() and not c.name.startswith(".")),
                key=lambda c: c.name.lower(),
            )[:max_subdirs]
            for c in children:
                add(str(c), "subdir")
        except (PermissionError, OSError):
            pass

    if include_git_roots:
        for root in _discover_git_roots(cwd):
            add(root, "git_root")

    return out


def filter_candidates(
    candidates: list[RepoCandidate],
    query: str,
) -> list[RepoCandidate]:
    """Return candidates whose label or path contains the query (case-insensitive).

    Empty queries return the full list unchanged.
    """
    if not query or not query.strip():
        return list(candidates)
    needle = query.strip().lower()
    return [c for c in candidates if needle in c.label.lower() or needle in c.path.lower()]


def _discover_git_roots(start: Path) -> list[str]:
    """Walk upward from ``start`` to find git repository roots.

    Returns absolute paths. If ``start`` is itself a git repo, that root
    is returned first; subsequent roots are added if subdirs of the
    search also live in separate git repos (rare, but possible with
    nested worktrees).
    """
    roots: list[str] = []
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(start),
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if result.returncode == 0:
            top = result.stdout.strip()
            if top:
                roots.append(top)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return roots


# -----------------------------------------------------------------------------
# List item widget
# -----------------------------------------------------------------------------


class _RepoListItem(ListItem):
    """A ListItem that stores its RepoCandidate as an attribute."""

    def __init__(self, candidate: RepoCandidate) -> None:
        super().__init__(Static(candidate.display))
        self.candidate = candidate


# -----------------------------------------------------------------------------
# Modal screen
# -----------------------------------------------------------------------------


class RepoSelector(ModalScreen[str | None]):
    """A modal for selecting the active workspace.

    Returns the selected repo path on enter, or ``None`` on dismiss.
    The caller is responsible for actually switching the active repo
    (e.g. by updating ``app._directory``).
    """

    BINDINGS = [
        Binding("escape", "dismiss_selector", "Cancel", show=True),
        Binding("ctrl+c", "dismiss_selector", "Cancel", show=False),
    ]

    DEFAULT_CSS = """
    RepoSelector {
        align: center middle;
        background: $background 80%;
    }

    RepoSelector > Vertical {
        width: 90%;
        max-width: 100;
        height: auto;
        max-height: 90vh;
        background: $surface;
        border: round $border;
        padding: 1 2;
    }

    RepoSelector #repo-filter {
        margin-bottom: 1;
    }

    RepoSelector #repo-list {
        height: auto;
        max-height: 30;
    }

    RepoSelector #repo-help {
        margin-top: 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        cwd: Path,
        recent_repos: list[str] | None = None,
        *,
        include_subdirs: bool = True,
        include_git_roots: bool = True,
    ) -> None:
        super().__init__()
        self._cwd = cwd
        self._all_candidates = build_repo_candidates(
            cwd,
            recent_repos or [],
            include_subdirs=include_subdirs,
            include_git_roots=include_git_roots,
        )
        self._query = ""

    # ----- compose -----

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("[b]Workspace Selector[/b]  pick a repo to switch to", id="repo-title")
            yield Input(placeholder="filter by path or label…", id="repo-filter")
            yield ListView(id="repo-list")
            yield Static(
                "Enter  select   ·   Esc  cancel   ·   Type  filter",
                id="repo-help",
            )

    def on_mount(self) -> None:
        self._refresh_list()
        # Focus the filter so the user can start typing immediately
        self.query_one("#repo-filter", Input).focus()

    # ----- list management -----

    def _refresh_list(self) -> None:
        filtered = filter_candidates(self._all_candidates, self._query)
        list_view = self.query_one("#repo-list", ListView)
        list_view.clear()
        if not filtered:
            list_view.append(ListItem(Static("[dim]no matches[/dim]")))
            return
        for cand in filtered:
            list_view.append(_RepoListItem(cand))
        # Highlight the first item so Enter picks it
        if list_view.children:
            first = list_view.children[0]
            if isinstance(first, ListItem):
                list_view.index = list_view.children.index(first)

    # ----- events -----

    @on(Input.Changed, "#repo-filter")
    def on_filter_changed(self, event: Input.Changed) -> None:
        self._query = event.value
        self._refresh_list()

    @on(ListView.Selected, "#repo-list")
    def on_list_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, _RepoListItem):
            self.dismiss(item.candidate.path)

    # ----- actions -----

    def action_dismiss_selector(self) -> None:
        self.dismiss(None)
