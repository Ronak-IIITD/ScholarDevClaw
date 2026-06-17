"""Diff / patch viewer modal for the ScholarDevClaw TUI.

A modal screen that shows a unified diff for one or more files
(added / modified / deleted) with line-level color coding:

- additions     → [green]
- deletions     → [red]
- file headers  → [cyan]
- hunk headers  → [dim]@@ ... @@[/dim]
- context lines → default

The viewer accepts a :class:`PatchDiff` (a list of :class:`FileDiff`)
and renders a tabbed interface, one tab per file. Tab cycles between
files, ``Esc`` dismisses, ``j/k`` scroll line-by-line.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Literal

from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

FileStatus = Literal["added", "modified", "deleted", "renamed"]


@dataclass(frozen=True)
class FileDiff:
    """A single file's diff."""

    path: str
    status: FileStatus
    original: str = ""
    modified: str = ""
    additions: int = 0
    deletions: int = 0

    def short_label(self) -> str:
        marker = {
            "added": "+",
            "modified": "~",
            "deleted": "-",
            "renamed": "→",
        }.get(self.status, "?")
        return f"{marker} {self.path}"


@dataclass(frozen=True)
class PatchDiff:
    """A complete patch — one or more file diffs plus metadata."""

    title: str
    files: tuple[FileDiff, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_additions(self) -> int:
        return sum(f.additions for f in self.files)

    @property
    def total_deletions(self) -> int:
        return sum(f.deletions for f in self.files)


# -----------------------------------------------------------------------------
# Diff generation (pure functions — fully testable)
# -----------------------------------------------------------------------------


def make_unified_diff(
    file_diff: FileDiff,
    *,
    context: int = 3,
) -> str:
    """Generate a unified diff for a single FileDiff.

    Returns a multi-line string suitable for display in a Static widget.
    Color markup (``[green]...[/green]`` etc.) is added so additions,
    deletions, and context lines render with semantic colors.
    """
    from_label = file_diff.path if file_diff.status != "added" else "/dev/null"
    to_label = file_diff.path if file_diff.status != "deleted" else "/dev/null"

    if file_diff.status == "added":
        original_lines: list[str] = []
        modified_lines = file_diff.modified.splitlines(keepends=True)
    elif file_diff.status == "deleted":
        original_lines = file_diff.original.splitlines(keepends=True)
        modified_lines = []
    else:
        original_lines = file_diff.original.splitlines(keepends=True)
        modified_lines = file_diff.modified.splitlines(keepends=True)

    raw_text = "\n".join(original_lines) + "\n" if original_lines else ""
    mod_text = "\n".join(modified_lines) + "\n" if modified_lines else ""

    # Use Rust native diff if available, fall back to Python difflib
    try:
        from scholardevclaw_native import unified_diff as _rust_diff

        diff_text = _rust_diff(
            raw_text,
            mod_text,
            context,
            from_label,
            to_label,
        )
    except (ImportError, Exception):
        raw = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=from_label,
            tofile=to_label,
            n=context,
            lineterm="",
        )
        diff_text = "\n".join(raw)

    return _colorize_unified(diff_text)


def _colorize_unified(text: str) -> str:
    """Apply Rich/textual color markup to a unified diff string."""
    out: list[str] = []
    for line in text.split("\n"):
        if line.startswith("+++") or line.startswith("---"):
            out.append(f"[cyan]{line}[/cyan]")
        elif line.startswith("@@"):
            out.append(f"[dim]{line}[/dim]")
        elif line.startswith("+"):
            out.append(f"[green]{line}[/green]")
        elif line.startswith("-"):
            out.append(f"[red]{line}[/red]")
        else:
            out.append(line)
    return "\n".join(out)


def compute_additions_deletions(file_diff: FileDiff) -> tuple[int, int]:
    """Count addition and deletion lines in a unified diff."""
    adds = 0
    dels = 0
    for original, modified in zip(
        file_diff.original.splitlines(),
        file_diff.modified.splitlines(),
        strict=False,
    ):
        if original != modified:
            dels += 1
            adds += 1
    # Account for pure additions or pure deletions
    orig_lines = file_diff.original.splitlines()
    mod_lines = file_diff.modified.splitlines()
    if len(mod_lines) > len(orig_lines):
        adds += len(mod_lines) - len(orig_lines)
    if len(orig_lines) > len(mod_lines):
        dels += len(orig_lines) - len(mod_lines)
    return adds, dels


def make_file_diff_from_text(
    path: str,
    original: str,
    modified: str,
) -> FileDiff:
    """Construct a FileDiff and populate additions/deletions automatically."""
    status: FileStatus
    if not original:
        status = "added"
    elif not modified:
        status = "deleted"
    else:
        status = "modified"
    diff = FileDiff(
        path=path,
        status=status,
        original=original,
        modified=modified,
    )
    adds, dels = compute_additions_deletions(diff)
    return FileDiff(
        path=diff.path,
        status=diff.status,
        original=diff.original,
        modified=diff.modified,
        additions=adds,
        deletions=dels,
    )


def summarize_patch(patch: PatchDiff) -> str:
    """One-line summary like '5 files: +42 -13'."""
    return (
        f"{patch.file_count} file{'s' if patch.file_count != 1 else ''}: "
        f"[green]+{patch.total_additions}[/green] "
        f"[red]-{patch.total_deletions}[/red]"
    )


# -----------------------------------------------------------------------------
# Modal screen
# -----------------------------------------------------------------------------


class DiffViewer(ModalScreen[None]):
    """A modal for viewing a patch's diff with file-by-file navigation.

    Returns ``None`` on dismiss. The viewer is display-only — no edits
    can be made from within the modal.
    """

    BINDINGS = [
        Binding("escape", "dismiss_viewer", "Close", show=True),
        Binding("ctrl+c", "dismiss_viewer", "Close", show=False),
        Binding("tab", "next_file", "Next file", show=True),
        Binding("shift+tab", "prev_file", "Prev file", show=True),
        Binding("right", "next_file", "→", show=False),
        Binding("left", "prev_file", "←", show=False),
    ]

    DEFAULT_CSS = """
    DiffViewer {
        align: center middle;
        background: $background 80%;
    }

    DiffViewer > Vertical {
        width: 95%;
        max-width: 140;
        height: 90%;
        background: $surface;
        border: round $border;
        padding: 1 2;
    }

    DiffViewer #diff-title {
        height: 1;
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    DiffViewer #diff-summary {
        height: 1;
        color: $text-muted;
        margin-bottom: 1;
    }

    DiffViewer #diff-content {
        height: 1fr;
        width: 100%;
        background: $background;
        border: round $border;
        padding: 0 1;
    }

    DiffViewer #diff-help {
        margin-top: 1;
        color: $text-muted;
    }
    """

    def __init__(self, patch: PatchDiff) -> None:
        super().__init__()
        self._patch = patch
        self._file_index = 0

    # ----- compose -----

    def compose(self):  # type: ignore[no-untyped-def]
        with Vertical():
            yield Static(f"[b]{self._patch.title}[/b]", id="diff-title")
            yield Static(summarize_patch(self._patch), id="diff-summary")
            yield Static(self._render_current(), id="diff-content")
            yield Static(
                "Tab / → next  ·  Shift+Tab / ← prev  ·  Esc close",
                id="diff-help",
            )

    def on_mount(self) -> None:
        self._update_file_label()

    # ----- rendering -----

    def _render_current(self) -> str:
        if not self._patch.files:
            return "[dim]no files in patch[/dim]"
        fd = self._patch.files[self._file_index]
        label = (
            f"[b]{fd.short_label()}[/b]   "
            f"[green]+{fd.additions}[/green] [red]-{fd.deletions}[/red]\n\n"
        )
        if fd.status == "added":
            body = "[dim](new file)[/dim]\n" + _show_new_file(fd.modified)
        elif fd.status == "deleted":
            body = "[dim](deleted)[/dim]\n" + _show_deleted_file(fd.original)
        else:
            body = make_unified_diff(fd)
        return label + body

    def _update_file_label(self) -> None:
        if not self._patch.files:
            return
        try:
            content = self.query_one("#diff-content", Static)
        except Exception:
            return
        content.update(self._render_current())
        # Update the summary line with file position
        try:
            summary = self.query_one("#diff-summary", Static)
        except Exception:
            return
        position = f"  ·  file {self._file_index + 1}/{self._patch.file_count}"
        summary.update(summarize_patch(self._patch) + position)

    # ----- actions -----

    def action_next_file(self) -> None:
        if not self._patch.files:
            return
        self._file_index = (self._file_index + 1) % self._patch.file_count
        self._update_file_label()

    def action_prev_file(self) -> None:
        if not self._patch.files:
            return
        self._file_index = (self._file_index - 1) % self._patch.file_count
        self._update_file_label()

    def action_dismiss_viewer(self) -> None:
        self.dismiss(None)


# -----------------------------------------------------------------------------
# Display helpers for new/deleted files
# -----------------------------------------------------------------------------


def _show_new_file(content: str) -> str:
    """Render a new file's content with every line marked as an addition."""
    if not content:
        return "[dim](empty)[/dim]"
    return "\n".join(f"[green]+ {line}[/green]" for line in content.splitlines())


def _show_deleted_file(content: str) -> str:
    """Render a deleted file's content with every line marked as a deletion."""
    if not content:
        return "[dim](empty)[/dim]"
    return "\n".join(f"[red]- {line}[/red]" for line in content.splitlines())


# -----------------------------------------------------------------------------
# Convenience: build a PatchDiff from a patch payload dict
# -----------------------------------------------------------------------------


def patch_diff_from_payload(payload: Any) -> PatchDiff:
    """Convert a patch payload (dict or list) to a PatchDiff.

    Accepts the shape produced by the agent bridge:
        {"newFiles": [{"path": ..., "content": ...}, ...],
         "transformations": [{"targetFile": ..., "original": ..., "modified": ...}, ...],
         "branchName": "..."}

    Returns a :class:`PatchDiff` suitable for the viewer.
    """
    files: list[FileDiff] = []

    if isinstance(payload, dict):
        # New files
        for nf in payload.get("newFiles") or payload.get("new_files") or []:
            path = str(nf.get("path") or nf.get("name") or "(new)")
            content = str(nf.get("content") or "")
            files.append(make_file_diff_from_text(path, "", content))

        # Transformations
        for tr in payload.get("transformations") or []:
            path = str(tr.get("targetFile") or tr.get("target_file") or tr.get("path") or "?")
            original = str(tr.get("original") or tr.get("before") or "")
            modified = str(tr.get("modified") or tr.get("after") or "")
            files.append(make_file_diff_from_text(path, original, modified))

        # Removed files (deletions)
        for rm in payload.get("removedFiles") or payload.get("removed_files") or []:
            path = str(rm.get("path") or rm.get("name") or "(removed)")
            content = str(rm.get("content") or "")
            files.append(make_file_diff_from_text(path, content, ""))
    elif isinstance(payload, list):
        # Treat each item as a transformation-style entry
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path") or entry.get("targetFile") or "?")
            original = str(entry.get("original") or entry.get("before") or "")
            modified = str(entry.get("modified") or entry.get("after") or "")
            files.append(make_file_diff_from_text(path, original, modified))

    title = "Patch"
    if isinstance(payload, dict):
        branch = payload.get("branchName") or payload.get("branch_name")
        if branch:
            title = f"Patch ({branch})"

    return PatchDiff(title=title, files=tuple(files))
