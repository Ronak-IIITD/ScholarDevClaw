"""Tests for the diff / patch viewer modal."""

from __future__ import annotations

import pytest

from scholardevclaw.tui.diff_viewer import (
    DiffViewer,
    FileDiff,
    PatchDiff,
    _colorize_unified,
    compute_additions_deletions,
    make_file_diff_from_text,
    make_unified_diff,
    patch_diff_from_payload,
    summarize_patch,
)

# -----------------------------------------------------------------------------
# FileDiff
# -----------------------------------------------------------------------------


class TestFileDiff:
    def test_construction(self) -> None:
        fd = FileDiff(path="src/foo.py", status="modified", original="a", modified="b")
        assert fd.path == "src/foo.py"
        assert fd.status == "modified"
        assert fd.additions == 0  # default
        assert fd.deletions == 0

    def test_short_label_added(self) -> None:
        fd = FileDiff(path="new.py", status="added")
        assert fd.short_label() == "+ new.py"

    def test_short_label_modified(self) -> None:
        fd = FileDiff(path="mod.py", status="modified")
        assert fd.short_label() == "~ mod.py"

    def test_short_label_deleted(self) -> None:
        fd = FileDiff(path="gone.py", status="deleted")
        assert fd.short_label() == "- gone.py"

    def test_short_label_renamed(self) -> None:
        fd = FileDiff(path="x.py", status="renamed")
        assert fd.short_label() == "→ x.py"

    def test_short_label_unknown(self) -> None:
        fd = FileDiff(path="x.py", status="unknown")  # type: ignore[arg-type]
        assert fd.short_label().startswith("?")


# -----------------------------------------------------------------------------
# PatchDiff
# -----------------------------------------------------------------------------


class TestPatchDiff:
    def test_empty_patch(self) -> None:
        p = PatchDiff(title="empty")
        assert p.file_count == 0
        assert p.total_additions == 0
        assert p.total_deletions == 0

    def test_totals(self) -> None:
        files = (
            FileDiff(path="a.py", status="added", additions=10, deletions=2),
            FileDiff(path="b.py", status="modified", additions=5, deletions=5),
        )
        p = PatchDiff(title="t", files=files)
        assert p.file_count == 2
        assert p.total_additions == 15
        assert p.total_deletions == 7

    def test_metadata_default(self) -> None:
        p = PatchDiff(title="t")
        assert p.metadata == {}

    def test_frozen(self) -> None:
        p = PatchDiff(title="t")
        with pytest.raises((AttributeError, Exception)):
            p.title = "x"  # type: ignore[misc]


# -----------------------------------------------------------------------------
# Diff generation helpers
# -----------------------------------------------------------------------------


class TestColorizeUnified:
    def test_empty(self) -> None:
        assert _colorize_unified("") == ""

    def test_colorizes_plus_line(self) -> None:
        out = _colorize_unified("+hello")
        assert "[green]" in out
        assert "+hello" in out
        assert "[/green]" in out

    def test_colorizes_minus_line(self) -> None:
        out = _colorize_unified("-world")
        assert "[red]" in out
        assert "-world" in out

    def test_colorizes_hunk_header(self) -> None:
        out = _colorize_unified("@@ -1,3 +1,3 @@")
        assert "[dim]" in out
        assert "@@" in out

    def test_colorizes_file_header(self) -> None:
        out = _colorize_unified("+++ b/foo.py")
        assert "[cyan]" in out

    def test_context_line_unchanged(self) -> None:
        out = _colorize_unified(" normal line")
        assert "[" not in out  # no markup


class TestComputeAdditionsDeletions:
    def test_pure_addition(self) -> None:
        fd = FileDiff(path="x", status="added", original="", modified="a\nb\nc")
        adds, dels = compute_additions_deletions(fd)
        assert adds == 3
        assert dels == 0

    def test_pure_deletion(self) -> None:
        fd = FileDiff(path="x", status="deleted", original="a\nb\nc", modified="")
        adds, dels = compute_additions_deletions(fd)
        assert adds == 0
        assert dels == 3

    def test_modification_balanced(self) -> None:
        fd = FileDiff(path="x", status="modified", original="a\nb", modified="c\nd")
        adds, dels = compute_additions_deletions(fd)
        assert adds == 2
        assert dels == 2

    def test_no_change(self) -> None:
        fd = FileDiff(path="x", status="modified", original="a", modified="a")
        adds, dels = compute_additions_deletions(fd)
        assert adds == 0
        assert dels == 0


class TestMakeFileDiffFromText:
    def test_added_status_when_empty_original(self) -> None:
        fd = make_file_diff_from_text("x.py", "", "hello\n")
        assert fd.status == "added"
        assert fd.additions == 1
        assert fd.deletions == 0

    def test_deleted_status_when_empty_modified(self) -> None:
        fd = make_file_diff_from_text("x.py", "hello\n", "")
        assert fd.status == "deleted"
        assert fd.additions == 0
        assert fd.deletions == 1

    def test_modified_status_when_both_nonempty(self) -> None:
        fd = make_file_diff_from_text("x.py", "a", "b")
        assert fd.status == "modified"


class TestMakeUnifiedDiff:
    def test_added_file(self) -> None:
        fd = make_file_diff_from_text("new.py", "", "line1\nline2")
        diff = make_unified_diff(fd)
        assert "new.py" in diff
        assert "[green]" in diff
        assert "+ line1" in diff or "+line1" in diff

    def test_deleted_file(self) -> None:
        fd = make_file_diff_from_text("old.py", "line1\nline2", "")
        diff = make_unified_diff(fd)
        assert "[red]" in diff
        assert "- line1" in diff or "-line1" in diff

    def test_modified_file(self) -> None:
        fd = make_file_diff_from_text("m.py", "old", "new")
        diff = make_unified_diff(fd)
        assert "[red]" in diff
        assert "[green]" in diff


class TestSummarizePatch:
    def test_empty(self) -> None:
        s = summarize_patch(PatchDiff(title="t"))
        assert "0 files" in s
        assert "+0" in s
        assert "-0" in s

    def test_singular_file(self) -> None:
        p = PatchDiff(title="t", files=(FileDiff(path="x", status="added", additions=5),))
        s = summarize_patch(p)
        assert "1 file" in s  # not "1 files"
        assert "1 files" not in s
        assert "+5" in s

    def test_plural_files(self) -> None:
        p = PatchDiff(
            title="t",
            files=(
                FileDiff(path="a", status="modified", additions=3, deletions=1),
                FileDiff(path="b", status="added", additions=2),
            ),
        )
        s = summarize_patch(p)
        assert "2 files" in s
        assert "+5" in s
        assert "-1" in s


# -----------------------------------------------------------------------------
# patch_diff_from_payload
# -----------------------------------------------------------------------------


class TestPatchDiffFromPayload:
    def test_empty_payload(self) -> None:
        p = patch_diff_from_payload({})
        assert p.file_count == 0

    def test_new_files_camel_case(self) -> None:
        p = patch_diff_from_payload({"newFiles": [{"path": "a.py", "content": "print(1)\n"}]})
        assert p.file_count == 1
        assert p.files[0].path == "a.py"
        assert p.files[0].status == "added"
        assert p.files[0].additions == 1

    def test_new_files_snake_case(self) -> None:
        p = patch_diff_from_payload({"new_files": [{"path": "a.py", "content": "x\ny\n"}]})
        assert p.file_count == 1
        assert p.files[0].status == "added"

    def test_transformations_camel_case(self) -> None:
        p = patch_diff_from_payload(
            {
                "transformations": [
                    {
                        "targetFile": "src/foo.py",
                        "original": "old\n",
                        "modified": "new\n",
                    }
                ]
            }
        )
        assert p.file_count == 1
        assert p.files[0].path == "src/foo.py"
        assert p.files[0].status == "modified"

    def test_transformations_snake_case(self) -> None:
        p = patch_diff_from_payload(
            {
                "transformations": [
                    {
                        "target_file": "src/foo.py",
                        "before": "old",
                        "after": "new",
                    }
                ]
            }
        )
        assert p.file_count == 1

    def test_removed_files(self) -> None:
        p = patch_diff_from_payload({"removedFiles": [{"path": "x.py", "content": "y\n"}]})
        assert p.file_count == 1
        assert p.files[0].status == "deleted"

    def test_branch_name_in_title(self) -> None:
        p = patch_diff_from_payload({"branchName": "feat/foo"})
        assert "feat/foo" in p.title

    def test_list_payload(self) -> None:
        p = patch_diff_from_payload([{"path": "a.py", "original": "1", "modified": "2"}])
        assert p.file_count == 1

    def test_list_with_non_dict_skipped(self) -> None:
        p = patch_diff_from_payload(
            [{"path": "a.py", "original": "1", "modified": "2"}, "not_a_dict", 42]
        )
        assert p.file_count == 1

    def test_default_title(self) -> None:
        p = patch_diff_from_payload({})
        assert p.title == "Patch"


# -----------------------------------------------------------------------------
# DiffViewer (construction only)
# -----------------------------------------------------------------------------


class TestDiffViewer:
    def test_constructible_empty(self) -> None:
        v = DiffViewer(PatchDiff(title="t"))
        assert v._file_index == 0
        assert v._patch.file_count == 0

    def test_constructible_with_files(self) -> None:
        files = (
            FileDiff(path="a.py", status="added", additions=5),
            FileDiff(path="b.py", status="modified", additions=2, deletions=1),
        )
        v = DiffViewer(PatchDiff(title="t", files=files))
        assert v._file_index == 0
        assert v._patch.file_count == 2

    def test_modal_screen_subclass(self) -> None:
        from textual.screen import ModalScreen

        assert issubclass(DiffViewer, ModalScreen)

    def test_next_file_wraps(self) -> None:
        files = (
            FileDiff(path="a.py", status="added", additions=1),
            FileDiff(path="b.py", status="added", additions=1),
        )
        v = DiffViewer(PatchDiff(title="t", files=files))
        v.action_next_file()
        assert v._file_index == 1
        v.action_next_file()
        assert v._file_index == 0  # wraps

    def test_prev_file_wraps(self) -> None:
        files = (
            FileDiff(path="a.py", status="added", additions=1),
            FileDiff(path="b.py", status="added", additions=1),
        )
        v = DiffViewer(PatchDiff(title="t", files=files))
        v.action_prev_file()
        assert v._file_index == 1  # wraps backward

    def test_next_file_empty_no_op(self) -> None:
        v = DiffViewer(PatchDiff(title="t"))
        v.action_next_file()  # should not raise
        assert v._file_index == 0
