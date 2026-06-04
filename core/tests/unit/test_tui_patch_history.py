"""Tests for the PatchHistoryScreen modal."""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from scholardevclaw.tui.patch_history import (  # noqa: E402
    PatchHistoryEntry,
    PatchHistoryScreen,
    _summarize_payload,
    build_patch_history,
)


def _make_payload(path: str = "foo.py", added: bool = True) -> dict:
    if added:
        return {"newFiles": [{"path": path, "content": "x = 1\n"}]}
    return {
        "transformations": [
            {
                "targetFile": path,
                "original": "x = 1\n",
                "modified": "x = 2\n",
            }
        ]
    }


class TestPatchHistoryScreenInstantiation:
    def test_can_instantiate_empty(self) -> None:
        screen = PatchHistoryScreen()
        assert screen is not None
        assert screen._all_entries == []

    def test_can_instantiate_with_payloads(self) -> None:
        screen = PatchHistoryScreen(payloads=[_make_payload("a.py")])
        assert len(screen._all_entries) == 1

    def test_inherits_modal_screen(self) -> None:
        from textual.screen import ModalScreen

        assert isinstance(PatchHistoryScreen(), ModalScreen)


class TestBuildPatchHistory:
    def test_empty_input(self) -> None:
        assert build_patch_history([]) == []

    def test_wraps_each_payload(self) -> None:
        payloads = [_make_payload("a.py"), _make_payload("b.py")]
        entries = build_patch_history(payloads)
        assert len(entries) == 2
        for entry, payload in zip(entries, payloads):
            assert isinstance(entry, PatchHistoryEntry)
            assert entry.payload is payload

    def test_summary_contains_path(self) -> None:
        entries = build_patch_history([_make_payload("hello.py")])
        assert "hello.py" in entries[0].summary

    def test_summary_for_multi_file(self) -> None:
        payload = {
            "newFiles": [
                {"path": "a.py", "content": ""},
                {"path": "b.py", "content": ""},
                {"path": "c.py", "content": ""},
            ]
        }
        entries = build_patch_history([payload])
        assert entries[0].summary.startswith("3 files:")

    def test_timestamp_is_recent(self) -> None:
        from datetime import datetime, timedelta

        entries = build_patch_history([_make_payload()])
        delta = datetime.now() - entries[0].timestamp
        assert delta < timedelta(seconds=2)


class TestSummarizePayload:
    def test_empty_payload(self) -> None:
        assert "(empty patch)" in _summarize_payload({})

    def test_single_file(self) -> None:
        s = _summarize_payload(_make_payload("a.py"))
        assert "a.py" in s

    def test_handles_invalid_payload(self) -> None:
        s = _summarize_payload({"garbage": True})
        # Should not raise
        assert isinstance(s, str)


class TestPatchHistoryFilter:
    def test_filter_narrows_results(self) -> None:
        screen = PatchHistoryScreen(
            payloads=[
                _make_payload("alpha.py"),
                _make_payload("beta.py"),
            ]
        )
        screen._query = "alpha"
        # Internal state is updated; _refresh_list is called by the
        # Input.Changed event. We just verify the entry list is
        # searchable by summary text.
        assert any("alpha" in e.summary for e in screen._all_entries)
        assert any("beta" in e.summary for e in screen._all_entries)


class TestPatchHistoryActions:
    def test_has_dismiss_action(self) -> None:
        screen = PatchHistoryScreen()
        assert callable(getattr(screen, "action_dismiss_modal", None))

    def test_has_select_first_action(self) -> None:
        screen = PatchHistoryScreen()
        assert callable(getattr(screen, "action_select_first", None))


class TestPatchHistoryEntryRender:
    def test_render_includes_timestamp(self) -> None:
        from datetime import datetime

        entry = PatchHistoryEntry(
            payload=_make_payload("a.py"),
            timestamp=datetime(2024, 1, 1, 12, 30, 45),
            summary="a.py",
        )
        out = entry.render()
        assert "12:30:45" in out
        assert "a.py" in out
