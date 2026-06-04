"""Tests for the ReverseSearchScreen modal."""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from scholardevclaw.tui.reverse_search import ReverseSearchScreen  # noqa: E402


class TestReverseSearchScreenInstantiation:
    def test_can_instantiate(self) -> None:
        screen = ReverseSearchScreen()
        assert screen is not None

    def test_default_history_empty(self) -> None:
        screen = ReverseSearchScreen()
        assert screen._history == []
        assert screen._query == ""
        assert screen._matches == []

    def test_accepts_history(self) -> None:
        screen = ReverseSearchScreen(history=["foo", "bar", "baz"])
        # Most-recent-first ordering
        assert screen._history == ["baz", "bar", "foo"]

    def test_inherits_modal_screen(self) -> None:
        from textual.screen import ModalScreen

        assert isinstance(ReverseSearchScreen(), ModalScreen)


class TestReverseSearchMatching:
    def test_refresh_matches_finds_substring(self) -> None:
        screen = ReverseSearchScreen(history=["analyze foo", "search bar", "analyze bar"])
        screen._query = "analyze"
        screen._refresh_matches()
        # Most-recent-first history was ["analyze bar", "search bar", "analyze foo"]
        assert screen._matches == ["analyze bar", "analyze foo"]

    def test_refresh_matches_case_insensitive(self) -> None:
        screen = ReverseSearchScreen(history=["Analyze Foo", "search bar"])
        screen._query = "analyze"
        screen._refresh_matches()
        assert screen._matches == ["Analyze Foo"]

    def test_refresh_matches_no_query(self) -> None:
        screen = ReverseSearchScreen(history=["foo", "bar"])
        screen._query = ""
        screen._refresh_matches()
        assert screen._matches == []

    def test_refresh_matches_no_hits(self) -> None:
        screen = ReverseSearchScreen(history=["foo", "bar"])
        screen._query = "zzz"
        screen._refresh_matches()
        assert screen._matches == []


class TestReverseSearchNavigation:
    def test_find_prev_advances(self) -> None:
        screen = ReverseSearchScreen(history=["a", "b", "c"])
        screen._query = "a"
        screen._refresh_matches()
        # Wrap to first match
        screen._match_index = 0
        screen.action_find_prev()
        assert screen._match_index == 1 % len(screen._matches)

    def test_find_next_wraps(self) -> None:
        screen = ReverseSearchScreen(history=["a", "b", "c"])
        screen._query = "a"
        screen._refresh_matches()
        screen._match_index = 0
        screen.action_find_next()
        # Goes to last match
        assert screen._match_index == len(screen._matches) - 1

    def test_find_prev_on_empty_matches_is_noop(self) -> None:
        screen = ReverseSearchScreen(history=["foo"])
        screen._query = "zzz"
        screen._refresh_matches()
        before = screen._match_index
        screen.action_find_prev()
        assert screen._match_index == before


class TestReverseSearchAccept:
    def test_accept_with_match_returns_command(self) -> None:
        # We can't easily exercise .dismiss() in a unit test, so test the
        # return value the screen would pass to dismiss.
        screen = ReverseSearchScreen(history=["analyze foo", "search bar"])
        screen._query = "analyze"
        screen._refresh_matches()
        # Simulate the inner logic of action_accept
        chosen = screen._matches[screen._match_index] if screen._matches else screen._query
        assert chosen == "analyze foo"

    def test_accept_without_match_returns_query(self) -> None:
        screen = ReverseSearchScreen(history=["foo"])
        screen._query = "zzz"
        screen._refresh_matches()
        chosen = screen._matches[screen._match_index] if screen._matches else screen._query
        assert chosen == "zzz"


class TestReverseSearchActions:
    def test_has_cancel_action(self) -> None:
        screen = ReverseSearchScreen()
        assert callable(getattr(screen, "action_cancel", None))

    def test_has_accept_action(self) -> None:
        screen = ReverseSearchScreen()
        assert callable(getattr(screen, "action_accept", None))

    def test_has_find_prev_action(self) -> None:
        screen = ReverseSearchScreen()
        assert callable(getattr(screen, "action_find_prev", None))

    def test_has_find_next_action(self) -> None:
        screen = ReverseSearchScreen()
        assert callable(getattr(screen, "action_find_next", None))
