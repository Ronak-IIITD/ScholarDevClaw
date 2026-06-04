"""Tests for LogView severity / search filtering."""

from __future__ import annotations

from scholardevclaw.tui.widgets import LogView

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


class _MountSpy:
    """A mount-replacement that records every (widget, kwargs) call."""

    def __init__(self, target: LogView) -> None:
        self.target = target
        self.calls: list[tuple[object, dict]] = []
        self.children: list[object] = []

    def __call__(self, *widgets: object, **kwargs: object) -> None:
        for w in widgets:
            self.calls.append((w, dict(kwargs)))
            self.children.append(w)
        # The first widget in the call becomes a real "child" we can
        # access via the parent. For tests that need to remove()
        # widgets, we just no-op since _enforce_max_visible only checks
        # the .children list.
        try:
            self.target.children.extend(widgets)  # type: ignore[attr-defined]
        except Exception:
            pass


def _install_mount_spy(log: LogView) -> _MountSpy:
    spy = _MountSpy(log)
    object.__setattr__(log, "mount", spy)  # type: ignore[method-assign]
    return spy


# -----------------------------------------------------------------------------
# Filter state
# -----------------------------------------------------------------------------


class TestLogViewFilterState:
    def test_default_severity_is_all(self) -> None:
        log = LogView()
        assert log.severity_filter == "all"

    def test_default_search_is_empty(self) -> None:
        log = LogView()
        assert log.search_filter == ""

    def test_default_buffer_empty(self) -> None:
        log = LogView()
        assert log.buffer_size == 0
        assert log.visible_count == 0

    def test_severity_filter_levels_complete(self) -> None:
        expected = {"all", "info", "success", "warning", "error", "system", "debug"}
        assert set(LogView.SEVERITY_FILTER_LEVELS) == expected


# -----------------------------------------------------------------------------
# Severity filter
# -----------------------------------------------------------------------------


class TestLogViewSeverityFilter:
    def test_set_severity_filter_updates_state(self) -> None:
        log = LogView()
        log.set_severity_filter("warning")
        assert log.severity_filter == "warning"

    def test_set_severity_filter_unknown_falls_back_to_all(self) -> None:
        log = LogView()
        log.set_severity_filter("not-a-level")  # type: ignore[arg-type]
        assert log.severity_filter == "all"

    def test_cycle_severity_advances(self) -> None:
        log = LogView()
        next_level = log.cycle_severity_filter()
        assert next_level == "info"
        next_level = log.cycle_severity_filter()
        assert next_level == "success"
        next_level = log.cycle_severity_filter()
        assert next_level == "warning"

    def test_cycle_severity_wraps(self) -> None:
        log = LogView()
        # Walking through all levels once should wrap "debug" back to "all"
        for _ in range(len(LogView.SEVERITY_FILTER_LEVELS)):
            log.cycle_severity_filter()
        assert log.severity_filter == "all"

    def test_add_log_always_appends_to_buffer(self) -> None:
        log = LogView()
        _install_mount_spy(log)
        log.set_severity_filter("warning")
        log.add_log("info message", level="info")
        log.add_log("warning message", level="warning")
        log.add_log("error message", level="error")
        # Buffer should have all 3
        assert log.buffer_size == 3
        # Visible count should be only 1 (the warning)
        assert log.visible_count == 1

    def test_change_filter_re_renders_buffer(self) -> None:
        log = LogView()
        _install_mount_spy(log)
        log.add_log("info message", level="info")
        log.add_log("warning message", level="warning")
        log.add_log("error message", level="error")
        # Default: all visible
        assert log.visible_count == 3
        # Switch to error filter
        log.set_severity_filter("error")
        assert log.visible_count == 1
        # Switch to all
        log.set_severity_filter("all")
        assert log.visible_count == 3


# -----------------------------------------------------------------------------
# Search filter
# -----------------------------------------------------------------------------


class TestLogViewSearchFilter:
    def test_set_search_filter_updates_state(self) -> None:
        log = LogView()
        log.set_search_filter("foo")
        assert log.search_filter == "foo"

    def test_set_search_filter_none_clears(self) -> None:
        log = LogView()
        log.set_search_filter("foo")
        log.set_search_filter(None)  # type: ignore[arg-type]
        assert log.search_filter == ""

    def test_search_filter_substring_match(self) -> None:
        log = LogView()
        _install_mount_spy(log)
        log.add_log("hello world", level="info")
        log.add_log("goodbye", level="info")
        log.add_log("hello again", level="info")
        log.set_search_filter("hello")
        assert log.visible_count == 2

    def test_search_filter_case_insensitive(self) -> None:
        log = LogView()
        _install_mount_spy(log)
        log.add_log("Hello World", level="info")
        log.set_search_filter("hello")
        assert log.visible_count == 1

    def test_search_filter_no_match_hides_all(self) -> None:
        log = LogView()
        _install_mount_spy(log)
        log.add_log("alpha", level="info")
        log.add_log("beta", level="info")
        log.set_search_filter("zzz")
        assert log.visible_count == 0

    def test_search_filter_clearing_restores_all(self) -> None:
        log = LogView()
        _install_mount_spy(log)
        log.add_log("alpha", level="info")
        log.add_log("beta", level="info")
        log.set_search_filter("alp")
        assert log.visible_count == 1
        log.set_search_filter("")
        assert log.visible_count == 2


# -----------------------------------------------------------------------------
# Combined filters
# -----------------------------------------------------------------------------


class TestLogViewCombinedFilters:
    def test_severity_and_search_together(self) -> None:
        log = LogView()
        _install_mount_spy(log)
        log.add_log("INFO: hello", level="info")
        log.add_log("ERROR: hello", level="error")
        log.add_log("ERROR: world", level="error")
        log.set_severity_filter("error")
        log.set_search_filter("hello")
        # Only "ERROR: hello" matches both
        assert log.visible_count == 1

    def test_clear_filters_resets_both(self) -> None:
        log = LogView()
        _install_mount_spy(log)
        log.add_log("a", level="info")
        log.add_log("b", level="error")
        log.set_severity_filter("error")
        log.set_search_filter("a")
        log.clear_filters()
        assert log.severity_filter == "all"
        assert log.search_filter == ""
        assert log.visible_count == 2


# -----------------------------------------------------------------------------
# Filter header visibility
# -----------------------------------------------------------------------------


class TestLogViewFilterHeader:
    def test_no_header_when_no_filter(self) -> None:
        log = LogView()
        assert log._filter_header is None

    def test_header_appears_with_severity(self) -> None:
        log = LogView()
        _install_mount_spy(log)
        log.set_severity_filter("warning")
        # Header should be created and mounted
        assert log._filter_header is not None
        assert len(log._filter_label) > 0
        assert "warning" in log._filter_label

    def test_header_appears_with_search(self) -> None:
        log = LogView()
        _install_mount_spy(log)
        log.set_search_filter("foo")
        assert log._filter_header is not None

    def test_header_disappears_when_cleared(self) -> None:
        log = LogView()
        _install_mount_spy(log)
        log.set_severity_filter("warning")
        assert log._filter_header is not None
        log.set_severity_filter("all")
        assert log._filter_header is None


# -----------------------------------------------------------------------------
# clear_logs preserves filters but clears buffer
# -----------------------------------------------------------------------------


class TestLogViewClearLogs:
    def test_clear_logs_empties_buffer(self) -> None:
        log = LogView()
        _install_mount_spy(log)
        log.add_log("a", level="info")
        log.add_log("b", level="info")
        assert log.buffer_size == 2
        log.clear_logs()
        assert log.buffer_size == 0
        assert log.visible_count == 0

    def test_clear_logs_preserves_filters(self) -> None:
        log = LogView()
        _install_mount_spy(log)
        log.set_severity_filter("warning")
        log.add_log("info", level="info")
        log.clear_logs()
        # Filter state is preserved
        assert log.severity_filter == "warning"
