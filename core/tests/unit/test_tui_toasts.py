"""Tests for the toast notification system."""

from __future__ import annotations

import pytest

from scholardevclaw.tui.toasts import (
    DEFAULT_DURATION_SECONDS,
    SEVERITY_COLORS,
    SEVERITY_ICONS,
    Toast,
    ToastStack,
    ToastWidget,
    _wrap,
    show_toast,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------


class TestSeverityConstants:
    """Severity → icon/color mappings must be complete."""

    def test_severity_icons_keys(self) -> None:
        assert set(SEVERITY_ICONS.keys()) == {"info", "success", "warning", "error"}

    def test_severity_colors_keys(self) -> None:
        assert set(SEVERITY_COLORS.keys()) == {"info", "success", "warning", "error"}

    def test_severity_icons_non_empty(self) -> None:
        for sev, icon in SEVERITY_ICONS.items():
            assert icon, f"empty icon for {sev}"

    def test_severity_colors_non_empty(self) -> None:
        for sev, color in SEVERITY_COLORS.items():
            assert color, f"empty color for {sev}"
            assert color.startswith("$"), f"color for {sev} not a CSS var"

    def test_default_duration_positive(self) -> None:
        assert DEFAULT_DURATION_SECONDS > 0


# -----------------------------------------------------------------------------
# Toast dataclass
# -----------------------------------------------------------------------------


class TestToast:
    """The Toast data class."""

    def test_minimal_construction(self) -> None:
        t = Toast(message="hi")
        assert t.message == "hi"
        assert t.severity == "info"
        assert t.title == ""
        assert t.duration == DEFAULT_DURATION_SECONDS
        assert t.action_label == ""
        assert t.action is None
        assert t.dismissed is False

    def test_full_construction(self) -> None:
        cb = lambda: None  # noqa: E731
        t = Toast(
            message="body",
            severity="warning",
            title="heads up",
            duration=2.5,
            action_label="retry",
            action=cb,
        )
        assert t.severity == "warning"
        assert t.title == "heads up"
        assert t.duration == 2.5
        assert t.action_label == "retry"
        assert t.action is cb

    def test_icon_property(self) -> None:
        assert Toast(message="x", severity="info").icon == SEVERITY_ICONS["info"]
        assert Toast(message="x", severity="success").icon == SEVERITY_ICONS["success"]
        assert Toast(message="x", severity="warning").icon == SEVERITY_ICONS["warning"]
        assert Toast(message="x", severity="error").icon == SEVERITY_ICONS["error"]

    def test_color_property(self) -> None:
        assert Toast(message="x", severity="info").color == SEVERITY_COLORS["info"]
        assert Toast(message="x", severity="success").color == SEVERITY_COLORS["success"]

    def test_has_action(self) -> None:
        assert Toast(message="x").has_action is False
        assert Toast(message="x", action_label="go").has_action is False
        assert Toast(message="x", action=lambda: None).has_action is False
        assert Toast(message="x", action_label="go", action=lambda: None).has_action is True

    def test_format_line_with_title(self) -> None:
        t = Toast(message="body", severity="success", title="Done")
        line = t.format_line()
        assert "Done" in line
        assert SEVERITY_ICONS["success"] in line

    def test_format_line_without_title(self) -> None:
        t = Toast(message="x", severity="info")
        line = t.format_line()
        assert "INFO" in line

    def test_format_message(self) -> None:
        assert Toast(message="hello world").format_message() == "hello world"

    def test_created_at_set(self) -> None:
        t = Toast(message="x")
        assert t.created_at > 0


# -----------------------------------------------------------------------------
# Word-wrap helper
# -----------------------------------------------------------------------------


class TestWrap:
    """_wrap should break on word boundaries."""

    def test_short_text_single_line(self) -> None:
        assert _wrap("hi", 10) == ["hi"]

    def test_empty_text(self) -> None:
        assert _wrap("", 10) == [""]

    def test_exact_width(self) -> None:
        assert _wrap("abcdef", 6) == ["abcdef"]

    def test_overflow_wraps(self) -> None:
        result = _wrap("the quick brown fox", 10)
        assert all(len(line) <= 10 for line in result)
        # Original words preserved
        joined = " ".join(result)
        assert joined == "the quick brown fox"

    def test_zero_width_returns_whole(self) -> None:
        assert _wrap("hello world", 0) == ["hello world"]


# -----------------------------------------------------------------------------
# ToastStack
# -----------------------------------------------------------------------------


class TestToastStack:
    """Stack management inside a mounted App."""

    @pytest.mark.asyncio
    async def test_initial_count(self) -> None:
        from textual.app import App

        class _Probe(App[None]):
            pass

        app = _Probe()
        async with app.run_test() as pilot:
            await pilot.pause()
            stack = ToastStack()
            app.screen.mount(stack)
            await pilot.pause()
            assert stack.count == 0
            assert stack.toasts == []

    @pytest.mark.asyncio
    async def test_add_toast_increments(self) -> None:
        from textual.app import App

        class _Probe(App[None]):
            pass

        app = _Probe()
        async with app.run_test() as pilot:
            stack = ToastStack()
            app.screen.mount(stack)
            await pilot.pause()
            t = Toast(message="a")
            stack.add_toast(t)
            await pilot.pause()
            assert stack.count == 1

    @pytest.mark.asyncio
    async def test_remove_toast_decrements(self) -> None:
        from textual.app import App

        class _Probe(App[None]):
            pass

        app = _Probe()
        async with app.run_test() as pilot:
            stack = ToastStack()
            app.screen.mount(stack)
            await pilot.pause()
            t = Toast(message="a")
            stack.add_toast(t)
            await pilot.pause()
            stack.remove_toast(t)
            await pilot.pause()
            assert stack.count == 0
            assert t.dismissed is True

    @pytest.mark.asyncio
    async def test_remove_unknown_toast_noop(self) -> None:
        from textual.app import App

        class _Probe(App[None]):
            pass

        app = _Probe()
        async with app.run_test() as pilot:
            stack = ToastStack()
            app.screen.mount(stack)
            await pilot.pause()
            t = Toast(message="a")
            # Should not raise even though t was never added
            stack.remove_toast(t)
            assert stack.count == 0

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        from textual.app import App

        class _Probe(App[None]):
            pass

        app = _Probe()
        async with app.run_test() as pilot:
            stack = ToastStack()
            app.screen.mount(stack)
            await pilot.pause()
            stack.add_toast(Toast(message="a"))
            stack.add_toast(Toast(message="b"))
            await pilot.pause()
            assert stack.count == 2
            stack.clear()
            await pilot.pause()
            assert stack.count == 0


# -----------------------------------------------------------------------------
# ToastWidget (CSS class assignment only — no App)
# -----------------------------------------------------------------------------


class TestToastWidget:
    """Widget construction + CSS class."""

    def test_severity_class_info(self) -> None:
        w = ToastWidget(Toast(message="x", severity="info"))
        assert w.has_class("-info")

    def test_severity_class_success(self) -> None:
        w = ToastWidget(Toast(message="x", severity="success"))
        assert w.has_class("-success")

    def test_severity_class_warning(self) -> None:
        w = ToastWidget(Toast(message="x", severity="warning"))
        assert w.has_class("-warning")

    def test_severity_class_error(self) -> None:
        w = ToastWidget(Toast(message="x", severity="error"))
        assert w.has_class("-error")

    def test_stores_toast(self) -> None:
        t = Toast(message="hello", severity="success")
        w = ToastWidget(t)
        assert w.toast is t


# -----------------------------------------------------------------------------
# show_toast (with a minimal App fixture)
# -----------------------------------------------------------------------------


class TestShowToast:
    """The top-level helper needs a real App to mount the stack."""

    @pytest.mark.asyncio
    async def test_show_toast_creates_widget(self) -> None:
        from textual.app import App

        class _Probe(App[None]):
            pass

        app = _Probe()
        async with app.run_test() as pilot:
            toast = show_toast(app, "patch generated", severity="success", duration=0)
            await pilot.pause()
            stack = app.screen.query("ToastStack").first()
            assert isinstance(stack, ToastStack)
            assert stack.count == 1
            assert toast.severity == "success"
            assert toast.dismissed is False

    @pytest.mark.asyncio
    async def test_show_toast_with_action(self) -> None:
        from textual.app import App

        called: list[bool] = []

        def on_action() -> None:
            called.append(True)

        class _Probe(App[None]):
            pass

        app = _Probe()
        async with app.run_test() as pilot:
            t = show_toast(
                app,
                "patch ready",
                severity="success",
                action_label="view",
                action=on_action,
                duration=0,
            )
            await pilot.pause()
            assert t.has_action is True
            assert t.action is on_action
            # Manually invoke the action
            assert t.action is not None
            t.action()
            assert called == [True]

    @pytest.mark.asyncio
    async def test_show_toast_default_severity_is_info(self) -> None:
        from textual.app import App

        class _Probe(App[None]):
            pass

        app = _Probe()
        async with app.run_test() as pilot:
            t = show_toast(app, "hello", duration=0)
            await pilot.pause()
            assert t.severity == "info"
