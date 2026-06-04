"""Toast notification system for the ScholarDevClaw TUI.

Provides a lightweight, theme-consistent toast widget that:
- Auto-dismisses after a configurable duration
- Supports 4 severities: info, success, warning, error
- Supports an optional action button that fires a callback on press
- Stacks multiple toasts in the bottom-right corner without overlapping
- Uses ASCII/Unicode icons that render in any terminal

Usage:
    from .toasts import show_toast, ToastSeverity

    show_toast(self.app, "Saved 12 runs to history", severity="success")
    show_toast(
        self.app,
        "Patch generated",
        severity="success",
        action_label="View diff",
        action=lambda: push_diff_viewer(),
    )

The widget is intentionally self-contained — it has no dependencies on
pipeline state, history, or agent state, which keeps it easy to test in
isolation.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from textual.app import App
from textual.containers import Container
from textual.widgets import Static

# -----------------------------------------------------------------------------
# Public types
# -----------------------------------------------------------------------------

ToastSeverity = Literal["info", "success", "warning", "error"]

SEVERITY_ICONS: dict[ToastSeverity, str] = {
    "info": "ℹ",
    "success": "✓",
    "warning": "⚠",
    "error": "✗",
}

SEVERITY_COLORS: dict[ToastSeverity, str] = {
    "info": "$text-accent",
    "success": "$success",
    "warning": "$warning",
    "error": "$error",
}

DEFAULT_DURATION_SECONDS = 4.0


# -----------------------------------------------------------------------------
# Toast data
# -----------------------------------------------------------------------------


@dataclass
class Toast:
    """A single toast notification."""

    message: str
    severity: ToastSeverity = "info"
    title: str = ""
    duration: float = DEFAULT_DURATION_SECONDS
    action_label: str = ""
    action: Callable[[], None] | None = None
    created_at: float = field(default_factory=time.time)
    dismissed: bool = False

    @property
    def icon(self) -> str:
        return SEVERITY_ICONS.get(self.severity, "•")

    @property
    def color(self) -> str:
        """The Rich/textual color variable for this severity."""
        return SEVERITY_COLORS.get(self.severity, "$text")

    @property
    def has_action(self) -> bool:
        return bool(self.action_label and self.action is not None)

    def format_line(self) -> str:
        """Format the first line (icon + title)."""
        if self.title:
            return f"{self.icon} {self.title}"
        return f"{self.icon} {self.severity.upper()}"

    def format_message(self) -> str:
        """Format the body message."""
        return self.message


# -----------------------------------------------------------------------------
# Widget
# -----------------------------------------------------------------------------


class ToastWidget(Static):
    """A single toast notification widget.

    Displays the severity icon + title on the first line and the message
    below, with an optional action button hint on the right.

    The widget itself doesn't auto-dismiss — that is handled by the
    surrounding ToastStack which manages timers.
    """

    DEFAULT_CSS = """
    ToastWidget {
        width: 56;
        height: auto;
        padding: 0 1;
        margin: 0 1 1 1;
        background: $surface;
        color: $text;
        border: round $border;
    }

    ToastWidget.-info {
        border: round $text-accent;
    }

    ToastWidget.-success {
        border: round $success;
    }

    ToastWidget.-warning {
        border: round $warning;
    }

    ToastWidget.-error {
        border: round $error;
    }

    ToastWidget:hover {
        background: $surface;
    }
    """

    def __init__(self, toast: Toast, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.toast = toast
        self.add_class(f"-{toast.severity}")

    def render(self) -> str:  # type: ignore[override]
        """Render the toast content."""
        lines: list[str] = []
        header = self.toast.format_line()
        if self.toast.has_action:
            header = f"{header}  [dim][[{self.toast.action_label}][/dim]]"
        lines.append(header)
        # Word-wrap message at width-4 to keep it readable
        msg = self.toast.format_message()
        for chunk in _wrap(msg, width=52):
            lines.append(f"  {chunk}")
        return "\n".join(lines)


class ToastStack(Container):
    """A vertical stack of ToastWidgets.

    The stack is anchored to the bottom-right corner of the parent
    screen. Toasts are added at the top (newest first) and removed
    after their duration elapses.

    The stack does not own timer state directly — it accepts a
    `notifier` callable that the App schedules with `set_timer`.
    """

    DEFAULT_CSS = """
    ToastStack {
        layer: overlay;
        width: auto;
        height: auto;
        align: right bottom;
        background: transparent;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._toasts: list[ToastWidget] = []

    def add_toast(self, toast: Toast) -> None:
        """Add a toast to the top of the stack."""
        widget = ToastWidget(toast, id=f"toast-{int(toast.created_at * 1000)}")
        self._toasts.insert(0, widget)
        self.mount(widget)

    def remove_toast(self, toast: Toast) -> None:
        """Remove a toast from the stack and unmount its widget."""
        for widget in list(self._toasts):
            if widget.toast is toast or widget.toast.created_at == toast.created_at:
                widget.remove()
                self._toasts.remove(widget)
                toast.dismissed = True
                return

    def clear(self) -> None:
        """Remove all toasts."""
        for widget in list(self._toasts):
            widget.remove()
        self._toasts.clear()

    @property
    def count(self) -> int:
        return len(self._toasts)

    @property
    def toasts(self) -> list[ToastWidget]:
        return list(self._toasts)


# -----------------------------------------------------------------------------
# App-level helper
# -----------------------------------------------------------------------------


def _ensure_toast_stack(app: App[Any]) -> ToastStack:
    """Return the singleton ToastStack for the app, creating if needed.

    The stack is mounted on the active screen with an overlay layer so
    it floats above other content.
    """
    screen = app.screen
    existing = screen.query("ToastStack")
    if existing:
        first = existing.first()
        if isinstance(first, ToastStack):
            return first
    stack = ToastStack()
    screen.mount(stack)
    return stack


def show_toast(
    app: App[Any],
    message: str,
    *,
    severity: ToastSeverity = "info",
    title: str = "",
    duration: float = DEFAULT_DURATION_SECONDS,
    action_label: str = "",
    action: Callable[[], None] | None = None,
) -> Toast:
    """Show a toast notification on the current screen.

    Returns the Toast object. The toast will auto-dismiss after
    `duration` seconds. If `action` and `action_label` are provided,
    the toast can be activated by pressing the key bound to it.
    """
    toast = Toast(
        message=message,
        severity=severity,
        title=title,
        duration=duration,
        action_label=action_label,
        action=action,
    )
    stack = _ensure_toast_stack(app)
    stack.add_toast(toast)
    if duration > 0:
        app.set_timer(duration, lambda: stack.remove_toast(toast))
    return toast


# -----------------------------------------------------------------------------
# Word-wrap helper
# -----------------------------------------------------------------------------


def _wrap(text: str, width: int) -> list[str]:
    """Wrap text to `width` columns, breaking on word boundaries."""
    if width <= 0:
        return [text]
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines
