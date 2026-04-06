"""Animated widgets for the TUI."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from textual import work
from textual.app import ComposeResult
from textual.message import Message
from textual.widgets import Static


class Spinner(Static):
    """Animated loading spinner widget.

    Displays a rotating spinner animation while work is in progress.
    """

    DEFAULT_CSS = """
    Spinner {
        width: auto;
        height: 1;
        color: $accent;
    }
    """

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(self.FRAMES[0], **kwargs)
        self._frame = 0
        self._running = False
        self._last_update = 0.0

    def on_mount(self) -> None:
        self._running = True
        self._update_frame()

    def on_unmount(self) -> None:
        self._running = False

    def _update_frame(self) -> None:
        if not self._running:
            return

        now = time.perf_counter()
        if now - self._last_update > 0.1:
            self._frame = (self._frame + 1) % len(self.FRAMES)
            self.update(self.FRAMES[self._frame])
            self._last_update = now

        self.set_timer(0.05, self._update_frame)


class Pulse(Static):
    """Pulsing dot to indicate activity.

    More subtle than a full spinner - good for background tasks.
    """

    DEFAULT_CSS = """
    Pulse {
        width: auto;
        height: 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("●", **kwargs)
        self._phase = 0.0
        self._running = False

    def on_mount(self) -> None:
        self._running = True
        self._pulse()

    def on_unmount(self) -> None:
        self._running = False

    def _pulse(self) -> None:
        if not self._running:
            return

        self._phase += 0.15
        if self._phase > 6.28:
            self._phase = 0.0

        # Opacity cycles from 0.3 to 1.0
        opacity = 0.3 + 0.7 * (0.5 + 0.5 * (self._phase % 3.14159) / 1.5708)
        opacity = min(1.0, max(0.3, opacity))

        self.update(f"●")
        # Note: Textual doesn't support opacity directly in basic Static
        # The animation still provides subtle timing variation

        self.set_timer(0.1, self._pulse)


class ProgressBar(Static):
    """Simple ASCII progress bar.

    Shows progress as a filled bar with percentage.
    """

    DEFAULT_CSS = """
    ProgressBar {
        width: 100%;
        height: 1;
        color: $text-muted;
    }
    """

    def __init__(self, width: int = 20, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._width = width
        self._progress = 0.0

    def set_progress(self, progress: float) -> None:
        """Set progress from 0.0 to 1.0."""
        self._progress = max(0.0, min(1.0, progress))
        self._update_bar()

    def _update_bar(self) -> None:
        filled = int(self._progress * self._width)
        bar = "█" * filled + "░" * (self._width - filled)
        pct = int(self._progress * 100)
        self.update(f"[{bar}] {pct}%")


class TypingIndicator(Static):
    """Typing indicator with animated dots.

    Shows "..." that cycles - used for AI responses.
    """

    DEFAULT_CSS = """
    TypingIndicator {
        width: auto;
        height: 1;
        color: $text-muted;
    }
    """

    FRAMES = ["   ", ".  ", ".. ", "..."]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(self.FRAMES[0], **kwargs)
        self._frame = 0
        self._running = False

    def on_mount(self) -> None:
        self._running = True
        self._cycle()

    def on_unmount(self) -> None:
        self._running = False

    def _cycle(self) -> None:
        if not self._running:
            return

        self._frame = (self._frame + 1) % len(self.FRAMES)
        self.update(self.FRAMES[self._frame])
        self.set_timer(0.3, self._cycle)


class Marquee(Static):
    """Scrolling text marquee.

    For long messages that need to scroll across the screen.
    """

    DEFAULT_CSS = """
    Marquee {
        width: 100%;
        height: 1;
        color: $accent;
    }
    """

    def __init__(self, text: str = "", speed: float = 0.1, **kwargs: Any) -> None:
        super().__init__(text, **kwargs)
        self._text = text
        self._speed = speed
        self._offset = 0
        self._running = False

    def set_text(self, text: str) -> None:
        self._text = text
        self._offset = 0

    def on_mount(self) -> None:
        self._running = True
        self._scroll()

    def on_unmount(self) -> None:
        self._running = False

    def _scroll(self) -> None:
        if not self._running:
            return

        if self._text:
            self._offset = (self._offset + 1) % len(self._text)
            display = self._text[self._offset :] + self._text[: self._offset]
            self.update(display[:20])  # Truncate to width

        self.set_timer(self._speed, self._scroll)
