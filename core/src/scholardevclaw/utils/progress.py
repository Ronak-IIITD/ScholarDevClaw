from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence


@dataclass
class ProgressConfig:
    bar_width: int = 40
    fill_char: str = "█"
    empty_char: str = "░"
    show_percent: bool = True
    show_eta: bool = True
    show_rate: bool = True
    show_count: bool = True
    stream: Any = None

    def __post_init__(self):
        if self.stream is None:
            self.stream = sys.stderr


class ProgressBar:
    def __init__(
        self,
        total: int,
        description: str = "",
        config: ProgressConfig | None = None,
    ):
        self.total = total
        self.current = 0
        self.description = description
        self.config = config or ProgressConfig()
        self._start_time: float | None = None
        self._last_update = 0.0

    def start(self) -> "ProgressBar":
        self._start_time = time.time()
        self._update_display()
        return self

    def update(self, n: int = 1) -> None:
        self.current += n
        self._update_display()

    def set(self, n: int) -> None:
        self.current = n
        self._update_display()

    def close(self) -> None:
        if self.config.stream:
            self.config.stream.write("\n")
            self.config.stream.flush()

    def _update_display(self) -> None:
        now = time.time()
        if now - self._last_update < 0.05 and self.current < self.total:
            return
        self._last_update = now

        parts = []

        if self.description:
            parts.append(self.description)

        percent = (self.current / self.total * 100) if self.total > 0 else 0
        filled = int(self.config.bar_width * self.current / self.total) if self.total > 0 else 0
        bar = self.config.fill_char * filled + self.config.empty_char * (
            self.config.bar_width - filled
        )
        parts.append(f"[{bar}]")

        if self.config.show_percent:
            parts.append(f"{percent:5.1f}%")

        if self.config.show_count:
            parts.append(f"({self.current}/{self.total})")

        if self.config.show_eta and self._start_time and self.current > 0:
            elapsed = now - self._start_time
            rate = self.current / elapsed
            remaining = (self.total - self.current) / rate if rate > 0 else 0
            parts.append(f"ETA: {self._format_time(remaining)}")

        if self.config.show_rate and self._start_time and self.current > 0:
            elapsed = now - self._start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            parts.append(f"[{rate:.1f} it/s]")

        line = " ".join(parts)
        if self.config.stream:
            self.config.stream.write(f"\r{line}")
            self.config.stream.flush()

    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def __enter__(self) -> "ProgressBar":
        return self.start()

    def __exit__(self, *args) -> None:
        self.close()


def progress_iter(
    iterable: Sequence[Any] | Iterator[Any],
    description: str = "",
    total: int | None = None,
    config: ProgressConfig | None = None,
) -> Iterator[Any]:
    if total is not None:
        count = total
    elif hasattr(iterable, "__len__"):
        count = len(iterable)  # type: ignore
    else:
        items = list(iterable)
        count = len(items)
        iterable = iter(items)

    with ProgressBar(count, description, config) as bar:
        for item in iterable:
            yield item
            bar.update()


def progress_range(
    stop: int,
    description: str = "",
    config: ProgressConfig | None = None,
) -> range:
    return range(stop)


class Spinner:
    CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, message: str = "", stream: Any = None):
        self.message = message
        self.stream = stream or sys.stderr
        self._index = 0
        self._active = False

    def start(self) -> "Spinner":
        self._active = True
        self._update()
        return self

    def update(self, message: str) -> None:
        self.message = message
        if self._active:
            self._update()

    def stop(self, final_message: str | None = None) -> None:
        self._active = False
        if self.stream:
            if final_message:
                self.stream.write(f"\r{final_message}\n")
            else:
                self.stream.write("\r" + " " * (len(self.message) + 10) + "\r")
            self.stream.flush()

    def _update(self) -> None:
        if self.stream:
            char = self.CHARS[self._index % len(self.CHARS)]
            self.stream.write(f"\r{char} {self.message}")
            self.stream.flush()
            self._index += 1

    def tick(self) -> None:
        if self._active:
            self._update()

    def __enter__(self) -> "Spinner":
        return self.start()

    def __exit__(self, *args) -> None:
        self.stop()


def with_progress(description: str):
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with Spinner(description):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class MultiProgress:
    def __init__(self, config: ProgressConfig | None = None):
        self.config = config or ProgressConfig()
        self._bars: list[ProgressBar] = []

    def add_bar(self, total: int, description: str = "") -> ProgressBar:
        bar = ProgressBar(total, description, self.config)
        self._bars.append(bar)
        return bar

    def update_all(self) -> None:
        if self.config.stream:
            self.config.stream.write(f"\033[{len(self._bars)}F")
            for bar in self._bars:
                bar._update_display()
                self.config.stream.write("\n")
            self.config.stream.flush()

    def close(self) -> None:
        for bar in self._bars:
            bar.close()
