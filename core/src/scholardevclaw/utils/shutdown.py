from __future__ import annotations

import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable
from contextlib import contextmanager
import atexit

from .errors import setup_logger


@dataclass
class ShutdownState:
    shutting_down: bool = False
    reason: str = ""
    handlers_called: int = 0
    start_time: float = 0.0
    timeout_seconds: float = 30.0


class GracefulShutdown:
    """Handles graceful shutdown with cleanup handlers."""

    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout_seconds = timeout_seconds
        self.state = ShutdownState(timeout_seconds=timeout_seconds)
        self._handlers: list[Callable[[], None]] = []
        self._lock = threading.Lock()
        self._logger = setup_logger("scholardevclaw.shutdown")

        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        atexit.register(self._atexit_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        signal_name = signal.Signals(signum).name
        self._logger.info(f"Received signal {signal_name}")
        self.shutdown(reason=f"Signal {signal_name}")

    def _atexit_handler(self) -> None:
        if not self.state.shutting_down:
            self.shutdown(reason="Program exit")

    def register_handler(self, handler: Callable[[], None]) -> None:
        with self._lock:
            self._handlers.append(handler)

    def shutdown(self, reason: str = "Requested") -> bool:
        with self._lock:
            if self.state.shutting_down:
                return False

            self.state.shutting_down = True
            self.state.reason = reason
            self.state.start_time = time.time()

        self._logger.info(f"Shutting down: {reason}")

        timeout_time = time.time() + self.timeout_seconds

        for handler in self._handlers:
            if time.time() > timeout_time:
                self._logger.error("Shutdown timeout exceeded")
                break

            try:
                handler()
                self.state.handlers_called += 1
            except Exception as e:
                self._logger.error(f"Shutdown handler failed: {e}")

        duration = time.time() - self.state.start_time
        self._logger.info(
            f"Shutdown complete: {self.state.handlers_called} handlers called in {duration:.2f}s"
        )

        return True

    def is_shutting_down(self) -> bool:
        return self.state.shutting_down

    def check_shutdown(self) -> None:
        if self.state.shutting_down:
            raise ShutdownError(f"System is shutting down: {self.state.reason}")


class ShutdownError(Exception):
    """Raised during shutdown."""

    pass


shutdown_manager = GracefulShutdown()


@contextmanager
def shutdown_guard():
    """Context manager that checks for shutdown on entry."""
    shutdown_manager.check_shutdown()
    yield


class ResourceManager:
    """Manages resources with cleanup."""

    def __init__(self):
        self._resources: dict[str, Any] = {}
        self._cleanup_funcs: dict[str, Callable] = {}
        self._lock = threading.Lock()

    def register(self, name: str, resource: Any, cleanup: Callable | None = None) -> None:
        with self._lock:
            self._resources[name] = resource
            if cleanup:
                self._cleanup_funcs[name] = cleanup

    def get(self, name: str) -> Any | None:
        with self._lock:
            return self._resources.get(name)

    def release(self, name: str) -> None:
        with self._lock:
            if name in self._cleanup_funcs:
                try:
                    self._cleanup_funcs[name]()
                except Exception:
                    pass
                del self._cleanup_funcs[name]

            if name in self._resources:
                del self._resources[name]

    def release_all(self) -> None:
        with self._lock:
            for name in list(self._cleanup_funcs.keys()):
                try:
                    self._cleanup_funcs[name]()
                except Exception:
                    pass

            self._cleanup_funcs.clear()
            self._resources.clear()


resource_manager = ResourceManager()


def setup_graceful_shutdown() -> None:
    """Setup graceful shutdown handlers."""

    def cleanup_resources():
        resource_manager.release_all()

    shutdown_manager.register_handler(cleanup_resources)


from typing import Any
