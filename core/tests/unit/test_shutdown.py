from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from scholardevclaw.utils.shutdown import GracefulShutdown, ShutdownError


@pytest.fixture
def shutdown(monkeypatch):
    monkeypatch.setattr(GracefulShutdown, "_setup_signal_handlers", lambda self: None)
    return GracefulShutdown(timeout_seconds=0.5)


def test_shutdown_runs_handlers_and_sets_state(shutdown):
    called = []
    shutdown.register_handler(lambda: called.append("h1"))

    ok = shutdown.shutdown(reason="test")

    assert ok is True
    assert shutdown.state.shutting_down is True
    assert shutdown.state.reason == "test"
    assert shutdown.state.handlers_called == 1
    assert called == ["h1"]


def test_shutdown_is_idempotent(shutdown):
    assert shutdown.shutdown(reason="first") is True
    assert shutdown.shutdown(reason="second") is False


def test_check_shutdown_raises_after_shutdown(shutdown):
    shutdown.shutdown(reason="maintenance", emit_logs=False)
    with pytest.raises(ShutdownError):
        shutdown.check_shutdown()


def test_atexit_handler_disables_logging_emission(shutdown, monkeypatch):
    calls: list[tuple[str, bool]] = []

    def fake_shutdown(reason="Requested", *, emit_logs=True):
        calls.append((reason, emit_logs))
        shutdown.state.shutting_down = True
        return True

    monkeypatch.setattr(shutdown, "shutdown", fake_shutdown)

    shutdown._atexit_handler()

    assert calls == [("Program exit", False)]


def test_best_effort_log_skips_when_stream_closed(shutdown):
    class ClosedStream:
        closed = True

    handler = logging.StreamHandler()
    handler.stream = ClosedStream()
    shutdown._logger.handlers = [handler]

    # Should not raise even though stream is closed.
    shutdown._log("info", "ignored")


def test_best_effort_log_swallows_logger_errors(shutdown):
    class OpenStream:
        closed = False

    handler = logging.StreamHandler()
    handler.stream = OpenStream()
    shutdown._logger.handlers = [handler]

    class BadLogger:
        handlers = [handler]

        def info(self, *_args, **_kwargs):
            raise ValueError("stream closed")

    shutdown._logger = BadLogger()  # type: ignore[assignment]

    # Should swallow exceptions raised by logger methods.
    shutdown._log("info", "ignored")


def test_shutdown_handler_exception_is_counted_as_non_fatal(shutdown):
    shutdown.register_handler(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    shutdown.register_handler(lambda: None)

    ok = shutdown.shutdown(reason="test", emit_logs=False)

    assert ok is True
    assert shutdown.state.handlers_called == 1


def test_shutdown_timeout_stops_handler_loop(shutdown, monkeypatch):
    called = []

    def slow_handler():
        called.append("slow")

    def fast_handler():
        called.append("fast")

    shutdown.register_handler(slow_handler)
    shutdown.register_handler(fast_handler)

    timeline = iter([0.0, 0.0, 1.0, 1.0])
    monkeypatch.setattr("scholardevclaw.utils.shutdown.time.time", lambda: next(timeline))

    ok = shutdown.shutdown(reason="timeout", emit_logs=False)

    assert ok is True
    assert called == []


def test_signal_handler_calls_shutdown_with_signal_reason(shutdown, monkeypatch):
    captured = SimpleNamespace(reason=None)

    def fake_shutdown(reason="Requested", *, emit_logs=True):
        captured.reason = reason
        shutdown.state.shutting_down = True
        return True

    monkeypatch.setattr(shutdown, "shutdown", fake_shutdown)
    shutdown._signal_handler(2, None)

    assert captured.reason == "Signal SIGINT"
