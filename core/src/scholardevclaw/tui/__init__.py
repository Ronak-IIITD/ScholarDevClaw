"""Textual TUI entrypoints."""


def run_tui():
    from .app import run_tui as _run_tui

    return _run_tui()


__all__ = [
    "run_tui",
    "ClipboardManager",
    "ImageAttachment",
    "ImageInputHandler",
    "copy_to_clipboard",
    "get_clipboard_text",
]


def __getattr__(name):
    if name in (
        "ClipboardManager",
        "ImageAttachment",
        "ImageInputHandler",
        "copy_to_clipboard",
        "get_clipboard_text",
    ):
        from .clipboard import (
            ClipboardManager,  # noqa: F401
            ImageAttachment,  # noqa: F401
            ImageInputHandler,  # noqa: F401
            copy_to_clipboard,  # noqa: F401
            get_clipboard_text,  # noqa: F401
        )

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
