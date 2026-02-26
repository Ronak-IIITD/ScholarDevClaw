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
            ClipboardManager,
            ImageAttachment,
            ImageInputHandler,
            copy_to_clipboard,
            get_clipboard_text,
        )

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
