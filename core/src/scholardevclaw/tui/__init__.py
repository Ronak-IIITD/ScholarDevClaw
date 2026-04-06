"""Textual TUI entrypoints."""


def run_tui():
    from .app import run_tui as _run_tui

    return _run_tui()


__all__ = [
    "run_tui",
    "ScholarDevClawApp",
    "ClipboardManager",
    "ImageAttachment",
    "ImageInputHandler",
    "copy_to_clipboard",
    "get_clipboard_text",
    "WelcomeScreen",
    "HelpOverlay",
    "CommandPalette",
    "PhaseTracker",
    "LogView",
    "StatusBar",
    "HistoryPane",
    "AgentStatus",
    # Theme and styling
    "theme",
    "COLORS",
    "ANIMATION",
    # Animated widgets
    "Spinner",
    "Pulse",
    "ProgressBar",
    "TypingIndicator",
    "Marquee",
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

    if name in ("WelcomeScreen", "HelpOverlay", "CommandPalette"):
        from .screens import (
            CommandPalette,  # noqa: F401
            HelpOverlay,  # noqa: F401
            WelcomeScreen,  # noqa: F401
        )

        return locals()[name]

    if name in (
        "PhaseTracker",
        "LogView",
        "StatusBar",
        "HistoryPane",
        "AgentStatus",
    ):
        from .widgets import (
            AgentStatus,  # noqa: F401
            HistoryPane,  # noqa: F401
            LogView,  # noqa: F401
            PhaseTracker,  # noqa: F401
            StatusBar,  # noqa: F401
        )

        return locals()[name]

    if name in ("Spinner", "Pulse", "ProgressBar", "TypingIndicator", "Marquee"):
        from .widgets_animated import (
            Marquee,  # noqa: F401
            ProgressBar,  # noqa: F401
            Pulse,  # noqa: F401
            Spinner,  # noqa: F401
            TypingIndicator,  # noqa: F401
        )

        return locals()[name]

    if name in ("theme", "COLORS", "ANIMATION"):
        from . import theme

        if name == "theme":
            return theme
        if name == "COLORS":
            return theme.COLORS
        if name == "ANIMATION":
            return theme.ANIMATION

    if name == "ScholarDevClawApp":
        from .app import ScholarDevClawApp  # noqa: F401

        return ScholarDevClawApp

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
