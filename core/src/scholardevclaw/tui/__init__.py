"""Textual TUI entrypoints."""


def run_tui(*, yes_mode: bool = False):
    from .app import run_tui as _run_tui

    return _run_tui(yes_mode=yes_mode)


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
    # New conversation widgets
    "ConversationView",
    "MessageBubble",
    "ActionBar",
    "ProgressViz",
    "StreamingIndicator",
    "ContextPanel",
    "WelcomeMessage",
    "ConversationMessage",
    "MessageRole",
    "MessageStatus",
    "make_user_message",
    "make_assistant_message",
    "make_system_message",
    "make_tool_message",
    # Inline workflow widgets
    "WorkflowCard",
    "InlineInput",
    "InlineConfirmBar",
    "InlineProgressCard",
    # Inline diff/patch review widgets
    "InlineDiffCard",
    "InlinePatchReview",
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

    if name in (
        "ConversationView",
        "MessageBubble",
        "ActionBar",
        "ProgressViz",
        "StreamingIndicator",
        "ContextPanel",
        "WelcomeMessage",
        "ConversationMessage",
        "MessageRole",
        "MessageStatus",
        "make_user_message",
        "make_assistant_message",
        "make_system_message",
        "make_tool_message",
        "WorkflowCard",
        "InlineInput",
        "InlineConfirmBar",
        "InlineProgressCard",
        "InlineDiffCard",
        "InlinePatchReview",
    ):
        from .widgets_new import (
            ActionBar,  # noqa: F401
            ContextPanel,  # noqa: F401
            ConversationMessage,  # noqa: F401
            ConversationView,  # noqa: F401
            InlineConfirmBar,  # noqa: F401
            InlineDiffCard,  # noqa: F401
            InlineInput,  # noqa: F401
            InlinePatchReview,  # noqa: F401
            InlineProgressCard,  # noqa: F401
            MessageBubble,  # noqa: F401
            MessageRole,  # noqa: F401
            MessageStatus,  # noqa: F401
            ProgressViz,  # noqa: F401
            StreamingIndicator,  # noqa: F401
            WelcomeMessage,  # noqa: F401
            WorkflowCard,  # noqa: F401
            make_assistant_message,  # noqa: F401
            make_system_message,  # noqa: F401
            make_tool_message,  # noqa: F401
            make_user_message,  # noqa: F401
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
