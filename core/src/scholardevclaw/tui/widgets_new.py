"""
New conversation-centric widgets for ScholarDevClaw TUI.

These widgets replace the old log-based interface with a modern,
conversation-first design inspired by Claude Code and OpenCode.

Key widgets:
- ConversationView: Main scrolling conversation area
- MessageBubble: User/assistant messages with markdown rendering
- ActionBar: Inline buttons for diff/patch actions
- ProgressViz: Rich progress visualization
- StreamingIndicator: Animated typing indicator
- ContextPanel: Collapsible side panel for context
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from textual import events
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Markdown, Static

from .theme import COLORS as TUI_COLORS


# -----------------------------------------------------------------------------
# Message Types
# -----------------------------------------------------------------------------


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MessageStatus(str, Enum):
    PENDING = "pending"
    STREAMING = "streaming"
    COMPLETE = "complete"
    ERROR = "error"


# -----------------------------------------------------------------------------
# Message Data
# -----------------------------------------------------------------------------


@dataclass
class ConversationMessage:
    """A single message in the conversation."""

    id: str
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    status: MessageStatus = MessageStatus.COMPLETE
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_user(self) -> bool:
        return self.role == MessageRole.USER

    @property
    def is_assistant(self) -> bool:
        return self.role == MessageRole.ASSISTANT

    @property
    def is_streaming(self) -> bool:
        return self.status == MessageStatus.STREAMING

    @property
    def time_str(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")


# -----------------------------------------------------------------------------
# ConversationView - Main scrolling area
# -----------------------------------------------------------------------------


class ConversationView(VerticalScroll):
    """Main scrolling conversation area.

    Displays messages in a vertical stream with user messages on the right
    and assistant messages on the left (or full-width for simplicity).
    """

    can_focus = True

    class MessageActivated(Message):
        """Fired when a message is clicked/activated."""

        def __init__(self, message_id: str) -> None:
            super().__init__()
            self.message_id = message_id

    DEFAULT_CSS = """
    ConversationView {
        width: 100%;
        height: 1fr;
        padding: 0;
        background: $background;
        overflow-y: auto;
        scrollbar-size: 1 1;
    }

    ConversationView:focus {
        border: none;
    }

    ConversationView .conversation-empty {
        width: 100%;
        height: 100%;
        content-align: center middle;
        color: $text-muted;
        text-style: italic;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._messages: list[ConversationMessage] = []
        self._message_widgets: dict[str, MessageBubble] = {}

    def add_message(self, message: ConversationMessage) -> MessageBubble:
        """Add a message to the conversation."""
        self._messages.append(message)
        bubble = MessageBubble(message)
        self._message_widgets[message.id] = bubble
        self.mount(bubble)
        # Auto-scroll to bottom
        self.scroll_end(animate=False)
        return bubble

    def update_message(self, message_id: str, content: str) -> None:
        """Update an existing message's content (for streaming)."""
        bubble = self._message_widgets.get(message_id)
        if bubble:
            bubble.update_content(content)
            self.scroll_end(animate=False)

    def set_message_status(self, message_id: str, status: MessageStatus) -> None:
        """Update a message's status."""
        bubble = self._message_widgets.get(message_id)
        if bubble:
            bubble.set_status(status)

    def clear_messages(self) -> None:
        """Clear all messages."""
        self._messages.clear()
        self._message_widgets.clear()
        self.remove_children()

    def get_messages(self) -> list[ConversationMessage]:
        """Return all messages."""
        return list(self._messages)

    def scroll_to_message(self, message_id: str) -> None:
        """Scroll to a specific message."""
        bubble = self._message_widgets.get(message_id)
        if bubble:
            bubble.scroll_visible()


# -----------------------------------------------------------------------------
# MessageBubble - Individual message display
# -----------------------------------------------------------------------------


class MessageBubble(Vertical):
    """A single message bubble with role indicator and content.

    Renders markdown for assistant messages, plain text for user messages.
    Shows streaming indicator when status is STREAMING.
    """

    DEFAULT_CSS = """
    MessageBubble {
        width: 100%;
        height: auto;
        min-height: 1;
        padding: 0 2;
        margin: 0 0 1 0;
    }

    MessageBubble.-user {
        background: $surface;
    }

    MessageBubble.-assistant {
        background: $assistant-bg;
    }

    MessageBubble.-system {
        background: $background;
    }

    MessageBubble.-tool {
        background: $code-bg;
        border-left: tall $accent;
    }

    MessageBubble .message-header {
        width: 100%;
        height: 1;
        padding: 0 0 0 0;
        margin: 0 0 0 0;
    }

    MessageBubble.-user .message-header {
        color: $accent;
    }

    MessageBubble.-assistant .message-header {
        color: $success;
    }

    MessageBubble.-system .message-header {
        color: $text-muted;
    }

    MessageBubble.-tool .message-header {
        color: $info;
    }

    MessageBubble .message-content {
        width: 100%;
        height: auto;
        padding: 0;
        margin: 0;
        color: $text;
    }

    MessageBubble .message-actions {
        width: 100%;
        height: auto;
        padding: 0;
        margin: 0;
    }

    MessageBubble .streaming-indicator {
        width: auto;
        height: 1;
        color: $accent;
        margin: 0 0 0 0;
    }
    """

    ROLE_LABELS = {
        MessageRole.USER: "You",
        MessageRole.ASSISTANT: "ScholarDevClaw",
        MessageRole.SYSTEM: "System",
        MessageRole.TOOL: "Tool",
    }

    ROLE_ICONS = {
        MessageRole.USER: "▶",
        MessageRole.ASSISTANT: "◆",
        MessageRole.SYSTEM: "▸",
        MessageRole.TOOL: "⚙",
    }

    def __init__(self, message: ConversationMessage, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._message = message
        self._content_widget: Static | Markdown | None = None

        # Apply role-based class
        role_class = f"-{message.role.value}"
        self.add_class(role_class)

    def compose(self):
        # Header with role and timestamp
        icon = self.ROLE_ICONS.get(self._message.role, "•")
        label = self.ROLE_LABELS.get(self._message.role, "Unknown")
        time_str = self._message.time_str
        status_suffix = ""
        if self._message.status == MessageStatus.STREAMING:
            status_suffix = " ..."
        elif self._message.status == MessageStatus.ERROR:
            status_suffix = " [error]"

        yield Static(
            f"{icon} {label}  {time_str}{status_suffix}",
            classes="message-header",
        )

        # Content - use Markdown for assistant, Static for others
        if self._message.is_assistant:
            try:
                content = Markdown(self._message.content)
            except Exception:
                content = Static(self._message.content)
        else:
            content = Static(self._message.content)

        content.add_class("message-content")
        self._content_widget = content
        yield content

        # Action bar placeholder
        yield Static("", classes="message-actions")

    def update_content(self, content: str) -> None:
        """Update the message content (for streaming)."""
        self._message.content = content
        if self._content_widget:
            if isinstance(self._content_widget, Markdown):
                # Recreate markdown widget with new content
                try:
                    new_content = Markdown(content)
                    new_content.add_class("message-content")
                    self._content_widget.remove()
                    self._content_widget = new_content
                    # Mount after header
                    header = self.query_one(".message-header")
                    self.mount(new_content, after=header)
                except Exception:
                    self._content_widget.update(content)
            else:
                self._content_widget.update(content)

    def set_status(self, status: MessageStatus) -> None:
        """Update the message status."""
        self._message.status = status
        # Refresh header to show status change
        try:
            header = self.query_one(".message-header", Static)
            icon = self.ROLE_ICONS.get(self._message.role, "•")
            label = self.ROLE_LABELS.get(self._message.role, "Unknown")
            time_str = self._message.time_str
            status_suffix = ""
            if status == MessageStatus.STREAMING:
                status_suffix = " ..."
            elif status == MessageStatus.ERROR:
                status_suffix = " [error]"
            header.update(f"{icon} {label}  {time_str}{status_suffix}")
        except Exception:
            pass


# -----------------------------------------------------------------------------
# ActionBar - Inline action buttons
# -----------------------------------------------------------------------------


@dataclass
class ActionButton:
    """An action button in the action bar."""

    key: str
    label: str
    icon: str = ""
    shortcut: str = ""
    variant: str = "default"  # default, primary, success, error
    enabled: bool = True


class ActionBar(Horizontal):
    """Inline action bar with buttons for message actions.

    Appears below messages for diff review, copy, etc.
    """

    class ActionClicked(Message):
        """Fired when an action button is clicked."""

        def __init__(self, action_key: str) -> None:
            super().__init__()
            self.action_key = action_key

    DEFAULT_CSS = """
    ActionBar {
        width: 100%;
        height: auto;
        padding: 0 0 0 2;
        margin: 0;
    }

    ActionBar .action-btn {
        width: auto;
        height: 1;
        padding: 0 1;
        margin: 0 1 0 0;
        color: $text-muted;
        background: transparent;
    }

    ActionBar .action-btn:hover {
        color: $text;
        background: $surface;
    }

    ActionBar .action-btn.-primary {
        color: $accent;
    }

    ActionBar .action-btn.-success {
        color: $success;
    }

    ActionBar .action-btn.-error {
        color: $error;
    }

    ActionBar .action-btn.-disabled {
        color: $border;
    }
    """

    def __init__(self, actions: list[ActionButton] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._actions = actions or []

    def set_actions(self, actions: list[ActionButton]) -> None:
        """Replace all actions."""
        self._actions = actions
        self.remove_children()
        for action in actions:
            self._mount_action(action)

    def _mount_action(self, action: ActionButton) -> None:
        """Mount a single action button."""
        icon_part = f"{action.icon} " if action.icon else ""
        shortcut_part = f" [{action.shortcut}]" if action.shortcut else ""
        label = f"{icon_part}{action.label}{shortcut_part}"

        btn = Static(label, classes="action-btn")
        if action.variant != "default":
            btn.add_class(f"-{action.variant}")
        if not action.enabled:
            btn.add_class("-disabled")

        btn._action_key = action.key  # type: ignore[attr-defined]
        self.mount(btn)

    def on_click(self, event: events.Click) -> None:
        """Handle click on an action button."""
        # Textual Click event uses `control` not `target`
        control = getattr(event, "control", None)
        if control is not None and hasattr(control, "_action_key"):
            self.post_message(self.ActionClicked(control._action_key))  # type: ignore[attr-defined]

    def on_key(self, event: events.Key) -> None:
        """Handle keyboard shortcuts for actions."""
        for action in self._actions:
            if action.shortcut and event.key == action.shortcut.lower():
                self.post_message(self.ActionClicked(action.key))
                event.stop()
                return


# -----------------------------------------------------------------------------
# ProgressViz - Rich progress visualization
# -----------------------------------------------------------------------------


class ProgressViz(Vertical):
    """Rich progress visualization with phases, tokens, and time.

    Shows a multi-step progress bar with per-phase status.
    """

    DEFAULT_CSS = """
    ProgressViz {
        width: 100%;
        height: auto;
        padding: 0 2;
        margin: 0 0 1 0;
        background: $surface;
    }

    ProgressViz .progress-header {
        width: 100%;
        height: 1;
        color: $accent;
        text-style: bold;
        padding: 0;
        margin: 0 0 0 0;
    }

    ProgressViz .progress-bar-line {
        width: 100%;
        height: 1;
        color: $text-muted;
        padding: 0;
        margin: 0;
    }

    ProgressViz .progress-phases {
        width: 100%;
        height: auto;
        padding: 0;
        margin: 0;
    }

    ProgressViz .progress-phase-line {
        width: 100%;
        height: 1;
        color: $text-muted;
        padding: 0;
        margin: 0;
    }

    ProgressViz .progress-phase-line.-complete {
        color: $success;
    }

    ProgressViz .progress-phase-line.-running {
        color: $accent;
    }

    ProgressViz .progress-phase-line.-pending {
        color: $text-muted;
    }

    ProgressViz .progress-phase-line.-error {
        color: $error;
    }

    ProgressViz .progress-stats {
        width: 100%;
        height: 1;
        color: $text-muted;
        padding: 0;
        margin: 0;
    }
    """

    PHASE_ICONS = {
        "complete": "✓",
        "running": "⟳",
        "pending": "⏳",
        "error": "✗",
        "skipped": "⊘",
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._phases: list[dict[str, Any]] = []
        self._overall_progress: float = 0.0
        self._tokens: int = 0
        self._elapsed: float = 0.0
        self._title: str = "Progress"

    def set_title(self, title: str) -> None:
        """Set the progress title."""
        self._title = title
        self._refresh()

    def set_phases(self, phases: list[dict[str, Any]]) -> None:
        """Set the phases list.

        Each phase dict: {"name": str, "status": str, "duration": float|None}
        """
        self._phases = phases
        self._refresh()

    def update_phase(self, name: str, status: str, duration: float | None = None) -> None:
        """Update a single phase's status."""
        for phase in self._phases:
            if phase["name"] == name:
                phase["status"] = status
                if duration is not None:
                    phase["duration"] = duration
                break
        self._refresh()

    def set_progress(self, progress: float) -> None:
        """Set overall progress (0.0 - 1.0)."""
        self._overall_progress = max(0.0, min(1.0, progress))
        self._refresh()

    def set_stats(self, tokens: int = 0, elapsed: float = 0.0) -> None:
        """Set token count and elapsed time."""
        self._tokens = tokens
        self._elapsed = elapsed
        self._refresh()

    def _refresh(self) -> None:
        """Re-render the progress visualization."""
        try:
            # Remove old children
            self.remove_children()

            # Header
            icon = "⟳" if self._overall_progress < 1.0 else "✓"
            self.mount(Static(f"{icon} {self._title}", classes="progress-header"))

            # Progress bar
            width = 30
            filled = int(self._overall_progress * width)
            bar = "█" * filled + "░" * (width - filled)
            pct = int(self._overall_progress * 100)
            self.mount(Static(f"  [{bar}] {pct}%", classes="progress-bar-line"))

            # Phases
            for phase in self._phases:
                name = phase.get("name", "?")
                status = phase.get("status", "pending")
                duration = phase.get("duration")

                icon = self.PHASE_ICONS.get(status, "?")
                duration_str = f"  {duration:.1f}s" if duration else ""
                status_class = f"-{status}" if status in self.PHASE_ICONS else ""

                line = Static(
                    f"  ├─ {icon} {name}{duration_str}",
                    classes=f"progress-phase-line{status_class}",
                )
                self.mount(line)

            # Stats line
            if self._tokens or self._elapsed:
                parts: list[str] = []
                if self._tokens:
                    if self._tokens >= 1000:
                        parts.append(f"{self._tokens / 1000:.1f}k tokens")
                    else:
                        parts.append(f"{self._tokens} tokens")
                if self._elapsed:
                    parts.append(f"{self._elapsed:.1f}s")
                self.mount(Static(f"  {' │ '.join(parts)}", classes="progress-stats"))

        except Exception:
            pass


# -----------------------------------------------------------------------------
# StreamingIndicator - Animated typing indicator
# -----------------------------------------------------------------------------


class StreamingIndicator(Static):
    """Animated typing indicator shown while LLM is streaming."""

    DEFAULT_CSS = """
    StreamingIndicator {
        width: auto;
        height: 1;
        color: $accent;
        padding: 0 0 0 2;
        margin: 0;
    }
    """

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._frame = 0
        self._running = False

    def start(self) -> None:
        """Start the animation."""
        self._running = True
        self._frame = 0
        self._tick()

    def stop(self) -> None:
        """Stop the animation."""
        self._running = False

    def _tick(self) -> None:
        if not self._running:
            return
        self._frame = (self._frame + 1) % len(self.FRAMES)
        try:
            self.update(f"  {self.FRAMES[self._frame]} Thinking...")
        except Exception:
            pass
        self.set_timer(0.1, self._tick)


# -----------------------------------------------------------------------------
# ContextPanel - Collapsible side panel
# -----------------------------------------------------------------------------


class ContextPanel(Vertical):
    """Collapsible side panel for context information.

    Shows git status, file tree, run history, inspector.
    """

    DEFAULT_CSS = """
    ContextPanel {
        width: 40;
        min-width: 30;
        max-width: 60;
        height: 1fr;
        padding: 0 1;
        background: $surface;
        border-left: tall $border;
    }

    ContextPanel .panel-header {
        width: 100%;
        height: 1;
        color: $accent;
        text-style: bold;
        padding: 0;
        margin: 0 0 1 0;
    }

    ContextPanel .panel-section {
        width: 100%;
        height: auto;
        padding: 0;
        margin: 0 0 1 0;
    }

    ContextPanel .panel-section-title {
        width: 100%;
        height: 1;
        color: $text-muted;
        text-style: bold;
        padding: 0;
        margin: 0;
    }

    ContextPanel .panel-section-content {
        width: 100%;
        height: auto;
        color: $text;
        padding: 0;
        margin: 0;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._sections: dict[str, dict[str, str]] = {}

    def set_section(self, title: str, content: str) -> None:
        """Set a section's content."""
        self._sections[title] = {"content": content}
        self._refresh()

    def clear_section(self, title: str) -> None:
        """Clear a section."""
        self._sections.pop(title, None)
        self._refresh()

    def _refresh(self) -> None:
        """Re-render all sections."""
        try:
            self.remove_children()

            if not self._sections:
                self.mount(Static("No context", classes="panel-header"))
                return

            self.mount(Static("Context", classes="panel-header"))

            for title, data in self._sections.items():
                self.mount(Static(title, classes="panel-section-title"))
                self.mount(Static(data["content"], classes="panel-section-content"))

        except Exception:
            pass


# -----------------------------------------------------------------------------
# WelcomeMessage - First-run welcome
# -----------------------------------------------------------------------------


class WelcomeMessage(Static):
    """A welcome message shown when the conversation is empty."""

    DEFAULT_CSS = """
    WelcomeMessage {
        width: 100%;
        height: auto;
        padding: 2 4;
        color: $text-muted;
        content-align: center top;
    }
    """

    WELCOME_TEXT = """[bold cyan]ScholarDevClaw[/bold cyan]

Research → Code assistant

[i]Ask anything about research papers, code, or implementations.[/i]

[dim]Tips:[/dim]
  [dim]• Type naturally — "Add RMSNorm to the transformer"[/dim]
  [dim]• Ctrl+K for command palette[/dim]
  [dim]• Ctrl+P for paper workflow[/dim]
  [dim]• Ctrl+H for help[/dim]"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(self.WELCOME_TEXT, **kwargs)


# -----------------------------------------------------------------------------
# Convenience: build ConversationMessage
# -----------------------------------------------------------------------------


def make_user_message(content: str, **kwargs: Any) -> ConversationMessage:
    """Create a user message."""
    return ConversationMessage(
        id=f"msg-{int(time.time() * 1000)}",
        role=MessageRole.USER,
        content=content,
        **kwargs,
    )


def make_assistant_message(content: str = "", **kwargs: Any) -> ConversationMessage:
    """Create an assistant message (can start empty for streaming)."""
    return ConversationMessage(
        id=f"msg-{int(time.time() * 1000)}",
        role=MessageRole.ASSISTANT,
        content=content,
        **kwargs,
    )


def make_system_message(content: str, **kwargs: Any) -> ConversationMessage:
    """Create a system message."""
    return ConversationMessage(
        id=f"msg-{int(time.time() * 1000)}",
        role=MessageRole.SYSTEM,
        content=content,
        **kwargs,
    )


def make_tool_message(content: str, **kwargs: Any) -> ConversationMessage:
    """Create a tool output message."""
    return ConversationMessage(
        id=f"msg-{int(time.time() * 1000)}",
        role=MessageRole.TOOL,
        content=content,
        **kwargs,
    )
