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
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from textual import events, on
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Input, Markdown, Static

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
        self._render_start_time = 0.0
        self._render_count = 0
        self._last_render_time = 0.0
        self._fps_samples: list[float] = []
        self._memory_samples: list[float] = []
        # Search state
        self._search_query = ""
        self._search_results: list[int] = []  # Indices of matching messages
        self._search_index = -1  # Current result index
        self._search_filters: dict[str, Any] = {
            "role": None,
            "status": None,
            "date_from": None,
            "date_to": None,
            "use_regex": False,
            "case_sensitive": False,
        }

    def add_message(self, message: ConversationMessage) -> MessageBubble:
        """Add a message to the conversation."""

        self._messages.append(message)
        bubble = MessageBubble(message)
        self._message_widgets[message.id] = bubble
        self.mount(bubble)
        # Auto-scroll to bottom
        self.scroll_end(animate=False)
        return bubble

    def _start_render_timing(self) -> None:
        """Start timing a render operation."""
        import time

        self._render_start_time = time.time()

    def _end_render_timing(self) -> None:
        """End timing a render operation and record metrics."""
        import time

        self._render_count += 1
        render_time = time.time() - self._render_start_time
        self._last_render_time = render_time
        self._fps_samples.append(1.0 / max(render_time, 0.001))
        # Keep only last 60 samples for performance monitoring
        if len(self._fps_samples) > 60:
            self._fps_samples.pop(0)

    def get_performance_stats(self) -> dict[str, float]:
        """Get performance statistics for the conversation view."""

        stats = {
            "render_count": self._render_count,
            "last_render_time_ms": self._last_render_time * 1000,
            "average_fps": sum(self._fps_samples) / len(self._fps_samples)
            if self._fps_samples
            else 0,
            "min_fps": min(self._fps_samples) if self._fps_samples else 0,
            "max_fps": max(self._fps_samples) if self._fps_samples else 0,
        }
        return stats

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

    def _get_message_bubble_at(self, x: int, y: int) -> MessageBubble | None:
        """Get the MessageBubble widget at the given coordinates."""
        for bubble in self._message_widgets.values():
            if bubble.region.contains_point((x, y)):
                return bubble
        return None

    def on_mouse_down(self, event: events.MouseDown) -> None:
        """Handle mouse down events for message selection."""
        # Find the message bubble under the mouse
        target = self._get_message_bubble_at(event.x, event.y)
        if target:
            # Select this message
            self._select_message(target._message.id)
            # Start drag selection if shift is held
            if event.shift:
                self._start_drag_selection(target._message.id)

    def on_mouse_up(self, event: events.MouseUp) -> None:
        """Handle mouse up events to end drag selection."""
        if hasattr(self, "_drag_selection_start"):
            delattr(self, "_drag_selection_start")

    def on_mouse_move(self, event: events.MouseMove) -> None:
        """Handle mouse move events for drag selection."""
        if hasattr(self, "_drag_selection_start"):
            # Find the message bubble under the mouse
            target = self._get_message_bubble_at(event.x, event.y)
            if target:
                self._update_drag_selection(target._message.id)

    def _select_message(self, message_id: str) -> None:
        """Select a single message."""
        # Clear previous selection
        for bubble in self._message_widgets.values():
            bubble.remove_class("selected")

        # Select the target message
        bubble = self._message_widgets.get(message_id)
        if bubble:
            bubble.add_class("selected")

    def _start_drag_selection(self, message_id: str) -> None:
        """Start drag selection from a message."""
        self._drag_selection_start = message_id
        self._drag_selection_end = message_id

    def _update_drag_selection(self, message_id: str) -> None:
        """Update drag selection range."""
        if hasattr(self, "_drag_selection_start"):
            self._drag_selection_end = message_id
            # Update selection highlighting
            self._update_selection_range()

    def _update_selection_range(self) -> None:
        """Update selection highlighting for drag selection range."""
        if not hasattr(self, "_drag_selection_start") or not hasattr(self, "_drag_selection_end"):
            return

        start_id = self._drag_selection_start
        end_id = self._drag_selection_end

        # Find indices of start and end messages
        start_idx = None
        end_idx = None
        for i, msg in enumerate(self._messages):
            if msg.id == start_id:
                start_idx = i
            if msg.id == end_id:
                end_idx = i

        if start_idx is None or end_idx is None:
            return

        # Ensure start_idx <= end_idx
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        # Clear all selections first
        for bubble in self._message_widgets.values():
            bubble.remove_class("selected")

        # Select messages in range
        for i in range(start_idx, end_idx + 1):
            msg = self._messages[i]
            bubble = self._message_widgets.get(msg.id)
            if bubble:
                bubble.add_class("selected")

    # -------------------------------------------------------------------------
    # Search & Filtering
    # -------------------------------------------------------------------------

    def search(self, query: str, **filters: Any) -> list[int]:
        """Search messages with optional filters.

        Args:
            query: Search query (text or regex pattern)
            **filters: Optional filters:
                - role: MessageRole to filter by
                - status: MessageStatus to filter by
                - date_from: datetime to filter from
                - date_to: datetime to filter to
                - use_regex: Whether to treat query as regex
                - case_sensitive: Whether search is case sensitive

        Returns:
            List of message indices matching the search
        """
        import re

        self._search_query = query
        self._search_filters.update(filters)
        self._search_results = []
        self._search_index = -1

        if not query and not any(filters.values()):
            return []

        use_regex = filters.get("use_regex", self._search_filters.get("use_regex", False))
        case_sensitive = filters.get(
            "case_sensitive", self._search_filters.get("case_sensitive", False)
        )
        role_filter = filters.get("role", self._search_filters.get("role"))
        status_filter = filters.get("status", self._search_filters.get("status"))
        date_from = filters.get("date_from", self._search_filters.get("date_from"))
        date_to = filters.get("date_to", self._search_filters.get("date_to"))

        flags = 0 if case_sensitive else re.IGNORECASE

        try:
            pattern = re.compile(query, flags) if use_regex and query else None
        except re.error:
            pattern = None

        for i, msg in enumerate(self._messages):
            # Apply role filter
            if role_filter and msg.role != role_filter:
                continue

            # Apply status filter
            if status_filter and msg.status != status_filter:
                continue

            # Apply date filters
            msg_time = msg.timestamp
            if date_from and msg_time < date_from:
                continue
            if date_to and msg_time > date_to:
                continue

            # Apply text search
            if query:
                content = msg.content
                if pattern:
                    if not pattern.search(content):
                        continue
                elif query.lower() not in content.lower():
                    continue

            self._search_results.append(i)

        if self._search_results:
            self._search_index = 0
            self._highlight_current_result()

        return self._search_results

    def _highlight_current_result(self) -> None:
        """Highlight the current search result."""
        # Clear previous highlights
        for bubble in self._message_widgets.values():
            bubble.remove_class("search-match")
            bubble.remove_class("search-current")

        if 0 <= self._search_index < len(self._search_results):
            msg_idx = self._search_results[self._search_index]
            msg = self._messages[msg_idx]
            bubble = self._message_widgets.get(msg.id)
            if bubble:
                bubble.add_class("search-current")
                bubble.scroll_visible()

        # Highlight all matches
        for idx in self._search_results:
            msg = self._messages[idx]
            bubble = self._message_widgets.get(msg.id)
            if bubble:
                bubble.add_class("search-match")

    def next_search_result(self) -> bool:
        """Navigate to next search result. Returns True if moved."""
        if not self._search_results:
            return False
        self._search_index = (self._search_index + 1) % len(self._search_results)
        self._highlight_current_result()
        return True

    def prev_search_result(self) -> bool:
        """Navigate to previous search result. Returns True if moved."""
        if not self._search_results:
            return False
        self._search_index = (self._search_index - 1) % len(self._search_results)
        self._highlight_current_result()
        return True

    def clear_search(self) -> None:
        """Clear search results and highlights."""
        self._search_query = ""
        self._search_results = []
        self._search_index = -1
        for bubble in self._message_widgets.values():
            bubble.remove_class("search-match")
            bubble.remove_class("search-current")

    def get_search_info(self) -> dict[str, Any]:
        """Get current search state info."""
        return {
            "query": self._search_query,
            "total_matches": len(self._search_results),
            "current_index": self._search_index + 1 if self._search_index >= 0 else 0,
            "filters": self._search_filters.copy(),
        }


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

    MessageBubble.selected {
        border: tall $accent;
        background: $surface;
    }

    MessageBubble.search-match {
        border: tall $warning;
        background: $warning-bg;
    }

    MessageBubble.search-current {
        border: tall $accent;
        background: $accent-bg;
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
        self._render_start_time = 0.0
        self._render_count = 0
        self._last_render_time = 0.0

        # Apply role-based class
        role_class = f"-{message.role.value}"
        self.add_class(role_class)

    def compose(self):
        import time

        self._render_start_time = time.time()

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

    def _end_render_timing(self) -> None:
        """End timing a render operation and record metrics."""
        import time

        self._render_count += 1
        render_time = time.time() - self._render_start_time
        self._last_render_time = render_time

    def get_performance_stats(self) -> dict[str, float]:
        """Get performance statistics for the message bubble."""
        stats = {
            "render_count": self._render_count,
            "last_render_time_ms": self._last_render_time * 1000,
        }
        return stats

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
# WorkflowCard - Rich workflow stage display
# -----------------------------------------------------------------------------


class WorkflowCard(Vertical):
    """A rich card for displaying workflow stages inline in the conversation.

    Shows structured data like paper metadata, understanding, plan, etc.
    with a header, body, and optional action bar.
    """

    DEFAULT_CSS = """
    WorkflowCard {
        width: 100%;
        height: auto;
        min-height: 1;
        padding: 0 2;
        margin: 0 0 1 0;
        background: $surface;
        border-left: tall $accent;
    }

    WorkflowCard .card-header {
        width: 100%;
        height: 1;
        color: $accent;
        text-style: bold;
        padding: 0;
        margin: 0 0 0 0;
    }

    WorkflowCard .card-body {
        width: 100%;
        height: auto;
        padding: 0;
        margin: 0;
        color: $text;
    }

    WorkflowCard .card-body-row {
        width: 100%;
        height: 1;
        padding: 0;
        margin: 0;
    }

    WorkflowCard .card-body-row.-label {
        color: $text-muted;
    }

    WorkflowCard .card-body-row.-value {
        color: $text;
    }

    WorkflowCard .card-body-row.-success {
        color: $success;
    }

    WorkflowCard .card-body-row.-warning {
        color: $warning;
    }

    WorkflowCard .card-body-row.-error {
        color: $error;
    }

    WorkflowCard .card-body-row.-accent {
        color: $accent;
    }

    WorkflowCard .card-progress {
        width: 100%;
        height: 1;
        color: $text-muted;
        padding: 0;
        margin: 0;
    }

    WorkflowCard .card-actions {
        width: 100%;
        height: auto;
        padding: 0;
        margin: 0;
    }
    """

    def __init__(self, title: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._body_rows: list[tuple[str, str]] = []  # (text, style_class)
        self._progress: float | None = None
        self._progress_text: str = ""

    def set_title(self, title: str) -> None:
        """Set the card title."""
        self._title = title
        self._refresh()

    def add_row(self, text: str, style: str = "value") -> None:
        """Add a body row with style: value, label, success, warning, error, accent."""
        self._body_rows.append((text, f"-{style}"))
        self._refresh()

    def clear_rows(self) -> None:
        """Clear all body rows."""
        self._body_rows.clear()
        self._refresh()

    def set_rows(self, rows: list[tuple[str, str]]) -> None:
        """Replace all body rows."""
        self._body_rows = [(text, f"-{style}") for text, style in rows]
        self._refresh()

    def set_progress(self, progress: float, text: str = "") -> None:
        """Set progress bar (0.0-1.0) with optional text."""
        self._progress = max(0.0, min(1.0, progress))
        self._progress_text = text
        self._refresh()

    def clear_progress(self) -> None:
        """Remove progress bar."""
        self._progress = None
        self._progress_text = ""
        self._refresh()

    def _refresh(self) -> None:
        """Re-render the card."""
        try:
            self.remove_children()

            # Header
            if self._title:
                self.mount(Static(self._title, classes="card-header"))

            # Body rows
            for text, style_class in self._body_rows:
                self.mount(Static(text, classes=f"card-body-row {style_class}"))

            # Progress bar
            if self._progress is not None:
                width = 20
                filled = int(self._progress * width)
                bar = "█" * filled + "░" * (width - filled)
                pct = int(self._progress * 100)
                progress_str = f"  [{bar}] {pct}%"
                if self._progress_text:
                    progress_str += f" {self._progress_text}"
                self.mount(Static(progress_str, classes="card-progress"))

        except Exception:
            pass


# -----------------------------------------------------------------------------
# InlineInput - Input field embedded in conversation
# -----------------------------------------------------------------------------


class InlineInput(Vertical):
    """An input field that appears inline in the conversation flow.

    Used for asking the user questions without opening a modal.
    """

    class Submitted(Message):
        """Fired when the user submits the input."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    class Cancelled(Message):
        """Fired when the user cancels the input."""

        pass

    DEFAULT_CSS = """
    InlineInput {
        width: 100%;
        height: auto;
        min-height: 3;
        padding: 0 2;
        margin: 0 0 1 0;
        background: $surface;
        border-left: tall $accent;
    }

    InlineInput .inline-label {
        width: 100%;
        height: 1;
        color: $accent;
        text-style: bold;
        padding: 0;
        margin: 0 0 0 0;
    }

    InlineInput Input {
        width: 100%;
        height: 1;
        border: solid $border;
        padding: 0 1;
        margin: 0;
    }

    InlineInput Input:focus {
        border: solid $accent;
    }

    InlineInput .inline-hint {
        width: 100%;
        height: 1;
        color: $text-muted;
        padding: 0;
        margin: 0;
    }
    """

    def __init__(
        self,
        label: str = "",
        placeholder: str = "",
        hint: str = "",
        value: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._placeholder = placeholder
        self._hint = hint
        self._initial_value = value

    def compose(self):
        if self._label:
            yield Static(self._label, classes="inline-label")
        yield Input(
            value=self._initial_value,
            placeholder=self._placeholder,
            id="inline-input-field",
        )
        if self._hint:
            yield Static(self._hint, classes="inline-hint")

    def on_mount(self) -> None:
        try:
            self.query_one("#inline-input-field", Input).focus()
        except Exception:
            pass

    @on(Input.Submitted, "#inline-input-field")
    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.post_message(self.Submitted(event.value))

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.post_message(self.Cancelled())
            event.stop()


# -----------------------------------------------------------------------------
# InlineConfirmBar - Approval/confirm buttons inline
# -----------------------------------------------------------------------------


class InlineConfirmBar(Horizontal):
    """Inline confirmation bar with Approve/Reject buttons.

    Appears below messages for approval gates.
    """

    class Confirmed(Message):
        """Fired when the user confirms."""

        def __init__(self, decision: str = "approve") -> None:
            super().__init__()
            self.decision = decision

    DEFAULT_CSS = """
    InlineConfirmBar {
        width: 100%;
        height: auto;
        min-height: 1;
        padding: 0 2;
        margin: 0 0 1 0;
    }

    InlineConfirmBar .confirm-label {
        width: auto;
        height: 1;
        color: $warning;
        text-style: bold;
        padding: 0 1 0 0;
        margin: 0;
    }

    InlineConfirmBar .confirm-btn {
        width: auto;
        height: 1;
        padding: 0 1;
        margin: 0 1 0 0;
        color: $text-muted;
        background: transparent;
    }

    InlineConfirmBar .confirm-btn:hover {
        color: $text;
        background: $surface;
    }

    InlineConfirmBar .confirm-btn.-approve {
        color: $success;
    }

    InlineConfirmBar .confirm-btn.-reject {
        color: $error;
    }
    """

    def __init__(self, label: str = "Approval gate:", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._label = label

    def compose(self):
        yield Static(self._label, classes="confirm-label")
        yield Static("✓ Approve [y]", classes="confirm-btn -approve")
        yield Static("✗ Reject [n]", classes="confirm-btn -reject")

    def on_click(self, event: events.Click) -> None:
        control = getattr(event, "control", None)
        if control is None:
            return
        text = str(getattr(control, "_text", "") or "")
        if "Approve" in text:
            self.post_message(self.Confirmed("approve"))
        elif "Reject" in text:
            self.post_message(self.Confirmed("reject"))

    def on_key(self, event: events.Key) -> None:
        if event.key == "y":
            self.post_message(self.Confirmed("approve"))
            event.stop()
        elif event.key == "n":
            self.post_message(self.Confirmed("reject"))
            event.stop()


# -----------------------------------------------------------------------------
# InlineProgressCard - Progress visualization inline
# -----------------------------------------------------------------------------


class InlineProgressCard(WorkflowCard):
    """A progress card specifically for showing workflow progress.

    Extends WorkflowCard with phase tracking and token/elapsed display.
    """

    PHASE_ICONS = {
        "complete": "✓",
        "running": "⟳",
        "pending": "⏳",
        "error": "✗",
        "skipped": "⊘",
    }

    def __init__(self, title: str = "Progress", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)
        self._phases: list[dict[str, Any]] = []
        self._tokens: int = 0
        self._elapsed: float = 0.0

    def set_phases(self, phases: list[dict[str, Any]]) -> None:
        """Set phases list. Each: {"name": str, "status": str, "duration": float|None}"""
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

    def set_stats(self, tokens: int = 0, elapsed: float = 0.0) -> None:
        """Set token count and elapsed time."""
        self._tokens = tokens
        self._elapsed = elapsed
        self._refresh()

    def _refresh(self) -> None:
        """Re-render with phases and stats."""
        try:
            self.remove_children()

            # Header
            icon = "⟳" if (self._progress or 0) < 1.0 else "✓"
            if self._title:
                self.mount(Static(f"{icon} {self._title}", classes="card-header"))

            # Progress bar
            if self._progress is not None:
                width = 20
                filled = int(self._progress * width)
                bar = "█" * filled + "░" * (width - filled)
                pct = int(self._progress * 100)
                self.mount(Static(f"  [{bar}] {pct}%", classes="card-progress"))

            # Phases
            for phase in self._phases:
                name = phase.get("name", "?")
                status = phase.get("status", "pending")
                duration = phase.get("duration")
                p_icon = self.PHASE_ICONS.get(status, "?")
                duration_str = f"  {duration:.1f}s" if duration else ""
                style = f"-{status}" if status in self.PHASE_ICONS else ""
                self.mount(
                    Static(
                        f"  ├─ {p_icon} {name}{duration_str}",
                        classes=f"card-body-row {style}",
                    )
                )

            # Stats
            if self._tokens or self._elapsed:
                parts: list[str] = []
                if self._tokens:
                    if self._tokens >= 1000:
                        parts.append(f"{self._tokens / 1000:.1f}k tokens")
                    else:
                        parts.append(f"{self._tokens} tokens")
                if self._elapsed:
                    parts.append(f"{self._elapsed:.1f}s")
                self.mount(Static(f"  {' │ '.join(parts)}", classes="card-progress"))

        except Exception:
            pass


# -----------------------------------------------------------------------------
# InlineDiffCard - File diff displayed inline
# -----------------------------------------------------------------------------


class InlineDiffCard(Vertical):
    """A card that shows a file's diff inline with color-coded lines.

    Used for inline patch review in the conversation flow.
    """

    DEFAULT_CSS = """
    InlineDiffCard {
        width: 100%;
        height: auto;
        min-height: 1;
        padding: 0 2;
        margin: 0 0 1 0;
        background: $background;
        border-left: tall $accent;
    }

    InlineDiffCard .diff-file-header {
        width: 100%;
        height: 1;
        color: $accent;
        text-style: bold;
        padding: 0;
        margin: 0 0 0 0;
    }

    InlineDiffCard .diff-file-header.-added {
        color: $success;
    }

    InlineDiffCard .diff-file-header.-deleted {
        color: $error;
    }

    InlineDiffCard .diff-file-header.-modified {
        color: $accent;
    }

    InlineDiffCard .diff-stats {
        width: 100%;
        height: 1;
        color: $text-muted;
        padding: 0;
        margin: 0 0 0 0;
    }

    InlineDiffCard .diff-content {
        width: 100%;
        height: auto;
        padding: 0;
        margin: 0;
        color: $text;
    }

    InlineDiffCard .diff-line {
        width: 100%;
        height: 1;
        padding: 0;
        margin: 0;
    }

    InlineDiffCard .diff-line.-addition {
        color: $success;
    }

    InlineDiffCard .diff-line.-deletion {
        color: $error;
    }

    InlineDiffCard .diff-line.-header {
        color: $info;
    }

    InlineDiffCard .diff-line.-hunk {
        color: $text-muted;
    }

    InlineDiffCard .diff-actions {
        width: 100%;
        height: auto;
        padding: 0;
        margin: 0;
    }
    """

    STATUS_ICONS = {
        "added": "+",
        "modified": "~",
        "deleted": "-",
        "renamed": "→",
    }

    def __init__(self, file_path: str = "", status: str = "modified", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._file_path = file_path
        self._status = status
        self._additions = 0
        self._deletions = 0
        self._diff_lines: list[tuple[str, str]] = []  # (line, style_class)

    def set_file_info(self, path: str, status: str, additions: int = 0, deletions: int = 0) -> None:
        """Set file metadata."""
        self._file_path = path
        self._status = status
        self._additions = additions
        self._deletions = deletions
        self._refresh()

    def set_diff_lines(self, lines: list[tuple[str, str]]) -> None:
        """Set diff lines. Each tuple: (line_text, style_class).

        Style classes: addition, deletion, header, hunk, context
        """
        self._diff_lines = lines
        self._refresh()

    def add_diff_line(self, line: str, style: str = "context") -> None:
        """Add a single diff line."""
        self._diff_lines.append((line, style))
        self._refresh()

    def clear_diff_lines(self) -> None:
        """Clear all diff lines."""
        self._diff_lines.clear()
        self._refresh()

    def _refresh(self) -> None:
        """Re-render the diff card."""
        try:
            self.remove_children()

            # File header
            icon = self.STATUS_ICONS.get(self._status, "?")
            header_class = f"diff-file-header -{self._status}"
            self.mount(
                Static(
                    f"{icon} {self._file_path}",
                    classes=header_class,
                )
            )

            # Stats line
            self.mount(
                Static(
                    f"+{self._additions} -{self._deletions}",
                    classes="diff-stats",
                )
            )

            # Diff content
            for line_text, style in self._diff_lines:
                self.mount(Static(line_text, classes=f"diff-line -{style}"))

        except Exception:
            pass


# -----------------------------------------------------------------------------
# InlinePatchReview - Complete patch review with navigation
# -----------------------------------------------------------------------------


class InlinePatchReview(Vertical):
    """Complete patch review widget with file tabs and navigation.

    Shows a summary, file tabs, diff content, and action buttons.
    """

    class FileAction(Message):
        """Fired when a file-level action is taken."""

        def __init__(self, file_path: str, action: str) -> None:
            super().__init__()
            self.file_path = file_path
            self.action = action  # "accept", "reject", "regenerate"

    class AllFilesAction(Message):
        """Fired when an all-files action is taken."""

        def __init__(self, action: str) -> None:
            super().__init__()
            self.action = action  # "accept_all", "reject_all"

    DEFAULT_CSS = """
    InlinePatchReview {
        width: 100%;
        height: auto;
        min-height: 5;
        padding: 0 2;
        margin: 0 0 1 0;
        background: $surface;
        border-left: tall $accent;
    }

    InlinePatchReview .patch-header {
        width: 100%;
        height: 1;
        color: $accent;
        text-style: bold;
        padding: 0;
        margin: 0 0 0 0;
    }

    InlinePatchReview .patch-summary {
        width: 100%;
        height: 1;
        color: $text-muted;
        padding: 0;
        margin: 0 0 0 0;
    }

    InlinePatchReview .patch-tabs {
        width: 100%;
        height: 1;
        color: $text-muted;
        padding: 0;
        margin: 0 0 0 0;
    }

    InlinePatchReview .patch-tabs .tab-active {
        color: $accent;
        text-style: bold;
    }

    InlinePatchReview .patch-diff {
        width: 100%;
        height: auto;
        max-height: 30;
        padding: 0;
        margin: 0 0 0 0;
        background: $background;
        border: solid $border;
        overflow-y: auto;
    }

    InlinePatchReview .patch-diff-line {
        width: 100%;
        height: 1;
        padding: 0 1;
        margin: 0;
    }

    InlinePatchReview .patch-diff-line.-addition {
        color: $success;
    }

    InlinePatchReview .patch-diff-line.-deletion {
        color: $error;
    }

    InlinePatchReview .patch-diff-line.-header {
        color: $info;
    }

    InlinePatchReview .patch-diff-line.-hunk {
        color: $text-muted;
    }

    InlinePatchReview .patch-actions {
        width: 100%;
        height: auto;
        padding: 0;
        margin: 0;
    }

    InlinePatchReview .patch-action-btn {
        width: auto;
        height: 1;
        padding: 0 1;
        margin: 0 1 0 0;
        color: $text-muted;
        background: transparent;
    }

    InlinePatchReview .patch-action-btn:hover {
        color: $text;
        background: $surface;
    }

    InlinePatchReview .patch-action-btn.-accept {
        color: $success;
    }

    InlinePatchReview .patch-action-btn.-reject {
        color: $error;
    }

    InlinePatchReview .patch-action-btn.-regenerate {
        color: $warning;
    }

    InlinePatchReview .patch-action-btn.-accept-all {
        color: $success;
        text-style: bold;
    }

    InlinePatchReview .patch-action-btn.-reject-all {
        color: $error;
        text-style: bold;
    }
    """

    def __init__(self, title: str = "Patch Review", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._files: list[
            dict[str, Any]
        ] = []  # Each: {path, status, additions, deletions, diff_lines}
        self._current_file_index = 0
        self._file_decisions: dict[str, str] = {}  # path -> "accept"/"reject"/"regenerate"

    def set_files(self, files: list[dict[str, Any]]) -> None:
        """Set the files list.

        Each file dict: {path, status, additions, deletions, diff_lines}
        diff_lines: list of (line_text, style_class) tuples
        """
        self._files = files
        self._current_file_index = 0
        self._refresh()

    def add_file(
        self,
        path: str,
        status: str,
        additions: int = 0,
        deletions: int = 0,
        diff_lines: list[tuple[str, str]] | None = None,
    ) -> None:
        """Add a file to the review."""
        self._files.append(
            {
                "path": path,
                "status": status,
                "additions": additions,
                "deletions": deletions,
                "diff_lines": diff_lines or [],
            }
        )
        self._refresh()

    def set_file_decision(self, path: str, decision: str) -> None:
        """Record a decision for a file."""
        self._file_decisions[path] = decision
        self._refresh()

    def next_file(self) -> None:
        """Navigate to the next file."""
        if self._files:
            self._current_file_index = (self._current_file_index + 1) % len(self._files)
            self._refresh()

    def prev_file(self) -> None:
        """Navigate to the previous file."""
        if self._files:
            self._current_file_index = (self._current_file_index - 1) % len(self._files)
            self._refresh()

    def _get_total_stats(self) -> tuple[int, int]:
        """Get total additions and deletions."""
        total_adds = sum(f.get("additions", 0) for f in self._files)
        total_dels = sum(f.get("deletions", 0) for f in self._files)
        return total_adds, total_dels

    def _refresh(self) -> None:
        """Re-render the patch review."""
        try:
            self.remove_children()

            # Header
            self.mount(Static(self._title, classes="patch-header"))

            # Summary
            total_adds, total_dels = self._get_total_stats()
            file_count = len(self._files)
            summary = (
                f"{file_count} file{'s' if file_count != 1 else ''}: +{total_adds} -{total_dels}"
            )
            self.mount(Static(summary, classes="patch-summary"))

            # File tabs
            if self._files:
                tab_parts: list[str] = []
                for i, f in enumerate(self._files):
                    path = f.get("path", "?")
                    # Shorten path for display
                    short = path.split("/")[-1] if "/" in path else path
                    decision = self._file_decisions.get(path, "")
                    suffix = f" [{decision}]" if decision else ""
                    if i == self._current_file_index:
                        tab_parts.append(f"[{short}{suffix}]")
                    else:
                        tab_parts.append(f"{short}{suffix}")
                self.mount(Static("  ".join(tab_parts), classes="patch-tabs"))

                # Current file diff
                current = self._files[self._current_file_index]
                diff_lines = current.get("diff_lines", [])
                if diff_lines:
                    for line_text, style in diff_lines[:50]:  # Cap at 50 lines
                        self.mount(Static(line_text, classes=f"patch-diff-line -{style}"))
                    if len(diff_lines) > 50:
                        self.mount(
                            Static(
                                f"  ... {len(diff_lines) - 50} more lines",
                                classes="patch-diff-line -hunk",
                            )
                        )
                else:
                    self.mount(Static("  (no diff available)", classes="patch-diff-line -hunk"))

            # Action buttons
            self.mount(Static("", classes="patch-actions"))
            current_path = self._files[self._current_file_index]["path"] if self._files else ""
            decision = self._file_decisions.get(current_path, "")

            # Per-file actions
            accept_label = "✓ Accept" if decision != "accept" else "✓ Accepted"
            reject_label = "✗ Reject" if decision != "reject" else "✗ Rejected"
            regen_label = "↻ Regenerate" if decision != "regenerate" else "↻ Regenerating"

            self.mount(Static(accept_label, classes="patch-action-btn -accept"))
            self.mount(Static(reject_label, classes="patch-action-btn -reject"))
            self.mount(Static(regen_label, classes="patch-action-btn -regenerate"))

            # All-files actions
            self.mount(Static("", classes="patch-actions"))
            self.mount(Static("✓ Accept All [A]", classes="patch-action-btn -accept-all"))
            self.mount(Static("✗ Reject All [X]", classes="patch-action-btn -reject-all"))

        except Exception:
            pass

    def on_click(self, event: events.Click) -> None:
        """Handle click on action buttons."""
        control = getattr(event, "control", None)
        if control is None:
            return
        text = str(getattr(control, "_text", "") or "")
        current_path = self._files[self._current_file_index]["path"] if self._files else ""

        if "Accept All" in text:
            self.post_message(self.AllFilesAction("accept_all"))
        elif "Reject All" in text:
            self.post_message(self.AllFilesAction("reject_all"))
        elif "Accept" in text and "All" not in text:
            self.post_message(self.FileAction(current_path, "accept"))
        elif "Reject" in text and "All" not in text:
            self.post_message(self.FileAction(current_path, "reject"))
        elif "Regenerate" in text:
            self.post_message(self.FileAction(current_path, "regenerate"))

    def on_key(self, event: events.Key) -> None:
        """Handle keyboard shortcuts."""
        if event.key == "right" or event.key == "tab":
            self.next_file()
            event.stop()
        elif event.key == "left" or event.key == "shift+tab":
            self.prev_file()
            event.stop()
        elif event.key == "a":
            # Accept current file
            current_path = self._files[self._current_file_index]["path"] if self._files else ""
            if current_path:
                self.post_message(self.FileAction(current_path, "accept"))
            event.stop()
        elif event.key == "x":
            # Reject current file
            current_path = self._files[self._current_file_index]["path"] if self._files else ""
            if current_path:
                self.post_message(self.FileAction(current_path, "reject"))
            event.stop()
        elif event.key == "g":
            # Regenerate current file
            current_path = self._files[self._current_file_index]["path"] if self._files else ""
            if current_path:
                self.post_message(self.FileAction(current_path, "regenerate"))
            event.stop()
        elif event.key == "A":
            # Accept all
            self.post_message(self.AllFilesAction("accept_all"))
            event.stop()
        elif event.key == "X":
            # Reject all
            self.post_message(self.AllFilesAction("reject_all"))
            event.stop()


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


# -----------------------------------------------------------------------------
# Error Handling Widgets
# -----------------------------------------------------------------------------


class ErrorBanner(Static):
    """A banner widget for displaying error messages with recovery options."""

    DEFAULT_CSS = """
    ErrorBanner {
        width: 100%;
        height: auto;
        padding: 1;
        margin: 0 0 1 0;
        background: $error-bg;
        color: $error-fg;
        border: tall $error;
        text-align: center;
    }

    ErrorBanner:hover {
        background: $error-bg-hover;
    }
    """

    def __init__(
        self, message: str, action: Callable[[], None] | None = None, **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self._action = action
        self._dismissed = False

    def on_click(self, event: events.Click) -> None:
        """Handle click events on the error banner."""
        if self._action and not self._dismissed:
            self._action()
            self._dismissed = True
            self.remove()

    def dismiss(self) -> None:
        """Dismiss the error banner."""
        self.remove()
