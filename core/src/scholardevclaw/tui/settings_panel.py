"""Settings panel modal for the ScholarDevClaw TUI.

A keyboard-navigable settings dialog that exposes the most common
runtime configuration knobs grouped by category:

- **LLM**: provider (cycle), model (free text)
- **Behavior**: default mode (cycle), YOLO mode (toggle)
- **Appearance**: Textual theme (cycle)

The panel does not persist anything to disk by itself — it returns the
new settings dict on save and the caller is responsible for applying
changes. This keeps the panel decoupled from the various config
backends (env vars, ``~/.scholardevclaw/``, in-memory state).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

# -----------------------------------------------------------------------------
# Setting model
# -----------------------------------------------------------------------------

SettingKind = Literal["choice", "text", "toggle"]


@dataclass(frozen=True)
class Setting:
    """A single configurable value."""

    key: str
    label: str
    kind: SettingKind
    current: str
    options: tuple[str, ...] = ()
    help_text: str = ""

    def cycle(self) -> str:
        """Return the next value for a choice or toggle setting."""
        if self.kind == "text":
            return self.current
        if self.kind == "toggle":
            return "off" if self.current == "on" else "on"
        if not self.options:
            return self.current
        try:
            idx = self.options.index(self.current)
        except ValueError:
            idx = 0
        return self.options[(idx + 1) % len(self.options)]

    def display(self) -> str:
        if self.kind == "toggle":
            return "●  on" if self.current == "on" else "○  off"
        if self.kind == "choice":
            return f"‹  {self.current}  ›"
        return f"{self.current}"


# -----------------------------------------------------------------------------
# Section header + row widgets
# -----------------------------------------------------------------------------


class _SectionHeader(Static):
    """A non-interactive header for a settings group."""

    DEFAULT_CSS = """
    _SectionHeader {
        height: 1;
        margin-top: 1;
        color: $accent;
        text-style: bold;
    }
    """


class _SettingRow(Static):
    """A single settings row: label on the left, value on the right.

    The row stores its index in the panel so navigation can move
    focus by index without querying the DOM. The :class:`Setting`
    itself is immutable; the row tracks its own ``_current`` value
    so the panel can compute diffs at save time.
    """

    DEFAULT_CSS = """
    _SettingRow {
        height: 1;
        padding: 0 1;
    }

    _SettingRow.-focused {
        background: $surface;
    }
    """

    def __init__(self, setting: Setting, index: int) -> None:
        super().__init__()
        self.setting = setting
        self.index = index
        self._current = setting.current
        self._focused = False
        self._render_row()

    @property
    def current_value(self) -> str:
        return self._current

    @property
    def initial_value(self) -> str:
        return self.setting.current

    def set_focused(self, focused: bool) -> None:
        self._focused = focused
        self.set_class(focused, "-focused")
        self._refresh_display()

    def set_value(self, value: str) -> None:
        self._current = value
        self._refresh_display()

    def _refresh_display(self) -> None:
        marker = "▶" if self._focused else " "
        cursor = " ✎" if self._focused and self.setting.kind == "text" else ""
        help_part = (
            f"  [dim italic]{self.setting.help_text}[/dim italic]" if self.setting.help_text else ""
        )
        # Build a temporary Setting-like object for display()
        temp = Setting(
            key=self.setting.key,
            label=self.setting.label,
            kind=self.setting.kind,
            current=self._current,
            options=self.setting.options,
            help_text=self.setting.help_text,
        )
        self.update(f"{marker} [b]{self.setting.label}[/b]{cursor:<3}  {temp.display()}{help_part}")
        # Build a temporary Setting-like object for display()
        temp = Setting(
            key=self.setting.key,
            label=self.setting.label,
            kind=self.setting.kind,
            current=self._current,
            options=self.setting.options,
            help_text=self.setting.help_text,
        )
        self.update(f"{marker} [b]{self.setting.label}[/b]{cursor:<3}  {temp.display()}{help_part}")
        self._render_row()

    def _render_row(self) -> None:
        marker = "▶" if self._focused else " "
        cursor = " ✎" if self._focused and self.setting.kind == "text" else ""
        help_part = (
            f"  [dim italic]{self.setting.help_text}[/dim italic]" if self.setting.help_text else ""
        )
        self.update(
            f"{marker} [b]{self.setting.label}[/b]{cursor:<3}  {self.setting.display()}{help_part}"
        )


# -----------------------------------------------------------------------------
# Modal screen
# -----------------------------------------------------------------------------


class SettingsPanel(ModalScreen[dict[str, str] | None]):
    """A modal for viewing and changing TUI settings.

    Returns the updated settings dict on save, or ``None`` on cancel.
    The settings dict only contains keys whose values were changed.
    """

    BINDINGS = [
        Binding("escape", "dismiss_panel", "Cancel", show=True),
        Binding("ctrl+c", "dismiss_panel", "Cancel", show=False),
        Binding("ctrl+s", "save_settings", "Save", show=True),
        Binding("up", "focus_prev", "↑", show=False),
        Binding("down", "focus_next", "↓", show=False),
        Binding("enter", "activate_focused", "Edit", show=True),
        Binding("space", "activate_focused", "Edit", show=False),
    ]

    DEFAULT_CSS = """
    SettingsPanel {
        align: center middle;
        background: $background 80%;
    }

    SettingsPanel > Vertical {
        width: 90%;
        max-width: 100;
        height: auto;
        max-height: 90vh;
        background: $surface;
        border: round $border;
        padding: 1 2;
    }

    SettingsPanel #settings-title {
        height: 1;
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    SettingsPanel #settings-help {
        margin-top: 1;
        color: $text-muted;
    }

    SettingsPanel #settings-input {
        display: none;
    }

    SettingsPanel.-editing #settings-input {
        display: block;
    }
    """

    def __init__(self, settings: list[Setting]) -> None:
        super().__init__()
        self._initial_settings = list(settings)
        self._rows: list[_SettingRow] = []
        self._focus_index = 0
        # Build the rows eagerly so introspection (and the test
        # suite) sees the full set without requiring the modal to
        # be mounted.
        for s in self._initial_settings:
            self._rows.append(_SettingRow(s, len(self._rows)))

    # ----- compose -----

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("[b]Settings[/b]  (Ctrl+S to save, Esc to cancel)", id="settings-title")
            yield from self._iter_widgets()
            yield Input(id="settings-input")
            yield Static(
                "↑↓ navigate  ·  Enter edit  ·  Ctrl+S save  ·  Esc cancel",
                id="settings-help",
            )

    def _iter_widgets(self) -> Any:
        """Yield section headers and the (already-built) setting rows.

        Splitting rows from headers in :meth:`__init__` keeps the
        panel navigable by index even when the modal is not yet
        mounted; this method just lays them out in the correct
        visual order.
        """
        sections: dict[str, list[_SettingRow]] = {}
        for row in self._rows:
            section = row.setting.key.split(".")[0] if "." in row.setting.key else "general"
            sections.setdefault(section, []).append(row)

        for section_label, rows in sections.items():
            yield _SectionHeader(f"── {section_label.upper()} ──")
            yield from rows

    def on_mount(self) -> None:
        self._refresh_focus()

    # ----- focus management -----

    def _refresh_focus(self) -> None:
        for i, row in enumerate(self._rows):
            row.set_focused(i == self._focus_index)

    # ----- actions -----

    def action_focus_next(self) -> None:
        if not self._rows:
            return
        self._focus_index = (self._focus_index + 1) % len(self._rows)
        self._refresh_focus()

    def action_focus_prev(self) -> None:
        if not self._rows:
            return
        self._focus_index = (self._focus_index - 1) % len(self._rows)
        self._refresh_focus()

    def action_activate_focused(self) -> None:
        if not self._rows:
            return
        setting = self._rows[self._focus_index].setting
        if setting.kind == "text":
            # Open the input pre-filled with the current value
            self.add_class("-editing")
            inp = self.query_one("#settings-input", Input)
            inp.value = setting.current
            inp.focus()
        else:
            # Cycle choice / toggle
            new_value = setting.cycle()
            self._rows[self._focus_index].set_value(new_value)

    def action_save_settings(self) -> None:
        changed: dict[str, str] = {}
        for row in self._rows:
            if row.current_value != row.initial_value:
                changed[row.setting.key] = row.current_value
        self.dismiss(changed if changed else None)

    def action_dismiss_panel(self) -> None:
        self.dismiss(None)

    # ----- input handling -----

    @on(Input.Submitted, "#settings-input")
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Save text input value and close the editing state."""
        if self._focus_index < len(self._rows):
            self._rows[self._focus_index].set_value(event.value.strip())
        self.remove_class("-editing")
        # Return focus to the row
        self.set_focus(self._rows[self._focus_index] if self._rows else self)

    @on(Input.Blurred, "#settings-input")
    def on_input_blurred(self) -> None:
        self.remove_class("-editing")
