"""Patch history browser modal.

Lists every patch that has been recorded via
:func:`ScholarDevClawApp.set_last_patch` during this session, most
recent first. Selecting an entry dismisses the modal with the chosen
patch payload so the caller can push a :class:`DiffViewer` for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, ListItem, ListView, Static

from .diff_viewer import patch_diff_from_payload

HINT_TEXT = (
    "[dim]Enter opens the selected patch in the diff viewer · "
    "Esc closes · ↑/↓ navigate · Type to filter[/dim]"
)


@dataclass
class PatchHistoryEntry:
    """One row in the patch history browser."""

    payload: dict[str, Any]
    timestamp: datetime
    summary: str

    def render(self) -> str:
        ts = self.timestamp.strftime("%H:%M:%S")
        return f"[dim]{ts}[/dim]  {self.summary}"


def _summarize_payload(payload: dict[str, Any]) -> str:
    """Produce a single-line summary of a patch payload."""
    try:
        diff = patch_diff_from_payload(payload)
    except Exception as exc:  # noqa: BLE001
        return f"[red](could not parse: {exc})[/red]"
    if not diff.files:
        return "[dim](empty patch)[/dim]"
    parts: list[str] = []
    for fd in diff.files:
        label = fd.short_label() if hasattr(fd, "short_label") else fd.path
        parts.append(label)
    if len(parts) == 1:
        return parts[0]
    return f"{len(parts)} files: " + ", ".join(parts[:3]) + ("…" if len(parts) > 3 else "")


def build_patch_history(
    payloads: list[dict[str, Any]],
) -> list[PatchHistoryEntry]:
    """Wrap raw payloads in :class:`PatchHistoryEntry` records.

    Each entry gets the current timestamp because the underlying payload
    doesn't always carry one. The list order is preserved (caller is
    expected to pass most-recent first).
    """
    now = datetime.now()
    return [
        PatchHistoryEntry(
            payload=p,
            timestamp=now,
            summary=_summarize_payload(p),
        )
        for p in payloads
    ]


class _PatchListItem(ListItem):
    """A ListItem that carries a :class:`PatchHistoryEntry`."""

    def __init__(self, entry: PatchHistoryEntry) -> None:
        super().__init__(Static(entry.render()))
        self.entry = entry


class PatchHistoryScreen(ModalScreen[dict[str, Any] | None]):
    """Browse every patch recorded this session.

    Dismisses with the selected payload (or ``None`` if cancelled).
    """

    BINDINGS = [
        ("escape", "dismiss_modal", "Close"),
        ("enter", "select_first", "Open"),
    ]

    DEFAULT_CSS = """
    PatchHistoryScreen {
        align: center top;
        background: $background 60%;
        layer: overlay;
    }

    PatchHistoryScreen > Vertical {
        width: 80%;
        height: auto;
        max-height: 90%;
        max-width: 110;
        margin-top: 1;
        padding: 1 2;
        border: tall $accent;
        background: $surface;
    }

    PatchHistoryScreen #patch-history-title {
        width: 100%;
        height: 1;
        margin-bottom: 1;
        color: $accent;
        text-style: bold;
    }

    PatchHistoryScreen Input {
        width: 100%;
        height: 3;
        margin-bottom: 1;
    }

    PatchHistoryScreen #patch-history-list {
        width: 100%;
        height: auto;
        max-height: 18;
        background: $background;
        border: round $border;
    }

    PatchHistoryScreen #patch-history-hint {
        width: 100%;
        height: auto;
        margin-top: 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        payloads: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._all_entries: list[PatchHistoryEntry] = build_patch_history(payloads or [])
        self._query: str = ""

    # ----- compose -----

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Patch history", id="patch-history-title")
            yield Input(placeholder="filter…", id="patch-history-filter")
            yield ListView(id="patch-history-list")
            yield Static(HINT_TEXT, id="patch-history-hint")

    def on_mount(self) -> None:
        self._refresh_list()
        try:
            self.query_one("#patch-history-filter", Input).focus()
        except Exception:
            pass

    # ----- list management -----

    def _refresh_list(self) -> None:
        list_view = self.query_one("#patch-history-list", ListView)
        list_view.clear()
        if not self._all_entries:
            list_view.append(ListItem(Static("[dim]no patches recorded yet[/dim]")))
            return
        q = self._query.lower()
        visible = (
            [e for e in self._all_entries if q in e.summary.lower()]
            if q
            else list(self._all_entries)
        )
        if not visible:
            list_view.append(ListItem(Static("[dim]no matches[/dim]")))
            return
        for entry in visible:
            list_view.append(_PatchListItem(entry))
        # Highlight first match so Enter picks it
        if list_view.children:
            first = list_view.children[0]
            if isinstance(first, ListItem):
                try:
                    list_view.index = list_view.children.index(first)
                except Exception:
                    pass

    # ----- events -----

    @on(Input.Changed, "#patch-history-filter")
    def on_filter_changed(self, event: Input.Changed) -> None:
        self._query = event.value
        self._refresh_list()

    @on(ListView.Selected, "#patch-history-list")
    def on_list_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, _PatchListItem):
            self.dismiss(item.entry.payload)

    # ----- actions -----

    def action_select_first(self) -> None:
        try:
            list_view = self.query_one("#patch-history-list", ListView)
        except Exception:
            return
        if not list_view.children:
            return
        first = list_view.children[0]
        if isinstance(first, _PatchListItem):
            self.dismiss(first.entry.payload)

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)
