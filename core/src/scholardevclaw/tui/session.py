"""Session persistence for the TUI.

Saves and restores window state, command history, and user preferences
across sessions using a JSON file in the user's config directory.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class SessionState:
    """Persisted session state."""

    # Window/layout state
    theme: str = "default"
    directory: str = "."
    mode: str = "analyze"
    provider: str = "setup"
    model: str = "auto"

    # History
    command_history: list[str] = field(default_factory=list)
    recent_repos: list[str] = field(default_factory=list)

    # UI state
    log_filter_severity: str = "all"
    log_filter_search: str = ""
    inspector_visible: bool = True
    history_pane_visible: bool = True

    # Metadata
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1


DEFAULT_SESSION_PATH = Path.home() / ".config" / "scholardevclaw" / "session.json"


def get_session_path() -> Path:
    """Get the session file path, respecting XDG_CONFIG_HOME."""
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home) / "scholardevclaw" / "session.json"
    return DEFAULT_SESSION_PATH


def load_session() -> SessionState:
    """Load session state from disk.

    Returns a default SessionState if file doesn't exist or is invalid.
    """
    path = get_session_path()
    if not path.exists():
        return SessionState()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Filter out unknown fields for forward compatibility
        known_fields = {f.name for f in SessionState.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return SessionState(**filtered)
    except Exception:
        return SessionState()


def save_session(state: SessionState) -> None:
    """Save session state to disk atomically."""
    path = get_session_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Update timestamp
    state.last_updated = datetime.now().isoformat()

    # Write to temp file then rename for atomicity
    temp_path = path.with_suffix(".tmp")
    try:
        temp_path.write_text(
            json.dumps(asdict(state), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temp_path.replace(path)
    except Exception:
        # Best effort - don't crash the app on save failure
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


def update_session_from_app(app: Any) -> SessionState:
    """Extract current state from the app and return a SessionState."""
    state = load_session()

    # Update from app attributes (guarded with hasattr/getattr)
    state.theme = getattr(app, "theme", state.theme) or "default"
    state.directory = getattr(app, "_directory", state.directory) or "."
    state.mode = getattr(app, "_mode", state.mode) or "analyze"
    state.provider = getattr(app, "_provider", state.provider) or "setup"
    state.model = getattr(app, "_model", state.model) or "auto"

    # Command history
    if hasattr(app, "_command_history"):
        state.command_history = list(app._command_history)[-100:]  # Keep last 100

    # Recent repos from artifacts
    if hasattr(app, "_recent_run_artifacts"):
        repos = []
        for artifact in app._recent_run_artifacts:
            if hasattr(artifact, "repo_path") and artifact.repo_path:
                repos.append(artifact.repo_path)
        # Deduplicate preserving order
        seen: set[str] = set()
        unique_repos: list[str] = []
        for r in repos:
            if r not in seen:
                seen.add(r)
                unique_repos.append(r)
        state.recent_repos = unique_repos[:20]

    # Log filter state
    try:
        log_view = app.query_one("#main-output", None)
        if log_view and hasattr(log_view, "severity_filter"):
            state.log_filter_severity = log_view.severity_filter
        if log_view and hasattr(log_view, "search_filter"):
            state.log_filter_search = log_view.search_filter
    except Exception:
        pass

    return state


def apply_session_to_app(app: Any, state: SessionState) -> None:
    """Apply session state to the app."""
    # Theme
    try:
        app.theme = state.theme
    except Exception:
        pass

    # Directory
    try:
        app._directory = state.directory
    except Exception:
        pass

    # Mode/provider/model
    try:
        app._mode = state.mode
    except Exception:
        pass
    try:
        app._provider = state.provider
    except Exception:
        pass
    try:
        app._model = state.model
    except Exception:
        pass

    # Command history
    try:
        app._command_history = list(state.command_history)
        app._history_index = len(app._command_history)
    except Exception:
        pass

    # Log filter state
    try:
        log_view = app.query_one("#main-output", None)
        if log_view:
            if hasattr(log_view, "set_severity_filter"):
                log_view.set_severity_filter(state.log_filter_severity)
            if hasattr(log_view, "set_search_filter"):
                log_view.set_search_filter(state.log_filter_search)
    except Exception:
        pass

    # Sync status bar
    try:
        if hasattr(app, "_sync_status_bar"):
            app._sync_status_bar()
    except Exception:
        pass
