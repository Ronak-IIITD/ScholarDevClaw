"""
Atomic file write utilities for safe state persistence.

Provides functions for writing files atomically using temp + rename pattern,
preventing TOCTOU race conditions and partial writes.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path | str, content: str, mode: int | None = None) -> None:
    """Write text to a file atomically using temp + rename.

    Creates a temporary file in the same directory, writes content to it,
    then atomically renames to the target path. This prevents partial writes
    and race conditions.

    Parameters
    ----------
    path : Path | str
        Target file path.
    content : str
        Content to write.
    mode : int | None
        Optional file permissions (default: owner read/write 0600).
    """
    path = Path(path)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        os.write(fd, content.encode())
        os.close(fd)
        fd = -1
        if mode is None:
            mode = stat.S_IRUSR | stat.S_IWUSR
        os.chmod(tmp_path, mode)
        os.rename(tmp_path, str(path))
    except Exception:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def atomic_write_json(path: Path | str, data: dict[str, Any], **json_kwargs: Any) -> None:
    """Write JSON data to a file atomically.

    Parameters
    ----------
    path : Path | str
        Target file path.
    data : dict
        JSON-serializable data.
    **json_kwargs
        Additional arguments passed to json.dumps().
    """
    import json

    content = json.dumps(data, **json_kwargs)
    atomic_write_text(path, content)


def atomic_write_jsonl(path: Path | str, lines: list[str]) -> None:
    """Write JSONL (JSON Lines) data to a file atomically.

    Parameters
    ----------
    path : Path | str
        Target file path.
    lines : list[str]
        List of JSON strings (one per line).
    """
    content = "\n".join(lines)
    if lines:
        content += "\n"
    atomic_write_text(path, content)
