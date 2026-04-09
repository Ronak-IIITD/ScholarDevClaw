from __future__ import annotations

import os
from pathlib import Path


def parse_allowed_repo_roots() -> list[Path]:
    """Parse SCHOLARDEVCLAW_ALLOWED_REPO_DIRS into resolved root paths."""
    return [
        Path(p.strip()).expanduser().resolve()
        for p in os.environ.get("SCHOLARDEVCLAW_ALLOWED_REPO_DIRS", "").split(":")
        if p.strip()
    ]


def is_path_within_allowed_roots(path: Path, allowed_roots: list[Path]) -> bool:
    """Return True if path is equal to or a descendant of any allowed root."""
    return any(path == root or root in path.parents for root in allowed_roots)


def enforce_allowed_repo_path(path: Path) -> Path:
    """Enforce configured allowed roots for a repo path.

    Raises PermissionError when allowed roots are configured and *path* is outside.
    """
    allowed_roots = parse_allowed_repo_roots()
    if allowed_roots and not is_path_within_allowed_roots(path, allowed_roots):
        allowed_text = ", ".join(str(p) for p in allowed_roots)
        raise PermissionError(
            f"Repository path is outside allowed roots: {path}. Allowed roots: {allowed_text}"
        )
    return path
