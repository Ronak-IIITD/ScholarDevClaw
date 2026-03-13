"""
Multi-repo manager — coordinate analysis across multiple repositories.

Manages a collection of ``RepoProfile`` objects, each representing a single
repository's analysis snapshot.  Provides batch analysis, profile caching,
and the data layer consumed by ``CrossRepoAnalyzer`` and
``KnowledgeTransferEngine``.
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class RepoProfileStatus(str, enum.Enum):
    """Status of a repo profile within a multi-repo workspace."""

    PENDING = "pending"
    ANALYZING = "analyzing"
    READY = "ready"
    ERROR = "error"
    STALE = "stale"


@dataclass
class RepoProfile:
    """Snapshot of a single repository's analysis results.

    Stores the full analysis payload produced by ``run_analyze()`` together
    with research suggestions and descriptive metadata.  Two profiles can
    be compared by the ``CrossRepoAnalyzer``.
    """

    # Identity
    repo_path: str
    name: str  # human-friendly name (defaults to dir basename)
    repo_id: str = ""  # deterministic hash of resolved path

    # Analysis payload (from run_analyze)
    languages: list[str] = field(default_factory=list)
    language_stats: list[dict[str, Any]] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    patterns: dict[str, list[str]] = field(default_factory=dict)
    element_count: int = 0

    # Research suggestions (from run_suggest)
    suggestions: list[dict[str, Any]] = field(default_factory=list)

    # Metadata
    status: RepoProfileStatus = RepoProfileStatus.PENDING
    analyzed_at: float = 0.0  # epoch timestamp
    analysis_duration_ms: float = 0.0
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.repo_id:
            resolved = str(Path(self.repo_path).expanduser().resolve())
            self.repo_id = hashlib.sha256(resolved.encode()).hexdigest()[:12]
        if not self.name:
            self.name = Path(self.repo_path).name

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_path": self.repo_path,
            "name": self.name,
            "repo_id": self.repo_id,
            "languages": self.languages,
            "language_stats": self.language_stats,
            "frameworks": self.frameworks,
            "entry_points": self.entry_points,
            "test_files": self.test_files,
            "patterns": self.patterns,
            "element_count": self.element_count,
            "suggestions": self.suggestions,
            "status": self.status.value,
            "analyzed_at": self.analyzed_at,
            "analysis_duration_ms": self.analysis_duration_ms,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepoProfile:
        status_raw = data.get("status", "pending")
        try:
            status = RepoProfileStatus(status_raw)
        except ValueError:
            status = RepoProfileStatus.PENDING
        return cls(
            repo_path=data["repo_path"],
            name=data.get("name", ""),
            repo_id=data.get("repo_id", ""),
            languages=data.get("languages", []),
            language_stats=data.get("language_stats", []),
            frameworks=data.get("frameworks", []),
            entry_points=data.get("entry_points", []),
            test_files=data.get("test_files", []),
            patterns=data.get("patterns", {}),
            element_count=data.get("element_count", 0),
            suggestions=data.get("suggestions", []),
            status=status,
            analyzed_at=data.get("analyzed_at", 0.0),
            analysis_duration_ms=data.get("analysis_duration_ms", 0.0),
            error=data.get("error"),
        )


LogCallback = Callable[[str], None]

# ---------------------------------------------------------------------------
# Workspace persistence
# ---------------------------------------------------------------------------

_WORKSPACE_DIR = ".scholardevclaw"
_WORKSPACE_FILE = "multi_repo_workspace.json"


def _default_workspace_path() -> Path:
    return Path.home() / _WORKSPACE_DIR / _WORKSPACE_FILE


def _load_workspace(path: Path | None = None) -> dict[str, Any]:
    ws = path or _default_workspace_path()
    if not ws.exists():
        return {"profiles": {}}
    try:
        return json.loads(ws.read_text())
    except Exception:
        return {"profiles": {}}


def _save_workspace(data: dict[str, Any], path: Path | None = None) -> None:
    ws = path or _default_workspace_path()
    ws.parent.mkdir(parents=True, exist_ok=True)
    ws.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# MultiRepoManager
# ---------------------------------------------------------------------------


class MultiRepoManager:
    """Coordinate analysis across multiple repositories.

    Usage::

        mgr = MultiRepoManager()
        mgr.add_repo("/path/to/repo_a")
        mgr.add_repo("/path/to/repo_b")
        profiles = mgr.analyze_all()
    """

    def __init__(self, *, workspace_path: Path | None = None) -> None:
        self._ws_path = workspace_path
        self._workspace = _load_workspace(self._ws_path)
        self._profiles: dict[str, RepoProfile] = {}

        # Hydrate from persisted workspace
        for repo_id, raw in self._workspace.get("profiles", {}).items():
            try:
                profile = RepoProfile.from_dict(raw)
                self._profiles[profile.repo_id] = profile
            except Exception:
                continue

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        self._workspace["profiles"] = {pid: p.to_dict() for pid, p in self._profiles.items()}
        _save_workspace(self._workspace, self._ws_path)

    # ------------------------------------------------------------------
    # Repo management
    # ------------------------------------------------------------------

    def add_repo(self, repo_path: str, *, name: str | None = None) -> RepoProfile:
        """Add a repository to the workspace.

        Returns an existing profile if the repo was already added.
        """
        resolved = str(Path(repo_path).expanduser().resolve())
        profile = RepoProfile(repo_path=resolved, name=name or Path(resolved).name)

        if profile.repo_id in self._profiles:
            existing = self._profiles[profile.repo_id]
            if name and name != existing.name:
                existing.name = name
                self._persist()
            return existing

        self._profiles[profile.repo_id] = profile
        self._persist()
        logger.info("Added repo: %s (%s)", profile.name, profile.repo_id)
        return profile

    def remove_repo(self, repo_id_or_path: str) -> bool:
        """Remove a repository from the workspace."""
        profile = self._resolve_repo(repo_id_or_path)
        if profile is None:
            return False
        del self._profiles[profile.repo_id]
        self._persist()
        logger.info("Removed repo: %s", profile.name)
        return True

    def get_profile(self, repo_id_or_path: str) -> RepoProfile | None:
        """Look up a profile by repo_id or path."""
        return self._resolve_repo(repo_id_or_path)

    def list_profiles(self) -> list[RepoProfile]:
        """Return all profiles in the workspace."""
        return list(self._profiles.values())

    def _resolve_repo(self, repo_id_or_path: str) -> RepoProfile | None:
        """Resolve a repo by ID, name, or path."""
        if repo_id_or_path in self._profiles:
            return self._profiles[repo_id_or_path]

        # Try by resolved path
        try:
            resolved = str(Path(repo_id_or_path).expanduser().resolve())
            rid = hashlib.sha256(resolved.encode()).hexdigest()[:12]
            if rid in self._profiles:
                return self._profiles[rid]
        except Exception:
            pass

        # Try by name
        for p in self._profiles.values():
            if p.name == repo_id_or_path:
                return p
        return None

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze_repo(
        self,
        repo_id_or_path: str,
        *,
        log_callback: LogCallback | None = None,
    ) -> RepoProfile:
        """Run analysis + suggestions on a single repo and update its profile."""
        from scholardevclaw.application.pipeline import run_analyze, run_suggest

        profile = self._resolve_repo(repo_id_or_path)
        if profile is None:
            # Auto-add if passed a path
            profile = self.add_repo(repo_id_or_path)

        profile.status = RepoProfileStatus.ANALYZING
        self._persist()

        t0 = time.monotonic()
        try:
            # Analyze
            if log_callback:
                log_callback(f"Analyzing {profile.name}...")
            analyze_result = run_analyze(profile.repo_path, log_callback=log_callback)
            if not analyze_result.ok:
                raise RuntimeError(analyze_result.error or "Analysis failed")

            payload = analyze_result.payload
            profile.languages = payload.get("languages", [])
            profile.language_stats = payload.get("language_stats", [])
            profile.frameworks = payload.get("frameworks", [])
            profile.entry_points = payload.get("entry_points", [])
            profile.test_files = payload.get("test_files", [])
            profile.patterns = payload.get("patterns", {})

            # Count elements from language_stats
            total_elements = 0
            for ls in profile.language_stats:
                total_elements += ls.get("function_count", 0) + ls.get("class_count", 0)
            # Fallback: use file_count sum
            if total_elements == 0:
                total_elements = sum(ls.get("file_count", 0) for ls in profile.language_stats)
            profile.element_count = total_elements

            # Suggestions
            if log_callback:
                log_callback(f"Getting suggestions for {profile.name}...")
            suggest_result = run_suggest(profile.repo_path, log_callback=log_callback)
            if suggest_result.ok:
                profile.suggestions = suggest_result.payload.get("suggestions", [])

            profile.status = RepoProfileStatus.READY
            profile.error = None
        except Exception as exc:
            profile.status = RepoProfileStatus.ERROR
            profile.error = str(exc)
            logger.error("Analysis failed for %s: %s", profile.name, exc)
        finally:
            elapsed = (time.monotonic() - t0) * 1000
            profile.analyzed_at = time.time()
            profile.analysis_duration_ms = round(elapsed, 2)
            self._persist()

        return profile

    def analyze_all(
        self,
        *,
        log_callback: LogCallback | None = None,
    ) -> list[RepoProfile]:
        """Analyze all repos in the workspace.

        Returns the list of updated profiles.
        """
        results = []
        total = len(self._profiles)
        for i, profile in enumerate(list(self._profiles.values()), 1):
            if log_callback:
                log_callback(f"\n--- Repository {i}/{total}: {profile.name} ---")
            updated = self.analyze_repo(profile.repo_id, log_callback=log_callback)
            results.append(updated)
        return results

    def get_ready_profiles(self) -> list[RepoProfile]:
        """Return only profiles in READY state."""
        return [p for p in self._profiles.values() if p.status == RepoProfileStatus.READY]

    def clear_workspace(self) -> None:
        """Remove all profiles from the workspace."""
        self._profiles.clear()
        self._persist()
