from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .types import RollbackSnapshot, RollbackStatus


class RollbackStore:
    def __init__(self, store_dir: str | None = None):
        if store_dir:
            self.store_dir = Path(store_dir)
        else:
            default_dir = os.environ.get("SCHOLARDEVCLAW_ROLLBACK_DIR")
            if default_dir:
                self.store_dir = Path(default_dir)
            else:
                self.store_dir = Path.home() / ".scholardevclaw" / "rollbacks"

        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _get_repo_hash(self, repo_path: str) -> str:
        return hashlib.sha256(Path(repo_path).resolve().as_posix().encode()).hexdigest()[:16]

    def _get_repo_dir(self, repo_path: str) -> Path:
        repo_hash = self._get_repo_hash(repo_path)
        repo_dir = self.store_dir / repo_hash
        repo_dir.mkdir(parents=True, exist_ok=True)
        return repo_dir

    def _get_snapshot_path(self, repo_path: str, snapshot_id: str) -> Path:
        return self._get_repo_dir(repo_path) / f"{snapshot_id}.json"

    def _get_index_path(self, repo_path: str) -> Path:
        return self._get_repo_dir(repo_path) / "index.json"

    def generate_snapshot_id(self, repo_path: str, spec: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        spec_slug = spec.lower().replace(" ", "-").replace("/", "-")[:20]
        random_suffix = hashlib.sha256(
            f"{repo_path}{spec}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:6]
        return f"{timestamp}_{spec_slug}_{random_suffix}"

    def save(self, snapshot: RollbackSnapshot) -> None:
        snapshot_path = self._get_snapshot_path(snapshot.repo_path, snapshot.id)
        with open(snapshot_path, "w") as f:
            json.dump(snapshot.to_dict(), f, indent=2)
        self._update_index(snapshot.repo_path, snapshot)

    def load(self, repo_path: str, snapshot_id: str) -> RollbackSnapshot | None:
        snapshot_path = self._get_snapshot_path(repo_path, snapshot_id)
        if not snapshot_path.exists():
            return None
        try:
            with open(snapshot_path) as f:
                data = json.load(f)
            return RollbackSnapshot.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def delete(self, repo_path: str, snapshot_id: str) -> bool:
        snapshot_path = self._get_snapshot_path(repo_path, snapshot_id)
        if snapshot_path.exists():
            snapshot_path.unlink()
            self._remove_from_index(repo_path, snapshot_id)
            return True
        return False

    def list_snapshots(
        self,
        repo_path: str,
        status: RollbackStatus | None = None,
        limit: int = 50,
    ) -> list[RollbackSnapshot]:
        repo_dir = self._get_repo_dir(repo_path)
        snapshots: list[RollbackSnapshot] = []

        for snapshot_file in sorted(repo_dir.glob("*.json"), reverse=True):
            if snapshot_file.name == "index.json":
                continue
            try:
                with open(snapshot_file) as f:
                    data = json.load(f)
                snapshot = RollbackSnapshot.from_dict(data)
                if status is None or snapshot.status == status:
                    snapshots.append(snapshot)
                if len(snapshots) >= limit:
                    break
            except (json.JSONDecodeError, KeyError):
                continue

        return snapshots

    def get_latest(self, repo_path: str) -> RollbackSnapshot | None:
        snapshots = self.list_snapshots(repo_path, limit=1)
        return snapshots[0] if snapshots else None

    def get_latest_applied(self, repo_path: str) -> RollbackSnapshot | None:
        snapshots = self.list_snapshots(repo_path, status=RollbackStatus.APPLIED, limit=1)
        return snapshots[0] if snapshots else None

    def _update_index(self, repo_path: str, snapshot: RollbackSnapshot) -> None:
        index_path = self._get_index_path(repo_path)
        index: list[dict[str, Any]] = []

        if index_path.exists():
            try:
                with open(index_path) as f:
                    index = json.load(f)
            except (json.JSONDecodeError, KeyError):
                index = []

        existing = next((i for i, e in enumerate(index) if e.get("id") == snapshot.id), None)
        entry = {
            "id": snapshot.id,
            "spec": snapshot.spec,
            "timestamp": snapshot.timestamp,
            "status": snapshot.status.value,
            "description": snapshot.description,
        }

        if existing is not None:
            index[existing] = entry
        else:
            index.insert(0, entry)

        index = index[:100]

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    def _remove_from_index(self, repo_path: str, snapshot_id: str) -> None:
        index_path = self._get_index_path(repo_path)
        if not index_path.exists():
            return

        try:
            with open(index_path) as f:
                index = json.load(f)
            index = [e for e in index if e.get("id") != snapshot_id]
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)
        except (json.JSONDecodeError, KeyError):
            pass

    def list_repos(self) -> list[dict[str, Any]]:
        repos: list[dict[str, Any]] = []
        for repo_dir in self.store_dir.iterdir():
            if repo_dir.is_dir():
                index_path = repo_dir / "index.json"
                if index_path.exists():
                    try:
                        with open(index_path) as f:
                            index = json.load(f)
                        if index:
                            repos.append(
                                {
                                    "hash": repo_dir.name,
                                    "snapshot_count": len(index),
                                    "latest": index[0] if index else None,
                                }
                            )
                    except (json.JSONDecodeError, KeyError):
                        pass
        return repos

    def cleanup_old_snapshots(self, repo_path: str, keep_count: int = 20) -> int:
        snapshots = self.list_snapshots(repo_path, limit=1000)
        removed = 0

        rolled_back = [s for s in snapshots if s.status == RollbackStatus.ROLLED_BACK]
        applied = [s for s in snapshots if s.status == RollbackStatus.APPLIED]
        other = [
            s
            for s in snapshots
            if s.status not in (RollbackStatus.ROLLED_BACK, RollbackStatus.APPLIED)
        ]

        to_keep_rolled = rolled_back[:keep_count]
        to_keep_applied = applied[:keep_count]

        all_to_keep = {s.id for s in to_keep_rolled + to_keep_applied + other}

        for snapshot in snapshots:
            if snapshot.id not in all_to_keep:
                self.delete(repo_path, snapshot.id)
                removed += 1

        return removed
