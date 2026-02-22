from __future__ import annotations

from typing import Any, Callable

from .manager import RollbackManager
from .store import RollbackStore
from .types import (
    ChangeRecord,
    ChangeType,
    FileSnapshot,
    GitSnapshot,
    RollbackResult,
    RollbackSnapshot,
    RollbackStatus,
)

LogCallback = Callable[[str], None]


def get_rollback_store(store_dir: str | None = None) -> RollbackStore:
    return RollbackStore(store_dir)


def get_rollback_manager(store: RollbackStore | None = None) -> RollbackManager:
    return RollbackManager(store)


def create_rollback_snapshot(
    repo_path: str,
    spec: str,
    *,
    description: str = "",
    integration_run_id: str | None = None,
    log_callback: LogCallback | None = None,
) -> RollbackSnapshot:
    manager = RollbackManager()
    return manager.create_snapshot(
        repo_path,
        spec,
        description=description,
        integration_run_id=integration_run_id,
        log_callback=log_callback,
    )


def run_rollback(
    repo_path: str,
    snapshot_id: str | None = None,
    *,
    force: bool = False,
    log_callback: LogCallback | None = None,
) -> RollbackResult:
    manager = RollbackManager()
    return manager.rollback(
        repo_path,
        snapshot_id,
        force=force,
        log_callback=log_callback,
    )


def list_rollback_snapshots(
    repo_path: str,
    status: RollbackStatus | None = None,
) -> list[dict[str, Any]]:
    manager = RollbackManager()
    snapshots = manager.list_snapshots(repo_path, status=status)
    return [
        {
            "id": s.id,
            "spec": s.spec,
            "timestamp": s.timestamp,
            "status": s.status.value,
            "description": s.description,
            "changes_count": len(s.changes),
        }
        for s in snapshots
    ]


def get_rollback_status(repo_path: str, snapshot_id: str | None = None) -> dict[str, Any]:
    manager = RollbackManager()
    if snapshot_id:
        snapshot = manager.get_snapshot(repo_path, snapshot_id)
    else:
        snapshot = manager.get_latest_applied(repo_path)

    if not snapshot:
        return {"found": False, "message": "No snapshot found"}

    return {
        "found": True,
        "id": snapshot.id,
        "spec": snapshot.spec,
        "status": snapshot.status.value,
        "timestamp": snapshot.timestamp,
        "description": snapshot.description,
        "changes_count": len(snapshot.changes),
        "git_branch": snapshot.git_snapshot.branch if snapshot.git_snapshot else None,
        "rolled_back_at": snapshot.rolled_back_at,
    }
