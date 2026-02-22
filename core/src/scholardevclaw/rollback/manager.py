from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

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


def _log(logs: list[str], message: str, log_callback: LogCallback | None = None) -> None:
    logs.append(message)
    if log_callback is not None:
        log_callback(message)


def _compute_checksum(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class RollbackManager:
    def __init__(self, store: RollbackStore | None = None):
        self.store = store or RollbackStore()

    def create_snapshot(
        self,
        repo_path: str,
        spec: str,
        *,
        description: str = "",
        integration_run_id: str | None = None,
        log_callback: LogCallback | None = None,
    ) -> RollbackSnapshot:
        logs: list[str] = []
        path = Path(repo_path).expanduser().resolve()

        snapshot_id = self.store.generate_snapshot_id(str(path), spec)
        _log(logs, f"Creating rollback snapshot: {snapshot_id}", log_callback)

        git_snapshot = self._capture_git_state(path, logs, log_callback)

        snapshot = RollbackSnapshot(
            id=snapshot_id,
            repo_path=str(path),
            spec=spec,
            timestamp=datetime.now().isoformat(),
            status=RollbackStatus.PENDING,
            git_snapshot=git_snapshot,
            changes=[],
            integration_run_id=integration_run_id,
            description=description,
        )

        self.store.save(snapshot)
        _log(logs, f"Snapshot created successfully", log_callback)

        return snapshot

    def record_file_change(
        self,
        repo_path: str,
        snapshot_id: str,
        file_path: str,
        change_type: ChangeType,
        content_before: str | None = None,
        content_after: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        snapshot = self.store.load(repo_path, snapshot_id)
        if not snapshot:
            raise ValueError(f"Snapshot not found: {snapshot_id}")

        path = Path(repo_path).expanduser().resolve()
        file_path_resolved = Path(file_path)
        if not file_path_resolved.is_absolute():
            file_path_resolved = path / file_path
        relative_path = str(file_path_resolved.relative_to(path))

        before_snapshot = None
        if content_before is not None:
            before_snapshot = FileSnapshot(
                path=relative_path,
                content=content_before,
                checksum=_compute_checksum(content_before),
                exists=True,
            )
        elif change_type in (ChangeType.FILE_MODIFIED, ChangeType.FILE_DELETED):
            full_path = path / relative_path
            if full_path.exists():
                try:
                    content = full_path.read_text()
                    before_snapshot = FileSnapshot(
                        path=relative_path,
                        content=content,
                        checksum=_compute_checksum(content),
                        exists=True,
                    )
                except Exception:
                    before_snapshot = FileSnapshot(
                        path=relative_path,
                        content=None,
                        checksum=None,
                        exists=False,
                    )

        after_snapshot = None
        if content_after is not None:
            after_snapshot = FileSnapshot(
                path=relative_path,
                content=content_after,
                checksum=_compute_checksum(content_after),
                exists=True,
            )
        elif change_type == ChangeType.FILE_CREATED:
            full_path = path / relative_path
            if full_path.exists():
                try:
                    content = full_path.read_text()
                    after_snapshot = FileSnapshot(
                        path=relative_path,
                        content=content,
                        checksum=_compute_checksum(content),
                        exists=True,
                    )
                except Exception:
                    pass

        change = ChangeRecord(
            change_type=change_type,
            path=relative_path,
            before=before_snapshot,
            after=after_snapshot,
            metadata=metadata or {},
        )

        snapshot.changes.append(change)
        self.store.save(snapshot)

    def record_branch_creation(
        self,
        repo_path: str,
        snapshot_id: str,
        branch_name: str,
    ) -> None:
        snapshot = self.store.load(repo_path, snapshot_id)
        if not snapshot:
            raise ValueError(f"Snapshot not found: {snapshot_id}")

        if snapshot.git_snapshot:
            snapshot.git_snapshot.created_branch = branch_name

        change = ChangeRecord(
            change_type=ChangeType.BRANCH_CREATED,
            path=branch_name,
            metadata={"branch_name": branch_name},
        )
        snapshot.changes.append(change)
        self.store.save(snapshot)

    def record_commit_creation(
        self,
        repo_path: str,
        snapshot_id: str,
        commit_sha: str,
        message: str = "",
    ) -> None:
        snapshot = self.store.load(repo_path, snapshot_id)
        if not snapshot:
            raise ValueError(f"Snapshot not found: {snapshot_id}")

        if snapshot.git_snapshot:
            snapshot.git_snapshot.created_commit_sha = commit_sha

        change = ChangeRecord(
            change_type=ChangeType.COMMIT_CREATED,
            path=commit_sha,
            metadata={"commit_sha": commit_sha, "message": message},
        )
        snapshot.changes.append(change)
        self.store.save(snapshot)

    def mark_applied(self, repo_path: str, snapshot_id: str) -> None:
        snapshot = self.store.load(repo_path, snapshot_id)
        if not snapshot:
            raise ValueError(f"Snapshot not found: {snapshot_id}")

        snapshot.status = RollbackStatus.APPLIED
        self.store.save(snapshot)

    def rollback(
        self,
        repo_path: str,
        snapshot_id: str | None = None,
        *,
        force: bool = False,
        log_callback: LogCallback | None = None,
    ) -> RollbackResult:
        logs: list[str] = []
        path = Path(repo_path).expanduser().resolve()

        if snapshot_id:
            snapshot = self.store.load(str(path), snapshot_id)
        else:
            snapshot = self.store.get_latest_applied(str(path))

        if not snapshot:
            error = "No snapshot found to rollback"
            _log(logs, error, log_callback)
            return RollbackResult(
                ok=False,
                snapshot_id="",
                status=RollbackStatus.FAILED,
                error=error,
                logs=logs,
            )

        _log(logs, f"Rolling back snapshot: {snapshot.id}", log_callback)

        if snapshot.status != RollbackStatus.APPLIED and not force:
            error = f"Snapshot status is {snapshot.status.value}, not 'applied'. Use --force to rollback anyway."
            _log(logs, error, log_callback)
            return RollbackResult(
                ok=False,
                snapshot_id=snapshot.id,
                status=snapshot.status,
                error=error,
                logs=logs,
            )

        changes_reverted = 0
        files_restored: list[str] = []
        branches_deleted: list[str] = []

        try:
            git_result = self._rollback_git_state(path, snapshot, logs, log_callback, force)
            if git_result.get("branches_deleted"):
                branches_deleted.extend(git_result["branches_deleted"])

            for change in reversed(snapshot.changes):
                if change.change_type in (ChangeType.FILE_MODIFIED, ChangeType.FILE_DELETED):
                    result = self._rollback_file_change(path, change, logs, log_callback)
                    if result:
                        changes_reverted += 1
                        files_restored.append(change.path)

            snapshot.status = RollbackStatus.ROLLED_BACK
            snapshot.rolled_back_at = datetime.now().isoformat()
            self.store.save(snapshot)

            _log(
                logs,
                f"Rollback complete: {changes_reverted} changes reverted, {len(files_restored)} files restored",
                log_callback,
            )

            return RollbackResult(
                ok=True,
                snapshot_id=snapshot.id,
                status=RollbackStatus.ROLLED_BACK,
                changes_reverted=changes_reverted,
                files_restored=files_restored,
                branches_deleted=branches_deleted,
                logs=logs,
            )

        except Exception as e:
            snapshot.status = RollbackStatus.PARTIAL
            snapshot.rollback_error = str(e)
            self.store.save(snapshot)

            _log(logs, f"Rollback failed: {e}", log_callback)
            return RollbackResult(
                ok=False,
                snapshot_id=snapshot.id,
                status=RollbackStatus.PARTIAL,
                error=str(e),
                logs=logs,
            )

    def _capture_git_state(
        self,
        path: Path,
        logs: list[str],
        log_callback: LogCallback | None,
    ) -> GitSnapshot:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(path),
                capture_output=True,
                text=True,
                timeout=10,
            )
            branch = result.stdout.strip() if result.returncode == 0 else None

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(path),
                capture_output=True,
                text=True,
                timeout=10,
            )
            commit_sha = result.stdout.strip() if result.returncode == 0 else None

            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(path),
                capture_output=True,
                text=True,
                timeout=10,
            )
            is_clean = result.returncode == 0 and not result.stdout.strip()

            _log(
                logs,
                f"Git state: branch={branch}, commit={commit_sha[:8] if commit_sha else 'none'}, clean={is_clean}",
                log_callback,
            )

            return GitSnapshot(
                branch=branch,
                commit_sha=commit_sha,
                is_clean=is_clean,
            )

        except Exception as e:
            _log(logs, f"Warning: Could not capture git state: {e}", log_callback)
            return GitSnapshot()

    def _rollback_git_state(
        self,
        path: Path,
        snapshot: RollbackSnapshot,
        logs: list[str],
        log_callback: LogCallback | None,
        force: bool,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"branches_deleted": []}

        if not snapshot.git_snapshot:
            return result

        git_snapshot = snapshot.git_snapshot

        if git_snapshot.created_branch:
            try:
                current_result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=str(path),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                current_branch = (
                    current_result.stdout.strip() if current_result.returncode == 0 else None
                )

                if current_branch == git_snapshot.created_branch:
                    target_branch = git_snapshot.branch or "main"
                    subprocess.run(
                        ["git", "checkout", target_branch],
                        cwd=str(path),
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    _log(logs, f"Switched to branch: {target_branch}", log_callback)

                subprocess.run(
                    ["git", "branch", "-D", git_snapshot.created_branch],
                    cwd=str(path),
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                result["branches_deleted"].append(git_snapshot.created_branch)
                _log(logs, f"Deleted branch: {git_snapshot.created_branch}", log_callback)

            except Exception as e:
                _log(logs, f"Warning: Could not delete branch: {e}", log_callback)

        elif git_snapshot.commit_sha:
            try:
                subprocess.run(
                    ["git", "reset", "--hard", git_snapshot.commit_sha],
                    cwd=str(path),
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                _log(logs, f"Reset to commit: {git_snapshot.commit_sha[:8]}", log_callback)
            except Exception as e:
                _log(logs, f"Warning: Could not reset git state: {e}", log_callback)

        return result

    def _rollback_file_change(
        self,
        path: Path,
        change: ChangeRecord,
        logs: list[str],
        log_callback: LogCallback | None,
    ) -> bool:
        file_path = path / change.path

        try:
            if change.change_type == ChangeType.FILE_DELETED:
                if change.before and change.before.content:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(change.before.content)
                    _log(logs, f"Restored deleted file: {change.path}", log_callback)
                    return True

            elif change.change_type == ChangeType.FILE_MODIFIED:
                if change.before and change.before.content:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(change.before.content)
                    _log(logs, f"Reverted modified file: {change.path}", log_callback)
                    return True

            elif change.change_type == ChangeType.FILE_CREATED:
                if file_path.exists():
                    file_path.unlink()
                    _log(logs, f"Removed created file: {change.path}", log_callback)
                    return True

        except Exception as e:
            _log(logs, f"Warning: Could not rollback {change.path}: {e}", log_callback)

        return False

    def list_snapshots(
        self,
        repo_path: str,
        status: RollbackStatus | None = None,
    ) -> list[RollbackSnapshot]:
        return self.store.list_snapshots(repo_path, status=status)

    def get_snapshot(self, repo_path: str, snapshot_id: str) -> RollbackSnapshot | None:
        return self.store.load(repo_path, snapshot_id)

    def get_latest_applied(self, repo_path: str) -> RollbackSnapshot | None:
        return self.store.get_latest_applied(repo_path)

    def delete_snapshot(self, repo_path: str, snapshot_id: str) -> bool:
        return self.store.delete(repo_path, snapshot_id)
