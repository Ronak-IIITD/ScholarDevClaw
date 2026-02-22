from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class RollbackStatus(str, Enum):
    PENDING = "pending"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"
    PARTIAL = "partial"


class ChangeType(str, Enum):
    FILE_MODIFIED = "file_modified"
    FILE_CREATED = "file_created"
    FILE_DELETED = "file_deleted"
    BRANCH_CREATED = "branch_created"
    COMMIT_CREATED = "commit_created"


@dataclass
class FileSnapshot:
    path: str
    content: str | None = None
    checksum: str | None = None
    exists: bool = True


@dataclass
class ChangeRecord:
    change_type: ChangeType
    path: str
    before: FileSnapshot | None = None
    after: FileSnapshot | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GitSnapshot:
    branch: str | None = None
    commit_sha: str | None = None
    is_clean: bool = True
    created_branch: str | None = None
    created_commit_sha: str | None = None


@dataclass
class RollbackSnapshot:
    id: str
    repo_path: str
    spec: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: RollbackStatus = RollbackStatus.PENDING
    git_snapshot: GitSnapshot | None = None
    changes: list[ChangeRecord] = field(default_factory=list)
    integration_run_id: str | None = None
    description: str = ""
    rolled_back_at: str | None = None
    rollback_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "repo_path": self.repo_path,
            "spec": self.spec,
            "timestamp": self.timestamp,
            "status": self.status.value,
            "git_snapshot": {
                "branch": self.git_snapshot.branch,
                "commit_sha": self.git_snapshot.commit_sha,
                "is_clean": self.git_snapshot.is_clean,
                "created_branch": self.git_snapshot.created_branch,
                "created_commit_sha": self.git_snapshot.created_commit_sha,
            }
            if self.git_snapshot
            else None,
            "changes": [
                {
                    "change_type": c.change_type.value,
                    "path": c.path,
                    "before": {
                        "path": c.before.path,
                        "checksum": c.before.checksum,
                        "exists": c.before.exists,
                    }
                    if c.before
                    else None,
                    "after": {
                        "path": c.after.path,
                        "checksum": c.after.checksum,
                        "exists": c.after.exists,
                    }
                    if c.after
                    else None,
                    "metadata": c.metadata,
                }
                for c in self.changes
            ],
            "integration_run_id": self.integration_run_id,
            "description": self.description,
            "rolled_back_at": self.rolled_back_at,
            "rollback_error": self.rollback_error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RollbackSnapshot:
        git_data = data.get("git_snapshot")
        git_snapshot = None
        if git_data:
            git_snapshot = GitSnapshot(
                branch=git_data.get("branch"),
                commit_sha=git_data.get("commit_sha"),
                is_clean=git_data.get("is_clean", True),
                created_branch=git_data.get("created_branch"),
                created_commit_sha=git_data.get("created_commit_sha"),
            )

        changes = []
        for c in data.get("changes", []):
            before = None
            if c.get("before"):
                before = FileSnapshot(
                    path=c["before"]["path"],
                    checksum=c["before"].get("checksum"),
                    exists=c["before"].get("exists", True),
                )
            after = None
            if c.get("after"):
                after = FileSnapshot(
                    path=c["after"]["path"],
                    checksum=c["after"].get("checksum"),
                    exists=c["after"].get("exists", True),
                )
            changes.append(
                ChangeRecord(
                    change_type=ChangeType(c["change_type"]),
                    path=c["path"],
                    before=before,
                    after=after,
                    metadata=c.get("metadata", {}),
                )
            )

        return cls(
            id=data["id"],
            repo_path=data["repo_path"],
            spec=data["spec"],
            timestamp=data["timestamp"],
            status=RollbackStatus(data["status"]),
            git_snapshot=git_snapshot,
            changes=changes,
            integration_run_id=data.get("integration_run_id"),
            description=data.get("description", ""),
            rolled_back_at=data.get("rolled_back_at"),
            rollback_error=data.get("rollback_error"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RollbackResult:
    ok: bool
    snapshot_id: str
    status: RollbackStatus
    changes_reverted: int = 0
    files_restored: list[str] = field(default_factory=list)
    branches_deleted: list[str] = field(default_factory=list)
    error: str | None = None
    logs: list[str] = field(default_factory=list)
