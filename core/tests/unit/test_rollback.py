import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from scholardevclaw.rollback import (
    RollbackManager,
    RollbackStore,
    RollbackStatus,
    RollbackSnapshot,
    RollbackResult,
    ChangeRecord,
    ChangeType,
    FileSnapshot,
    GitSnapshot,
    create_rollback_snapshot,
    get_rollback_status,
    list_rollback_snapshots,
    run_rollback,
)


@pytest.fixture
def temp_store_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_repo():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "test_repo"
        repo_path.mkdir()
        (repo_path / "model.py").write_text("class Model:\n    pass\n")
        yield repo_path


@pytest.fixture
def store(temp_store_dir):
    return RollbackStore(temp_store_dir)


@pytest.fixture
def manager(store):
    return RollbackManager(store)


class TestRollbackStore:
    def test_store_initialization(self, temp_store_dir):
        store = RollbackStore(temp_store_dir)
        assert store.store_dir == Path(temp_store_dir)

    def test_default_store_dir(self):
        store = RollbackStore()
        expected = Path.home() / ".scholardevclaw" / "rollbacks"
        assert store.store_dir == expected

    def test_generate_snapshot_id(self, store):
        snapshot_id = store.generate_snapshot_id("/path/to/repo", "rmsnorm")
        assert snapshot_id is not None
        assert len(snapshot_id) > 10
        assert "rmsnorm" in snapshot_id.lower()

    def test_save_and_load_snapshot(self, store, temp_repo):
        snapshot = RollbackSnapshot(
            id="test-snapshot-123",
            repo_path=str(temp_repo),
            spec="rmsnorm",
            status=RollbackStatus.PENDING,
        )
        store.save(snapshot)

        loaded = store.load(str(temp_repo), "test-snapshot-123")
        assert loaded is not None
        assert loaded.id == "test-snapshot-123"
        assert loaded.spec == "rmsnorm"
        assert loaded.status == RollbackStatus.PENDING

    def test_delete_snapshot(self, store, temp_repo):
        snapshot = RollbackSnapshot(
            id="test-snapshot-delete",
            repo_path=str(temp_repo),
            spec="rmsnorm",
        )
        store.save(snapshot)

        assert store.delete(str(temp_repo), "test-snapshot-delete") is True
        assert store.load(str(temp_repo), "test-snapshot-delete") is None

    def test_list_snapshots(self, store, temp_repo):
        for i in range(3):
            snapshot = RollbackSnapshot(
                id=f"test-snapshot-{i}",
                repo_path=str(temp_repo),
                spec="rmsnorm",
                status=RollbackStatus.APPLIED if i == 0 else RollbackStatus.PENDING,
            )
            store.save(snapshot)

        snapshots = store.list_snapshots(str(temp_repo))
        assert len(snapshots) == 3

        applied = store.list_snapshots(str(temp_repo), status=RollbackStatus.APPLIED)
        assert len(applied) == 1
        assert applied[0].id == "test-snapshot-0"

    def test_get_latest(self, store, temp_repo):
        import time

        snapshot1 = RollbackSnapshot(
            id="snapshot_a",
            repo_path=str(temp_repo),
            spec="rmsnorm",
            timestamp="2026-01-01T10:00:00",
        )
        store.save(snapshot1)
        time.sleep(0.1)
        snapshot2 = RollbackSnapshot(
            id="snapshot_b",
            repo_path=str(temp_repo),
            spec="swiglu",
            timestamp="2026-01-02T10:00:00",
        )
        store.save(snapshot2)

        latest = store.get_latest(str(temp_repo))
        assert latest is not None
        assert latest.id == "snapshot_b"

    def test_cleanup_old_snapshots(self, store, temp_repo):
        for i in range(30):
            status = RollbackStatus.ROLLED_BACK if i < 20 else RollbackStatus.APPLIED
            snapshot = RollbackSnapshot(
                id=f"test-snapshot-{i}",
                repo_path=str(temp_repo),
                spec="rmsnorm",
                status=status,
            )
            store.save(snapshot)

        removed = store.cleanup_old_snapshots(str(temp_repo), keep_count=5)
        assert removed > 0

        remaining = store.list_snapshots(str(temp_repo))
        assert len(remaining) <= 15


class TestRollbackManager:
    def test_create_snapshot(self, manager, temp_repo):
        snapshot = manager.create_snapshot(
            str(temp_repo),
            "rmsnorm",
            description="Test snapshot",
        )

        assert snapshot.id is not None
        assert snapshot.repo_path == str(temp_repo)
        assert snapshot.spec == "rmsnorm"
        assert snapshot.status == RollbackStatus.PENDING

    def test_create_snapshot_with_git(self, manager, temp_repo):
        snapshot = manager.create_snapshot(str(temp_repo), "rmsnorm")

        assert snapshot.git_snapshot is not None

    def test_record_file_change(self, manager, temp_repo):
        snapshot = manager.create_snapshot(str(temp_repo), "rmsnorm")

        manager.record_file_change(
            str(temp_repo),
            snapshot.id,
            "model.py",
            ChangeType.FILE_MODIFIED,
            content_before="old content",
            content_after="new content",
        )

        updated = manager.get_snapshot(str(temp_repo), snapshot.id)
        assert len(updated.changes) == 1
        assert updated.changes[0].change_type == ChangeType.FILE_MODIFIED
        assert updated.changes[0].before is not None
        assert updated.changes[0].before.checksum is not None

    def test_record_branch_creation(self, manager, temp_repo):
        snapshot = manager.create_snapshot(str(temp_repo), "rmsnorm")

        manager.record_branch_creation(str(temp_repo), snapshot.id, "integration/rmsnorm")

        updated = manager.get_snapshot(str(temp_repo), snapshot.id)
        assert updated.git_snapshot.created_branch == "integration/rmsnorm"
        assert any(c.change_type == ChangeType.BRANCH_CREATED for c in updated.changes)

    def test_mark_applied(self, manager, temp_repo):
        snapshot = manager.create_snapshot(str(temp_repo), "rmsnorm")
        assert snapshot.status == RollbackStatus.PENDING

        manager.mark_applied(str(temp_repo), snapshot.id)

        updated = manager.get_snapshot(str(temp_repo), snapshot.id)
        assert updated.status == RollbackStatus.APPLIED

    def test_rollback_nonexistent_snapshot(self, manager, temp_repo):
        result = manager.rollback(str(temp_repo), "nonexistent-id")
        assert result.ok is False
        assert "No snapshot found" in result.error

    def test_rollback_latest_applied(self, manager, temp_repo):
        snapshot = manager.create_snapshot(str(temp_repo), "rmsnorm")
        manager.mark_applied(str(temp_repo), snapshot.id)

        result = manager.rollback(str(temp_repo))
        assert result.snapshot_id == snapshot.id

    def test_rollback_wrong_status(self, manager, temp_repo):
        snapshot = manager.create_snapshot(str(temp_repo), "rmsnorm")

        result = manager.rollback(str(temp_repo), snapshot.id)
        assert result.ok is False
        assert "not 'applied'" in result.error

    def test_rollback_with_force(self, manager, temp_repo):
        snapshot = manager.create_snapshot(str(temp_repo), "rmsnorm")

        result = manager.rollback(str(temp_repo), snapshot.id, force=True)
        assert result.ok is True


class TestRollbackTypes:
    def test_rollback_snapshot_to_dict(self):
        snapshot = RollbackSnapshot(
            id="test-id",
            repo_path="/path/to/repo",
            spec="rmsnorm",
            status=RollbackStatus.APPLIED,
            changes=[
                ChangeRecord(
                    change_type=ChangeType.FILE_MODIFIED,
                    path="model.py",
                    before=FileSnapshot(path="model.py", checksum="abc123"),
                )
            ],
        )

        data = snapshot.to_dict()
        assert data["id"] == "test-id"
        assert data["status"] == "applied"
        assert len(data["changes"]) == 1

    def test_rollback_snapshot_from_dict(self):
        data = {
            "id": "test-id",
            "repo_path": "/path/to/repo",
            "spec": "rmsnorm",
            "timestamp": "2026-01-01T10:00:00",
            "status": "applied",
            "changes": [
                {
                    "change_type": "file_modified",
                    "path": "model.py",
                    "before": {"path": "model.py", "checksum": "abc123", "exists": True},
                    "after": None,
                    "metadata": {},
                }
            ],
        }

        snapshot = RollbackSnapshot.from_dict(data)
        assert snapshot.id == "test-id"
        assert snapshot.status == RollbackStatus.APPLIED
        assert len(snapshot.changes) == 1

    def test_git_snapshot(self):
        git = GitSnapshot(
            branch="main",
            commit_sha="abc123",
            is_clean=True,
            created_branch="integration/rmsnorm",
        )
        assert git.branch == "main"
        assert git.created_branch == "integration/rmsnorm"


class TestRollbackConvenienceFunctions:
    def test_create_rollback_snapshot(self, temp_repo):
        snapshot = create_rollback_snapshot(
            str(temp_repo),
            "rmsnorm",
            description="Test",
        )
        assert snapshot is not None
        assert snapshot.spec == "rmsnorm"

    def test_get_rollback_status(self, temp_repo):
        manager = RollbackManager()
        snapshot = manager.create_snapshot(str(temp_repo), "rmsnorm")
        manager.mark_applied(str(temp_repo), snapshot.id)

        status = get_rollback_status(str(temp_repo), snapshot.id)
        assert status["found"] is True
        assert status["status"] == "applied"

    def test_list_rollback_snapshots(self, temp_repo):
        manager = RollbackManager()
        manager.create_snapshot(str(temp_repo), "rmsnorm")
        manager.create_snapshot(str(temp_repo), "swiglu")

        snapshots = list_rollback_snapshots(str(temp_repo))
        assert len(snapshots) == 2

    def test_run_rollback(self, temp_repo):
        manager = RollbackManager()
        snapshot = manager.create_snapshot(str(temp_repo), "rmsnorm")
        manager.mark_applied(str(temp_repo), snapshot.id)

        result = run_rollback(str(temp_repo), snapshot.id)
        assert result.snapshot_id == snapshot.id
