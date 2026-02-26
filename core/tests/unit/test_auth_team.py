"""Tests for team and multi-user support."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from scholardevclaw.auth.team import (
    TeamRole,
    TeamPermission,
    TeamMember,
    Team,
    TeamInvite,
    TeamStore,
    TeamAccessControl,
    ROLE_PERMISSIONS,
)


class TestTeamRole:
    def test_roles_exist(self):
        assert TeamRole.ADMIN.value == "admin"
        assert TeamRole.DEVELOPER.value == "developer"
        assert TeamRole.VIEWER.value == "viewer"


class TestTeamPermission:
    def test_permissions_exist(self):
        assert TeamPermission.READ_KEYS.value == "read_keys"
        assert TeamPermission.WRITE_KEYS.value == "write_keys"
        assert TeamPermission.DELETE_KEYS.value == "delete_keys"


class TestRolePermissions:
    def test_admin_has_all_permissions(self):
        assert TeamPermission.READ_KEYS in ROLE_PERMISSIONS[TeamRole.ADMIN]
        assert TeamPermission.WRITE_KEYS in ROLE_PERMISSIONS[TeamRole.ADMIN]
        assert TeamPermission.MANAGE_USERS in ROLE_PERMISSIONS[TeamRole.ADMIN]

    def test_developer_has_limited_permissions(self):
        assert TeamPermission.READ_KEYS in ROLE_PERMISSIONS[TeamRole.DEVELOPER]
        assert TeamPermission.DELETE_KEYS not in ROLE_PERMISSIONS[TeamRole.DEVELOPER]

    def test_viewer_has_minimal_permissions(self):
        assert TeamPermission.READ_KEYS in ROLE_PERMISSIONS[TeamRole.VIEWER]
        assert TeamPermission.WRITE_KEYS not in ROLE_PERMISSIONS[TeamRole.VIEWER]


class TestTeamMember:
    def test_creation(self):
        member = TeamMember(
            id="member_1",
            email="test@example.com",
            name="Test User",
            role=TeamRole.DEVELOPER,
            joined_at=datetime.now().isoformat(),
        )
        assert member.email == "test@example.com"
        assert member.role == TeamRole.DEVELOPER

    def test_to_dict_roundtrip(self):
        member = TeamMember(
            id="member_1",
            email="test@example.com",
            name="Test User",
            role=TeamRole.DEVELOPER,
            joined_at=datetime.now().isoformat(),
        )
        data = member.to_dict()
        restored = TeamMember.from_dict(data)
        assert restored.id == member.id
        assert restored.email == member.email

    def test_has_permission(self):
        member = TeamMember(
            id="member_1",
            email="test@example.com",
            name="Test",
            role=TeamRole.DEVELOPER,
            joined_at=datetime.now().isoformat(),
        )
        assert member.has_permission(TeamPermission.READ_KEYS) is True
        assert member.has_permission(TeamPermission.DELETE_KEYS) is False


class TestTeam:
    def test_creation(self):
        team = Team(
            id="team_1",
            name="Test Team",
            created_at=datetime.now().isoformat(),
            owner_id="owner_1",
        )
        assert team.name == "Test Team"
        assert len(team.members) == 0

    def test_add_member(self):
        team = Team(
            id="team_1",
            name="Test Team",
            created_at=datetime.now().isoformat(),
            owner_id="owner_1",
        )
        member = team.add_member("test@example.com", "Test User", TeamRole.DEVELOPER)
        assert len(team.members) == 1
        assert member.email == "test@example.com"

    def test_remove_member(self):
        team = Team(
            id="team_1",
            name="Test Team",
            created_at=datetime.now().isoformat(),
            owner_id="owner_1",
        )
        member = team.add_member("test@example.com", "Test User", TeamRole.DEVELOPER)
        assert team.remove_member(member.id) is True
        assert len(team.members) == 0

    def test_get_member(self):
        team = Team(
            id="team_1",
            name="Test Team",
            created_at=datetime.now().isoformat(),
            owner_id="owner_1",
        )
        member = team.add_member("test@example.com", "Test User", TeamRole.DEVELOPER)
        assert team.get_member(member.id) is not None
        assert team.get_member("nonexistent") is None


class TestTeamStore:
    @pytest.fixture
    def store_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_create_team(self, store_dir):
        store = TeamStore(store_dir)
        team = store.create_team("Test Team", "owner@example.com", "Owner")
        assert team.name == "Test Team"
        assert len(team.members) == 1
        assert team.members[0].role == TeamRole.ADMIN

    def test_get_team(self, store_dir):
        store = TeamStore(store_dir)
        created = store.create_team("Test Team", "owner@example.com", "Owner")
        retrieved = store.get_team(created.id)
        assert retrieved is not None
        assert retrieved.name == "Test Team"

    def test_get_team_not_found(self, store_dir):
        store = TeamStore(store_dir)
        assert store.get_team("nonexistent") is None

    def test_delete_team(self, store_dir):
        store = TeamStore(store_dir)
        team = store.create_team("Test Team", "owner@example.com", "Owner")
        assert store.delete_team(team.id) is True
        assert store.get_team(team.id) is None

    def test_list_teams(self, store_dir):
        store = TeamStore(store_dir)
        store.create_team("Team 1", "owner1@example.com", "Owner 1")
        store.create_team("Team 2", "owner2@example.com", "Owner 2")
        teams = store.list_teams()
        assert len(teams) == 2


class TestTeamAccessControl:
    @pytest.fixture
    def store_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_check_permission(self, store_dir):
        store = TeamStore(store_dir)
        team = store.create_team("Test Team", "owner@example.com", "Owner")
        member = team.members[0]

        acl = TeamAccessControl(store_dir)
        assert acl.check_permission(team.id, member.id, TeamPermission.READ_KEYS) is True

    def test_check_permission_not_member(self, store_dir):
        store = TeamStore(store_dir)
        team = store.create_team("Test Team", "owner@example.com", "Owner")

        acl = TeamAccessControl(store_dir)
        assert acl.check_permission(team.id, "nonexistent", TeamPermission.READ_KEYS) is False

    def test_can_manage_users(self, store_dir):
        store = TeamStore(store_dir)
        team = store.create_team("Test Team", "owner@example.com", "Owner")
        member = team.members[0]

        acl = TeamAccessControl(store_dir)
        assert acl.can_manage_users(team.id, member.id) is True
