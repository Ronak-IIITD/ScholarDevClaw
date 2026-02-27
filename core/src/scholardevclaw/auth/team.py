"""Team and multi-user support for shared credential management.

Supports roles (admin, developer, viewer), team workspaces, and shared key access.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class TeamRole(Enum):
    """Team member roles."""

    ADMIN = "admin"
    DEVELOPER = "developer"
    VIEWER = "viewer"


class TeamPermission(Enum):
    """Granular team permissions."""

    READ_KEYS = "read_keys"
    WRITE_KEYS = "write_keys"
    ROTATE_KEYS = "rotate_keys"
    DELETE_KEYS = "delete_keys"
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT = "view_audit"
    EXPORT_CONFIG = "export_config"
    MANAGE_SETTINGS = "manage_settings"


ROLE_PERMISSIONS: dict[TeamRole, set[TeamPermission]] = {
    TeamRole.ADMIN: {
        TeamPermission.READ_KEYS,
        TeamPermission.WRITE_KEYS,
        TeamPermission.ROTATE_KEYS,
        TeamPermission.DELETE_KEYS,
        TeamPermission.MANAGE_USERS,
        TeamPermission.VIEW_AUDIT,
        TeamPermission.EXPORT_CONFIG,
        TeamPermission.MANAGE_SETTINGS,
    },
    TeamRole.DEVELOPER: {
        TeamPermission.READ_KEYS,
        TeamPermission.WRITE_KEYS,
        TeamPermission.ROTATE_KEYS,
        TeamPermission.VIEW_AUDIT,
    },
    TeamRole.VIEWER: {
        TeamPermission.READ_KEYS,
        TeamPermission.VIEW_AUDIT,
    },
}


@dataclass
class TeamMember:
    """A team member."""

    id: str
    email: str
    name: str
    role: TeamRole
    joined_at: str
    last_active: str | None = None
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "role": self.role.value,
            "joined_at": self.joined_at,
            "last_active": self.last_active,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TeamMember:
        return cls(
            id=data["id"],
            email=data["email"],
            name=data["name"],
            role=TeamRole(data["role"]),
            joined_at=data["joined_at"],
            last_active=data.get("last_active"),
            is_active=data.get("is_active", True),
        )

    def has_permission(self, permission: TeamPermission) -> bool:
        """Check if member has a specific permission."""
        return permission in ROLE_PERMISSIONS.get(self.role, set())


@dataclass
class Team:
    """A team with shared credentials."""

    id: str
    name: str
    created_at: str
    owner_id: str
    members: list[TeamMember] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "owner_id": self.owner_id,
            "members": [m.to_dict() for m in self.members],
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Team:
        return cls(
            id=data["id"],
            name=data["name"],
            created_at=data["created_at"],
            owner_id=data["owner_id"],
            members=[TeamMember.from_dict(m) for m in data.get("members", [])],
            settings=data.get("settings", {}),
        )

    def add_member(self, email: str, name: str, role: TeamRole = TeamRole.DEVELOPER) -> TeamMember:
        """Add a member to the team."""
        member = TeamMember(
            id=f"member_{secrets.token_hex(8)}",
            email=email,
            name=name,
            role=role,
            joined_at=datetime.now().isoformat(),
        )
        self.members.append(member)
        return member

    def remove_member(self, member_id: str) -> bool:
        """Remove a member from the team."""
        for i, m in enumerate(self.members):
            if m.id == member_id:
                self.members.pop(i)
                return True
        return False

    def get_member(self, member_id: str) -> TeamMember | None:
        """Get a team member by ID."""
        return next((m for m in self.members if m.id == member_id), None)

    def get_member_by_email(self, email: str) -> TeamMember | None:
        """Get a team member by email."""
        return next((m for m in self.members if m.email == email), None)


@dataclass
class TeamInvite:
    """Invitation to join a team."""

    id: str
    team_id: str
    email: str
    role: TeamRole
    invited_by: str
    created_at: str
    expires_at: str
    accepted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "team_id": self.team_id,
            "email": self.email,
            "role": self.role.value,
            "invited_by": self.invited_by,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "accepted": self.accepted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TeamInvite:
        return cls(
            id=data["id"],
            team_id=data["team_id"],
            email=data["email"],
            role=TeamRole(data["role"]),
            invited_by=data["invited_by"],
            created_at=data["created_at"],
            expires_at=data["expires_at"],
            accepted=data.get("accepted", False),
        )


class TeamStore:
    """Manages teams and memberships."""

    TEAMS_FILE = "teams.json"
    INVITES_FILE = "team_invites.json"

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.store_dir, 0o700)
        self.teams_file = self.store_dir / self.TEAMS_FILE
        self.invites_file = self.store_dir / self.INVITES_FILE

    def create_team(self, name: str, owner_email: str, owner_name: str) -> Team:
        """Create a new team."""
        from .types import APIKey

        team = Team(
            id=f"team_{secrets.token_hex(8)}",
            name=name,
            created_at=datetime.now().isoformat(),
            owner_id=f"user_{secrets.token_hex(8)}",
        )
        team.add_member(owner_email, owner_name, TeamRole.ADMIN)

        teams = self._load_teams()
        teams[team.id] = team.to_dict()
        self._save_teams(teams)

        return team

    def get_team(self, team_id: str) -> Team | None:
        """Get a team by ID."""
        teams = self._load_teams()
        data = teams.get(team_id)
        return Team.from_dict(data) if data else None

    def update_team(self, team: Team) -> None:
        """Update a team."""
        teams = self._load_teams()
        teams[team.id] = team.to_dict()
        self._save_teams(teams)

    def delete_team(self, team_id: str) -> bool:
        """Delete a team."""
        teams = self._load_teams()
        if team_id in teams:
            del teams[team_id]
            self._save_teams(teams)

            # Also delete invites for this team
            invites = self._load_invites()
            invites = {k: v for k, v in invites.items() if v.get("team_id") != team_id}
            self._save_invites(invites)
            return True
        return False

    def list_teams(self) -> list[Team]:
        """List all teams."""
        teams = self._load_teams()
        return [Team.from_dict(d) for d in teams.values()]

    def invite_member(
        self, team_id: str, email: str, role: TeamRole, invited_by: str, expires_in_days: int = 7
    ) -> TeamInvite:
        """Create an invite for a team member."""
        from datetime import timedelta

        invite = TeamInvite(
            id=f"invite_{secrets.token_hex(8)}",
            team_id=team_id,
            email=email,
            role=role,
            invited_by=invited_by,
            created_at=datetime.now().isoformat(),
            expires_at=(datetime.now() + timedelta(days=expires_in_days)).isoformat(),
        )

        invites = self._load_invites()
        invites[invite.id] = invite.to_dict()
        self._save_invites(invites)

        return invite

    def accept_invite(self, invite_id: str) -> bool:
        """Accept an invitation."""
        invites = self._load_invites()
        if invite_id not in invites:
            return False

        invite_data = invites[invite_id]
        if datetime.fromisoformat(invite_data["expires_at"]) < datetime.now():
            return False

        invite_data["accepted"] = True
        invites[invite_id] = invite_data
        self._save_invites(invites)

        team = self.get_team(invite_data["team_id"])
        if team:
            team.add_member(
                invite_data["email"], invite_data["email"], TeamRole(invite_data["role"])
            )
            self.update_team(team)

        return True

    def get_pending_invites(self, email: str) -> list[TeamInvite]:
        """Get pending invites for an email."""
        invites = self._load_invites()
        result = []
        for inv in invites.values():
            if inv["email"] == email and not inv.get("accepted", False):
                if datetime.fromisoformat(inv["expires_at"]) > datetime.now():
                    result.append(TeamInvite.from_dict(inv))
        return result

    def revoke_invite(self, invite_id: str) -> bool:
        """Revoke an invitation."""
        invites = self._load_invites()
        if invite_id in invites:
            del invites[invite_id]
            self._save_invites(invites)
            return True
        return False

    def _load_teams(self) -> dict[str, Any]:
        if not self.teams_file.exists():
            return {}
        try:
            return json.loads(self.teams_file.read_text())
        except json.JSONDecodeError:
            return {}

    def _save_teams(self, teams: dict[str, Any]) -> None:
        self.teams_file.write_text(json.dumps(teams, indent=2))
        os.chmod(self.teams_file, 0o600)

    def _load_invites(self) -> dict[str, Any]:
        if not self.invites_file.exists():
            return {}
        try:
            return json.loads(self.invites_file.read_text())
        except json.JSONDecodeError:
            return {}

    def _save_invites(self, invites: dict[str, Any]) -> None:
        self.invites_file.write_text(json.dumps(invites, indent=2))
        os.chmod(self.invites_file, 0o600)


class TeamAccessControl:
    """Check permissions for team actions."""

    def __init__(self, store_dir: str | Path):
        self.team_store = TeamStore(store_dir)

    def check_permission(self, team_id: str, user_id: str, permission: TeamPermission) -> bool:
        """Check if a user has a specific permission in a team."""
        team = self.team_store.get_team(team_id)
        if not team:
            return False

        member = team.get_member(user_id)
        if not member or not member.is_active:
            return False

        return member.has_permission(permission)

    def can_access_key(self, team_id: str, user_id: str, key_id: str, write: bool = False) -> bool:
        """Check if user can access a key."""
        perm = TeamPermission.WRITE_KEYS if write else TeamPermission.READ_KEYS
        return self.check_permission(team_id, user_id, perm)

    def can_manage_users(self, team_id: str, user_id: str) -> bool:
        """Check if user can manage team members."""
        return self.check_permission(team_id, user_id, TeamPermission.MANAGE_USERS)

    def can_view_audit(self, team_id: str, user_id: str) -> bool:
        """Check if user can view audit logs."""
        return self.check_permission(team_id, user_id, TeamPermission.VIEW_AUDIT)
