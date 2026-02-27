"""Audit logging for auth module"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any


class AuditEventType(str, Enum):
    KEY_ACCESSED = "key_accessed"
    KEY_ADDED = "key_added"
    KEY_REMOVED = "key_removed"
    KEY_ROTATED = "key_rotated"
    KEY_SCOPE_CHANGED = "key_scope_changed"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    PROFILE_UPDATED = "profile_updated"
    PROFILE_CREATED = "profile_created"
    CONFIG_CHANGED = "config_changed"


@dataclass
class AuditEvent:
    id: str
    timestamp: str
    event_type: AuditEventType
    key_id: str | None = None
    key_fingerprint: str | None = None
    provider: str | None = None
    user_email: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "key_id": self.key_id,
            "key_fingerprint": self.key_fingerprint,
            "provider": self.provider,
            "user_email": self.user_email,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "details": self.details,
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEvent":
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            event_type=AuditEventType(data["event_type"]),
            key_id=data.get("key_id"),
            key_fingerprint=data.get("key_fingerprint"),
            provider=data.get("provider"),
            user_email=data.get("user_email"),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            details=data.get("details", {}),
            success=data.get("success", True),
        )


class AuditLogger:
    def __init__(self, store_dir: str | None = None):
        if store_dir:
            self.store_dir = Path(store_dir)
        else:
            default_dir = os.environ.get("SCHOLARDEVCLAW_AUTH_DIR")
            if default_dir:
                self.store_dir = Path(default_dir)
            else:
                self.store_dir = Path.home() / ".scholardevclaw"

        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.audit_file = self.store_dir / "audit.log"

    def log(
        self,
        event_type: AuditEventType,
        key_id: str | None = None,
        key_fingerprint: str | None = None,
        provider: str | None = None,
        user_email: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        details: dict[str, Any] | None = None,
        success: bool = True,
    ) -> AuditEvent:
        """Log an audit event.

        SECURITY: Accepts only key_fingerprint, never raw key material.
        Callers must pre-compute fingerprints via key.get_fingerprint().
        """
        import secrets

        event = AuditEvent(
            id=f"evt_{secrets.token_hex(8)}",
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            key_id=key_id,
            key_fingerprint=key_fingerprint,
            provider=provider,
            user_email=user_email,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
            success=success,
        )

        self._append_event(event)
        return event

    def _append_event(self, event: AuditEvent) -> None:
        is_new = not self.audit_file.exists()
        with open(self.audit_file, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")
        if is_new:
            try:
                os.chmod(self.audit_file, 0o600)
            except OSError:
                pass

    def get_events(
        self,
        key_id: str | None = None,
        event_type: AuditEventType | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        if not self.audit_file.exists():
            return []

        events = []
        with open(self.audit_file) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    event = AuditEvent.from_dict(data)

                    if key_id and event.key_id != key_id:
                        continue
                    if event_type and event.event_type != event_type:
                        continue

                    events.append(event)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def get_key_usage_history(self, key_id: str, limit: int = 50) -> list[AuditEvent]:
        return self.get_events(key_id=key_id, limit=limit)

    def get_failed_logins(self, limit: int = 20) -> list[AuditEvent]:
        return self.get_events(event_type=AuditEventType.LOGIN_FAILED, limit=limit)

    def clear_old_events(self, days: int = 90) -> int:
        if not self.audit_file.exists():
            return 0

        cutoff = datetime.now() - timedelta(days=days)
        remaining_events = []

        with open(self.audit_file) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    event = AuditEvent.from_dict(data)
                    event_time = datetime.fromisoformat(event.timestamp)
                    if event_time > cutoff:
                        remaining_events.append(line)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        with open(self.audit_file, "w") as f:
            f.writelines(remaining_events)
        try:
            os.chmod(self.audit_file, 0o600)
        except OSError:
            pass

        return len(remaining_events)
