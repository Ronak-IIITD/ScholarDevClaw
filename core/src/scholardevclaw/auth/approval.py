"""Key request and approval workflow.

Allows team members to request API keys and get admin approval.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any


class RequestStatus(Enum):
    """Status of a key request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class RequestType(Enum):
    """Type of key request."""

    NEW_KEY = "new_key"
    KEY_ROTATION = "key_rotation"
    KEY_RENEWAL = "key_renewal"
    ACCESS_EXTENSION = "access_extension"


@dataclass
class KeyRequest:
    """A request for an API key."""

    id: str
    team_id: str
    requester_id: str
    requester_email: str
    request_type: RequestType
    provider: str
    reason: str
    requested_at: str
    expires_at: str
    status: RequestStatus = RequestStatus.PENDING
    approver_id: str | None = None
    approved_at: str | None = None
    rejection_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "team_id": self.team_id,
            "requester_id": self.requester_id,
            "requester_email": self.requester_email,
            "request_type": self.request_type.value,
            "provider": self.provider,
            "reason": self.reason,
            "requested_at": self.requested_at,
            "expires_at": self.expires_at,
            "status": self.status.value,
            "approver_id": self.approver_id,
            "approved_at": self.approved_at,
            "rejection_reason": self.rejection_reason,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KeyRequest:
        return cls(
            id=data["id"],
            team_id=data["team_id"],
            requester_id=data["requester_id"],
            requester_email=data["requester_email"],
            request_type=RequestType(data["request_type"]),
            provider=data["provider"],
            reason=data["reason"],
            requested_at=data["requested_at"],
            expires_at=data["expires_at"],
            status=RequestStatus(data["status"]),
            approver_id=data.get("approver_id"),
            approved_at=data.get("approved_at"),
            rejection_reason=data.get("rejection_reason"),
            metadata=data.get("metadata", {}),
        )

    def is_pending(self) -> bool:
        """Check if request is still pending."""
        return self.status == RequestStatus.PENDING

    def is_expired(self) -> bool:
        """Check if request has expired."""
        return datetime.fromisoformat(self.expires_at) < datetime.now()


@dataclass
class ApprovalNotification:
    """Notification about a request."""

    id: str
    request_id: str
    recipient_id: str
    type: str  # "new_request", "approved", "rejected"
    message: str
    created_at: str
    read: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "request_id": self.request_id,
            "recipient_id": self.recipient_id,
            "type": self.type,
            "message": self.message,
            "created_at": self.created_at,
            "read": self.read,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApprovalNotification:
        return cls(
            id=data["id"],
            request_id=data["request_id"],
            recipient_id=data["recipient_id"],
            type=data["type"],
            message=data["message"],
            created_at=data["created_at"],
            read=data.get("read", False),
        )


class ApprovalWorkflow:
    """Manage key request and approval workflow."""

    REQUESTS_FILE = "key_requests.json"
    NOTIFICATIONS_FILE = "notifications.json"

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.store_dir, 0o700)
        self.requests_file = self.store_dir / self.REQUESTS_FILE
        self.notifications_file = self.store_dir / self.NOTIFICATIONS_FILE

    def create_request(
        self,
        team_id: str,
        requester_id: str,
        requester_email: str,
        request_type: RequestType,
        provider: str,
        reason: str,
        expires_in_days: int = 7,
        metadata: dict[str, Any] | None = None,
    ) -> KeyRequest:
        """Create a new key request."""
        now = datetime.now()
        request = KeyRequest(
            id=f"req_{secrets.token_hex(8)}",
            team_id=team_id,
            requester_id=requester_id,
            requester_email=requester_email,
            request_type=request_type,
            provider=provider,
            reason=reason,
            requested_at=now.isoformat(),
            expires_at=(now + timedelta(days=expires_in_days)).isoformat(),
            metadata=metadata or {},
        )

        requests = self._load_requests()
        requests[request.id] = request.to_dict()
        self._save_requests(requests)

        return request

    def approve_request(self, request_id: str, approver_id: str) -> KeyRequest | None:
        """Approve a key request."""
        requests = self._load_requests()
        if request_id not in requests:
            return None

        request = KeyRequest.from_dict(requests[request_id])

        if not request.is_pending():
            return None

        request.status = RequestStatus.APPROVED
        request.approver_id = approver_id
        request.approved_at = datetime.now().isoformat()
        requests[request.id] = request.to_dict()
        self._save_requests(requests)

        # Create notification
        self._create_notification(
            request.requester_id,
            "approved",
            f"Your {request.request_type.value} request for {request.provider} has been approved",
        )

        return request

    def reject_request(self, request_id: str, approver_id: str, reason: str) -> KeyRequest | None:
        """Reject a key request."""
        requests = self._load_requests()
        if request_id not in requests:
            return None

        request = KeyRequest.from_dict(requests[request_id])

        if not request.is_pending():
            return None

        request.status = RequestStatus.REJECTED
        request.approver_id = approver_id
        request.rejection_reason = reason
        requests[request.id] = request.to_dict()
        self._save_requests(requests)

        # Create notification
        self._create_notification(
            request.requester_id,
            "rejected",
            f"Your {request.request_type.value} request for {request.provider} was rejected: {reason}",
        )

        return request

    def cancel_request(self, request_id: str, user_id: str) -> bool:
        """Cancel a pending request."""
        requests = self._load_requests()
        if request_id not in requests:
            return False

        request = KeyRequest.from_dict(requests[request_id])

        if request.requester_id != user_id:
            return False

        if not request.is_pending():
            return False

        request.status = RequestStatus.CANCELLED
        requests[request.id] = request.to_dict()
        self._save_requests(requests)

        return True

    def get_request(self, request_id: str) -> KeyRequest | None:
        """Get a request by ID."""
        requests = self._load_requests()
        data = requests.get(request_id)
        return KeyRequest.from_dict(data) if data else None

    def get_pending_requests(self, team_id: str | None = None) -> list[KeyRequest]:
        """Get all pending requests."""
        requests = self._load_requests()
        result = []
        for r in requests.values():
            req = KeyRequest.from_dict(r)
            if req.is_pending():
                if team_id is None or req.team_id == team_id:
                    result.append(req)
        return sorted(result, key=lambda x: x.requested_at, reverse=True)

    def get_requests_by_requester(self, requester_id: str) -> list[KeyRequest]:
        """Get all requests by a user."""
        requests = self._load_requests()
        return sorted(
            [
                KeyRequest.from_dict(r)
                for r in requests.values()
                if r["requester_id"] == requester_id
            ],
            key=lambda x: x.requested_at,
            reverse=True,
        )

    def get_requests_by_team(self, team_id: str) -> list[KeyRequest]:
        """Get all requests for a team."""
        requests = self._load_requests()
        return sorted(
            [KeyRequest.from_dict(r) for r in requests.values() if r["team_id"] == team_id],
            key=lambda x: x.requested_at,
            reverse=True,
        )

    def expire_old_requests(self) -> int:
        """Mark expired requests as expired."""
        requests = self._load_requests()
        expired_count = 0

        for rid, r in requests.items():
            req = KeyRequest.from_dict(r)
            if req.is_pending() and req.is_expired():
                req.status = RequestStatus.EXPIRED
                requests[rid] = req.to_dict()
                expired_count += 1

        if expired_count > 0:
            self._save_requests(requests)

        return expired_count

    def _create_notification(self, recipient_id: str, notif_type: str, message: str) -> None:
        """Create a notification."""
        notif = ApprovalNotification(
            id=f"notif_{secrets.token_hex(8)}",
            request_id="",
            recipient_id=recipient_id,
            type=notif_type,
            message=message,
            created_at=datetime.now().isoformat(),
        )

        notifications = self._load_notifications()
        notifications[notif.id] = notif.to_dict()
        self._save_notifications(notifications)

    def get_notifications(
        self, user_id: str, unread_only: bool = False
    ) -> list[ApprovalNotification]:
        """Get notifications for a user."""
        notifications = self._load_notifications()
        result = []
        for n in notifications.values():
            if n["recipient_id"] == user_id:
                if unread_only and n.get("read", False):
                    continue
                result.append(ApprovalNotification.from_dict(n))
        return sorted(result, key=lambda x: x.created_at, reverse=True)

    def mark_notification_read(self, notification_id: str) -> bool:
        """Mark a notification as read."""
        notifications = self._load_notifications()
        if notification_id in notifications:
            notifications[notification_id]["read"] = True
            self._save_notifications(notifications)
            return True
        return False

    def _load_requests(self) -> dict[str, Any]:
        if not self.requests_file.exists():
            return {}
        try:
            return json.loads(self.requests_file.read_text())
        except json.JSONDecodeError:
            return {}

    def _save_requests(self, requests: dict[str, Any]) -> None:
        self.requests_file.write_text(json.dumps(requests, indent=2))
        os.chmod(self.requests_file, 0o600)

    def _load_notifications(self) -> dict[str, Any]:
        if not self.notifications_file.exists():
            return {}
        try:
            return json.loads(self.notifications_file.read_text())
        except json.JSONDecodeError:
            return {}

    def _save_notifications(self, notifications: dict[str, Any]) -> None:
        self.notifications_file.write_text(json.dumps(notifications, indent=2))
        os.chmod(self.notifications_file, 0o600)


class RequestValidator:
    """Validate key requests against policies."""

    def __init__(self, workflow: ApprovalWorkflow):
        self.workflow = workflow

    def can_request(self, team_id: str, requester_id: str, provider: str) -> tuple[bool, str]:
        """Check if user can make a request."""
        pending = self.workflow.get_pending_requests(team_id)

        # Check for duplicate pending requests
        for req in pending:
            if req.requester_id == requester_id and req.provider == provider:
                if req.is_pending():
                    return False, f"You already have a pending request for {provider}"

        # Check request rate limit (max 3 requests per day)
        today = datetime.now().date()
        daily_requests = [
            r
            for r in pending
            if r.requester_id == requester_id and r.requested_at[:10] == today.isoformat()
        ]
        if len(daily_requests) >= 3:
            return False, "You have reached the maximum number of requests per day (3)"

        return True, "OK"

    def validate_request_reason(self, reason: str) -> tuple[bool, str]:
        """Validate the reason for a request."""
        if not reason or len(reason.strip()) < 10:
            return False, "Reason must be at least 10 characters"

        if len(reason) > 500:
            return False, "Reason must be less than 500 characters"

        return True, "OK"
