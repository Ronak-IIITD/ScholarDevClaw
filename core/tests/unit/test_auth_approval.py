"""Tests for key request/approval workflow."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from scholardevclaw.auth.approval import (
    RequestStatus,
    RequestType,
    KeyRequest,
    ApprovalNotification,
    ApprovalWorkflow,
    RequestValidator,
)


class TestRequestStatus:
    def test_statuses(self):
        assert RequestStatus.PENDING.value == "pending"
        assert RequestStatus.APPROVED.value == "approved"
        assert RequestStatus.REJECTED.value == "rejected"


class TestRequestType:
    def test_types(self):
        assert RequestType.NEW_KEY.value == "new_key"
        assert RequestType.KEY_ROTATION.value == "key_rotation"


class TestKeyRequest:
    def test_creation(self):
        request = KeyRequest(
            id="req_1",
            team_id="team_1",
            requester_id="user_1",
            requester_email="user@example.com",
            request_type=RequestType.NEW_KEY,
            provider="anthropic",
            reason="Need access for project",
            requested_at=datetime.now().isoformat(),
            expires_at=(datetime.now() + timedelta(days=7)).isoformat(),
        )
        assert request.requester_email == "user@example.com"
        assert request.status == RequestStatus.PENDING

    def test_is_pending(self):
        request = KeyRequest(
            id="req_1",
            team_id="team_1",
            requester_id="user_1",
            requester_email="user@example.com",
            request_type=RequestType.NEW_KEY,
            provider="anthropic",
            reason="Need access",
            requested_at=datetime.now().isoformat(),
            expires_at=(datetime.now() + timedelta(days=7)).isoformat(),
        )
        assert request.is_pending() is True

    def test_is_expired(self):
        request = KeyRequest(
            id="req_1",
            team_id="team_1",
            requester_id="user_1",
            requester_email="user@example.com",
            request_type=RequestType.NEW_KEY,
            provider="anthropic",
            reason="Need access",
            requested_at=datetime.now().isoformat(),
            expires_at=(datetime.now() - timedelta(days=1)).isoformat(),
        )
        assert request.is_expired() is True


class TestApprovalWorkflow:
    @pytest.fixture
    def store_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_create_request(self, store_dir):
        workflow = ApprovalWorkflow(store_dir)
        request = workflow.create_request(
            team_id="team_1",
            requester_id="user_1",
            requester_email="user@example.com",
            request_type=RequestType.NEW_KEY,
            provider="anthropic",
            reason="Need access for project",
        )
        assert request.status == RequestStatus.PENDING
        assert request.team_id == "team_1"

    def test_approve_request(self, store_dir):
        workflow = ApprovalWorkflow(store_dir)
        request = workflow.create_request(
            team_id="team_1",
            requester_id="user_1",
            requester_email="user@example.com",
            request_type=RequestType.NEW_KEY,
            provider="anthropic",
            reason="Need access",
        )
        approved = workflow.approve_request(request.id, "approver_1")
        assert approved is not None
        assert approved.status == RequestStatus.APPROVED
        assert approved.approver_id == "approver_1"

    def test_reject_request(self, store_dir):
        workflow = ApprovalWorkflow(store_dir)
        request = workflow.create_request(
            team_id="team_1",
            requester_id="user_1",
            requester_email="user@example.com",
            request_type=RequestType.NEW_KEY,
            provider="anthropic",
            reason="Need access",
        )
        rejected = workflow.reject_request(request.id, "approver_1", "Not authorized")
        assert rejected is not None
        assert rejected.status == RequestStatus.REJECTED
        assert rejected.rejection_reason == "Not authorized"

    def test_cancel_request(self, store_dir):
        workflow = ApprovalWorkflow(store_dir)
        request = workflow.create_request(
            team_id="team_1",
            requester_id="user_1",
            requester_email="user@example.com",
            request_type=RequestType.NEW_KEY,
            provider="anthropic",
            reason="Need access",
        )
        cancelled = workflow.cancel_request(request.id, "user_1")
        assert cancelled is True

    def test_cancel_request_not_requester(self, store_dir):
        workflow = ApprovalWorkflow(store_dir)
        request = workflow.create_request(
            team_id="team_1",
            requester_id="user_1",
            requester_email="user@example.com",
            request_type=RequestType.NEW_KEY,
            provider="anthropic",
            reason="Need access",
        )
        cancelled = workflow.cancel_request(request.id, "other_user")
        assert cancelled is False

    def test_get_pending_requests(self, store_dir):
        workflow = ApprovalWorkflow(store_dir)
        workflow.create_request(
            team_id="team_1",
            requester_id="user_1",
            requester_email="user1@example.com",
            request_type=RequestType.NEW_KEY,
            provider="anthropic",
            reason="Need access",
        )
        workflow.create_request(
            team_id="team_1",
            requester_id="user_2",
            requester_email="user2@example.com",
            request_type=RequestType.NEW_KEY,
            provider="openai",
            reason="Need access",
        )
        pending = workflow.get_pending_requests("team_1")
        assert len(pending) == 2

    def test_expire_old_requests(self, store_dir):
        workflow = ApprovalWorkflow(store_dir)
        # Create a request that's already expired
        request_data = {
            "id": "req_expired",
            "team_id": "team_1",
            "requester_id": "user_1",
            "requester_email": "user@example.com",
            "request_type": "new_key",
            "provider": "anthropic",
            "reason": "Need access",
            "requested_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() - timedelta(days=1)).isoformat(),
            "status": "pending",
        }
        requests_file = store_dir / "key_requests.json"
        requests_file.write_text(json.dumps({"req_expired": request_data}))

        expired = workflow.expire_old_requests()
        assert expired == 1


class TestRequestValidator:
    @pytest.fixture
    def store_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_can_request_allowed(self, store_dir):
        workflow = ApprovalWorkflow(store_dir)
        validator = RequestValidator(workflow)
        can, msg = validator.can_request("team_1", "user_1", "anthropic")
        assert can is True

    def test_can_request_duplicate_pending(self, store_dir):
        workflow = ApprovalWorkflow(store_dir)
        workflow.create_request(
            team_id="team_1",
            requester_id="user_1",
            requester_email="user@example.com",
            request_type=RequestType.NEW_KEY,
            provider="anthropic",
            reason="Need access",
        )
        validator = RequestValidator(workflow)
        can, msg = validator.can_request("team_1", "user_1", "anthropic")
        assert can is False
        assert "pending" in msg.lower()

    def test_validate_reason_too_short(self, store_dir):
        workflow = ApprovalWorkflow(store_dir)
        validator = RequestValidator(workflow)
        can, msg = validator.validate_request_reason("short")
        assert can is False

    def test_validate_reason_valid(self, store_dir):
        workflow = ApprovalWorkflow(store_dir)
        validator = RequestValidator(workflow)
        can, msg = validator.validate_request_reason("I need this for a project")
        assert can is True
