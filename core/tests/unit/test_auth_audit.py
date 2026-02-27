"""Tests for audit logging"""

import tempfile
import json

import pytest


class TestAuditLogging:
    """Tests for audit logging functionality"""

    @pytest.fixture
    def temp_audit_dir(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("SCHOLARDEVCLAW_AUTH_DIR", tmpdir)
            yield tmpdir

    def test_audit_logger_initialization(self, temp_audit_dir):
        from scholardevclaw.auth.audit import AuditLogger, AuditEventType

        logger = AuditLogger(temp_audit_dir)
        logger.log(event_type=AuditEventType.KEY_ADDED, provider="test")
        assert logger.audit_file.exists()

    def test_audit_log_event(self, temp_audit_dir):
        from scholardevclaw.auth.audit import AuditLogger, AuditEventType

        logger = AuditLogger(temp_audit_dir)
        event = logger.log(
            event_type=AuditEventType.KEY_ADDED,
            key_id="key_123",
            key_fingerprint="abcdef0123456789",
            provider="anthropic",
            details={"name": "test-key"},
        )

        assert event.event_type == AuditEventType.KEY_ADDED
        assert event.key_id == "key_123"

    def test_audit_log_persists(self, temp_audit_dir):
        from scholardevclaw.auth.audit import AuditLogger, AuditEventType

        logger = AuditLogger(temp_audit_dir)
        logger.log(
            event_type=AuditEventType.KEY_ADDED,
            key_id="key_123",
            provider="anthropic",
        )

        events = logger.get_events()
        assert len(events) == 1

    def test_get_events_by_key_id(self, temp_audit_dir):
        from scholardevclaw.auth.audit import AuditLogger, AuditEventType

        logger = AuditLogger(temp_audit_dir)
        logger.log(event_type=AuditEventType.KEY_ADDED, key_id="key_1", provider="anthropic")
        logger.log(event_type=AuditEventType.KEY_ACCESSED, key_id="key_2", provider="openai")
        logger.log(event_type=AuditEventType.KEY_ACCESSED, key_id="key_1", provider="anthropic")

        events = logger.get_events(key_id="key_1")
        assert len(events) == 2

    def test_get_events_by_type(self, temp_audit_dir):
        from scholardevclaw.auth.audit import AuditLogger, AuditEventType

        logger = AuditLogger(temp_audit_dir)
        logger.log(event_type=AuditEventType.KEY_ADDED, key_id="key_1")
        logger.log(event_type=AuditEventType.KEY_ACCESSED, key_id="key_2")
        logger.log(event_type=AuditEventType.KEY_ADDED, key_id="key_3")

        events = logger.get_events(event_type=AuditEventType.KEY_ADDED)
        assert len(events) == 2

    def test_get_failed_logins(self, temp_audit_dir):
        from scholardevclaw.auth.audit import AuditLogger, AuditEventType

        logger = AuditLogger(temp_audit_dir)
        logger.log(
            event_type=AuditEventType.LOGIN_FAILED, success=False, details={"reason": "invalid key"}
        )
        logger.log(event_type=AuditEventType.LOGIN_SUCCESS, success=True)
        logger.log(
            event_type=AuditEventType.LOGIN_FAILED, success=False, details={"reason": "expired"}
        )

        failed = logger.get_failed_logins()
        assert len(failed) == 2
        assert all(not e.success for e in failed)

    def test_audit_event_serialization(self, temp_audit_dir):
        from scholardevclaw.auth.audit import AuditLogger, AuditEventType, AuditEvent

        logger = AuditLogger(temp_audit_dir)
        event = logger.log(
            event_type=AuditEventType.KEY_ROTATED,
            key_id="key_123",
            provider="anthropic",
            details={"reason": "scheduled"},
        )

        data = event.to_dict()
        restored = AuditEvent.from_dict(data)
        assert restored.event_type == AuditEventType.KEY_ROTATED
        assert restored.details["reason"] == "scheduled"

    def test_audit_key_fingerprint(self, temp_audit_dir):
        from scholardevclaw.auth.audit import AuditLogger, AuditEventType

        logger = AuditLogger(temp_audit_dir)
        import hashlib

        fp = hashlib.sha256(b"sk_secret_key_12345").hexdigest()
        event = logger.log(
            event_type=AuditEventType.KEY_ACCESSED,
            key_fingerprint=fp,
        )

        assert event.key_fingerprint is not None
        assert "sk_secret" not in event.key_fingerprint
        assert len(event.key_fingerprint) == 64


class TestAuthStoreAuditIntegration:
    """Test that auth store properly logs events"""

    @pytest.fixture
    def temp_auth_dir(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("SCHOLARDEVCLAW_AUTH_DIR", tmpdir)
            yield tmpdir

    def test_add_key_logs_event(self, temp_auth_dir):
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_dir)
        store.add_api_key("sk_test", "test-key", AuthProvider.ANTHROPIC)

        events = store._audit.get_events()
        assert len(events) >= 1
        assert events[0].event_type.value == "key_added"

    def test_remove_key_logs_event(self, temp_auth_dir):
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_dir)
        key = store.add_api_key("sk_test", "test-key", AuthProvider.ANTHROPIC)
        store.remove_api_key(key.id)

        events = store._audit.get_events(event_type=None)
        event_types = [e.event_type.value for e in events]
        assert "key_removed" in event_types

    def test_rotate_key_logs_event(self, temp_auth_dir):
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_dir)
        key = store.add_api_key("sk_old", "test-key", AuthProvider.ANTHROPIC)
        store.rotate_api_key(key.id, "sk_new", reason="Test rotation")

        events = store._audit.get_events()
        event_types = [e.event_type.value for e in events]
        assert "key_rotated" in event_types
