import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from scholardevclaw.auth import AuthStore
from scholardevclaw.auth.types import (
    APIKey,
    AuthConfig,
    AuthProvider,
    AuthStatus,
    KeyRotationEntry,
    KeyScope,
    SubscriptionTier,
    UserProfile,
)


@pytest.fixture
def temp_auth_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def store(temp_auth_dir):
    return AuthStore(temp_auth_dir)


class TestAPIKey:
    def test_generate_key_prefix(self):
        key = APIKey.generate_key("sk")
        assert key.startswith("sk_")
        assert len(key) > 10

    def test_mask_short_key(self):
        api_key = APIKey(
            id="key_1",
            name="short",
            provider=AuthProvider.CUSTOM,
            key="abcd",
        )
        assert api_key.mask() == "***"

    def test_mask_long_key(self):
        api_key = APIKey(
            id="key_2",
            name="long",
            provider=AuthProvider.CUSTOM,
            key="sk_1234567890abcdef",
        )
        masked = api_key.mask()
        assert masked.startswith("sk_12345")
        assert masked.endswith("cdef")
        assert "..." in masked

    def test_is_valid_inactive(self):
        api_key = APIKey(
            id="key_3",
            name="inactive",
            provider=AuthProvider.CUSTOM,
            key="sk_test",
            is_active=False,
        )
        assert api_key.is_valid() is False

    def test_is_valid_expired(self):
        expired = (datetime.now() - timedelta(days=1)).isoformat()
        api_key = APIKey(
            id="key_4",
            name="expired",
            provider=AuthProvider.CUSTOM,
            key="sk_test",
            expires_at=expired,
        )
        assert api_key.is_valid() is False

    def test_is_valid_future_expiry(self):
        future = (datetime.now() + timedelta(days=1)).isoformat()
        api_key = APIKey(
            id="key_5",
            name="valid",
            provider=AuthProvider.CUSTOM,
            key="sk_test",
            expires_at=future,
        )
        assert api_key.is_valid() is True

    def test_from_dict_roundtrip(self):
        data = {
            "id": "key_6",
            "name": "roundtrip",
            "provider": "openai",
            "key": "sk_test",
            "created_at": "2026-02-20T10:00:00",
            "last_used": "2026-02-21T12:00:00",
            "expires_at": None,
            "is_active": True,
            "metadata": {"scope": "test"},
        }
        api_key = APIKey.from_dict(data)
        assert api_key.provider == AuthProvider.OPENAI
        assert api_key.to_dict()["name"] == "roundtrip"

    def test_validate_key_format_anthropic_valid(self):
        is_valid, msg = APIKey.validate_key_format("sk-ant-abcdef", AuthProvider.ANTHROPIC)
        assert is_valid is True

    def test_validate_key_format_anthropic_invalid(self):
        is_valid, msg = APIKey.validate_key_format("sk-wrong", AuthProvider.ANTHROPIC)
        assert is_valid is False
        assert "sk-ant" in msg

    def test_validate_key_format_openai_valid(self):
        is_valid, msg = APIKey.validate_key_format("sk-testkey123", AuthProvider.OPENAI)
        assert is_valid is True

    def test_validate_key_format_openai_invalid(self):
        is_valid, msg = APIKey.validate_key_format("wrong-key", AuthProvider.OPENAI)
        assert is_valid is False
        assert "sk-" in msg

    def test_validate_key_format_github_valid(self):
        is_valid, msg = APIKey.validate_key_format("ghp_abcdef", AuthProvider.GITHUB)
        assert is_valid is True

    def test_validate_key_format_github_pat_valid(self):
        is_valid, msg = APIKey.validate_key_format("github_pat_abcdef", AuthProvider.GITHUB)
        assert is_valid is True

    def test_validate_key_format_empty(self):
        is_valid, msg = APIKey.validate_key_format("", AuthProvider.OPENAI)
        assert is_valid is False
        assert "empty" in msg.lower()

    def test_validate_key_format_too_short(self):
        is_valid, msg = APIKey.validate_key_format("abc", AuthProvider.OPENAI)
        assert is_valid is False
        assert "short" in msg.lower()

    def test_validate_key_name_valid(self):
        is_valid, msg = APIKey.is_valid_key_name("my-api-key")
        assert is_valid is True

    def test_validate_key_name_empty(self):
        is_valid, msg = APIKey.is_valid_key_name("")
        assert is_valid is False

    def test_validate_key_name_too_long(self):
        is_valid, msg = APIKey.is_valid_key_name("a" * 101)
        assert is_valid is False
        assert "long" in msg.lower()

    def test_validate_key_name_invalid_chars(self):
        is_valid, msg = APIKey.is_valid_key_name("key@#$%")
        assert is_valid is False
        assert "invalid" in msg.lower()

    def test_validate_email_valid(self):
        assert APIKey.is_valid_email("test@example.com") is True
        assert APIKey.is_valid_email("user.name+tag@domain.co.uk") is True

    def test_validate_email_invalid(self):
        assert APIKey.is_valid_email("") is False
        assert APIKey.is_valid_email("not-an-email") is False
        assert APIKey.is_valid_email("@example.com") is False
        assert APIKey.is_valid_email("test@") is False

    def test_get_fingerprint(self):
        api_key = APIKey(
            id="key_1",
            name="test",
            provider=AuthProvider.CUSTOM,
            key="secret_key_12345",
        )
        fp = api_key.get_fingerprint()
        assert len(fp) == 16
        assert isinstance(fp, str)

    def test_get_fingerprint_unique(self):
        key1 = APIKey(id="k1", name="n", provider=AuthProvider.CUSTOM, key="secret1")
        key2 = APIKey(id="k2", name="n", provider=AuthProvider.CUSTOM, key="secret2")
        assert key1.get_fingerprint() != key2.get_fingerprint()

    def test_is_valid_invalid_expires_at_format(self):
        api_key = APIKey(
            id="key_bad",
            name="bad",
            provider=AuthProvider.CUSTOM,
            key="sk_test",
            expires_at="not-a-date",
        )
        assert api_key.is_valid() is False


class TestUserProfile:
    def test_profile_roundtrip(self):
        profile = UserProfile(
            id="user_1",
            email="test@example.com",
            name="Test User",
            subscription_tier=SubscriptionTier.PRO,
        )
        data = profile.to_dict()
        restored = UserProfile.from_dict(data)
        assert restored.email == "test@example.com"
        assert restored.subscription_tier == SubscriptionTier.PRO


class TestAuthConfig:
    def test_get_active_key_default(self):
        key1 = APIKey(
            id="key_1",
            name="primary",
            provider=AuthProvider.ANTHROPIC,
            key="sk_1",
        )
        key2 = APIKey(
            id="key_2",
            name="secondary",
            provider=AuthProvider.OPENAI,
            key="sk_2",
        )
        config = AuthConfig(api_keys=[key1, key2], default_key_id="key_1")
        active = config.get_active_key()
        assert active.id == "key_1"

    def test_get_active_key_fallback_provider(self):
        key1 = APIKey(
            id="key_1",
            name="inactive",
            provider=AuthProvider.ANTHROPIC,
            key="sk_1",
            is_active=False,
        )
        key2 = APIKey(
            id="key_2",
            name="openai",
            provider=AuthProvider.OPENAI,
            key="sk_2",
        )
        config = AuthConfig(api_keys=[key1, key2], default_provider=AuthProvider.OPENAI)
        active = config.get_active_key()
        assert active.id == "key_2"

    def test_get_active_key_any_valid(self):
        key1 = APIKey(
            id="key_1",
            name="inactive",
            provider=AuthProvider.ANTHROPIC,
            key="sk_1",
            is_active=False,
        )
        key2 = APIKey(
            id="key_2",
            name="custom",
            provider=AuthProvider.CUSTOM,
            key="sk_2",
        )
        config = AuthConfig(api_keys=[key1, key2])
        active = config.get_active_key()
        assert active.id == "key_2"

    def test_get_key_missing(self):
        config = AuthConfig(api_keys=[])
        assert config.get_key("missing") is None


class TestAuthStore:
    def test_store_initialization(self, temp_auth_dir):
        store = AuthStore(temp_auth_dir)
        assert store.store_dir == Path(temp_auth_dir)
        assert store.auth_file == Path(temp_auth_dir) / "auth.json"

    def test_is_authenticated_false(self, store):
        assert store.is_authenticated() is False

    def test_add_api_key_and_status(self, store):
        key = store.add_api_key(
            key="sk_test",
            name="primary",
            provider=AuthProvider.ANTHROPIC,
        )
        status = store.get_status()
        assert status.is_authenticated is True
        assert status.active_keys == 1
        assert status.provider == "anthropic"
        assert key.id == store.get_config().default_key_id

    def test_get_api_key_env_override(self, store, monkeypatch):
        monkeypatch.setenv("SCHOLARDEVCLAW_API_KEY", "env_key")
        assert store.get_api_key() == "env_key"

    def test_get_api_key_updates_last_used(self, store):
        key = store.add_api_key(
            key="sk_test",
            name="primary",
            provider=AuthProvider.ANTHROPIC,
        )
        assert key.last_used is None
        retrieved = store.get_api_key()
        assert retrieved == "sk_test"
        updated = store.get_config().get_key(key.id)
        assert updated.last_used is not None

    def test_remove_api_key(self, store):
        key = store.add_api_key(
            key="sk_test",
            name="primary",
            provider=AuthProvider.CUSTOM,
        )
        removed = store.remove_api_key(key.id)
        assert removed is True
        assert store.get_config().get_key(key.id) is None

    def test_remove_api_key_missing(self, store):
        assert store.remove_api_key("missing") is False

    def test_set_default_key(self, store):
        key1 = store.add_api_key("sk_1", "k1", AuthProvider.CUSTOM)
        key2 = store.add_api_key("sk_2", "k2", AuthProvider.OPENAI, set_default=False)

        ok = store.set_default_key(key2.id)
        assert ok is True
        assert store.get_config().default_key_id == key2.id

    def test_set_default_key_missing(self, store):
        assert store.set_default_key("missing") is False

    def test_logout_clears_env_file(self, store, temp_auth_dir):
        store.add_api_key("sk_test", "k1", AuthProvider.CUSTOM)
        env_file = Path(temp_auth_dir) / ".env"
        assert env_file.exists()

        result = store.logout()
        assert result is True
        assert env_file.exists() is False
        assert store.is_authenticated() is False

    def test_clear_all_removes_auth_file(self, store):
        store.add_api_key("sk_test", "k1", AuthProvider.CUSTOM)
        assert store.auth_file.exists()
        store.clear_all()
        assert store.auth_file.exists() is False

    def test_invalid_auth_json_recovery(self, temp_auth_dir):
        auth_file = Path(temp_auth_dir) / "auth.json"
        auth_file.write_text("{invalid json}")
        store = AuthStore(temp_auth_dir)
        assert store.is_authenticated() is False

    def test_create_profile_and_update(self, store):
        profile = store.create_profile(email="user@test.com", name="Test")
        assert profile.email == "user@test.com"
        updated = store.update_profile(name="Updated")
        assert updated.name == "Updated"

    def test_update_profile_missing(self, store):
        assert store.update_profile(name="Missing") is None

    def test_empty_key_handling(self, store):
        key = store.add_api_key("", "empty", AuthProvider.CUSTOM)
        assert key.key == ""

    def test_unicode_in_key_name(self, store):
        key = store.add_api_key("sk_test", " ключ 🔑 ", AuthProvider.CUSTOM)
        assert key.name == " ключ 🔑 "

    def test_get_api_key_no_keys(self, store):
        assert store.get_api_key() is None

    def test_get_api_key_by_provider(self, store):
        store.add_api_key("sk_ant", "ant", AuthProvider.ANTHROPIC)
        store.add_api_key("sk_oi", "openai", AuthProvider.OPENAI)

        # By default returns default key
        default_key = store.get_api_key()
        assert default_key is not None

    def test_get_api_key_inactive_provider(self, store):
        key = store.add_api_key("sk_ant", "ant", AuthProvider.ANTHROPIC, set_default=False)
        config = store.get_config()
        config.get_key(key.id).is_active = False
        store._save_config(config)

        assert store.get_api_key() is None

    def test_auth_config_from_dict_missing_fields(self):
        data = {"api_keys": []}
        config = AuthConfig.from_dict(data)
        assert config.default_key_id is None
        assert config.api_keys == []

    def test_provider_case_sensitivity(self):
        assert AuthProvider("anthropic") == AuthProvider.ANTHROPIC
        assert AuthProvider("openai") == AuthProvider.OPENAI
        assert AuthProvider("custom") == AuthProvider.CUSTOM

    def test_key_id_uniqueness(self, store):
        keys = []
        for i in range(5):
            key = store.add_api_key(f"sk_{i}", f"key{i}", AuthProvider.CUSTOM)
            keys.append(key.id)

        assert len(set(keys)) == 5  # All unique

    def test_api_key_serialization_roundtrip(self):
        key = APIKey(
            id="test_id",
            name="test_name",
            provider=AuthProvider.ANTHROPIC,
            key="sk_test",
            metadata={"foo": "bar"},
        )
        data = key.to_dict()
        restored = APIKey.from_dict(data)
        assert restored.id == key.id
        assert restored.metadata == key.metadata

    def test_key_rotation_entry(self):
        entry = KeyRotationEntry(
            rotated_at="2026-02-25T10:00:00",
            previous_fingerprint="abc123",
            rotated_by="user@test.com",
            reason="Scheduled rotation",
        )
        data = entry.to_dict()
        restored = KeyRotationEntry.from_dict(data)
        assert restored.previous_fingerprint == "abc123"
        assert restored.reason == "Scheduled rotation"

    def test_api_key_with_rotation_history(self):
        key = APIKey(
            id="test_id",
            name="test",
            provider=AuthProvider.ANTHROPIC,
            key="sk_test",
            rotation_history=[
                KeyRotationEntry(
                    rotated_at="2026-01-01T10:00:00",
                    previous_fingerprint="old_fp",
                )
            ],
        )
        data = key.to_dict()
        restored = APIKey.from_dict(data)
        assert len(restored.rotation_history) == 1

    def test_key_scope(self):
        key = APIKey(
            id="test",
            name="test",
            provider=AuthProvider.ANTHROPIC,
            key="sk_test",
            scope=KeyScope.READ_ONLY,
        )
        assert key.scope == KeyScope.READ_ONLY

    def test_key_scope_serialization(self):
        key = APIKey(
            id="test",
            name="test",
            provider=AuthProvider.ANTHROPIC,
            key="sk_test",
            scope=KeyScope.ADMIN,
        )
        data = key.to_dict()
        assert data["scope"] == "admin"
        restored = APIKey.from_dict(data)
        assert restored.scope == KeyScope.ADMIN

    def test_rotate_api_key(self, store):
        key = store.add_api_key("sk_old", "old-key", AuthProvider.CUSTOM)
        old_fp = key.get_fingerprint()

        rotated = store.rotate_api_key(key.id, "sk_new", reason="Scheduled rotation")
        assert rotated is not None
        assert rotated.key == "sk_new"
        assert rotated.get_fingerprint() != old_fp

    def test_rotate_api_key_missing(self, store):
        result = store.rotate_api_key("nonexistent", "sk_new")
        assert result is None

    def test_get_rotation_history(self, store):
        key = store.add_api_key("sk_1", "key1", AuthProvider.CUSTOM)
        original_id = key.id

        rotated = store.rotate_api_key(key.id, "sk_2", reason="Scheduled rotation")

        history = store.get_rotation_history(rotated.id)
        assert len(history) == 1
        assert history[0].reason == "Scheduled rotation"

    def test_set_key_scope(self, store):
        key = store.add_api_key("sk_test", "test", AuthProvider.CUSTOM)

        result = store.set_key_scope(key.id, KeyScope.READ_ONLY)
        assert result is True

        config = store.get_config()
        assert config.get_key(key.id).scope == KeyScope.READ_ONLY

    def test_set_key_scope_missing(self, store):
        result = store.set_key_scope("nonexistent", KeyScope.ADMIN)
        assert result is False

    def test_get_keys_needing_rotation(self, temp_auth_dir):
        from scholardevclaw.auth.store import AuthStore

        store = AuthStore(temp_auth_dir)
        store.add_api_key("sk_old", "old", AuthProvider.CUSTOM)

        keys = store.get_keys_needing_rotation(days=0)
        assert len(keys) == 1

    def test_mark_key_for_rotation(self, store):
        key = store.add_api_key("sk_test", "test", AuthProvider.CUSTOM)

        result = store.mark_key_for_rotation(key.id)
        assert result is True

        config = store.get_config()
        assert config.get_key(key.id).rotation_recommended_at is not None
