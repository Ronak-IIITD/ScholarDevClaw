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
