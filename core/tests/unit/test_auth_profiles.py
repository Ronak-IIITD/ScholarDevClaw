"""Tests for multi-profile/workspace support and key expiration."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from scholardevclaw.auth.store import AuthStore
from scholardevclaw.auth.types import AuthProvider, KeyScope


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def store(temp_dir):
    return AuthStore(str(temp_dir))


# -------------------------------------------------------------------
# Multi-profile (workspace) tests
# -------------------------------------------------------------------


class TestMultiProfile:
    def test_list_profiles_empty(self, store):
        assert store.list_profiles() == []

    def test_save_profile_as(self, store, temp_dir):
        store.add_api_key("sk-test-123", "key1", AuthProvider.CUSTOM)
        path = store.save_profile_as("work")
        assert path.exists()
        assert "work.json" in str(path)

    def test_save_and_list_profiles(self, store):
        store.add_api_key("sk-test-123", "key1", AuthProvider.CUSTOM)
        store.save_profile_as("work")
        store.save_profile_as("personal")
        profiles = store.list_profiles()
        assert "work" in profiles
        assert "personal" in profiles

    def test_load_profile(self, store):
        # Set up profile A
        store.add_api_key("sk-key-a", "key-a", AuthProvider.ANTHROPIC)
        store.create_profile(email="a@example.com")
        store.save_profile_as("profile-a")

        # Switch to profile B
        store.logout()
        store.add_api_key("sk-key-b", "key-b", AuthProvider.OPENAI)
        store.create_profile(email="b@example.com")
        store.save_profile_as("profile-b")

        # Load profile A
        result = store.load_profile("profile-a")
        assert result is True

        # Need fresh store to see loaded config
        store2 = AuthStore(str(store.store_dir))
        keys = store2.list_api_keys()
        assert len(keys) == 1
        assert keys[0].key == "sk-key-a"
        profile = store2.get_profile()
        assert profile is not None
        assert profile.email == "a@example.com"

    def test_load_profile_nonexistent(self, store):
        assert store.load_profile("nonexistent") is False

    def test_delete_profile(self, store):
        store.add_api_key("sk-test", "k", AuthProvider.CUSTOM)
        store.save_profile_as("to-delete")
        assert "to-delete" in store.list_profiles()
        assert store.delete_profile("to-delete") is True
        assert "to-delete" not in store.list_profiles()

    def test_delete_profile_nonexistent(self, store):
        assert store.delete_profile("nonexistent") is False

    def test_save_overwrites_existing_profile(self, store):
        store.add_api_key("sk-key-1", "key1", AuthProvider.CUSTOM)
        store.save_profile_as("test-profile")

        # Add another key and re-save
        store.add_api_key("sk-key-2", "key2", AuthProvider.OPENAI)
        store.save_profile_as("test-profile")

        # Load and verify it has both keys
        store2 = AuthStore(str(store.store_dir))
        store2.load_profile("test-profile")
        store3 = AuthStore(str(store.store_dir))
        keys = store3.list_api_keys()
        assert len(keys) == 2

    def test_profile_file_permissions(self, store, temp_dir):
        store.add_api_key("sk-test", "k", AuthProvider.CUSTOM)
        path = store.save_profile_as("secured")
        mode = oct(path.stat().st_mode)[-3:]
        assert mode == "600"

    def test_profile_file_contains_valid_json(self, store, temp_dir):
        store.add_api_key("sk-test-key", "k", AuthProvider.CUSTOM)
        store.create_profile(email="test@test.com")
        path = store.save_profile_as("check-json")
        data = json.loads(path.read_text())
        assert "api_keys" in data
        assert "profile" in data

    def test_multiple_profiles_independent(self, store):
        """Switching profiles should fully replace config."""
        store.add_api_key("sk-work-key", "work-key", AuthProvider.ANTHROPIC)
        store.save_profile_as("work")

        store.logout()
        store.add_api_key("sk-personal-key", "personal-key", AuthProvider.OPENAI)
        store.save_profile_as("personal")

        # Switch to work
        store.load_profile("work")
        store2 = AuthStore(str(store.store_dir))
        keys = store2.list_api_keys()
        assert len(keys) == 1
        assert keys[0].name == "work-key"

        # Switch to personal
        store2.load_profile("personal")
        store3 = AuthStore(str(store.store_dir))
        keys = store3.list_api_keys()
        assert len(keys) == 1
        assert keys[0].name == "personal-key"

    def test_profiles_dir_created_on_save(self, store, temp_dir):
        store.add_api_key("sk-test", "k", AuthProvider.CUSTOM)
        store.save_profile_as("test")
        profiles_dir = temp_dir / "profiles"
        assert profiles_dir.exists()

    def test_corrupted_profile_file(self, store, temp_dir):
        """Loading a corrupted profile should return False."""
        profiles_dir = temp_dir / "profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)
        (profiles_dir / "bad.json").write_text("{{invalid json")
        assert store.load_profile("bad") is False


# -------------------------------------------------------------------
# Key expiration tests
# -------------------------------------------------------------------


class TestKeyExpiration:
    def test_get_expiring_keys_none(self, store):
        store.add_api_key("sk-test-123", "key1", AuthProvider.CUSTOM)
        expiring = store.get_expiring_keys(within_days=7)
        assert len(expiring) == 0

    def test_get_expiring_keys_within_window(self, store):
        expires = (datetime.now() + timedelta(days=3)).isoformat()
        store.add_api_key("sk-test-123", "key1", AuthProvider.CUSTOM, expires_at=expires)
        expiring = store.get_expiring_keys(within_days=7)
        assert len(expiring) == 1

    def test_get_expiring_keys_outside_window(self, store):
        expires = (datetime.now() + timedelta(days=30)).isoformat()
        store.add_api_key("sk-test-123", "key1", AuthProvider.CUSTOM, expires_at=expires)
        expiring = store.get_expiring_keys(within_days=7)
        assert len(expiring) == 0

    def test_get_expiring_keys_already_expired(self, store):
        expires = (datetime.now() - timedelta(days=1)).isoformat()
        store.add_api_key("sk-test-123", "key1", AuthProvider.CUSTOM, expires_at=expires)
        expiring = store.get_expiring_keys(within_days=7)
        # Already expired keys are no longer active (is_valid() returns False)
        # But is_active is still True until deactivated — and expires_at is in the past
        # which means expires <= deadline is True, so it should be returned
        # Actually: the key has is_active=True and expires <= deadline
        assert len(expiring) == 1

    def test_deactivate_expired_keys(self, store):
        expires = (datetime.now() - timedelta(days=1)).isoformat()
        key = store.add_api_key("sk-test-123", "key1", AuthProvider.CUSTOM, expires_at=expires)
        deactivated = store.deactivate_expired_keys()
        assert len(deactivated) == 1
        assert deactivated[0].is_active is False

        # Verify persistence
        store2 = AuthStore(str(store.store_dir))
        keys = store2.list_api_keys()
        assert keys[0].is_active is False

    def test_deactivate_expired_keys_none_expired(self, store):
        expires = (datetime.now() + timedelta(days=30)).isoformat()
        store.add_api_key("sk-test-123", "key1", AuthProvider.CUSTOM, expires_at=expires)
        deactivated = store.deactivate_expired_keys()
        assert len(deactivated) == 0

    def test_deactivate_already_inactive(self, store):
        """Already inactive keys should not be deactivated again."""
        expires = (datetime.now() - timedelta(days=1)).isoformat()
        store.add_api_key("sk-test-123", "key1", AuthProvider.CUSTOM, expires_at=expires)
        store.deactivate_expired_keys()

        # Second call should not find anything
        store2 = AuthStore(str(store.store_dir))
        deactivated2 = store2.deactivate_expired_keys()
        assert len(deactivated2) == 0

    def test_set_key_expiry(self, store):
        key = store.add_api_key("sk-test-123", "key1", AuthProvider.CUSTOM)
        expires = (datetime.now() + timedelta(days=30)).isoformat()
        result = store.set_key_expiry(key.id, expires)
        assert result is True

        store2 = AuthStore(str(store.store_dir))
        keys = store2.list_api_keys()
        assert keys[0].expires_at == expires

    def test_set_key_expiry_invalid_format(self, store):
        key = store.add_api_key("sk-test-123", "key1", AuthProvider.CUSTOM)
        with pytest.raises(ValueError, match="Invalid date format"):
            store.set_key_expiry(key.id, "not-a-date")

    def test_set_key_expiry_missing_key(self, store):
        result = store.set_key_expiry("nonexistent", datetime.now().isoformat())
        assert result is False

    def test_get_expiring_keys_multiple(self, store):
        """Multiple keys with different expiry dates."""
        expires_soon = (datetime.now() + timedelta(days=2)).isoformat()
        expires_later = (datetime.now() + timedelta(days=20)).isoformat()

        store.add_api_key("sk-test-1", "soon", AuthProvider.CUSTOM, expires_at=expires_soon)
        store.add_api_key("sk-test-2", "later", AuthProvider.CUSTOM, expires_at=expires_later)
        store.add_api_key("sk-test-3", "never", AuthProvider.CUSTOM)

        expiring = store.get_expiring_keys(within_days=7)
        assert len(expiring) == 1
        assert expiring[0].name == "soon"

    def test_deactivate_does_not_affect_valid_keys(self, store):
        expires_past = (datetime.now() - timedelta(days=1)).isoformat()
        expires_future = (datetime.now() + timedelta(days=30)).isoformat()

        store.add_api_key("sk-old", "expired", AuthProvider.CUSTOM, expires_at=expires_past)
        store.add_api_key("sk-new", "valid", AuthProvider.CUSTOM, expires_at=expires_future)

        deactivated = store.deactivate_expired_keys()
        assert len(deactivated) == 1
        assert deactivated[0].name == "expired"

        store2 = AuthStore(str(store.store_dir))
        keys = store2.list_api_keys()
        valid_key = next(k for k in keys if k.name == "valid")
        assert valid_key.is_active is True

    def test_expiry_with_bad_format_in_data_gracefully_handled(self, store, temp_dir):
        """Keys with invalid expires_at format should be skipped, not crash."""
        key = store.add_api_key("sk-test", "k", AuthProvider.CUSTOM)
        # Manually set a bad expiry
        config = store.get_config()
        config.api_keys[0].expires_at = "not-a-valid-date"
        store._save_config(config)

        # Should not raise
        store2 = AuthStore(str(temp_dir))
        expiring = store2.get_expiring_keys(within_days=7)
        assert len(expiring) == 0


# -------------------------------------------------------------------
# File permission hardening tests
# -------------------------------------------------------------------


class TestFilePermissions:
    def test_auth_file_permissions(self, store, temp_dir):
        store.add_api_key("sk-test-key", "k", AuthProvider.CUSTOM)
        auth_file = temp_dir / "auth.json"
        assert auth_file.exists()
        mode = oct(auth_file.stat().st_mode)[-3:]
        assert mode == "600"

    def test_env_file_permissions(self, store, temp_dir):
        store.add_api_key("sk-test-key", "k", AuthProvider.CUSTOM)
        env_file = temp_dir / ".env"
        assert env_file.exists()
        mode = oct(env_file.stat().st_mode)[-3:]
        assert mode == "600"
