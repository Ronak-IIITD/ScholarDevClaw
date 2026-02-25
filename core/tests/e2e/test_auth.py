"""Integration tests for auth module - full workflow tests"""

import json
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_auth_env(monkeypatch):
    """Create a temporary auth directory and set environment variable"""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("SCHOLARDEVCLAW_AUTH_DIR", tmpdir)
        yield tmpdir


class TestAuthIntegration:
    """Integration tests for full auth workflows"""

    def test_full_auth_workflow(self, temp_auth_env):
        """Test complete workflow: add key, check status, list, remove"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider, AuthStatus

        store = AuthStore(temp_auth_env)

        # Initially not authenticated
        assert store.is_authenticated() is False
        status = store.get_status()
        assert status.is_authenticated is False
        assert status.active_keys == 0

        # Add an API key
        key1 = store.add_api_key(
            key="sk_test_anthropic_123",
            name="anthropic-primary",
            provider=AuthProvider.ANTHROPIC,
        )
        assert key1.id is not None
        assert key1.provider == AuthProvider.ANTHROPIC

        # Now should be authenticated
        assert store.is_authenticated() is True
        status = store.get_status()
        assert status.is_authenticated is True
        assert status.active_keys == 1
        assert status.provider == "anthropic"

        # Add another key
        key2 = store.add_api_key(
            key="sk_test_openai_456",
            name="openai-secondary",
            provider=AuthProvider.OPENAI,
            set_default=False,
        )
        assert key2.id != key1.id

        # List keys
        keys = store.list_api_keys()
        assert len(keys) == 2

        # Set default to openai
        store.set_default_key(key2.id)
        config = store.get_config()
        assert config.default_key_id == key2.id

        # Get the active key (should be openai now)
        active = store.get_api_key()
        assert active == "sk_test_openai_456"

        # Remove the anthropic key
        removed = store.remove_api_key(key1.id)
        assert removed is True

        # Verify removal
        keys = store.list_api_keys()
        assert len(keys) == 1
        assert keys[0].id == key2.id

        # Logout
        result = store.logout()
        assert result is True
        assert store.is_authenticated() is False

    def test_profile_workflow(self, temp_auth_env):
        """Test profile creation and update workflow"""
        from scholardevclaw.auth.store import AuthStore

        store = AuthStore(temp_auth_env)

        # Create profile
        profile = store.create_profile(
            email="test@example.com",
            name="Test User",
        )
        assert profile is not None
        assert profile.email == "test@example.com"
        assert profile.name == "Test User"

        # Get profile
        retrieved = store.get_profile()
        assert retrieved is not None
        assert retrieved.email == "test@example.com"

        # Update profile
        updated = store.update_profile(name="Updated Name")
        assert updated is not None
        assert updated.name == "Updated Name"
        assert updated.email == "test@example.com"  # unchanged

        # Update with new email
        updated2 = store.update_profile(email="new@example.com")
        assert updated2.email == "new@example.com"
        assert updated2.name == "Updated Name"

    def test_env_override_workflow(self, temp_auth_env, monkeypatch):
        """Test environment variable override"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_env)

        # Add a key
        store.add_api_key("sk_stored_key", "stored", AuthProvider.ANTHROPIC)

        # Without env override, get stored key
        assert store.get_api_key() == "sk_stored_key"

        # With env override
        monkeypatch.setenv("SCHOLARDEVCLAW_API_KEY", "sk_env_key")
        assert store.get_api_key() == "sk_env_key"

        # Remove env override
        monkeypatch.delenv("SCHOLARDEVCLAW_API_KEY", raising=False)
        assert store.get_api_key() == "sk_stored_key"

    def test_multiple_providers_workflow(self, temp_auth_env):
        """Test handling multiple providers"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_env)

        # Add keys for different providers
        store.add_api_key("sk_ant", "ant-key", AuthProvider.ANTHROPIC)
        store.add_api_key("sk_oi", "openai-key", AuthProvider.OPENAI)
        store.add_api_key("sk_gh", "github-key", AuthProvider.GITHUB)
        store.add_api_key("sk_go", "google-key", AuthProvider.GOOGLE)

        keys = store.list_api_keys()
        assert len(keys) == 4

        providers = {k.provider for k in keys}
        assert AuthProvider.ANTHROPIC in providers
        assert AuthProvider.OPENAI in providers
        assert AuthProvider.GITHUB in providers
        assert AuthProvider.GOOGLE in providers

        # Get status
        status = store.get_status()
        assert status.key_count == 4

    def test_expiring_keys_workflow(self, temp_auth_env):
        """Test expiring API keys - skipping since expires_at not yet implemented in add_api_key"""
        # The expires_at feature is stored in APIKey but not yet settable via add_api_key
        # This test verifies the current behavior
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_env)

        # Add a key (expires_at will be None by default)
        key = store.add_api_key(
            key="sk_valid",
            name="valid-key",
            provider=AuthProvider.ANTHROPIC,
        )
        # Key without expiry is considered valid
        assert key.is_valid() is True
        assert key.expires_at is None

        # Status should show 1 active key
        status = store.get_status()
        assert status.active_keys == 1

    def test_key_rotation_workflow(self, temp_auth_env):
        """Test rotating API keys"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_env)

        # Add initial key
        key1 = store.add_api_key("sk_v1", "v1", AuthProvider.ANTHROPIC)
        assert store.get_api_key() == "sk_v1"

        # Rotate - add new key
        key2 = store.add_api_key("sk_v2", "v2", AuthProvider.ANTHROPIC)

        # Set new as default
        store.set_default_key(key2.id)
        assert store.get_api_key() == "sk_v2"

        # Remove old key
        store.remove_api_key(key1.id)

        # Verify only new key remains
        keys = store.list_api_keys()
        assert len(keys) == 1
        assert keys[0].id == key2.id

    def test_auth_persistence(self, temp_auth_env):
        """Test that auth data persists across store instances"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        # Create first store and add data
        store1 = AuthStore(temp_auth_env)
        store1.add_api_key("sk_persist", "persist-key", AuthProvider.ANTHROPIC)
        store1.create_profile(email="persist@test.com", name="Persist")

        # Create new store instance (simulating restart)
        store2 = AuthStore(temp_auth_env)

        # Verify data persisted
        assert store2.is_authenticated() is True
        keys = store2.list_api_keys()
        assert len(keys) == 1
        assert keys[0].key == "sk_persist"

        profile = store2.get_profile()
        assert profile is not None
        assert profile.email == "persist@test.com"

    def test_auth_file_format(self, temp_auth_env):
        """Test that auth file is valid JSON with expected structure"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_env)
        store.add_api_key("sk_test", "test-key", AuthProvider.ANTHROPIC)
        store.create_profile(email="test@example.com", name="Test")

        # Read the auth file directly
        auth_file = Path(temp_auth_env) / "auth.json"
        assert auth_file.exists()

        with open(auth_file) as f:
            data = json.load(f)

        # Verify structure
        assert "api_keys" in data
        assert "default_key_id" in data
        assert "default_provider" in data
        assert "profile" in data
        assert len(data["api_keys"]) == 1
        assert data["api_keys"][0]["key"] == "sk_test"
        assert data["profile"]["email"] == "test@example.com"


class TestAuthCLIntegration:
    """Integration tests for CLI commands with store"""

    def test_cli_store_integration(self, temp_auth_env, capsys):
        """Test CLI commands work with AuthStore"""
        from scholardevclaw.auth.cli import cmd_auth
        from argparse import Namespace

        # Add key via CLI
        args = Namespace(
            auth_action="add",
            key="sk_cli_test",
            name="cli-key",
            provider="anthropic",
            default=True,
            output_json=False,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "API key added" in captured.out

        # Check status via CLI
        args = Namespace(
            auth_action="status",
            output_json=False,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "Authenticated" in captured.out

        # List via CLI
        args = Namespace(
            auth_action="list",
            output_json=False,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "cli-key" in captured.out

    def test_cli_json_output_integration(self, temp_auth_env, capsys):
        """Test CLI JSON output can be parsed"""
        from scholardevclaw.auth.cli import cmd_auth
        from argparse import Namespace
        import re

        # Add key
        args = Namespace(
            auth_action="add",
            key="sk_json",
            name="json-key",
            provider="openai",
            default=True,
            output_json=False,
        )
        cmd_auth(args)

        # Get status as JSON
        args = Namespace(
            auth_action="status",
            output_json=True,
        )
        cmd_auth(args)
        captured = capsys.readouterr()

        # Extract JSON from output (skip the add key output)
        json_match = re.search(r"(\{[\s\S]*\})", captured.out)
        assert json_match is not None
        data = json.loads(json_match.group(1))
        assert data["is_authenticated"] is True
        assert data["active_keys"] == 1
