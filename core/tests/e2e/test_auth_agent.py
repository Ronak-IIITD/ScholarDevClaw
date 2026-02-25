"""Integration tests for auth + agent workflow"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_auth_env(monkeypatch):
    """Create a temporary auth directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("SCHOLARDEVCLAW_AUTH_DIR", tmpdir)
        yield tmpdir


class TestAuthAgentIntegration:
    """Test auth integration with agent/CLI workflow"""

    def test_auth_then_agent_analyze(self, temp_auth_env):
        """Test that auth works before agent analyze command"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_env)
        key = store.add_api_key(
            key="sk-ant-test-key",
            name="test-key",
            provider=AuthProvider.ANTHROPIC,
        )

        assert store.is_authenticated()
        api_key = store.get_api_key()
        assert api_key == "sk-ant-test-key"

        status = store.get_status()
        assert status.is_authenticated
        assert status.provider == "anthropic"

    def test_auth_env_var_precedence(self, temp_auth_env, monkeypatch):
        """Test that SCHOLARDEVCLAW_API_KEY env var takes precedence"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_env)
        store.add_api_key("sk-stored", "stored", AuthProvider.CUSTOM)

        assert store.get_api_key() == "sk-stored"

        monkeypatch.setenv("SCHOLARDEVCLAW_API_KEY", "sk-env-override")
        assert store.get_api_key() == "sk-env-override"

        monkeypatch.delenv("SCHOLARDEVCLAW_API_KEY", raising=False)
        assert store.get_api_key() == "sk-stored"

    def test_multiple_keys_different_providers(self, temp_auth_env):
        """Test having keys for multiple providers"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_env)

        store.add_api_key("sk-ant", "anthropic", AuthProvider.ANTHROPIC)
        store.add_api_key("sk-oi", "openai", AuthProvider.OPENAI)
        store.add_api_key("ghp_gh", "github", AuthProvider.GITHUB)

        keys = store.list_api_keys()
        assert len(keys) == 3

        status = store.get_status()
        assert status.key_count == 3

    def test_key_activation_deactivation(self, temp_auth_env):
        """Test activating and deactivating keys"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_env)

        key = store.add_api_key("sk-test", "test", AuthProvider.ANTHROPIC)
        assert store.get_api_key() == "sk-test"

        config = store.get_config()
        config.get_key(key.id).is_active = False
        store._save_config(config)

        assert store.get_api_key() is None

    def test_profile_with_subscription(self, temp_auth_env):
        """Test profile with subscription tier"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider, SubscriptionTier

        store = AuthStore(temp_auth_env)
        store.add_api_key("sk-test", "test", AuthProvider.ANTHROPIC)

        profile = store.create_profile(
            email="pro@example.com",
            name="Pro User",
        )

        config = store.get_config()
        config.profile.subscription_tier = SubscriptionTier.PRO
        store._save_config(config)

        status = store.get_status()
        assert status.subscription_tier == "pro"

    def test_auth_backup_and_restore(self, temp_auth_env):
        """Test backing up and restoring auth data"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_env)
        store.add_api_key("sk-backup", "backup", AuthProvider.ANTHROPIC)
        store.create_profile(email="backup@example.com", name="Backup User")

        auth_file = Path(temp_auth_env) / "auth.json"

        with open(auth_file) as f:
            backup_data = f.read()

        store2 = AuthStore(temp_auth_env)
        store2.logout()

        with open(auth_file, "w") as f:
            f.write(backup_data)

        store3 = AuthStore(temp_auth_env)
        assert store3.is_authenticated()
        assert store3.get_profile().email == "backup@example.com"

    def test_concurrent_key_access(self, temp_auth_env):
        """Test concurrent access to auth store"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_env)

        for i in range(10):
            store.add_api_key(f"sk-{i}", f"key-{i}", AuthProvider.CUSTOM)

        keys = store.list_api_keys()
        assert len(keys) == 10

    def test_key_metadata_persistence(self, temp_auth_env):
        """Test that key metadata persists"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_env)

        key = store.add_api_key(
            key="sk-meta",
            name="metadata-key",
            provider=AuthProvider.CUSTOM,
        )

        config = store.get_config()
        config.get_key(key.id).metadata["source"] = "cli"
        config.get_key(key.id).metadata["version"] = "1.0"
        store._save_config(config)

        store2 = AuthStore(temp_auth_env)
        restored_key = store2.get_config().get_key(key.id)
        assert restored_key.metadata["source"] == "cli"
        assert restored_key.metadata["version"] == "1.0"


class TestAuthWithCLICommands:
    """Test auth with various CLI command patterns"""

    def test_login_and_run_analyze(self, temp_auth_env, monkeypatch):
        """Simulate login then run analyze"""
        from scholardevclaw.auth.cli import cmd_auth
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider
        from argparse import Namespace

        args = Namespace(
            auth_action="login",
            provider="anthropic",
            key="sk-ant-test",
            name="analyze-key",
            output_json=False,
        )
        cmd_auth(args)

        store = AuthStore(temp_auth_env)
        assert store.is_authenticated()

        api_key = store.get_api_key()
        assert api_key == "sk-ant-test"

    def test_switch_provider(self, temp_auth_env):
        """Test switching between providers"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_env)

        ant_key = store.add_api_key("sk-ant", "ant", AuthProvider.ANTHROPIC)
        oi_key = store.add_api_key("sk-oi", "openai", AuthProvider.OPENAI)

        store.set_default_key(ant_key.id)
        assert store.get_api_key() == "sk-ant"

        store.set_default_key(oi_key.id)
        assert store.get_api_key() == "sk-oi"

    def test_remove_default_key(self, temp_auth_env):
        """Test removing the default key"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_env)

        key1 = store.add_api_key("sk-1", "key1", AuthProvider.CUSTOM)
        key2 = store.add_api_key("sk-2", "key2", AuthProvider.CUSTOM)

        store.remove_api_key(key1.id)

        config = store.get_config()
        assert config.default_key_id == key2.id

    def test_auth_file_permissions(self, temp_auth_env):
        """Test auth file is created with proper permissions"""
        import stat
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_env)
        store.add_api_key("sk-test", "test", AuthProvider.CUSTOM)

        auth_file = Path(temp_auth_env) / "auth.json"
        assert auth_file.exists()

        mode = auth_file.stat().st_mode
        assert mode & stat.S_IRUSR
        assert mode & stat.S_IWUSR

    def test_large_number_of_keys(self, temp_auth_env):
        """Test handling many API keys"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_env)

        for i in range(50):
            store.add_api_key(f"sk-{i:03d}", f"key-{i}", AuthProvider.CUSTOM)

        keys = store.list_api_keys()
        assert len(keys) == 50

        status = store.get_status()
        assert status.key_count == 50
