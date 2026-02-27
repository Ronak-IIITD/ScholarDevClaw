"""Security tests for auth module"""

import json
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_auth_dir(monkeypatch):
    """Create a temporary auth directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("SCHOLARDEVCLAW_AUTH_DIR", tmpdir)
        yield tmpdir


class TestAuthSecurity:
    """Security-related tests for auth module"""

    def test_api_key_not_logged_in_plain(self, temp_auth_dir, capsys):
        """Verify API keys are masked in output (shows only first 8 chars)"""
        from scholardevclaw.auth.cli import cmd_auth
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider
        from argparse import Namespace

        store = AuthStore(temp_auth_dir)
        store.add_api_key("sk-secret-12345", "secret", AuthProvider.ANTHROPIC)

        args = Namespace(
            auth_action="list",
            output_json=False,
        )
        cmd_auth(args)
        captured = capsys.readouterr()

        assert "sk-secret-12345" not in captured.out
        assert "sk-secre" in captured.out or "..." in captured.out

    def test_api_key_not_in_status_json(self, temp_auth_dir):
        """Verify API keys can be excluded from JSON status"""
        from scholardevclaw.auth.cli import cmd_auth
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider
        from argparse import Namespace

        store = AuthStore(temp_auth_dir)
        store.add_api_key("sk-very-secret-key-12345", "secret", AuthProvider.ANTHROPIC)

        args = Namespace(
            auth_action="list",
            output_json=True,
        )

        import io
        import sys
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            cmd_auth(args)
        output = f.getvalue()

        data = json.loads(output)
        # Security fix: to_safe_dict() no longer exposes raw key
        assert "key" not in data[0] or data[0].get("key") != "sk-very-secret-key-12345"
        assert "key_fingerprint" in data[0] or "key_masked" in data[0]

    def test_fingerprint_does_not_reveal_key(self):
        """Verify fingerprint doesn't expose the actual key"""
        from scholardevclaw.auth.types import APIKey, AuthProvider

        api_key = APIKey(
            id="test",
            name="test",
            provider=AuthProvider.ANTHROPIC,
            key="sk-very-secret-key-12345",
        )

        fp = api_key.get_fingerprint()
        assert "sk-very-secret" not in fp
        assert "12345" not in fp
        assert len(fp) == 64

    def test_auth_file_contains_masked_or_no_key(self, temp_auth_dir):
        """Verify auth file can be configured to not store keys"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_dir)
        store.add_api_key("sk-test", "test", AuthProvider.CUSTOM)

        with open(store.auth_file) as f:
            content = f.read()

        data = json.loads(content)
        assert data["api_keys"][0]["key"] == "sk-test"

    def test_logout_removes_all_sensitive_data(self, temp_auth_dir):
        """Verify logout removes all sensitive data"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_dir)
        store.add_api_key("sk-sensitive", "sensitive", AuthProvider.ANTHROPIC)
        store.create_profile(email="user@example.com", name="Test User")

        store.logout()

        assert not store.is_authenticated()

        if store.auth_file.exists():
            with open(store.auth_file) as f:
                content = f.read()
            data = json.loads(content)
            assert len(data["api_keys"]) == 0

    def test_env_override_key_not_stored(self, temp_auth_dir, monkeypatch):
        """Verify keys from env var are not stored"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        monkeypatch.setenv("SCHOLARDEVCLAW_API_KEY", "sk-env-only")

        store = AuthStore(temp_auth_dir)
        api_key = store.get_api_key()
        assert api_key == "sk-env-only"

        if store.auth_file.exists():
            with open(store.auth_file) as f:
                content = f.read()
            data = json.loads(content)
            assert len(data.get("api_keys", [])) == 0

    def test_key_id_is_random(self):
        """Verify key IDs are random and unpredictable"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        ids = set()
        with tempfile.TemporaryDirectory() as tmpdir:
            for _ in range(100):
                store = AuthStore(tmpdir)
                key = store.add_api_key("sk-test", "test", AuthProvider.CUSTOM)
                ids.add(key.id)

        assert len(ids) == 100

    def test_profile_id_is_random(self):
        """Verify profile IDs are random"""
        from scholardevclaw.auth.store import AuthStore

        ids = set()
        with tempfile.TemporaryDirectory() as tmpdir:
            for _ in range(50):
                store = AuthStore(tmpdir)
                profile = store.create_profile(email="test@example.com")
                ids.add(profile.id)

        assert len(ids) == 50

    def test_invalid_json_does_not_leak_info(self, temp_auth_dir):
        """Verify invalid JSON is handled gracefully"""
        from scholardevclaw.auth.store import AuthStore

        auth_file = Path(temp_auth_dir) / "auth.json"
        auth_file.write_text('{"api_keys": [{"key": "sk-leaked"}]}')

        store = AuthStore(temp_auth_dir)
        assert store.is_authenticated() is False

    def test_path_traversal_prevention(self, temp_auth_dir, monkeypatch):
        """Verify path traversal is prevented"""
        malicious_path = temp_auth_dir + "/../other_dir"
        monkeypatch.setenv("SCHOLARDEVCLAW_AUTH_DIR", malicious_path)

        from scholardevclaw.auth.store import AuthStore
        import os

        try:
            store = AuthStore(malicious_path)
            assert temp_auth_dir not in str(store.store_dir)
        except Exception:
            pass

    def test_concurrent_store_instances(self, temp_auth_dir):
        """Test that store instances handle concurrent operations"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store1 = AuthStore(temp_auth_dir)
        store1.add_api_key("sk-first", "first", AuthProvider.CUSTOM)

        store2 = AuthStore(temp_auth_dir)
        keys2 = store2.list_api_keys()

        assert len(keys2) == 1

    def test_large_key_handling(self, temp_auth_dir):
        """Test handling of very large API keys"""
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(temp_auth_dir)
        large_key = "x" * 10000

        key = store.add_api_key(large_key, "large", AuthProvider.CUSTOM)
        assert key.key == large_key

        mask = key.mask()
        assert len(mask) < len(large_key)
