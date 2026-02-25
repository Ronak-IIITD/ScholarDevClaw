import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestAuthCLI:
    """Tests for auth CLI commands"""

    @pytest.fixture
    def temp_auth_dir(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("SCHOLARDEVCLAW_AUTH_DIR", tmpdir)
            yield tmpdir

    @pytest.fixture
    def mock_input(self, monkeypatch):
        inputs = []

        def mock_input_func(prompt=""):
            inputs.append(prompt)
            return inputs.pop(0) if inputs else ""

        monkeypatch.setattr("scholardevclaw.auth.cli.input", mock_input_func)
        return inputs

    @pytest.fixture
    def mock_getpass(self, monkeypatch):
        def mock_getpass_func(prompt=""):
            return "sk_test_key_12345"

        monkeypatch.setattr("scholardevclaw.auth.cli.getpass.getpass", mock_getpass_func)

    def test_auth_status_not_authenticated(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from argparse import Namespace

        args = Namespace(
            auth_action="status",
            output_json=False,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "Not authenticated" in captured.out

    def test_auth_status_with_key(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider
        from argparse import Namespace

        store = AuthStore(temp_auth_dir)
        store.add_api_key("sk_test", "test-key", AuthProvider.ANTHROPIC)

        args = Namespace(
            auth_action="status",
            output_json=False,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "Authenticated" in captured.out
        assert "API Keys:" in captured.out

    def test_auth_status_json(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider
        from argparse import Namespace

        store = AuthStore(temp_auth_dir)
        store.add_api_key("sk_test", "test-key", AuthProvider.ANTHROPIC)

        args = Namespace(
            auth_action="status",
            output_json=True,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["is_authenticated"] is True

    def test_auth_list_empty(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from argparse import Namespace

        args = Namespace(
            auth_action="list",
            output_json=False,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "No API keys found" in captured.out

    def test_auth_list_with_keys(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider
        from argparse import Namespace

        store = AuthStore(temp_auth_dir)
        store.add_api_key("sk_test1", "key1", AuthProvider.ANTHROPIC)
        store.add_api_key("sk_test2", "key2", AuthProvider.OPENAI)

        args = Namespace(
            auth_action="list",
            output_json=False,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "key1" in captured.out
        assert "key2" in captured.out

    def test_auth_list_json(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider
        from argparse import Namespace

        store = AuthStore(temp_auth_dir)
        store.add_api_key("sk_test", "my-key", AuthProvider.ANTHROPIC)

        args = Namespace(
            auth_action="list",
            output_json=True,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 1
        assert data[0]["name"] == "my-key"

    def test_auth_add_with_args(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from argparse import Namespace

        args = Namespace(
            auth_action="add",
            key="sk_test_add",
            name="added-key",
            provider="anthropic",
            default=True,
            output_json=False,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "API key added" in captured.out

    def test_auth_add_missing_key(self, temp_auth_dir, capsys, monkeypatch):
        from scholardevclaw.auth.cli import cmd_auth
        from argparse import Namespace

        monkeypatch.setattr("scholardevclaw.auth.cli.getpass.getpass", lambda p: "")

        args = Namespace(
            auth_action="add",
            key=None,
            name="test",
            provider="custom",
            default=False,
            output_json=False,
        )
        with pytest.raises(SystemExit) as exc:
            cmd_auth(args)
        assert exc.value.code == 1

    def test_auth_remove_existing(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider
        from argparse import Namespace

        store = AuthStore(temp_auth_dir)
        key = store.add_api_key("sk_test", "to-remove", AuthProvider.ANTHROPIC)

        args = Namespace(
            auth_action="remove",
            key_id=key.id,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "removed" in captured.out.lower()

    def test_auth_remove_missing(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from argparse import Namespace

        args = Namespace(
            auth_action="remove",
            key_id="nonexistent",
        )
        with pytest.raises(SystemExit) as exc:
            cmd_auth(args)
        assert exc.value.code == 1

    def test_auth_default_existing(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider
        from argparse import Namespace

        store = AuthStore(temp_auth_dir)
        key1 = store.add_api_key("sk_test1", "key1", AuthProvider.ANTHROPIC)
        key2 = store.add_api_key("sk_test2", "key2", AuthProvider.OPENAI, set_default=False)

        args = Namespace(
            auth_action="default",
            key_id=key2.id,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "Default key set" in captured.out

    def test_auth_default_missing(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from argparse import Namespace

        args = Namespace(
            auth_action="default",
            key_id="nonexistent",
        )
        with pytest.raises(SystemExit) as exc:
            cmd_auth(args)
        assert exc.value.code == 1

    def test_auth_login_with_args(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from argparse import Namespace

        args = Namespace(
            auth_action="login",
            provider="anthropic",
            key="sk_login_test",
            name="login-key",
            output_json=False,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "Logged in successfully" in captured.out

    def test_auth_login_invalid_provider(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from argparse import Namespace

        args = Namespace(
            auth_action="login",
            provider="invalid_provider",
            key="sk_test",
            name=None,
            output_json=False,
        )
        with pytest.raises(SystemExit) as exc:
            cmd_auth(args)
        assert exc.value.code == 1

    def test_auth_login_json_output(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from argparse import Namespace
        import re

        args = Namespace(
            auth_action="login",
            provider="openai",
            key="sk_json_test",
            name=None,
            output_json=True,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        # Check that output contains JSON with expected fields
        assert '"id":' in captured.out
        assert '"provider": "openai"' in captured.out
        assert '"key": "sk_json_test"' in captured.out

    def test_auth_logout_confirm_yes(self, temp_auth_dir, capsys, monkeypatch):
        from scholardevclaw.auth.cli import cmd_auth
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider
        from argparse import Namespace

        store = AuthStore(temp_auth_dir)
        store.add_api_key("sk_test", "key1", AuthProvider.ANTHROPIC)

        monkeypatch.setattr("builtins.input", lambda p: "y")

        args = Namespace(
            auth_action="logout",
            force=False,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "Logged out successfully" in captured.out

        # Verify by checking with a new store instance
        store2 = AuthStore(temp_auth_dir)
        assert not store2.is_authenticated()

    def test_auth_logout_force(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider
        from argparse import Namespace

        store = AuthStore(temp_auth_dir)
        store.add_api_key("sk_test", "key1", AuthProvider.ANTHROPIC)

        args = Namespace(
            auth_action="logout",
            force=True,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "Logged out successfully" in captured.out

        # Verify by checking with a new store instance
        store2 = AuthStore(temp_auth_dir)
        assert not store2.is_authenticated()

    def test_auth_logout_confirm_no(self, temp_auth_dir, capsys, monkeypatch):
        from scholardevclaw.auth.cli import cmd_auth
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider
        from argparse import Namespace

        store = AuthStore(temp_auth_dir)
        store.add_api_key("sk_test", "key1", AuthProvider.ANTHROPIC)

        monkeypatch.setattr("builtins.input", lambda p: "n")

        args = Namespace(
            auth_action="logout",
            force=False,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "Cancelled" in captured.out

        # Verify still authenticated
        store2 = AuthStore(temp_auth_dir)
        assert store2.is_authenticated()

    def test_auth_logout_no_credentials(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from argparse import Namespace

        args = Namespace(
            auth_action="logout",
            force=True,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "No credentials" in captured.out

    def test_auth_unknown_action(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from argparse import Namespace

        args = Namespace(
            auth_action="unknown",
        )
        with pytest.raises(SystemExit) as exc:
            cmd_auth(args)
        assert exc.value.code == 1

    def test_auth_setup_already_authenticated(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider
        from argparse import Namespace

        store = AuthStore(temp_auth_dir)
        store.add_api_key("sk_test", "key1", AuthProvider.ANTHROPIC)

        args = Namespace(
            auth_action="setup",
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "Already authenticated" in captured.out

    def test_auth_setup_with_env_key_accept(self, temp_auth_dir, capsys, monkeypatch):
        from scholardevclaw.auth.cli import cmd_auth
        from argparse import Namespace

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk_env_key_123")
        monkeypatch.setattr("builtins.input", lambda p: "y")

        args = Namespace(
            auth_action="setup",
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "Setup complete" in captured.out or "API key added" in captured.out

    def test_auth_setup_with_env_key_decline(self, temp_auth_dir, capsys, monkeypatch):
        from scholardevclaw.auth.cli import cmd_auth
        from scholardevclaw.auth.store import AuthStore
        from argparse import Namespace

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk_env_key_123")

        call_count = [0]

        def mock_input(prompt=""):
            call_count[0] += 1
            if call_count[0] == 1:
                return "n"  # decline env key
            return "sk_manual_key"  # provide key manually

        monkeypatch.setattr("builtins.input", mock_input)

        args = Namespace(
            auth_action="setup",
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "Setup complete" in captured.out or "API key added" in captured.out

    def test_auth_setup_empty_key_fails(self, temp_auth_dir, capsys, monkeypatch):
        from scholardevclaw.auth.cli import cmd_auth, _prompt_for_key
        from scholardevclaw.auth.types import AuthProvider
        from argparse import Namespace
        import getpass

        call_count = [0]

        def mock_input(prompt=""):
            call_count[0] += 1
            if call_count[0] == 1:
                return "1"  # select provider
            return ""  # empty name

        def mock_getpass(prompt=""):
            return ""  # empty key

        monkeypatch.setattr("builtins.input", mock_input)
        monkeypatch.setattr("getpass.getpass", mock_getpass)

        args = Namespace(
            auth_action="setup",
        )
        with pytest.raises(SystemExit):
            cmd_auth(args)

    def test_auth_multiple_providers(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider, AuthConfig
        from argparse import Namespace

        store = AuthStore(temp_auth_dir)
        store.add_api_key("sk_ant", "anthropic-key", AuthProvider.ANTHROPIC)
        store.add_api_key("sk_oi", "openai-key", AuthProvider.OPENAI)
        store.add_api_key("sk_gh", "github-key", AuthProvider.GITHUB)

        args = Namespace(
            auth_action="status",
            output_json=False,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "anthropic" in captured.out.lower()
        assert "openai" in captured.out.lower()
        assert "github" in captured.out.lower()


class TestAuthCLIErrorHandling:
    """Tests for error handling in auth CLI"""

    @pytest.fixture
    def temp_auth_dir(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("SCHOLARDEVCLAW_AUTH_DIR", tmpdir)
            yield tmpdir

    def test_corrupted_auth_file_recovery(self, temp_auth_dir, capsys):
        auth_file = Path(temp_auth_dir) / "auth.json"
        auth_file.write_text("{invalid json content")

        from scholardevclaw.auth.cli import cmd_auth
        from argparse import Namespace

        args = Namespace(
            auth_action="status",
            output_json=False,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "Not authenticated" in captured.out

    def test_add_key_with_invalid_provider(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from scholardevclaw.auth.store import AuthStore
        from argparse import Namespace

        args = Namespace(
            auth_action="add",
            key="sk_test",
            name="test-key",
            provider="random_invalid",
            default=False,
            output_json=False,
        )
        # Invalid provider raises ValueError
        with pytest.raises(ValueError):
            cmd_auth(args)

        # Verify key was NOT added
        store = AuthStore(temp_auth_dir)
        keys = store.list_api_keys()
        assert len(keys) == 0

    def test_auth_list_json_with_multiple_keys(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider
        from argparse import Namespace

        store = AuthStore(temp_auth_dir)
        store.add_api_key("sk_1", "key1", AuthProvider.ANTHROPIC)
        store.add_api_key("sk_2", "key2", AuthProvider.OPENAI)
        store.add_api_key("sk_3", "key3", AuthProvider.GITHUB)

        args = Namespace(
            auth_action="list",
            output_json=True,
        )
        cmd_auth(args)
        captured = capsys.readouterr()

        import json
        import re

        json_match = re.search(r"(\[[\s\S]*\])", captured.out)
        data = json.loads(json_match.group(1))
        assert len(data) == 3

    def test_auth_status_with_profile(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider
        from argparse import Namespace

        store = AuthStore(temp_auth_dir)
        store.add_api_key("sk_test", "test", AuthProvider.ANTHROPIC)
        store.create_profile(email="profile@test.com", name="Test Profile")

        args = Namespace(
            auth_action="status",
            output_json=False,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "profile@test.com" in captured.out
        assert "Test Profile" in captured.out

    def test_auth_remove_last_key(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider
        from argparse import Namespace

        store = AuthStore(temp_auth_dir)
        key = store.add_api_key("sk_test", "test", AuthProvider.ANTHROPIC)

        args = Namespace(
            auth_action="remove",
            key_id=key.id,
        )
        cmd_auth(args)
        captured = capsys.readouterr()

        store2 = AuthStore(temp_auth_dir)
        assert not store2.is_authenticated()

    def test_auth_default_after_remove(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider
        from argparse import Namespace

        store = AuthStore(temp_auth_dir)
        key1 = store.add_api_key("sk_1", "key1", AuthProvider.ANTHROPIC)
        key2 = store.add_api_key("sk_2", "key2", AuthProvider.OPENAI)

        store.set_default_key(key1.id)

        args = Namespace(
            auth_action="remove",
            key_id=key1.id,
        )
        cmd_auth(args)

        store2 = AuthStore(temp_auth_dir)
        config = store2.get_config()
        assert config.default_key_id == key2.id

    def test_auth_login_empty_key(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        import getpass
        from argparse import Namespace
        from unittest.mock import patch

        with patch("getpass.getpass", return_value=""):
            args = Namespace(
                auth_action="login",
                provider="anthropic",
                key=None,
                name=None,
                output_json=False,
            )
            with pytest.raises(SystemExit):
                cmd_auth(args)

    def test_auth_add_with_special_chars_in_name(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from argparse import Namespace

        args = Namespace(
            auth_action="add",
            key="sk_test",
            name="my-key_v1.0",
            provider="custom",
            default=True,
            output_json=False,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "API key added" in captured.out

    def test_auth_status_tier_display(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider, SubscriptionTier
        from argparse import Namespace

        store = AuthStore(temp_auth_dir)
        store.add_api_key("sk_test", "test", AuthProvider.ANTHROPIC)
        profile = store.create_profile(email="test@test.com")

        config = store.get_config()
        config.profile.subscription_tier = SubscriptionTier.PRO
        store._save_config(config)

        args = Namespace(
            auth_action="status",
            output_json=False,
        )
        cmd_auth(args)
        captured = capsys.readouterr()
        assert "Tier:" in captured.out

    def test_auth_list_empty_output(self, temp_auth_dir, capsys):
        from scholardevclaw.auth.cli import cmd_auth
        from argparse import Namespace

        args = Namespace(
            auth_action="list",
            output_json=True,
        )
        cmd_auth(args)
        captured = capsys.readouterr()

        import json

        data = json.loads(captured.out.strip())
        assert data == []

    def test_auth_setup_with_all_providers(self, temp_auth_dir, capsys, monkeypatch):
        from scholardevclaw.auth.cli import cmd_auth
        import getpass
        from argparse import Namespace
        from unittest.mock import patch

        responses = [
            "2",  # select OpenAI
            "sk_openai_key",  # provide key
            "",  # skip email
        ]
        response_iter = iter(responses)

        def mock_input(prompt=""):
            return next(response_iter)

        with patch("getpass.getpass", return_value="sk_openai_key"):
            with patch("builtins.input", mock_input):
                args = Namespace(
                    auth_action="setup",
                )
                cmd_auth(args)
                captured = capsys.readouterr()
                assert "Setup complete" in captured.out or "API key added" in captured.out
