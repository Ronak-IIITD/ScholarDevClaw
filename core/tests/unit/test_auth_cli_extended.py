"""Tests for new CLI commands: rotate, audit, export, import, encrypt, profiles, usage, expiry."""

import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scholardevclaw.auth.cli import (
    _cmd_audit,
    _cmd_encrypt,
    _cmd_expiry,
    _cmd_export,
    _cmd_import,
    _cmd_profiles,
    _cmd_rotate,
    _cmd_usage,
    cmd_auth,
)
from scholardevclaw.auth.store import AuthStore
from scholardevclaw.auth.types import AuthProvider


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def store(temp_dir):
    return AuthStore(str(temp_dir))


def _make_args(**kwargs):
    defaults = {
        "output_json": False,
        "key": None,
        "key_id": None,
        "name": None,
        "provider": None,
        "force": False,
        "default": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# -------------------------------------------------------------------
# Rotate command
# -------------------------------------------------------------------


class TestCmdRotate:
    def test_rotate_success(self, store, temp_dir, capsys):
        key = store.add_api_key("sk-old-key-12345", "old", AuthProvider.CUSTOM)
        args = _make_args(
            auth_action="rotate",
            key_id=key.id,
            new_key="sk-new-key-12345",
            reason="scheduled rotation",
        )
        _cmd_rotate(args, store)
        out = capsys.readouterr().out
        assert "rotated successfully" in out

    def test_rotate_missing_key_id(self, store, capsys):
        args = _make_args(auth_action="rotate", key_id=None, new_key=None, reason=None)
        with pytest.raises(SystemExit):
            _cmd_rotate(args, store)

    def test_rotate_nonexistent_key(self, store, capsys):
        args = _make_args(
            auth_action="rotate",
            key_id="nonexistent",
            new_key="sk-new",
            reason=None,
        )
        with pytest.raises(SystemExit):
            _cmd_rotate(args, store)

    def test_rotate_prompts_for_key(self, store, temp_dir, capsys):
        key = store.add_api_key("sk-old-key-12345", "old", AuthProvider.CUSTOM)
        args = _make_args(auth_action="rotate", key_id=key.id, new_key=None, reason=None)
        with patch("getpass.getpass", return_value="sk-prompted-key"):
            _cmd_rotate(args, store)
        out = capsys.readouterr().out
        assert "rotated successfully" in out


# -------------------------------------------------------------------
# Audit command
# -------------------------------------------------------------------


class TestCmdAudit:
    def test_audit_no_events(self, store, capsys):
        args = _make_args(auth_action="audit", limit=20, key_id=None, output_json=False)
        _cmd_audit(args, store)
        out = capsys.readouterr().out
        assert "No audit events" in out

    def test_audit_with_events(self, store, capsys):
        store.add_api_key("sk-test-123456789", "k", AuthProvider.CUSTOM)
        args = _make_args(auth_action="audit", limit=20, key_id=None, output_json=False)
        _cmd_audit(args, store)
        out = capsys.readouterr().out
        assert "Audit Log" in out

    def test_audit_json_output(self, store, capsys):
        store.add_api_key("sk-test-123456789", "k", AuthProvider.CUSTOM)
        args = _make_args(auth_action="audit", limit=20, key_id=None, output_json=True)
        _cmd_audit(args, store)
        out = capsys.readouterr().out
        events = json.loads(out)
        assert isinstance(events, list)

    def test_audit_disabled(self, temp_dir, capsys):
        store = AuthStore(str(temp_dir), enable_audit=False)
        args = _make_args(auth_action="audit", limit=20, key_id=None, output_json=False)
        with pytest.raises(SystemExit):
            _cmd_audit(args, store)


# -------------------------------------------------------------------
# Export command
# -------------------------------------------------------------------


class TestCmdExport:
    def test_export_json_stdout(self, store, capsys):
        store.add_api_key("sk-test-123456789", "my-key", AuthProvider.CUSTOM)
        args = _make_args(auth_action="export", format="json", output=None, redact=False)
        _cmd_export(args, store)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert len(data["api_keys"]) == 1

    def test_export_env_stdout(self, store, capsys):
        store.add_api_key("sk-test-123456789", "my-key", AuthProvider.CUSTOM)
        args = _make_args(auth_action="export", format="env", output=None, redact=False)
        _cmd_export(args, store)
        out = capsys.readouterr().out
        assert "SCHOLARDEVCLAW_API_KEY=" in out

    def test_export_to_file(self, store, temp_dir, capsys):
        store.add_api_key("sk-test-123456789", "my-key", AuthProvider.CUSTOM)
        output_path = str(temp_dir / "exported.json")
        args = _make_args(auth_action="export", format="json", output=output_path, redact=False)
        _cmd_export(args, store)
        out = capsys.readouterr().out
        assert "Exported to" in out
        assert Path(output_path).exists()

    def test_export_redacted(self, store, capsys):
        store.add_api_key("sk-test-123456789", "my-key", AuthProvider.CUSTOM)
        args = _make_args(auth_action="export", format="json", output=None, redact=True)
        _cmd_export(args, store)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["api_keys"][0]["key"] == "***REDACTED***"


# -------------------------------------------------------------------
# Import command
# -------------------------------------------------------------------


class TestCmdImport:
    def test_import_missing_source(self, store, capsys):
        args = _make_args(auth_action="import", source=None, format="auto")
        with pytest.raises(SystemExit):
            _cmd_import(args, store)

    def test_import_file_not_found(self, store, capsys):
        args = _make_args(auth_action="import", source="/nonexistent/file.json", format="auto")
        with pytest.raises(SystemExit):
            _cmd_import(args, store)

    def test_import_env_file(self, store, temp_dir, capsys):
        env_file = temp_dir / "keys.env"
        env_file.write_text("ANTHROPIC_API_KEY=sk-ant-imported-key\n")
        args = _make_args(auth_action="import", source=str(env_file), format="env")
        _cmd_import(args, store)
        out = capsys.readouterr().out
        assert "Imported 1" in out

    def test_import_json_file(self, store, temp_dir, capsys):
        json_file = temp_dir / "keys.json"
        from scholardevclaw.auth.import_export import AuthExporter
        from scholardevclaw.auth.types import APIKey, AuthConfig

        config = AuthConfig()
        config.api_keys.append(
            APIKey(id="key_1", name="test", provider=AuthProvider.CUSTOM, key="sk-test")
        )
        json_file.write_text(AuthExporter.to_json(config))

        args = _make_args(auth_action="import", source=str(json_file), format="json")
        _cmd_import(args, store)
        out = capsys.readouterr().out
        assert "Imported 1" in out

    def test_import_csv_file(self, store, temp_dir, capsys):
        csv_file = temp_dir / "keys.csv"
        csv_file.write_text("""Title,Username,Password,URL,Notes
My Key,u,sk-ant-key123,https://anthropic.com,
""")
        args = _make_args(auth_action="import", source=str(csv_file), format="1password")
        _cmd_import(args, store)
        out = capsys.readouterr().out
        assert "Imported 1" in out

    def test_import_auto_detect_csv(self, store, temp_dir, capsys):
        csv_file = temp_dir / "export.csv"
        csv_file.write_text("""Title,Username,Password,URL,Notes
Key,u,sk-test,https://example.com,
""")
        args = _make_args(auth_action="import", source=str(csv_file), format="auto")
        _cmd_import(args, store)
        out = capsys.readouterr().out
        assert "Imported 1" in out

    def test_import_auto_detect_env(self, store, temp_dir, capsys):
        env_file = temp_dir / "test.env"
        env_file.write_text("OPENAI_API_KEY=sk-openai-key\n")
        args = _make_args(auth_action="import", source=str(env_file), format="auto")
        _cmd_import(args, store)
        out = capsys.readouterr().out
        assert "Imported 1" in out


# -------------------------------------------------------------------
# Encrypt command
# -------------------------------------------------------------------


class TestCmdEncrypt:
    def test_encrypt_status_disabled(self, store, capsys):
        args = _make_args(auth_action="encrypt", encrypt_action="status")
        _cmd_encrypt(args, store)
        out = capsys.readouterr().out
        assert "disabled" in out

    def test_encrypt_enable(self, store, capsys):
        args = _make_args(auth_action="encrypt", encrypt_action="enable")
        with patch("getpass.getpass", side_effect=["testpass123", "testpass123"]):
            _cmd_encrypt(args, store)
        out = capsys.readouterr().out
        assert "Encryption enabled" in out

    def test_encrypt_enable_mismatch(self, store, capsys):
        args = _make_args(auth_action="encrypt", encrypt_action="enable")
        with patch("getpass.getpass", side_effect=["pass1", "pass2"]):
            with pytest.raises(SystemExit):
                _cmd_encrypt(args, store)

    def test_encrypt_enable_empty_password(self, store, capsys):
        args = _make_args(auth_action="encrypt", encrypt_action="enable")
        with patch("getpass.getpass", return_value=""):
            with pytest.raises(SystemExit):
                _cmd_encrypt(args, store)

    def test_encrypt_disable(self, store, capsys):
        # First enable
        store.enable_encryption("testpass")
        args = _make_args(auth_action="encrypt", encrypt_action="disable")
        with patch("getpass.getpass", return_value="testpass"):
            _cmd_encrypt(args, store)
        out = capsys.readouterr().out
        assert "disabled" in out

    def test_encrypt_disable_wrong_password(self, store, capsys):
        store.enable_encryption("testpass")
        args = _make_args(auth_action="encrypt", encrypt_action="disable")
        with patch("getpass.getpass", return_value="wrongpass"):
            with pytest.raises(SystemExit):
                _cmd_encrypt(args, store)

    def test_encrypt_status_enabled(self, store, capsys):
        store.enable_encryption("testpass")
        args = _make_args(auth_action="encrypt", encrypt_action="status")
        _cmd_encrypt(args, store)
        out = capsys.readouterr().out
        assert "enabled" in out

    def test_encrypt_unknown_action(self, store, capsys):
        args = _make_args(auth_action="encrypt", encrypt_action="unknown")
        _cmd_encrypt(args, store)
        out = capsys.readouterr().out
        assert "auth encrypt" in out


# -------------------------------------------------------------------
# Profiles command
# -------------------------------------------------------------------


class TestCmdProfiles:
    def test_profiles_list_empty(self, store, capsys):
        args = _make_args(auth_action="profiles", profile_action="list")
        _cmd_profiles(args, store)
        out = capsys.readouterr().out
        assert "No saved profiles" in out

    def test_profiles_save(self, store, capsys):
        store.add_api_key("sk-test", "k", AuthProvider.CUSTOM)
        args = _make_args(auth_action="profiles", profile_action="save", profile_name="test")
        _cmd_profiles(args, store)
        out = capsys.readouterr().out
        assert "Profile saved" in out

    def test_profiles_save_no_name(self, store, capsys):
        args = _make_args(auth_action="profiles", profile_action="save", profile_name=None)
        with pytest.raises(SystemExit):
            _cmd_profiles(args, store)

    def test_profiles_list_with_profiles(self, store, capsys):
        store.add_api_key("sk-test", "k", AuthProvider.CUSTOM)
        store.save_profile_as("work")
        store.save_profile_as("personal")
        args = _make_args(auth_action="profiles", profile_action="list")
        _cmd_profiles(args, store)
        out = capsys.readouterr().out
        assert "work" in out
        assert "personal" in out

    def test_profiles_load(self, store, capsys):
        store.add_api_key("sk-test", "k", AuthProvider.CUSTOM)
        store.save_profile_as("myprofile")
        args = _make_args(auth_action="profiles", profile_action="load", profile_name="myprofile")
        _cmd_profiles(args, store)
        out = capsys.readouterr().out
        assert "Switched to profile" in out

    def test_profiles_load_not_found(self, store, capsys):
        args = _make_args(auth_action="profiles", profile_action="load", profile_name="nonexistent")
        with pytest.raises(SystemExit):
            _cmd_profiles(args, store)

    def test_profiles_load_no_name(self, store, capsys):
        args = _make_args(auth_action="profiles", profile_action="load", profile_name=None)
        with pytest.raises(SystemExit):
            _cmd_profiles(args, store)

    def test_profiles_delete(self, store, capsys):
        store.add_api_key("sk-test", "k", AuthProvider.CUSTOM)
        store.save_profile_as("todelete")
        args = _make_args(auth_action="profiles", profile_action="delete", profile_name="todelete")
        _cmd_profiles(args, store)
        out = capsys.readouterr().out
        assert "Profile deleted" in out

    def test_profiles_delete_not_found(self, store, capsys):
        args = _make_args(
            auth_action="profiles", profile_action="delete", profile_name="nonexistent"
        )
        with pytest.raises(SystemExit):
            _cmd_profiles(args, store)

    def test_profiles_delete_no_name(self, store, capsys):
        args = _make_args(auth_action="profiles", profile_action="delete", profile_name=None)
        with pytest.raises(SystemExit):
            _cmd_profiles(args, store)

    def test_profiles_unknown_action(self, store, capsys):
        args = _make_args(auth_action="profiles", profile_action="unknown")
        _cmd_profiles(args, store)
        out = capsys.readouterr().out
        assert "auth profiles" in out


# -------------------------------------------------------------------
# Usage command
# -------------------------------------------------------------------


class TestCmdUsage:
    def test_usage_empty(self, store, capsys):
        args = _make_args(auth_action="usage", key_id=None, output_json=False)
        _cmd_usage(args, store)
        out = capsys.readouterr().out
        assert "No usage data" in out

    def test_usage_with_data(self, store, capsys):
        key = store.add_api_key("sk-test-123456789", "k", AuthProvider.CUSTOM)
        store.get_api_key_with_rate_check()
        args = _make_args(auth_action="usage", key_id=None, output_json=False)
        _cmd_usage(args, store)
        out = capsys.readouterr().out
        assert "Usage Stats" in out

    def test_usage_specific_key(self, store, capsys):
        key = store.add_api_key("sk-test-123456789", "k", AuthProvider.CUSTOM)
        store.get_api_key_with_rate_check()
        args = _make_args(auth_action="usage", key_id=key.id, output_json=False)
        _cmd_usage(args, store)
        out = capsys.readouterr().out
        assert "Usage Stats" in out

    def test_usage_json_output(self, store, capsys):
        key = store.add_api_key("sk-test-123456789", "k", AuthProvider.CUSTOM)
        store.get_api_key_with_rate_check()
        args = _make_args(auth_action="usage", key_id=None, output_json=True)
        _cmd_usage(args, store)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert isinstance(data, dict)


# -------------------------------------------------------------------
# Expiry command
# -------------------------------------------------------------------


class TestCmdExpiry:
    def test_expiry_check_none(self, store, capsys):
        store.add_api_key("sk-test", "k", AuthProvider.CUSTOM)
        args = _make_args(auth_action="expiry", expiry_action="check")
        _cmd_expiry(args, store)
        out = capsys.readouterr().out
        assert "No keys expiring" in out

    def test_expiry_check_with_expiring(self, store, capsys):
        expires = (datetime.now() + timedelta(days=5)).isoformat()
        store.add_api_key("sk-test", "expiring-key", AuthProvider.CUSTOM, expires_at=expires)
        args = _make_args(auth_action="expiry", expiry_action="check")
        _cmd_expiry(args, store)
        out = capsys.readouterr().out
        assert "expiring soon" in out

    def test_expiry_set(self, store, capsys):
        key = store.add_api_key("sk-test", "k", AuthProvider.CUSTOM)
        expires = (datetime.now() + timedelta(days=30)).isoformat()
        args = _make_args(
            auth_action="expiry", expiry_action="set", key_id=key.id, expires_at=expires
        )
        _cmd_expiry(args, store)
        out = capsys.readouterr().out
        assert "Expiry set" in out

    def test_expiry_set_missing_args(self, store, capsys):
        args = _make_args(auth_action="expiry", expiry_action="set", key_id=None, expires_at=None)
        with pytest.raises(SystemExit):
            _cmd_expiry(args, store)

    def test_expiry_set_invalid_date(self, store, capsys):
        key = store.add_api_key("sk-test", "k", AuthProvider.CUSTOM)
        args = _make_args(
            auth_action="expiry", expiry_action="set", key_id=key.id, expires_at="not-a-date"
        )
        with pytest.raises(SystemExit):
            _cmd_expiry(args, store)

    def test_expiry_deactivate(self, store, capsys):
        expires = (datetime.now() - timedelta(days=1)).isoformat()
        store.add_api_key("sk-test", "expired-key", AuthProvider.CUSTOM, expires_at=expires)
        args = _make_args(auth_action="expiry", expiry_action="deactivate")
        _cmd_expiry(args, store)
        out = capsys.readouterr().out
        assert "Deactivated" in out

    def test_expiry_deactivate_none(self, store, capsys):
        store.add_api_key("sk-test", "valid-key", AuthProvider.CUSTOM)
        args = _make_args(auth_action="expiry", expiry_action="deactivate")
        _cmd_expiry(args, store)
        out = capsys.readouterr().out
        assert "No expired keys" in out

    def test_expiry_unknown_action(self, store, capsys):
        args = _make_args(auth_action="expiry", expiry_action="unknown")
        _cmd_expiry(args, store)
        out = capsys.readouterr().out
        assert "auth expiry" in out


# -------------------------------------------------------------------
# Dispatch tests
# -------------------------------------------------------------------


class TestCmdAuthDispatch:
    def test_dispatch_rotate(self, store, temp_dir, capsys, monkeypatch):
        key = store.add_api_key("sk-test-123456", "k", AuthProvider.CUSTOM)
        args = _make_args(
            auth_action="rotate",
            key_id=key.id,
            new_key="sk-rotated-key-123",
            reason="test",
        )
        # Patch AuthStore to use our temp dir
        monkeypatch.setattr(
            "scholardevclaw.auth.cli.AuthStore",
            lambda: AuthStore(str(temp_dir)),
        )
        cmd_auth(args)
        out = capsys.readouterr().out
        assert "rotated" in out

    def test_dispatch_unknown(self, capsys, monkeypatch, temp_dir):
        monkeypatch.setattr(
            "scholardevclaw.auth.cli.AuthStore",
            lambda: AuthStore(str(temp_dir)),
        )
        args = _make_args(auth_action="badaction")
        with pytest.raises(SystemExit):
            cmd_auth(args)
