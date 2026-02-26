"""Tests for hardware key support."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from scholardevclaw.auth.hardware_keys import (
    HardwareKeyInfo,
    HardwareKeyManager,
    YubiKeyPIV,
)


class TestHardwareKeyInfo:
    def test_creation(self):
        info = HardwareKeyInfo(
            key_id="test_key",
            key_type="yubikey",
            serial="123456",
            label="Test Key",
            has_pin=True,
            slot=9,
        )
        assert info.key_id == "test_key"
        assert info.key_type == "yubikey"


class TestHardwareKeyManager:
    @pytest.fixture
    def store_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @patch("subprocess.run")
    def test_is_yubikey_available_true(self, mock_run, store_dir):
        mock_run.return_value = MagicMock(returncode=0)
        mgr = HardwareKeyManager(store_dir)
        assert mgr.is_yubikey_available() is True

    @patch("subprocess.run")
    def test_is_yubikey_available_false(self, mock_run, store_dir):
        mock_run.return_value = MagicMock(returncode=1)
        mgr = HardwareKeyManager(store_dir)
        assert mgr.is_yubikey_available() is False

    def test_is_yubikey_available_not_installed(self, store_dir):
        mgr = HardwareKeyManager(store_dir)
        with patch(
            "scholardevclaw.auth.hardware_keys.subprocess.run", side_effect=FileNotFoundError
        ):
            assert mgr.is_yubikey_available() is False

    def test_is_pkcs11_available_not_set(self, store_dir):
        mgr = HardwareKeyManager(store_dir)
        with patch.dict("os.environ", {}, clear=True):
            assert mgr.is_pkcs11_available() is False

    def test_is_pkcs11_available_set(self, store_dir):
        mgr = HardwareKeyManager(store_dir)
        with patch.dict("os.environ", {"PKCS11_MODULE": "/usr/lib/pkcs11.so"}):
            assert mgr.is_pkcs11_available() is True

    def test_store_key_reference(self, store_dir):
        mgr = HardwareKeyManager(store_dir)
        mgr.store_key_reference("test_key", "yubikey", "Test Key", slot=9)
        refs = mgr.list_key_references()
        assert "test_key" in refs
        assert refs["test_key"]["type"] == "yubikey"

    def test_remove_key_reference(self, store_dir):
        mgr = HardwareKeyManager(store_dir)
        mgr.store_key_reference("test_key", "yubikey", "Test Key")
        assert mgr.remove_key_reference("test_key") is True
        assert "test_key" not in mgr.list_key_references()


class TestYubiKeyPIV:
    @pytest.fixture
    def store_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_creation(self):
        piv = YubiKeyPIV(pin="123456")
        assert piv.pin == "123456"

    @patch("scholardevclaw.auth.hardware_keys.HardwareKeyManager.is_yubikey_available")
    def test_authenticate_not_available(self, mock_available, store_dir):
        mock_available.return_value = False
        piv = YubiKeyPIV()
        with pytest.raises(RuntimeError):
            piv.authenticate("123456")
