"""Tests for encryption at rest (encryption.py)."""

import json
import tempfile
from pathlib import Path

import pytest

from scholardevclaw.auth.encryption import (
    EncryptionManager,
    FallbackEncryptionManager,
    _derive_key,
    get_encryption_manager,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestDeriveKey:
    def test_derive_key_returns_bytes(self):
        key = _derive_key("password", b"salt1234567890123456789012345678")
        assert isinstance(key, bytes)

    def test_derive_key_deterministic(self):
        salt = b"salt1234567890123456789012345678"
        key1 = _derive_key("password", salt)
        key2 = _derive_key("password", salt)
        assert key1 == key2

    def test_derive_key_different_passwords(self):
        salt = b"salt1234567890123456789012345678"
        key1 = _derive_key("password1", salt)
        key2 = _derive_key("password2", salt)
        assert key1 != key2

    def test_derive_key_different_salts(self):
        key1 = _derive_key("password", b"salt1234567890123456789012345678")
        key2 = _derive_key("password", b"different_salt__________________")
        assert key1 != key2

    def test_derive_key_urlsafe_base64(self):
        key = _derive_key("mypass", b"salt1234567890123456789012345678")
        # Fernet keys are 44 bytes base64-encoded
        assert len(key) == 44


class TestEncryptionManager:
    def test_init_creates_dir(self, temp_dir):
        sub = temp_dir / "nested" / "dir"
        mgr = EncryptionManager(sub)
        assert sub.exists()

    def test_not_enabled_by_default(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        assert mgr.is_enabled is False

    def test_enable_sets_enabled(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        mgr.enable("testpassword")
        assert mgr.is_enabled is True

    def test_enable_creates_salt_file(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        mgr.enable("testpassword")
        assert (temp_dir / "encryption.salt").exists()

    def test_enable_creates_marker_file(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        mgr.enable("testpassword")
        assert (temp_dir / "encryption.enabled").exists()

    def test_encrypt_decrypt_roundtrip(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        mgr.enable("testpassword")
        ciphertext = mgr.encrypt("hello world")
        plaintext = mgr.decrypt(ciphertext)
        assert plaintext == "hello world"

    def test_encrypt_produces_different_ciphertext(self, temp_dir):
        """Fernet generates unique ciphertexts each time (has timestamp + IV)."""
        mgr = EncryptionManager(temp_dir)
        mgr.enable("testpassword")
        c1 = mgr.encrypt("hello")
        c2 = mgr.encrypt("hello")
        assert c1 != c2

    def test_encrypt_dict_decrypt_dict_roundtrip(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        mgr.enable("testpassword")
        data = {"api_keys": [{"id": "key_1", "name": "test"}], "version": 1}
        ciphertext = mgr.encrypt_dict(data)
        result = mgr.decrypt_dict(ciphertext)
        assert result == data

    def test_decrypt_wrong_password_raises(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        mgr.enable("correctpassword")
        ciphertext = mgr.encrypt("secret data")

        # Create a new manager with wrong password
        mgr2 = EncryptionManager(temp_dir)
        result = mgr2.unlock("wrongpassword")
        # Security fix: unlock() now verifies password against stored token
        assert result is False
        with pytest.raises(RuntimeError, match="Encryption not unlocked"):
            mgr2.decrypt(ciphertext)

    def test_encrypt_without_enable_raises(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        with pytest.raises(RuntimeError, match="Encryption not unlocked"):
            mgr.encrypt("test")

    def test_decrypt_without_enable_raises(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        with pytest.raises(RuntimeError, match="Encryption not unlocked"):
            mgr.decrypt("test")

    def test_unlock_success(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        mgr.enable("testpassword")
        ciphertext = mgr.encrypt("secret")

        # New manager instance
        mgr2 = EncryptionManager(temp_dir)
        assert mgr2.unlock("testpassword") is True
        assert mgr2.decrypt(ciphertext) == "secret"

    def test_unlock_no_salt_file(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        assert mgr.unlock("anypassword") is False

    def test_disable_removes_marker(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        mgr.enable("testpassword")
        assert mgr.is_enabled is True
        mgr.disable()
        assert mgr.is_enabled is False

    def test_disable_clears_fernet(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        mgr.enable("testpassword")
        mgr.disable()
        with pytest.raises(RuntimeError):
            mgr.encrypt("test")

    def test_enable_reuses_existing_salt(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        mgr.enable("pass1")
        salt1 = (temp_dir / "encryption.salt").read_bytes()
        ciphertext = mgr.encrypt("data")

        # Enable again — should reuse the same salt
        mgr2 = EncryptionManager(temp_dir)
        mgr2.enable("pass1")
        salt2 = (temp_dir / "encryption.salt").read_bytes()
        assert salt1 == salt2
        # Should be able to decrypt with same password
        assert mgr2.decrypt(ciphertext) == "data"

    def test_change_password_success(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        mgr.enable("oldpass")

        auth_file = temp_dir / "auth.json"
        data = {"api_keys": [{"id": "k1"}]}
        encrypted = mgr.encrypt_dict(data)
        auth_file.write_text(encrypted)

        result = mgr.change_password("oldpass", "newpass", auth_file)
        assert result is True

        # Verify new password works
        mgr2 = EncryptionManager(temp_dir)
        mgr2.unlock("newpass")
        decrypted = mgr2.decrypt_dict(auth_file.read_text().strip())
        assert decrypted == data

    def test_change_password_wrong_old_password(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        mgr.enable("correctpass")

        auth_file = temp_dir / "auth.json"
        encrypted = mgr.encrypt("test data")
        auth_file.write_text(encrypted)

        result = mgr.change_password("wrongpass", "newpass", auth_file)
        assert result is False

    def test_change_password_no_salt(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        auth_file = temp_dir / "auth.json"
        result = mgr.change_password("old", "new", auth_file)
        assert result is False

    def test_change_password_no_auth_file(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        mgr.enable("pass")
        auth_file = temp_dir / "nonexistent.json"
        result = mgr.change_password("pass", "newpass", auth_file)
        assert result is False

    def test_encrypt_empty_string(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        mgr.enable("pass")
        ciphertext = mgr.encrypt("")
        assert mgr.decrypt(ciphertext) == ""

    def test_encrypt_unicode(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        mgr.enable("pass")
        text = "Hello 世界 🔑 مرحبا"
        ciphertext = mgr.encrypt(text)
        assert mgr.decrypt(ciphertext) == text

    def test_encrypt_large_data(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        mgr.enable("pass")
        data = "x" * 100_000
        ciphertext = mgr.encrypt(data)
        assert mgr.decrypt(ciphertext) == data

    def test_encrypt_dict_nested(self, temp_dir):
        mgr = EncryptionManager(temp_dir)
        mgr.enable("pass")
        data = {
            "keys": [{"id": "k1", "nested": {"deep": True}}],
            "count": 42,
            "tags": ["a", "b"],
        }
        ciphertext = mgr.encrypt_dict(data)
        assert mgr.decrypt_dict(ciphertext) == data


class TestFallbackEncryptionManager:
    def test_is_enabled_always_false(self, temp_dir):
        mgr = FallbackEncryptionManager(temp_dir)
        assert mgr.is_enabled is False

    def test_enable_raises(self, temp_dir):
        mgr = FallbackEncryptionManager(temp_dir)
        with pytest.raises(RuntimeError, match="cryptography"):
            mgr.enable("pass")

    def test_unlock_raises(self, temp_dir):
        mgr = FallbackEncryptionManager(temp_dir)
        with pytest.raises(RuntimeError, match="cryptography"):
            mgr.unlock("pass")

    def test_disable_is_noop(self, temp_dir):
        mgr = FallbackEncryptionManager(temp_dir)
        mgr.disable()  # Should not raise

    def test_encrypt_raises(self, temp_dir):
        mgr = FallbackEncryptionManager(temp_dir)
        with pytest.raises(RuntimeError):
            mgr.encrypt("test")

    def test_decrypt_raises(self, temp_dir):
        mgr = FallbackEncryptionManager(temp_dir)
        with pytest.raises(RuntimeError):
            mgr.decrypt("test")

    def test_encrypt_dict_raises(self, temp_dir):
        mgr = FallbackEncryptionManager(temp_dir)
        with pytest.raises(RuntimeError):
            mgr.encrypt_dict({"key": "value"})

    def test_decrypt_dict_raises(self, temp_dir):
        mgr = FallbackEncryptionManager(temp_dir)
        with pytest.raises(RuntimeError):
            mgr.decrypt_dict("ciphertext")

    def test_change_password_raises(self, temp_dir):
        mgr = FallbackEncryptionManager(temp_dir)
        with pytest.raises(RuntimeError):
            mgr.change_password("old", "new", Path("auth.json"))


class TestGetEncryptionManager:
    def test_returns_real_manager(self, temp_dir):
        # Since we installed cryptography, should return real manager
        mgr = get_encryption_manager(temp_dir)
        assert isinstance(mgr, EncryptionManager)


class TestEncryptionWithStore:
    """Integration tests: encryption + AuthStore."""

    def test_enable_encryption_encrypts_data(self, temp_dir):
        from scholardevclaw.auth.store import AuthStore

        store = AuthStore(str(temp_dir))
        store.add_api_key("sk-test-key-123456", "my-key", AuthProvider.CUSTOM)

        store.enable_encryption("masterpass")

        # Auth file should NOT be plain JSON anymore
        raw = (temp_dir / "auth.json").read_text().strip()
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)

    def test_encrypted_data_readable_after_unlock(self, temp_dir):
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(str(temp_dir))
        store.add_api_key("sk-test-key-123456", "my-key", AuthProvider.CUSTOM)
        store.enable_encryption("masterpass")

        # Create new store, unlock, and read
        store2 = AuthStore(str(temp_dir))
        store2.unlock_encryption("masterpass")
        keys = store2.list_api_keys()
        assert len(keys) == 1
        assert keys[0].key == "sk-test-key-123456"

    def test_disable_encryption_returns_plaintext(self, temp_dir):
        from scholardevclaw.auth.store import AuthStore
        from scholardevclaw.auth.types import AuthProvider

        store = AuthStore(str(temp_dir))
        store.add_api_key("sk-test-key-123456", "my-key", AuthProvider.CUSTOM)
        store.enable_encryption("masterpass")

        # Disable
        store2 = AuthStore(str(temp_dir))
        assert store2.disable_encryption("masterpass") is True

        # Should be readable as plain JSON now
        raw = (temp_dir / "auth.json").read_text().strip()
        data = json.loads(raw)
        assert len(data["api_keys"]) == 1

    def test_is_encryption_enabled(self, temp_dir):
        from scholardevclaw.auth.store import AuthStore

        store = AuthStore(str(temp_dir))
        assert store.is_encryption_enabled() is False
        store.enable_encryption("pass")
        assert store.is_encryption_enabled() is True


# Import needed for store integration tests
from scholardevclaw.auth.types import AuthProvider
