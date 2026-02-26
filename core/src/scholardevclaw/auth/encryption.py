"""Encryption at rest for auth credentials.

Uses Fernet symmetric encryption derived from a user-supplied master password
via PBKDF2-HMAC-SHA256. Salt is persisted alongside the encrypted data so
re-derive is deterministic for the same password.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any


def _derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 32-byte Fernet key from password + salt using PBKDF2."""
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations=600_000)
    return base64.urlsafe_b64encode(dk)


class EncryptionManager:
    """Manages encryption/decryption of auth data at rest."""

    SALT_FILE = "encryption.salt"
    MARKER_FILE = "encryption.enabled"

    def __init__(self, store_dir: str | Path) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._fernet: Any | None = None

    @property
    def is_enabled(self) -> bool:
        return (self.store_dir / self.MARKER_FILE).exists()

    def enable(self, password: str) -> None:
        """Enable encryption with the given master password."""
        from cryptography.fernet import Fernet  # type: ignore[import-untyped]

        salt_path = self.store_dir / self.SALT_FILE
        if salt_path.exists():
            salt = salt_path.read_bytes()
        else:
            salt = secrets.token_bytes(32)
            salt_path.write_bytes(salt)

        key = _derive_key(password, salt)
        self._fernet = Fernet(key)

        # Write marker
        marker = self.store_dir / self.MARKER_FILE
        marker.write_text("enabled")

    def unlock(self, password: str) -> bool:
        """Unlock encryption with the given password. Returns True on success."""
        from cryptography.fernet import Fernet, InvalidToken  # type: ignore[import-untyped]

        salt_path = self.store_dir / self.SALT_FILE
        if not salt_path.exists():
            return False

        salt = salt_path.read_bytes()
        key = _derive_key(password, salt)
        self._fernet = Fernet(key)
        return True

    def disable(self) -> None:
        """Disable encryption."""
        self._fernet = None
        marker = self.store_dir / self.MARKER_FILE
        if marker.exists():
            marker.unlink()

    def encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext string, return base64 ciphertext."""
        if self._fernet is None:
            raise RuntimeError("Encryption not unlocked. Call enable() or unlock() first.")
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt base64 ciphertext, return plaintext string."""
        if self._fernet is None:
            raise RuntimeError("Encryption not unlocked. Call enable() or unlock() first.")
        from cryptography.fernet import InvalidToken  # type: ignore[import-untyped]

        try:
            return self._fernet.decrypt(ciphertext.encode()).decode()
        except InvalidToken:
            raise ValueError("Decryption failed — wrong password or corrupted data")

    def encrypt_dict(self, data: dict[str, Any]) -> str:
        """Serialize dict to JSON then encrypt."""
        return self.encrypt(json.dumps(data))

    def decrypt_dict(self, ciphertext: str) -> dict[str, Any]:
        """Decrypt then deserialize JSON to dict."""
        return json.loads(self.decrypt(ciphertext))

    def change_password(self, old_password: str, new_password: str, auth_file: Path) -> bool:
        """Re-encrypt auth data with a new password.

        Returns True on success, False if old password is wrong.
        """
        from cryptography.fernet import Fernet, InvalidToken  # type: ignore[import-untyped]

        salt_path = self.store_dir / self.SALT_FILE
        if not salt_path.exists():
            return False

        salt = salt_path.read_bytes()
        old_key = _derive_key(old_password, salt)
        old_fernet = Fernet(old_key)

        if not auth_file.exists():
            return False

        raw = auth_file.read_text().strip()
        try:
            plaintext = old_fernet.decrypt(raw.encode()).decode()
        except InvalidToken:
            return False

        # Generate new salt + key
        new_salt = secrets.token_bytes(32)
        new_key = _derive_key(new_password, new_salt)
        new_fernet = Fernet(new_key)

        encrypted = new_fernet.encrypt(plaintext.encode()).decode()
        auth_file.write_text(encrypted)
        salt_path.write_bytes(new_salt)

        self._fernet = new_fernet
        return True


class FallbackEncryptionManager:
    """No-op encryption manager when cryptography is not installed."""

    SALT_FILE = "encryption.salt"
    MARKER_FILE = "encryption.enabled"

    def __init__(self, store_dir: str | Path) -> None:
        self.store_dir = Path(store_dir)

    @property
    def is_enabled(self) -> bool:
        return False

    def enable(self, password: str) -> None:
        raise RuntimeError(
            "Encryption requires the 'cryptography' package. "
            "Install it with: pip install cryptography"
        )

    def unlock(self, password: str) -> bool:
        raise RuntimeError(
            "Encryption requires the 'cryptography' package. "
            "Install it with: pip install cryptography"
        )

    def disable(self) -> None:
        pass

    def encrypt(self, plaintext: str) -> str:
        raise RuntimeError("Encryption not available")

    def decrypt(self, ciphertext: str) -> str:
        raise RuntimeError("Encryption not available")

    def encrypt_dict(self, data: dict[str, Any]) -> str:
        raise RuntimeError("Encryption not available")

    def decrypt_dict(self, ciphertext: str) -> dict[str, Any]:
        raise RuntimeError("Encryption not available")

    def change_password(self, old_password: str, new_password: str, auth_file: Path) -> bool:
        raise RuntimeError("Encryption not available")


def get_encryption_manager(store_dir: str | Path) -> EncryptionManager | FallbackEncryptionManager:
    """Factory that returns the real manager if cryptography is installed, else fallback."""
    try:
        import cryptography  # noqa: F401

        return EncryptionManager(store_dir)
    except ImportError:
        return FallbackEncryptionManager(store_dir)
