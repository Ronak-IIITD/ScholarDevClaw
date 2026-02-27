from __future__ import annotations

import json
import os
import re
import secrets
import stat
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .types import (
    APIKey,
    AuthConfig,
    AuthProvider,
    AuthStatus,
    KeyRotationEntry,
    KeyScope,
    UserProfile,
)
from .audit import AuditLogger, AuditEventType
from .rate_limit import RateLimiter, RateLimitConfig


class AuthStore:
    """Production-grade local credential store.

    Features:
    - CRUD for API keys with provider detection
    - Multi-profile support with workspace switching
    - Key rotation with history tracking
    - Key scoping (read/write/admin)
    - Audit logging for all operations
    - Per-key rate limiting and usage tracking
    - File permission hardening (chmod 600)
    - Encryption at rest (optional, via EncryptionManager)
    - Import/export (JSON, env, 1Password CSV)
    - Expiration alerts and auto-deactivation
    """

    PROFILES_DIR = "profiles"

    def __init__(
        self,
        store_dir: str | None = None,
        enable_audit: bool = True,
        enable_rate_limit: bool = True,
    ):
        if store_dir:
            self.store_dir = Path(store_dir)
        else:
            default_dir = os.environ.get("SCHOLARDEVCLAW_AUTH_DIR")
            if default_dir:
                self.store_dir = Path(default_dir)
            else:
                self.store_dir = Path.home() / ".scholardevclaw"

        self.store_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.store_dir, 0o700)
        except OSError:
            pass
        self.auth_file = self.store_dir / "auth.json"
        self._config: AuthConfig | None = None
        self._audit: AuditLogger | None = AuditLogger(str(self.store_dir)) if enable_audit else None
        self._rate_limiter: RateLimiter | None = (
            RateLimiter(str(self.store_dir)) if enable_rate_limit else None
        )
        self._encryption: Any | None = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Internal persistence
    # ------------------------------------------------------------------

    def _harden_file(self, path: Path) -> None:
        """Set file permissions to owner-only read/write (0600)."""
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError as e:
            import logging

            logging.getLogger(__name__).warning(
                "Could not harden file permissions on %s: %s", path, e
            )

    @staticmethod
    def _validate_profile_name(profile_name: str) -> str:
        """Validate profile name to prevent path traversal attacks.

        Only allows alphanumeric characters, hyphens, and underscores.
        Raises ValueError for invalid names.
        """
        if not profile_name:
            raise ValueError("Profile name cannot be empty")
        if not re.match(r"^[a-zA-Z0-9_-]+$", profile_name):
            raise ValueError(
                f"Invalid profile name '{profile_name}': "
                "only alphanumeric characters, hyphens, and underscores are allowed"
            )
        if profile_name in (".", ".."):
            raise ValueError("Profile name cannot be '.' or '..'")
        if len(profile_name) > 128:
            raise ValueError("Profile name too long (max 128 characters)")
        return profile_name

    def _atomic_write_text(self, path: Path, content: str) -> None:
        """Write text to a file atomically using temp + rename.

        Prevents TOCTOU race conditions and partial writes.
        """
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
        try:
            os.write(fd, content.encode())
            os.close(fd)
            fd = -1
            os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)
            os.rename(tmp_path, str(path))
        except Exception:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _load_config(self) -> AuthConfig:
        if self._config is not None:
            return self._config

        if not self.auth_file.exists():
            self._config = AuthConfig()
            return self._config

        try:
            raw = self.auth_file.read_text().strip()

            # If encryption is enabled, try to decrypt
            if self._encryption and self._encryption.is_enabled:
                try:
                    data = self._encryption.decrypt_dict(raw)
                except Exception:
                    # Could be plaintext migration scenario
                    data = json.loads(raw)
            else:
                data = json.loads(raw)

            self._config = AuthConfig.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            self._config = AuthConfig()

        return self._config

    def _save_config(self, config: AuthConfig) -> None:
        self._config = config
        data = config.to_dict()

        if self._encryption and self._encryption.is_enabled:
            encrypted = self._encryption.encrypt_dict(data)
            self._atomic_write_text(self.auth_file, encrypted)
        else:
            self._atomic_write_text(self.auth_file, json.dumps(data, indent=2))

    def _write_env_file(self, api_key: APIKey) -> None:
        """Write env file with provider info only. Never writes plaintext keys when encryption is on."""
        if self._encryption and self._encryption.is_enabled:
            # Do NOT write plaintext key to .env when encryption is enabled
            return

        env_file = self.store_dir / ".env"
        env_content = f"""# ScholarDevClaw Configuration
# Generated: {datetime.now().isoformat()}

# API Key for {api_key.provider.value}
SCHOLARDEVCLAW_API_KEY={api_key.key}
SCHOLARDEVCLAW_API_PROVIDER={api_key.provider.value}
"""
        self._atomic_write_text(env_file, env_content)

    # ------------------------------------------------------------------
    # Encryption support
    # ------------------------------------------------------------------

    def enable_encryption(self, password: str) -> None:
        """Enable encryption at rest. Encrypts existing auth data."""
        from .encryption import get_encryption_manager

        mgr = get_encryption_manager(str(self.store_dir))
        mgr.enable(password)
        self._encryption = mgr

        # Re-save current config encrypted
        config = self._load_config()
        self._save_config(config)

        if self._audit:
            self._audit.log(
                event_type=AuditEventType.CONFIG_CHANGED,
                details={"action": "encryption_enabled"},
            )

    def unlock_encryption(self, password: str) -> bool:
        """Unlock encrypted auth store."""
        from .encryption import get_encryption_manager

        mgr = get_encryption_manager(str(self.store_dir))
        if not mgr.is_enabled:
            return False
        ok = mgr.unlock(password)
        if ok:
            self._encryption = mgr
            self._config = None  # Force reload
        return ok

    def disable_encryption(self, password: str) -> bool:
        """Disable encryption and store data in plaintext.

        Verifies the password by attempting to decrypt existing data before
        disabling. Returns False if the password is wrong or encryption is
        not enabled.
        """
        from .encryption import get_encryption_manager

        mgr = get_encryption_manager(str(self.store_dir))
        if not mgr.is_enabled:
            return False

        if not mgr.unlock(password):
            return False

        # Verify the password actually decrypts the auth file. unlock()
        # only derives a key — it does NOT validate correctness.
        if self.auth_file.exists():
            raw = self.auth_file.read_text().strip()
            if raw:
                try:
                    mgr.decrypt_dict(raw)
                except Exception:
                    # Wrong password — decrypt failed
                    return False

        self._encryption = mgr
        self._config = None
        config = self._load_config()

        # Disable and re-save as plaintext
        mgr.disable()
        self._encryption = None
        self._save_config(config)
        return True

    def is_encryption_enabled(self) -> bool:
        """Check if encryption is currently enabled."""
        from .encryption import get_encryption_manager

        mgr = get_encryption_manager(str(self.store_dir))
        return mgr.is_enabled

    # ------------------------------------------------------------------
    # Auth status
    # ------------------------------------------------------------------

    def is_authenticated(self) -> bool:
        config = self._load_config()
        return len(config.api_keys) > 0

    def get_status(self) -> AuthStatus:
        config = self._load_config()

        active_keys = sum(1 for k in config.api_keys if k.is_valid())
        default_key = config.get_active_key()

        return AuthStatus(
            is_authenticated=len(config.api_keys) > 0,
            has_api_key=active_keys > 0,
            user_email=config.profile.email if config.profile else None,
            user_name=config.profile.name if config.profile else None,
            provider=default_key.provider.value if default_key else None,
            key_count=len(config.api_keys),
            active_keys=active_keys,
            subscription_tier=(
                config.profile.subscription_tier.value if config.profile else "free"
            ),
        )

    def get_config(self) -> AuthConfig:
        return self._load_config()

    # ------------------------------------------------------------------
    # API key CRUD
    # ------------------------------------------------------------------

    def get_api_key(self, provider: AuthProvider | None = None) -> str | None:
        env_key = os.environ.get("SCHOLARDEVCLAW_API_KEY")
        if env_key:
            return env_key

        config = self._load_config()
        key = config.get_active_key(provider)

        if key:
            key.last_used = datetime.now().isoformat()
            self._save_config(config)

            if self._audit:
                self._audit.log(
                    event_type=AuditEventType.KEY_ACCESSED,
                    key_id=key.id,
                    key_fingerprint=key.get_fingerprint(),
                    provider=key.provider.value,
                )

            return key.key

        return None

    def get_api_key_with_rate_check(
        self, provider: AuthProvider | None = None
    ) -> tuple[str | None, str]:
        """Get API key with rate limit check.

        Returns (key, status_message). If rate-limited, key is None.
        """
        env_key = os.environ.get("SCHOLARDEVCLAW_API_KEY")
        if env_key:
            return env_key, "OK"

        config = self._load_config()
        key_obj = config.get_active_key(provider)

        if not key_obj:
            return None, "No active key found"

        # Check rate limit
        if self._rate_limiter:
            allowed, reason = self._rate_limiter.check_rate_limit(key_obj.id)
            if not allowed:
                return None, reason
            self._rate_limiter.record_usage(key_obj.id, key_obj.provider.value)

        key_obj.last_used = datetime.now().isoformat()
        self._save_config(config)

        if self._audit:
            self._audit.log(
                event_type=AuditEventType.KEY_ACCESSED,
                key_id=key_obj.id,
                key_fingerprint=key_obj.get_fingerprint(),
                provider=key_obj.provider.value,
            )

        return key_obj.key, "OK"

    def add_api_key(
        self,
        key: str,
        name: str,
        provider: AuthProvider,
        set_default: bool = True,
        validate: bool = False,
        expires_at: str | None = None,
        scope: KeyScope = KeyScope.READ_WRITE,
        metadata: dict[str, Any] | None = None,
    ) -> APIKey:
        config = self._load_config()

        if validate:
            if provider != AuthProvider.CUSTOM and provider != AuthProvider.LOCAL:
                is_valid, message = APIKey.validate_key_format(key, provider)
                if not is_valid:
                    raise ValueError(f"Invalid key format for {provider.value}: {message}")

            is_valid, message = APIKey.is_valid_key_name(name)
            if not is_valid:
                raise ValueError(f"Invalid key name: {message}")

        api_key = APIKey(
            id=f"key_{secrets.token_hex(8)}",
            name=name,
            provider=provider,
            key=key,
            expires_at=expires_at,
            scope=scope,
            metadata=metadata or {},
        )

        config.api_keys.append(api_key)

        if set_default or not config.default_key_id:
            config.default_key_id = api_key.id
            config.default_provider = provider

        self._save_config(config)
        self._write_env_file(api_key)

        if self._audit:
            self._audit.log(
                event_type=AuditEventType.KEY_ADDED,
                key_id=api_key.id,
                key_fingerprint=api_key.get_fingerprint(),
                provider=provider.value,
                details={"name": name, "scope": scope.value},
            )

        return api_key

    def remove_api_key(self, key_id: str) -> bool:
        config = self._load_config()

        key_to_remove = config.get_key(key_id)
        if not key_to_remove:
            return False

        original_count = len(config.api_keys)
        config.api_keys = [k for k in config.api_keys if k.id != key_id]

        if len(config.api_keys) == original_count:
            return False

        if config.default_key_id == key_id:
            config.default_key_id = config.api_keys[0].id if config.api_keys else None

        self._save_config(config)

        if self._audit and key_to_remove:
            self._audit.log(
                event_type=AuditEventType.KEY_REMOVED,
                key_id=key_id,
                key_fingerprint=key_to_remove.get_fingerprint(),
                provider=key_to_remove.provider.value,
                details={"name": key_to_remove.name},
            )

        return True

    def set_default_key(self, key_id: str) -> bool:
        config = self._load_config()

        key = config.get_key(key_id)
        if not key:
            return False

        config.default_key_id = key_id
        config.default_provider = key.provider
        self._save_config(config)
        return True

    def list_api_keys(self) -> list[APIKey]:
        config = self._load_config()
        return config.api_keys

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def create_profile(
        self,
        email: str | None = None,
        name: str | None = None,
        validate: bool = False,
    ) -> UserProfile:
        config = self._load_config()

        if validate and email:
            if not APIKey.is_valid_email(email):
                raise ValueError(f"Invalid email format: {email}")

        profile = UserProfile(
            id=f"user_{secrets.token_hex(8)}",
            email=email,
            name=name,
        )

        config.profile = profile
        self._save_config(config)

        if self._audit:
            self._audit.log(
                event_type=AuditEventType.PROFILE_CREATED,
                user_email=email,
                details={"name": name},
            )

        return profile

    def update_profile(
        self,
        email: str | None = None,
        name: str | None = None,
        preferences: dict[str, Any] | None = None,
    ) -> UserProfile | None:
        config = self._load_config()

        if not config.profile:
            return None

        if email is not None:
            config.profile.email = email
        if name is not None:
            config.profile.name = name
        if preferences is not None:
            config.profile.preferences.update(preferences)

        self._save_config(config)

        if self._audit:
            self._audit.log(
                event_type=AuditEventType.PROFILE_UPDATED,
                user_email=config.profile.email,
            )

        return config.profile

    def get_profile(self) -> UserProfile | None:
        config = self._load_config()
        return config.profile

    # ------------------------------------------------------------------
    # Multi-profile / workspace support
    # ------------------------------------------------------------------

    def list_profiles(self) -> list[str]:
        """List all saved profile names (workspaces)."""
        profiles_dir = self.store_dir / self.PROFILES_DIR
        if not profiles_dir.exists():
            return []
        return [p.stem for p in profiles_dir.iterdir() if p.suffix == ".json" and p.is_file()]

    def save_profile_as(self, profile_name: str) -> Path:
        """Save current config as a named profile."""
        self._validate_profile_name(profile_name)

        profiles_dir = self.store_dir / self.PROFILES_DIR
        profiles_dir.mkdir(parents=True, exist_ok=True)

        config = self._load_config()
        profile_path = profiles_dir / f"{profile_name}.json"

        self._atomic_write_text(profile_path, json.dumps(config.to_dict(), indent=2))
        return profile_path

    def load_profile(self, profile_name: str) -> bool:
        """Switch to a saved profile."""
        self._validate_profile_name(profile_name)

        profiles_dir = self.store_dir / self.PROFILES_DIR
        profile_path = profiles_dir / f"{profile_name}.json"

        if not profile_path.exists():
            return False

        try:
            with open(profile_path) as f:
                data = json.load(f)
            config = AuthConfig.from_dict(data)
            self._save_config(config)

            if self._audit:
                self._audit.log(
                    event_type=AuditEventType.CONFIG_CHANGED,
                    details={"action": "profile_switched", "profile": profile_name},
                )

            return True
        except (json.JSONDecodeError, KeyError):
            return False

    def delete_profile(self, profile_name: str) -> bool:
        """Delete a saved profile."""
        self._validate_profile_name(profile_name)

        profiles_dir = self.store_dir / self.PROFILES_DIR
        profile_path = profiles_dir / f"{profile_name}.json"

        if not profile_path.exists():
            return False

        profile_path.unlink()
        return True

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def logout(self) -> bool:
        config = self._load_config()

        if not config.api_keys and not config.profile:
            return False

        if self._audit:
            self._audit.log(event_type=AuditEventType.LOGOUT)

        self._config = AuthConfig()
        self._save_config(AuthConfig())

        env_file = self.store_dir / ".env"
        if env_file.exists():
            env_file.unlink()

        return True

    def clear_all(self) -> None:
        self._config = AuthConfig()
        self._save_config(AuthConfig())

        if self.auth_file.exists():
            self.auth_file.unlink()

        env_file = self.store_dir / ".env"
        if env_file.exists():
            env_file.unlink()

    # ------------------------------------------------------------------
    # Key rotation
    # ------------------------------------------------------------------

    def rotate_api_key(
        self,
        key_id: str,
        new_key: str,
        reason: str | None = None,
    ) -> APIKey | None:
        """Rotate an API key, keeping history of the rotation."""
        config = self._load_config()

        key = config.get_key(key_id)
        if not key:
            return None

        old_fingerprint = key.get_fingerprint()

        rotation_entry = KeyRotationEntry(
            rotated_at=datetime.now().isoformat(),
            previous_fingerprint=old_fingerprint,
            reason=reason,
        )

        key.rotation_history.append(rotation_entry)
        key.key = new_key
        key.last_used = None
        key.id = f"key_{secrets.token_hex(8)}"

        self._save_config(config)
        self._write_env_file(key)

        if self._audit:
            self._audit.log(
                event_type=AuditEventType.KEY_ROTATED,
                key_id=key.id,
                key_fingerprint=key.get_fingerprint(),
                provider=key.provider.value,
                details={
                    "reason": reason,
                    "previous_fingerprint": old_fingerprint,
                },
            )

        return key

    def get_rotation_history(self, key_id: str) -> list[KeyRotationEntry]:
        """Get rotation history for a key."""
        config = self._load_config()
        key = config.get_key(key_id)
        if not key:
            return []
        return key.rotation_history

    def set_key_scope(self, key_id: str, scope: KeyScope) -> bool:
        """Set the scope/permissions for a key."""
        config = self._load_config()
        key = config.get_key(key_id)
        if not key:
            return False

        old_scope = key.scope
        key.scope = scope
        self._save_config(config)

        if self._audit:
            self._audit.log(
                event_type=AuditEventType.KEY_SCOPE_CHANGED,
                key_id=key_id,
                provider=key.provider.value,
                details={
                    "old_scope": old_scope.value,
                    "new_scope": scope.value,
                },
            )

        return True

    def get_keys_needing_rotation(self, days: int = 90) -> list[APIKey]:
        """Get keys that should be rotated based on age."""
        config = self._load_config()
        cutoff = datetime.now() - timedelta(days=days)

        keys_needing_rotation = []
        for key in config.api_keys:
            if not key.is_valid():
                continue

            created = datetime.fromisoformat(key.created_at)
            if created < cutoff:
                keys_needing_rotation.append(key)

        return keys_needing_rotation

    def mark_key_for_rotation(self, key_id: str) -> bool:
        """Mark a key as recommended for rotation."""
        config = self._load_config()
        key = config.get_key(key_id)
        if not key:
            return False

        key.rotation_recommended_at = datetime.now().isoformat()
        self._save_config(config)
        return True

    # ------------------------------------------------------------------
    # Expiration alerts
    # ------------------------------------------------------------------

    def get_expiring_keys(self, within_days: int = 7) -> list[APIKey]:
        """Get keys expiring within the given number of days."""
        config = self._load_config()
        deadline = datetime.now() + timedelta(days=within_days)
        expiring: list[APIKey] = []

        for key in config.api_keys:
            if not key.is_active or not key.expires_at:
                continue
            try:
                expires = datetime.fromisoformat(key.expires_at)
                if expires <= deadline:
                    expiring.append(key)
            except ValueError:
                continue

        return expiring

    def deactivate_expired_keys(self) -> list[APIKey]:
        """Deactivate keys that have passed their expiry. Returns deactivated keys."""
        config = self._load_config()
        now = datetime.now()
        deactivated: list[APIKey] = []

        for key in config.api_keys:
            if not key.is_active or not key.expires_at:
                continue
            try:
                expires = datetime.fromisoformat(key.expires_at)
                if now > expires:
                    key.is_active = False
                    deactivated.append(key)

                    if self._audit:
                        self._audit.log(
                            event_type=AuditEventType.CONFIG_CHANGED,
                            key_id=key.id,
                            provider=key.provider.value,
                            details={"action": "auto_deactivated", "expired_at": key.expires_at},
                        )
            except ValueError:
                continue

        if deactivated:
            self._save_config(config)

        return deactivated

    def set_key_expiry(self, key_id: str, expires_at: str) -> bool:
        """Set expiry date for a key. Format: ISO 8601."""
        config = self._load_config()
        key = config.get_key(key_id)
        if not key:
            return False

        # Validate format
        try:
            datetime.fromisoformat(expires_at)
        except ValueError:
            raise ValueError(f"Invalid date format: {expires_at}. Use ISO 8601.")

        key.expires_at = expires_at
        self._save_config(config)
        return True

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def set_rate_limit(self, key_id: str, config: RateLimitConfig) -> bool:
        """Set rate limit for a key."""
        if not self._rate_limiter:
            return False
        cfg = self._load_config()
        if not cfg.get_key(key_id):
            return False
        self._rate_limiter.set_limit(key_id, config)
        return True

    def get_key_usage(self, key_id: str | None = None) -> dict[str, Any]:
        """Get usage statistics for a key or all keys."""
        if not self._rate_limiter:
            return {}
        if key_id:
            return self._rate_limiter.get_usage_stats(key_id).to_dict()
        return {k: v.to_dict() for k, v in self._rate_limiter.get_all_usage().items()}

    # ------------------------------------------------------------------
    # Import / export
    # ------------------------------------------------------------------

    def export_json(self, include_keys: bool = True) -> str:
        """Export current config as JSON."""
        from .import_export import AuthExporter

        config = self._load_config()
        return AuthExporter.to_json(config, include_keys=include_keys)

    def export_env(self, include_all: bool = False) -> str:
        """Export current config as .env format."""
        from .import_export import AuthExporter

        config = self._load_config()
        return AuthExporter.to_env(config, include_all=include_all)

    def import_keys_from_env(self, env_content: str) -> tuple[int, list[str]]:
        """Import keys from .env content. Returns (count, errors)."""
        from .import_export import AuthImporter

        keys, result = AuthImporter.from_env(env_content)
        config = self._load_config()

        for key in keys:
            config.api_keys.append(key)
            if not config.default_key_id:
                config.default_key_id = key.id
                config.default_provider = key.provider

        if keys:
            self._save_config(config)

        return result.imported_count, result.errors or []

    def import_keys_from_json(self, json_str: str) -> tuple[int, list[str]]:
        """Import keys from JSON. Returns (count, errors)."""
        from .import_export import AuthImporter

        imported_config, result = AuthImporter.from_json(json_str)
        if result.error_count > 0:
            return 0, result.errors or []

        config = self._load_config()
        for key in imported_config.api_keys:
            config.api_keys.append(key)
            if not config.default_key_id:
                config.default_key_id = key.id
                config.default_provider = key.provider

        if imported_config.api_keys:
            self._save_config(config)

        if imported_config.profile and not config.profile:
            config.profile = imported_config.profile
            self._save_config(config)

        return result.imported_count, []

    def import_keys_from_1password(self, csv_content: str) -> tuple[int, list[str]]:
        """Import keys from 1Password CSV. Returns (count, errors)."""
        from .import_export import AuthImporter

        keys, result = AuthImporter.from_1password_csv(csv_content)
        config = self._load_config()

        for key in keys:
            config.api_keys.append(key)
            if not config.default_key_id:
                config.default_key_id = key.id
                config.default_provider = key.provider

        if keys:
            self._save_config(config)

        return result.imported_count, result.errors or []
