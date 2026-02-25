from __future__ import annotations

import json
import os
import secrets
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


class AuthStore:
    def __init__(self, store_dir: str | None = None, enable_audit: bool = True):
        if store_dir:
            self.store_dir = Path(store_dir)
        else:
            default_dir = os.environ.get("SCHOLARDEVCLAW_AUTH_DIR")
            if default_dir:
                self.store_dir = Path(default_dir)
            else:
                self.store_dir = Path.home() / ".scholardevclaw"

        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.auth_file = self.store_dir / "auth.json"
        self._config: AuthConfig | None = None
        self._audit: AuditLogger | None = AuditLogger(str(self.store_dir)) if enable_audit else None
        if store_dir:
            self.store_dir = Path(store_dir)
        else:
            default_dir = os.environ.get("SCHOLARDEVCLAW_AUTH_DIR")
            if default_dir:
                self.store_dir = Path(default_dir)
            else:
                self.store_dir = Path.home() / ".scholardevclaw"

        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.auth_file = self.store_dir / "auth.json"
        self._config: AuthConfig | None = None

    def _load_config(self) -> AuthConfig:
        if self._config is not None:
            return self._config

        if not self.auth_file.exists():
            self._config = AuthConfig()
            return self._config

        try:
            with open(self.auth_file) as f:
                data = json.load(f)
            self._config = AuthConfig.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            self._config = AuthConfig()

        return self._config

    def _save_config(self, config: AuthConfig) -> None:
        self._config = config
        with open(self.auth_file, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

    def _write_env_file(self, api_key: APIKey) -> None:
        env_file = self.store_dir / ".env"
        env_content = f"""# ScholarDevClaw Configuration
# Generated: {datetime.now().isoformat()}

# API Key for {api_key.provider.value}
SCHOLARDEVCLAW_API_KEY={api_key.key}
SCHOLARDEVCLAW_API_PROVIDER={api_key.provider.value}
"""
        with open(env_file, "w") as f:
            f.write(env_content)

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
            subscription_tier=config.profile.subscription_tier.value if config.profile else "free",
        )

    def get_config(self) -> AuthConfig:
        return self._load_config()

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
                    key=key.key,
                    provider=key.provider.value,
                )

            return key.key

        return None

    def add_api_key(
        self,
        key: str,
        name: str,
        provider: AuthProvider,
        set_default: bool = True,
        validate: bool = False,
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
                key=key,
                provider=provider.value,
                details={"name": name},
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
                key=key_to_remove.key,
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
        return config.profile

    def get_profile(self) -> UserProfile | None:
        config = self._load_config()
        return config.profile

    def logout(self) -> bool:
        config = self._load_config()

        if not config.api_keys and not config.profile:
            return False

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

        import secrets

        key.id = f"key_{secrets.token_hex(8)}"

        self._save_config(config)
        self._write_env_file(key)

        if self._audit:
            self._audit.log(
                event_type=AuditEventType.KEY_ROTATED,
                key_id=key.id,
                key=new_key,
                provider=key.provider.value,
                details={"reason": reason, "previous_fingerprint": old_fingerprint},
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

        key.scope = scope
        self._save_config(config)
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
