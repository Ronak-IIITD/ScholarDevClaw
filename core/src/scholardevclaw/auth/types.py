from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any


class AuthProvider(str, Enum):
    LOCAL = "local"
    GITHUB = "github"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    CUSTOM = "custom"

    @property
    def key_prefix(self) -> str | None:
        prefixes = {
            AuthProvider.ANTHROPIC: "sk-ant",
            AuthProvider.OPENAI: "sk-",
            AuthProvider.GITHUB: "ghp_",
            AuthProvider.GOOGLE: "ya29.",
        }
        return prefixes.get(self)

    @property
    def key_format_hint(self) -> str:
        hints = {
            AuthProvider.ANTHROPIC: "sk-ant-...",
            AuthProvider.OPENAI: "sk-...",
            AuthProvider.GITHUB: "ghp_...",
            AuthProvider.GOOGLE: "ya29...",
            AuthProvider.CUSTOM: "custom API key format",
            AuthProvider.LOCAL: "local key",
        }
        return hints.get(self, "API key")


class SubscriptionTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class KeyScope(str, Enum):
    READ_ONLY = "read"
    READ_WRITE = "write"
    ADMIN = "admin"
    CUSTOM = "custom"


@dataclass
class KeyRotationEntry:
    rotated_at: str
    previous_fingerprint: str
    rotated_by: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rotated_at": self.rotated_at,
            "previous_fingerprint": self.previous_fingerprint,
            "rotated_by": self.rotated_by,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KeyRotationEntry":
        return cls(
            rotated_at=data["rotated_at"],
            previous_fingerprint=data["previous_fingerprint"],
            rotated_by=data.get("rotated_by"),
            reason=data.get("reason"),
        )


@dataclass
class APIKey:
    id: str
    name: str
    provider: AuthProvider
    key: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str | None = None
    expires_at: str | None = None
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    rotation_history: list[KeyRotationEntry] = field(default_factory=list)
    scope: KeyScope = KeyScope.READ_WRITE
    rotation_recommended_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider.value,
            "key": self.key,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "expires_at": self.expires_at,
            "is_active": self.is_active,
            "metadata": self.metadata,
            "rotation_history": [r.to_dict() for r in self.rotation_history],
            "scope": self.scope.value,
            "rotation_recommended_at": self.rotation_recommended_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "APIKey":
        rotation_history = [KeyRotationEntry.from_dict(r) for r in data.get("rotation_history", [])]
        scope = KeyScope(data.get("scope", "write"))

        return cls(
            id=data["id"],
            name=data["name"],
            provider=AuthProvider(data["provider"]),
            key=data["key"],
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_used=data.get("last_used"),
            expires_at=data.get("expires_at"),
            is_active=data.get("is_active", True),
            metadata=data.get("metadata", {}),
            rotation_history=rotation_history,
            scope=scope,
            rotation_recommended_at=data.get("rotation_recommended_at"),
        )

    @staticmethod
    def generate_key(prefix: str = "sk") -> str:
        random_bytes = secrets.token_hex(24)
        return f"{prefix}_{random_bytes}"

    @staticmethod
    def validate_key_format(key: str, provider: AuthProvider) -> tuple[bool, str]:
        """Validate key format based on provider. Returns (is_valid, message)."""
        if not key:
            return False, "API key cannot be empty"

        if len(key) < 8:
            return False, "API key seems too short"

        if provider == AuthProvider.ANTHROPIC:
            if not key.startswith("sk-ant"):
                return False, "Anthropic keys should start with 'sk-ant'"

        elif provider == AuthProvider.OPENAI:
            if not key.startswith("sk-"):
                return False, "OpenAI keys should start with 'sk-'"

        elif provider == AuthProvider.GITHUB:
            if not (key.startswith("ghp_") or key.startswith("github_pat_")):
                return False, "GitHub tokens should start with 'ghp_' or 'github_pat_'"

        elif provider == AuthProvider.GOOGLE:
            if not key.startswith("ya29.") and not key.startswith("1//"):
                return False, "Google tokens typically start with 'ya29.' or '1//'"

        return True, "Valid"

    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Validate email format."""
        if not email:
            return False
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email))

    @staticmethod
    def is_valid_key_name(name: str) -> tuple[bool, str]:
        """Validate key name."""
        if not name:
            return False, "Key name cannot be empty"
        if len(name) > 100:
            return False, "Key name too long (max 100 chars)"
        if not re.match(r"^[\w\s\-\.]+$", name):
            return False, "Key name contains invalid characters"
        return True, "Valid"

    def mask(self) -> str:
        if len(self.key) < 8:
            return "***"
        return f"{self.key[:8]}...{self.key[-4:]}"

    def is_valid(self) -> bool:
        if not self.is_active:
            return False
        if self.expires_at:
            try:
                expires = datetime.fromisoformat(self.expires_at)
                if datetime.now() > expires:
                    return False
            except ValueError:
                return False
        return True

    def get_fingerprint(self) -> str:
        """Get a SHA256 hash of the key for identification without revealing it."""
        return hashlib.sha256(self.key.encode()).hexdigest()[:16]


@dataclass
class UserProfile:
    id: str
    email: str | None = None
    name: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_login: str | None = None
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    preferences: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserProfile":
        return cls(
            id=data["id"],
            email=data.get("email"),
            name=data.get("name"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_login=data.get("last_login"),
            subscription_tier=SubscriptionTier(data.get("subscription_tier", "free")),
            preferences=data.get("preferences", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AuthConfig:
    profile: UserProfile | None = None
    api_keys: list[APIKey] = field(default_factory=list)
    default_provider: AuthProvider = AuthProvider.LOCAL
    default_key_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict() if self.profile else None,
            "api_keys": [k.to_dict() for k in self.api_keys],
            "default_provider": self.default_provider.value,
            "default_key_id": self.default_key_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuthConfig":
        profile = None
        if data.get("profile"):
            profile = UserProfile.from_dict(data["profile"])

        api_keys = [APIKey.from_dict(k) for k in data.get("api_keys", [])]

        return cls(
            profile=profile,
            api_keys=api_keys,
            default_provider=AuthProvider(data.get("default_provider", "local")),
            default_key_id=data.get("default_key_id"),
        )

    def get_active_key(self, provider: AuthProvider | None = None) -> APIKey | None:
        target_provider = provider or self.default_provider

        if self.default_key_id:
            for key in self.api_keys:
                if key.id == self.default_key_id and key.is_valid():
                    return key

        for key in self.api_keys:
            if key.provider == target_provider and key.is_valid():
                return key

        for key in self.api_keys:
            if key.is_valid():
                return key

        return None

    def get_key(self, key_id: str) -> APIKey | None:
        for key in self.api_keys:
            if key.id == key_id:
                return key
        return None


@dataclass
class AuthStatus:
    is_authenticated: bool
    has_api_key: bool
    user_email: str | None = None
    user_name: str | None = None
    provider: str | None = None
    key_count: int = 0
    active_keys: int = 0
    subscription_tier: str = "free"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
