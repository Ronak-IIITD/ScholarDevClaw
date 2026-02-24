from __future__ import annotations

import hashlib
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


class SubscriptionTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "APIKey":
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
        )

    @staticmethod
    def generate_key(prefix: str = "sk") -> str:
        random_bytes = secrets.token_hex(24)
        return f"{prefix}_{random_bytes}"

    def mask(self) -> str:
        if len(self.key) < 8:
            return "***"
        return f"{self.key[:8]}...{self.key[-4:]}"

    def is_valid(self) -> bool:
        if not self.is_active:
            return False
        if self.expires_at:
            expires = datetime.fromisoformat(self.expires_at)
            if datetime.now() > expires:
                return False
        return True


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
