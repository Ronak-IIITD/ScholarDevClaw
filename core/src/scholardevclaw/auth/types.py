from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AuthProvider(str, Enum):
    # Identity / legacy providers
    LOCAL = "local"
    GITHUB = "github"
    GOOGLE = "google"
    CUSTOM = "custom"

    # LLM API providers
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GITHUB_COPILOT = "github_copilot"
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"
    GROQ = "groq"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    COHERE = "cohere"
    OPENROUTER = "openrouter"
    TOGETHER = "together"
    FIREWORKS = "fireworks"

    @property
    def is_llm_provider(self) -> bool:
        """Whether this provider is an LLM backend (vs identity/storage only)."""
        return self in _LLM_PROVIDERS

    @property
    def requires_api_key(self) -> bool:
        """Whether this provider requires an API key (Ollama typically doesn't)."""
        return self != AuthProvider.OLLAMA

    @property
    def default_base_url(self) -> str | None:
        """Default API base URL for LLM providers."""
        return _DEFAULT_BASE_URLS.get(self)

    @property
    def key_prefix(self) -> str | None:
        prefixes = {
            AuthProvider.ANTHROPIC: "sk-ant",
            AuthProvider.OPENAI: "sk-",
            AuthProvider.GITHUB: "ghp_",
            AuthProvider.GITHUB_COPILOT: "ghp_",
            AuthProvider.GOOGLE: "ya29.",
            AuthProvider.GROQ: "gsk_",
            AuthProvider.DEEPSEEK: "sk-",
            AuthProvider.COHERE: "co-",
            AuthProvider.MISTRAL: "mis",
        }
        return prefixes.get(self)

    @property
    def key_format_hint(self) -> str:
        hints = {
            AuthProvider.ANTHROPIC: "sk-ant-...",
            AuthProvider.OPENAI: "sk-...",
            AuthProvider.GITHUB: "ghp_...",
            AuthProvider.GITHUB_COPILOT: "ghp_... or github_pat_...",
            AuthProvider.GOOGLE: "ya29...",
            AuthProvider.OLLAMA: "(no key needed — local)",
            AuthProvider.AZURE_OPENAI: "32-character hex key",
            AuthProvider.GROQ: "gsk_...",
            AuthProvider.MISTRAL: "API key from console.mistral.ai",
            AuthProvider.DEEPSEEK: "sk-... from platform.deepseek.com",
            AuthProvider.COHERE: "co-... from dashboard.cohere.com",
            AuthProvider.OPENROUTER: "sk-or-... from openrouter.ai",
            AuthProvider.TOGETHER: "API key from api.together.xyz",
            AuthProvider.FIREWORKS: "API key from fireworks.ai",
            AuthProvider.CUSTOM: "custom API key format",
            AuthProvider.LOCAL: "local key",
        }
        return hints.get(self, "API key")

    @property
    def env_var_name(self) -> str:
        """Standard environment variable name for this provider's API key."""
        return _ENV_VAR_NAMES.get(self, "SCHOLARDEVCLAW_API_KEY")

    @property
    def display_name(self) -> str:
        """Human-friendly display name."""
        return _DISPLAY_NAMES.get(self, self.value.title())


# Sets/dicts defined after enum to avoid forward reference issues
_LLM_PROVIDERS = {
    AuthProvider.ANTHROPIC,
    AuthProvider.OPENAI,
    AuthProvider.GITHUB_COPILOT,
    AuthProvider.OLLAMA,
    AuthProvider.AZURE_OPENAI,
    AuthProvider.GROQ,
    AuthProvider.MISTRAL,
    AuthProvider.DEEPSEEK,
    AuthProvider.COHERE,
    AuthProvider.OPENROUTER,
    AuthProvider.TOGETHER,
    AuthProvider.FIREWORKS,
}

_DEFAULT_BASE_URLS: dict[AuthProvider, str] = {
    AuthProvider.ANTHROPIC: "https://api.anthropic.com",
    AuthProvider.OPENAI: "https://api.openai.com/v1",
    AuthProvider.OLLAMA: "http://localhost:11434",
    AuthProvider.GROQ: "https://api.groq.com/openai/v1",
    AuthProvider.MISTRAL: "https://api.mistral.ai/v1",
    AuthProvider.DEEPSEEK: "https://api.deepseek.com",
    AuthProvider.COHERE: "https://api.cohere.com/v2",
    AuthProvider.OPENROUTER: "https://openrouter.ai/api/v1",
    AuthProvider.TOGETHER: "https://api.together.xyz/v1",
    AuthProvider.FIREWORKS: "https://api.fireworks.ai/inference/v1",
}

_ENV_VAR_NAMES: dict[AuthProvider, str] = {
    AuthProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    AuthProvider.OPENAI: "OPENAI_API_KEY",
    AuthProvider.GITHUB: "GITHUB_TOKEN",
    AuthProvider.GITHUB_COPILOT: "GITHUB_TOKEN",
    AuthProvider.GOOGLE: "GOOGLE_API_KEY",
    AuthProvider.OLLAMA: "OLLAMA_HOST",
    AuthProvider.AZURE_OPENAI: "AZURE_OPENAI_API_KEY",
    AuthProvider.GROQ: "GROQ_API_KEY",
    AuthProvider.MISTRAL: "MISTRAL_API_KEY",
    AuthProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
    AuthProvider.COHERE: "COHERE_API_KEY",
    AuthProvider.OPENROUTER: "OPENROUTER_API_KEY",
    AuthProvider.TOGETHER: "TOGETHER_API_KEY",
    AuthProvider.FIREWORKS: "FIREWORKS_API_KEY",
}

_DISPLAY_NAMES: dict[AuthProvider, str] = {
    AuthProvider.LOCAL: "Local",
    AuthProvider.GITHUB: "GitHub",
    AuthProvider.GOOGLE: "Google",
    AuthProvider.ANTHROPIC: "Anthropic (Claude)",
    AuthProvider.OPENAI: "OpenAI (GPT / Codex)",
    AuthProvider.GITHUB_COPILOT: "GitHub Copilot",
    AuthProvider.OLLAMA: "Ollama (Local LLM)",
    AuthProvider.AZURE_OPENAI: "Azure OpenAI",
    AuthProvider.GROQ: "Groq",
    AuthProvider.MISTRAL: "Mistral AI",
    AuthProvider.DEEPSEEK: "DeepSeek",
    AuthProvider.COHERE: "Cohere",
    AuthProvider.OPENROUTER: "OpenRouter",
    AuthProvider.TOGETHER: "Together AI",
    AuthProvider.FIREWORKS: "Fireworks AI",
    AuthProvider.CUSTOM: "Custom Provider",
}


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
    def from_dict(cls, data: dict[str, Any]) -> KeyRotationEntry:
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

    def to_safe_dict(self) -> dict[str, Any]:
        """Return a dict representation that excludes the raw API key.

        Use this instead of to_dict() for any output that may be shown to users,
        logged, exported, or serialized where key confidentiality matters.
        """
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider.value,
            "key_fingerprint": self.get_fingerprint(),
            "key_masked": self.mask(),
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
    def from_dict(cls, data: dict[str, Any]) -> APIKey:
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

        # Ollama typically doesn't need a key
        if provider == AuthProvider.OLLAMA:
            return True, "Valid (Ollama — key is optional)"

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

        elif provider == AuthProvider.GITHUB_COPILOT:
            if not (
                key.startswith("ghp_") or key.startswith("github_pat_") or key.startswith("ghu_")
            ):
                return (
                    False,
                    "GitHub Copilot tokens should start with 'ghp_', 'github_pat_', or 'ghu_'",
                )

        elif provider == AuthProvider.GOOGLE:
            if not key.startswith("ya29.") and not key.startswith("1//"):
                return False, "Google tokens typically start with 'ya29.' or '1//'"

        elif provider == AuthProvider.GROQ:
            if not key.startswith("gsk_"):
                return False, "Groq keys should start with 'gsk_'"

        elif provider == AuthProvider.COHERE:
            if not key.startswith("co-"):
                return False, "Cohere keys should start with 'co-'"

        elif provider == AuthProvider.OPENROUTER:
            if not key.startswith("sk-or-"):
                return False, "OpenRouter keys should start with 'sk-or-'"

        # For Azure, Mistral, DeepSeek, Together, Fireworks — no strict prefix,
        # just length check above is sufficient

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
        return hashlib.sha256(self.key.encode()).hexdigest()


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
    def from_dict(cls, data: dict[str, Any]) -> UserProfile:
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
    def from_dict(cls, data: dict[str, Any]) -> AuthConfig:
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
