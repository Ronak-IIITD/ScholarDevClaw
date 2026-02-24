from __future__ import annotations

import os
from .store import AuthStore
from .types import (
    APIKey,
    AuthConfig,
    AuthProvider,
    AuthStatus,
    UserProfile,
    SubscriptionTier,
)


def get_auth_store(store_dir: str | None = None) -> AuthStore:
    return AuthStore(store_dir)


def get_api_key(provider: AuthProvider | None = None) -> str | None:
    env_key = os.environ.get("SCHOLARDEVCLAW_API_KEY")
    if env_key:
        return env_key

    store = AuthStore()
    return store.get_api_key(provider)


def is_authenticated() -> bool:
    if os.environ.get("SCHOLARDEVCLAW_API_KEY"):
        return True

    store = AuthStore()
    return store.is_authenticated()


def get_auth_status() -> AuthStatus:
    store = AuthStore()
    return store.get_status()


def get_current_user() -> UserProfile | None:
    store = AuthStore()
    return store.get_profile()


__all__ = [
    "AuthStore",
    "APIKey",
    "AuthConfig",
    "AuthProvider",
    "AuthStatus",
    "UserProfile",
    "SubscriptionTier",
    "get_auth_store",
    "get_api_key",
    "is_authenticated",
    "get_auth_status",
    "get_current_user",
]
