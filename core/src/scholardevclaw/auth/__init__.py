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
    KeyScope,
    KeyRotationEntry,
)
from .audit import AuditLogger, AuditEventType, AuditEvent
from .rate_limit import RateLimiter, RateLimitConfig, KeyUsageStats
from .import_export import AuthExporter, AuthImporter, ImportResult
from .oauth import (
    OAuthToken,
    OAuthProvider,
    GoogleOAuthProvider,
    GitHubOAuthProvider,
    OAuthTokenStore,
    OAuthManager,
    create_oauth_provider,
)
from .hardware_keys import (
    HardwareKeyInfo,
    HardwareKeyManager,
    YubiKeyPIV,
)
from .team import (
    TeamRole,
    TeamPermission,
    TeamMember,
    Team,
    TeamInvite,
    TeamStore,
    TeamAccessControl,
)
from .analytics import (
    UsageRecord,
    DailyUsage,
    UsageAnalytics,
    UsageTracker,
    UsageDashboard,
)
from .rotation import (
    RotationPolicy,
    RotationResult,
    RotationProvider,
    RotationScheduler,
    get_rotation_provider,
)
from .approval import (
    RequestStatus,
    RequestType,
    KeyRequest,
    ApprovalNotification,
    ApprovalWorkflow,
    RequestValidator,
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
    # Core
    "AuthStore",
    "APIKey",
    "AuthConfig",
    "AuthProvider",
    "AuthStatus",
    "UserProfile",
    "SubscriptionTier",
    "KeyScope",
    "KeyRotationEntry",
    # Audit
    "AuditLogger",
    "AuditEventType",
    "AuditEvent",
    # Rate limiting
    "RateLimiter",
    "RateLimitConfig",
    "KeyUsageStats",
    # Import / export
    "AuthExporter",
    "AuthImporter",
    "ImportResult",
    # OAuth 2.0
    "OAuthToken",
    "OAuthProvider",
    "GoogleOAuthProvider",
    "GitHubOAuthProvider",
    "OAuthTokenStore",
    "OAuthManager",
    "create_oauth_provider",
    # Hardware keys
    "HardwareKeyInfo",
    "HardwareKeyManager",
    "YubiKeyPIV",
    # Team / multi-user
    "TeamRole",
    "TeamPermission",
    "TeamMember",
    "Team",
    "TeamInvite",
    "TeamStore",
    "TeamAccessControl",
    # Analytics
    "UsageRecord",
    "DailyUsage",
    "UsageAnalytics",
    "UsageTracker",
    "UsageDashboard",
    # Rotation automation
    "RotationPolicy",
    "RotationResult",
    "RotationProvider",
    "RotationScheduler",
    "get_rotation_provider",
    # Approval workflow
    "RequestStatus",
    "RequestType",
    "KeyRequest",
    "ApprovalNotification",
    "ApprovalWorkflow",
    "RequestValidator",
    # Convenience functions
    "get_auth_store",
    "get_api_key",
    "is_authenticated",
    "get_auth_status",
    "get_current_user",
]
