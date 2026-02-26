"""OAuth 2.0 authentication flows for Google and GitHub.

Supports authorization code flow with token refresh, PKCE for security,
and secure token storage.
"""

from __future__ import annotations

import json
import secrets
import time
import webbrowser
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from ..auth.types import APIKey, AuthProvider


@dataclass
class OAuthToken:
    """OAuth 2.0 token with refresh capability."""

    access_token: str
    token_type: str
    expires_at: float  # Unix timestamp
    refresh_token: str | None = None
    scope: str | None = None
    id_token: str | None = None  # For OIDC
    provider: str | None = None

    def is_expired(self, buffer_seconds: int = 60) -> bool:
        """Check if token is expired or about to expire."""
        return time.time() >= (self.expires_at - buffer_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_at": self.expires_at,
            "refresh_token": self.refresh_token,
            "scope": self.scope,
            "id_token": self.id_token,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OAuthToken:
        return cls(
            access_token=data["access_token"],
            token_type=data.get("token_type", "Bearer"),
            expires_at=data["expires_at"],
            refresh_token=data.get("refresh_token"),
            scope=data.get("scope"),
            id_token=data.get("id_token"),
            provider=data.get("provider"),
        )


class OAuthProvider(ABC):
    """Base class for OAuth 2.0 providers."""

    AUTHORIZATION_URL: str
    TOKEN_URL: str
    USERINFO_URL: str
    DEFAULT_SCOPE: str

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str = "http://localhost:8080/callback",
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self._code_verifier: str | None = None

    def generate_code_verifier(self) -> str:
        """Generate PKCE code verifier."""
        self._code_verifier = secrets.token_urlsafe(64)[:128]
        return self._code_verifier

    def generate_code_challenge(self) -> str:
        """Generate PKCE code challenge from verifier."""
        import hashlib
        import base64

        if not self._code_verifier:
            self.generate_code_verifier()

        digest = hashlib.sha256(self._code_verifier.encode()).digest()
        return base64.urlsafe_b64encode(digest).decode().rstrip("=")

    def get_authorization_url(self, scope: str | None = None) -> tuple[str, str]:
        """Get authorization URL and state token."""
        state = secrets.token_urlsafe(32)
        code_challenge = self.generate_code_challenge()

        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "scope": scope or self.DEFAULT_SCOPE,
        }

        url = self.AUTHORIZATION_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())
        return url, state

    def exchange_code(self, code: str) -> OAuthToken:
        """Exchange authorization code for tokens."""
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
        }
        if self._code_verifier:
            data["code_verifier"] = self._code_verifier

        response = requests.post(self.TOKEN_URL, data=data, timeout=30)
        response.raise_for_status()
        token_data = response.json()

        expires_in = token_data.get("expires_in", 3600)
        return OAuthToken(
            access_token=token_data["access_token"],
            token_type=token_data.get("token_type", "Bearer"),
            expires_at=time.time() + expires_in,
            refresh_token=token_data.get("refresh_token"),
            scope=token_data.get("scope"),
            id_token=token_data.get("id_token"),
            provider=self.provider_name,
        )

    def refresh_access_token(self, refresh_token: str) -> OAuthToken:
        """Refresh an expired access token."""
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }

        response = requests.post(self.TOKEN_URL, data=data, timeout=30)
        response.raise_for_status()
        token_data = response.json()

        expires_in = token_data.get("expires_in", 3600)
        return OAuthToken(
            access_token=token_data["access_token"],
            token_type=token_data.get("token_type", "Bearer"),
            expires_at=time.time() + expires_in,
            refresh_token=token_data.get("refresh_token", refresh_token),
            scope=token_data.get("scope"),
            provider=self.provider_name,
        )

    def get_user_info(self, access_token: str) -> dict[str, Any]:
        """Fetch user info from provider."""
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(self.USERINFO_URL, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider name for storage."""
        pass


class GoogleOAuthProvider(OAuthProvider):
    """Google OAuth 2.0 provider."""

    AUTHORIZATION_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
    DEFAULT_SCOPE = "openid email profile"

    @property
    def provider_name(self) -> str:
        return "google"


class GitHubOAuthProvider(OAuthProvider):
    """GitHub OAuth 2.0 provider."""

    AUTHORIZATION_URL = "https://github.com/login/oauth/authorize"
    TOKEN_URL = "https://github.com/login/oauth/access_token"
    USERINFO_URL = "https://api.github.com/user"
    DEFAULT_SCOPE = "read:user user:email"

    @property
    def provider_name(self) -> str:
        return "github"

    def get_user_info(self, access_token: str) -> dict[str, Any]:
        """GitHub also needs scope for emails."""
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github.v3+json",
        }
        response = requests.get(self.USERINFO_URL, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Get primary email
        emails_response = requests.get(
            "https://api.github.com/user/emails", headers=headers, timeout=30
        )
        if emails_response.ok:
            emails = emails_response.json()
            primary = next((e for e in emails if e.get("primary")), emails[0] if emails else {})
            data["email"] = primary.get("email", data.get("email"))

        return data


class OAuthTokenStore:
    """Manages OAuth tokens with secure storage and auto-refresh."""

    TOKEN_FILE = "oauth_tokens.json"

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.token_file = self.store_dir / self.TOKEN_FILE

    def save_token(self, provider: str, token: OAuthToken) -> None:
        """Save token for a provider."""
        tokens = self._load_tokens()
        tokens[provider] = token.to_dict()
        self.token_file.write_text(json.dumps(tokens, indent=2))

    def get_token(self, provider: str) -> OAuthToken | None:
        """Get token for a provider, auto-refresh if expired."""
        tokens = self._load_tokens()
        if provider not in tokens:
            return None

        token = OAuthToken.from_dict(tokens[provider])

        if token.is_expired() and token.refresh_token:
            try:
                token = self._refresh_token(provider, token.refresh_token)
            except Exception:
                return None

        return token

    def _refresh_token(self, provider: str, refresh_token: str) -> OAuthToken:
        """Refresh token and save new one."""
        token = self._get_provider(provider).refresh_access_token(refresh_token)
        self.save_token(provider, token)
        return token

    def _get_provider(self, provider: str) -> OAuthProvider:
        """Get provider instance (requires config)."""
        config = self._load_oauth_config()
        if provider not in config:
            raise ValueError(f"OAuth not configured for {provider}")

        cfg = config[provider]
        if provider == "google":
            return GoogleOAuthProvider(
                cfg["client_id"],
                cfg["client_secret"],
                cfg.get("redirect_uri", "http://localhost:8080/callback"),
            )
        elif provider == "github":
            return GitHubOAuthProvider(
                cfg["client_id"],
                cfg["client_secret"],
                cfg.get("redirect_uri", "http://localhost:8080/callback"),
            )
        else:
            raise ValueError(f"Unknown OAuth provider: {provider}")

    def remove_token(self, provider: str) -> bool:
        """Remove token for a provider."""
        tokens = self._load_tokens()
        if provider in tokens:
            del tokens[provider]
            self.token_file.write_text(json.dumps(tokens, indent=2))
            return True
        return False

    def list_providers(self) -> list[str]:
        """List providers with stored tokens."""
        return list(self._load_tokens().keys())

    def _load_tokens(self) -> dict[str, Any]:
        if not self.token_file.exists():
            return {}
        try:
            return json.loads(self.token_file.read_text())
        except json.JSONDecodeError:
            return {}

    def _load_oauth_config(self) -> dict[str, Any]:
        """Load OAuth configuration from auth config."""
        config_file = self.store_dir / "auth.json"
        if not config_file.exists():
            return {}
        try:
            data = json.loads(config_file.read_text())
            return data.get("oauth_config", {})
        except json.JSONDecodeError:
            return {}


class OAuthManager:
    """High-level OAuth flow manager."""

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.token_store = OAuthTokenStore(store_dir)
        self._providers: dict[str, OAuthProvider] = {}

    def register_provider(self, provider: OAuthProvider) -> None:
        """Register an OAuth provider."""
        self._providers[provider.provider_name] = provider

    def start_flow(self, provider: str) -> tuple[str, str]:
        """Start OAuth flow, return authorization URL and state."""
        if provider not in self._providers:
            raise ValueError(f"Provider {provider} not registered")

        p = self._providers[provider]
        url, state = p.get_authorization_url()

        # Store provider for callback
        self._pending_provider = provider

        # Open browser for user
        webbrowser.open(url)

        return url, state

    def complete_flow(self, code: str, state: str, expected_state: str) -> OAuthToken:
        """Complete OAuth flow with authorization code."""
        if state != expected_state:
            raise ValueError("State mismatch - possible CSRF attack")

        if not hasattr(self, "_pending_provider"):
            raise ValueError("No pending OAuth flow")

        provider = self._pending_provider
        if provider not in self._providers:
            raise ValueError(f"Provider {provider} not registered")

        token = self._providers[provider].exchange_code(code)
        self.token_store.save_token(provider, token)
        delattr(self, "_pending_provider")

        return token

    def get_valid_token(self, provider: str) -> OAuthToken | None:
        """Get valid token, auto-refreshing if needed."""
        return self.token_store.get_token(provider)

    def revoke_token(self, provider: str) -> bool:
        """Revoke and remove token."""
        token = self.token_store.get_token(provider)
        if not token:
            return False

        # Try to revoke with provider (best effort)
        try:
            if provider == "google":
                requests.post(
                    "https://oauth2.googleapis.com/revoke",
                    params={"token": token.access_token},
                    timeout=10,
                )
            elif provider == "github":
                # GitHub doesn't have a revocation endpoint for tokens
                pass
        except Exception:
            pass

        return self.token_store.remove_token(provider)

    def list_connected(self) -> list[str]:
        """List connected OAuth providers."""
        return self.token_store.list_providers()


def create_oauth_provider(
    provider: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str = "http://localhost:8080/callback",
) -> OAuthProvider:
    """Factory to create OAuth provider instances."""
    if provider == "google":
        return GoogleOAuthProvider(client_id, client_secret, redirect_uri)
    elif provider == "github":
        return GitHubOAuthProvider(client_id, client_secret, redirect_uri)
    else:
        raise ValueError(f"Unknown OAuth provider: {provider}")
