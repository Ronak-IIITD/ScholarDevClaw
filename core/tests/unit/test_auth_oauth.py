"""Tests for OAuth 2.0 authentication."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from scholardevclaw.auth.oauth import (
    OAuthToken,
    OAuthProvider,
    GoogleOAuthProvider,
    GitHubOAuthProvider,
    OAuthTokenStore,
    OAuthManager,
    create_oauth_provider,
)


class TestOAuthToken:
    def test_token_creation(self):
        token = OAuthToken(
            access_token="test_token",
            token_type="Bearer",
            expires_at=datetime.now().timestamp() + 3600,
            refresh_token="refresh",
            provider="google",
        )
        assert token.access_token == "test_token"
        assert not token.is_expired()

    def test_token_expired(self):
        token = OAuthToken(
            access_token="test_token",
            token_type="Bearer",
            expires_at=datetime.now().timestamp() - 100,
            provider="google",
        )
        assert token.is_expired()

    def test_token_expired_with_buffer(self):
        token = OAuthToken(
            access_token="test_token",
            token_type="Bearer",
            expires_at=datetime.now().timestamp() + 30,
            provider="google",
        )
        assert token.is_expired(buffer_seconds=60)

    def test_to_dict_roundtrip(self):
        token = OAuthToken(
            access_token="test_token",
            token_type="Bearer",
            expires_at=1234567890.0,
            refresh_token="refresh",
            scope="openid email",
            id_token="id_token",
            provider="google",
        )
        data = token.to_dict()
        restored = OAuthToken.from_dict(data)
        assert restored.access_token == token.access_token
        assert restored.refresh_token == token.refresh_token
        assert restored.provider == token.provider


class TestGoogleOAuthProvider:
    def test_provider_creation(self):
        provider = GoogleOAuthProvider("client_id", "client_secret")
        assert provider.client_id == "client_id"
        assert provider.provider_name == "google"

    def test_authorization_url_generation(self):
        provider = GoogleOAuthProvider(
            "client_id", "client_secret", "http://localhost:8080/callback"
        )
        url, state = provider.get_authorization_url()
        assert "accounts.google.com" in url
        assert "client_id=client_id" in url
        assert "response_type=code" in url
        assert len(state) > 0


class TestGitHubOAuthProvider:
    def test_provider_creation(self):
        provider = GitHubOAuthProvider("client_id", "client_secret")
        assert provider.client_id == "client_id"
        assert provider.provider_name == "github"

    def test_authorization_url_generation(self):
        provider = GitHubOAuthProvider("client_id", "client_secret")
        url, state = provider.get_authorization_url()
        assert "github.com/login/oauth/authorize" in url
        assert "client_id=client_id" in url


class TestOAuthTokenStore:
    @pytest.fixture
    def store_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_save_and_get_token(self, store_dir):
        store = OAuthTokenStore(store_dir)
        token = OAuthToken(
            access_token="test_token",
            token_type="Bearer",
            expires_at=datetime.now().timestamp() + 3600,
            provider="google",
        )
        store.save_token("google", token)
        retrieved = store.get_token("google")
        assert retrieved is not None
        assert retrieved.access_token == "test_token"

    def test_get_token_not_found(self, store_dir):
        store = OAuthTokenStore(store_dir)
        assert store.get_token("google") is None

    def test_remove_token(self, store_dir):
        store = OAuthTokenStore(store_dir)
        token = OAuthToken(
            access_token="test_token",
            token_type="Bearer",
            expires_at=datetime.now().timestamp() + 3600,
            provider="google",
        )
        store.save_token("google", token)
        assert store.remove_token("google") is True
        assert store.get_token("google") is None

    def test_list_providers(self, store_dir):
        store = OAuthTokenStore(store_dir)
        token = OAuthToken(
            access_token="test_token",
            token_type="Bearer",
            expires_at=datetime.now().timestamp() + 3600,
            provider="google",
        )
        store.save_token("google", token)
        assert store.list_providers() == ["google"]


class TestOAuthManager:
    @pytest.fixture
    def store_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_register_provider(self, store_dir):
        manager = OAuthManager(store_dir)
        provider = GoogleOAuthProvider("client_id", "client_secret")
        manager.register_provider(provider)
        assert "google" in manager._providers

    def test_list_connected(self, store_dir):
        manager = OAuthManager(store_dir)
        assert manager.list_connected() == []

    @patch("scholardevclaw.auth.oauth.webbrowser.open")
    def test_start_flow(self, mock_browser, store_dir):
        manager = OAuthManager(store_dir)
        provider = GoogleOAuthProvider("client_id", "client_secret")
        manager.register_provider(provider)
        url, state = manager.start_flow("google")
        assert "google" in url
        assert len(state) > 0


class TestCreateOAuthProvider:
    def test_create_google(self):
        provider = create_oauth_provider("google", "id", "secret")
        assert isinstance(provider, GoogleOAuthProvider)

    def test_create_github(self):
        provider = create_oauth_provider("github", "id", "secret")
        assert isinstance(provider, GitHubOAuthProvider)

    def test_create_unknown(self):
        with pytest.raises(ValueError):
            create_oauth_provider("unknown", "id", "secret")
