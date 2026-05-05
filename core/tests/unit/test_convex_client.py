"""Tests for ConvexClient - TUI↔Convex integration."""

from __future__ import annotations

from unittest import mock

import httpx
import pytest

from scholardevclaw.convex_client import ConvexClient, ConvexIntegration


class TestConvexClientInit:
    """Tests for ConvexClient initialization."""

    def test_init_with_env_vars(self, monkeypatch):
        """Test initialization with CONVEX_URL and auth key set."""
        monkeypatch.setenv("CONVEX_URL", "https://test.convex.cloud")
        monkeypatch.setenv("SCHOLARDEVCLAW_CONVEX_AUTH_KEY", "test-key-123")

        client = ConvexClient()

        assert client.url == "https://test.convex.cloud"
        assert client.auth_key == "test-key-123"

    def test_init_without_env_vars(self, monkeypatch):
        """Test initialization without env vars - should not raise."""
        monkeypatch.delenv("CONVEX_URL", raising=False)
        monkeypatch.delenv("SCHOLARDEVCLAW_CONVEX_AUTH_KEY", raising=False)

        client = ConvexClient()

        assert client.url is None
        assert client.auth_key is None


class TestConvexClientMutation:
    """Tests for ConvexClient mutation methods."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create client with mocked env vars."""
        monkeypatch.setenv("CONVEX_URL", "https://test.convex.cloud")
        monkeypatch.setenv("SCHOLARDEVCLAW_CONVEX_AUTH_KEY", "test-key")
        return ConvexClient()

    def test_call_mutation_missing_config(self, client):
        """Test that missing URL or auth key raises RuntimeError."""
        client.url = None
        client.auth_key = "some-key"

        with pytest.raises(RuntimeError, match="Convex configuration missing"):
            client._call_mutation("test:mutation", {})

    def test_call_mutation_success(self, client, monkeypatch):
        """Test successful mutation call."""
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {"id": "123"}
        mock_response.raise_for_status = mock.MagicMock()

        with mock.patch.object(httpx.Client, "post", return_value=mock_response):
            result = client._call_mutation("test:mutation", {"foo": "bar"})

        assert result == {"id": "123"}

    def test_call_mutation_http_error(self, client, monkeypatch):
        """Test mutation raises on HTTP error."""
        import httpx

        mock_response = mock.MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error", request=mock.MagicMock(), response=mock.MagicMock()
        )

        with mock.patch.object(httpx.Client, "post", return_value=mock_response):
            with pytest.raises(RuntimeError, match="Convex mutation test:mutation failed"):
                client._call_mutation("test:mutation", {})


class TestConvexClientQuery:
    """Tests for ConvexClient query methods."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create client with mocked env vars."""
        monkeypatch.setenv("CONVEX_URL", "https://test.convex.cloud")
        return ConvexClient()

    def test_call_query_missing_url(self, client):
        """Test that missing URL raises RuntimeError."""
        client.url = None

        with pytest.raises(RuntimeError, match="Convex URL not set"):
            client._call_query("test:query", {})

    def test_call_query_success(self, client, monkeypatch):
        """Test successful query call."""
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status = mock.MagicMock()

        with mock.patch.object(httpx.Client, "get", return_value=mock_response):
            result = client._call_query("test:query", {"id": "123"})

        assert result == {"data": "test"}

    def test_call_query_http_error(self, client, monkeypatch):
        """Test query raises on HTTP error."""
        import httpx

        mock_response = mock.MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error", request=mock.MagicMock(), response=mock.MagicMock()
        )

        with mock.patch.object(httpx.Client, "get", return_value=mock_response):
            with pytest.raises(RuntimeError, match="Convex query test:query failed"):
                client._call_query("test:query", {})


class TestCreateIntegration:
    """Tests for create_integration method."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create client with mocked env vars."""
        monkeypatch.setenv("CONVEX_URL", "https://test.convex.cloud")
        monkeypatch.setenv("SCHOLARDEVCLAW_CONVEX_AUTH_KEY", "test-key")
        return ConvexClient()

    def test_create_integration_with_repo_url(self, client, monkeypatch):
        """Test creating integration with repo URL only."""
        mock_response = mock.MagicMock()
        mock_response.json.return_value = "integration-123"
        mock_response.raise_for_status = mock.MagicMock()

        with mock.patch.object(httpx.Client, "post", return_value=mock_response):
            result = client.create_integration(repo_url="https://github.com/test/repo")

        assert result == "integration-123"

    def test_create_integration_with_paper_url(self, client, monkeypatch):
        """Test creating integration with paper URL."""
        mock_response = mock.MagicMock()
        mock_response.json.return_value = "integration-456"
        mock_response.raise_for_status = mock.MagicMock()

        with mock.patch.object(httpx.Client, "post", return_value=mock_response):
            result = client.create_integration(
                repo_url="https://github.com/test/repo",
                paper_url="https://arxiv.org/pdf/1234.5678",
            )

        assert result == "integration-456"

    def test_create_integration_autonomous_mode(self, client, monkeypatch):
        """Test creating integration with autonomous mode."""
        mock_response = mock.MagicMock()
        mock_response.json.return_value = "integration-789"
        mock_response.raise_for_status = mock.MagicMock()

        with mock.patch.object(httpx.Client, "post", return_value=mock_response):
            result = client.create_integration(
                repo_url="https://github.com/test/repo",
                mode="autonomous",
            )

        assert result == "integration-789"


class TestGetIntegrationStatus:
    """Tests for get_integration_status method."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create client with mocked env vars."""
        monkeypatch.setenv("CONVEX_URL", "https://test.convex.cloud")
        return ConvexClient()

    def test_get_status_success(self, client, monkeypatch):
        """Test successful status retrieval."""
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {
            "_id": "int-123",
            "status": "running",
            "currentPhase": 3,
            "repoUrl": "https://github.com/test/repo",
            "paperUrl": "https://arxiv.org/pdf/1234",
            "mode": "step_approval",
        }
        mock_response.raise_for_status = mock.MagicMock()

        with mock.patch.object(httpx.Client, "get", return_value=mock_response):
            result = client.get_integration_status("int-123")

        assert isinstance(result, ConvexIntegration)
        assert result.id == "int-123"
        assert result.status == "running"
        assert result.current_phase == 3
        assert result.repo_url == "https://github.com/test/repo"
        assert result.paper_url == "https://arxiv.org/pdf/1234"
        assert result.mode == "step_approval"

    def test_get_status_not_found(self, client, monkeypatch):
        """Test status retrieval for non-existent integration."""
        mock_response = mock.MagicMock()
        mock_response.json.return_value = None
        mock_response.raise_for_status = mock.MagicMock()

        with mock.patch.object(httpx.Client, "get", return_value=mock_response):
            with pytest.raises(RuntimeError, match="Integration not-found not found"):
                client.get_integration_status("not-found")

    def test_get_status_defaults(self, client, monkeypatch):
        """Test status retrieval with missing optional fields."""
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {
            "_id": "int-123",
            "status": "pending",
            "repoUrl": "https://github.com/test/repo",
        }
        mock_response.raise_for_status = mock.MagicMock()

        with mock.patch.object(httpx.Client, "get", return_value=mock_response):
            result = client.get_integration_status("int-123")

        assert result.current_phase == 0
        assert result.mode == "step_approval"
        assert result.paper_url is None
        assert result.paper_pdf_path is None


class TestCreateApproval:
    """Tests for create_approval method."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create client with mocked env vars."""
        monkeypatch.setenv("CONVEX_URL", "https://test.convex.cloud")
        monkeypatch.setenv("SCHOLARDEVCLAW_CONVEX_AUTH_KEY", "test-key")
        return ConvexClient()

    def test_create_approval_approved(self, client, monkeypatch):
        """Test creating approval with approved action."""
        mock_response = mock.MagicMock()
        mock_response.raise_for_status = mock.MagicMock()

        with mock.patch.object(httpx.Client, "post", return_value=mock_response):
            client.create_approval(
                integration_id="int-123",
                phase=2,
                action="approved",
                notes="Looks good",
            )

    def test_create_approval_rejected(self, client, monkeypatch):
        """Test creating approval with rejected action."""
        mock_response = mock.MagicMock()
        mock_response.raise_for_status = mock.MagicMock()

        with mock.patch.object(httpx.Client, "post", return_value=mock_response):
            client.create_approval(
                integration_id="int-123",
                phase=3,
                action="rejected",
                notes="Needs changes",
            )

    def test_create_approval_no_notes(self, client, monkeypatch):
        """Test creating approval without notes."""
        mock_response = mock.MagicMock()
        mock_response.raise_for_status = mock.MagicMock()

        with mock.patch.object(httpx.Client, "post", return_value=mock_response):
            client.create_approval(
                integration_id="int-123",
                phase=1,
                action="approved",
            )


class TestConvexIntegration:
    """Tests for ConvexIntegration dataclass."""

    def test_create_integration_all_fields(self):
        """Test creating ConvexIntegration with all fields."""
        integration = ConvexIntegration(
            id="int-123",
            status="completed",
            current_phase=5,
            repo_url="https://github.com/test/repo",
            paper_url="https://arxiv.org/pdf/1234",
            paper_pdf_path="/tmp/paper.pdf",
            mode="autonomous",
        )

        assert integration.id == "int-123"
        assert integration.status == "completed"
        assert integration.current_phase == 5
        assert integration.repo_url == "https://github.com/test/repo"
        assert integration.paper_url == "https://arxiv.org/pdf/1234"
        assert integration.paper_pdf_path == "/tmp/paper.pdf"
        assert integration.mode == "autonomous"

    def test_create_integration_defaults(self):
        """Test creating ConvexIntegration with default values."""
        integration = ConvexIntegration(
            id="int-123",
            status="pending",
            current_phase=0,
            repo_url="https://github.com/test/repo",
        )

        assert integration.paper_url is None
        assert integration.paper_pdf_path is None
        assert integration.mode == "step_approval"
