from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def _load_server(monkeypatch, **env):
    managed = {
        "SCHOLARDEVCLAW_API_AUTH_KEY",
        "SCHOLARDEVCLAW_ALLOWED_REPO_DIRS",
        "SCHOLARDEVCLAW_ENABLE_HSTS",
        "SCHOLARDEVCLAW_CORS_ORIGINS",
    }
    for key in managed:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    import scholardevclaw.api.server as server

    return importlib.reload(server)


def test_auth_required_for_non_exempt_paths(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, SCHOLARDEVCLAW_API_AUTH_KEY="secret")
    client = TestClient(server.app)

    resp = client.post("/repo/analyze", json={"repoPath": str(tmp_path)})

    assert resp.status_code == 401
    assert resp.json() == {"detail": "Unauthorized"}


def test_auth_exempt_paths_still_accessible(monkeypatch):
    server = _load_server(monkeypatch, SCHOLARDEVCLAW_API_AUTH_KEY="secret")
    client = TestClient(server.app)

    for path in ("/health", "/docs", "/openapi.json", "/metrics"):
        resp = client.get(path)
        assert resp.status_code != 401


def test_auth_with_valid_token_reaches_endpoint(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, SCHOLARDEVCLAW_API_AUTH_KEY="secret")
    client = TestClient(server.app)

    missing = tmp_path / "missing-repo"
    resp = client.post(
        "/repo/analyze",
        json={"repoPath": str(missing)},
        headers={"Authorization": "Bearer secret"},
    )

    assert resp.status_code == 404


def test_security_headers_applied(monkeypatch):
    server = _load_server(monkeypatch)
    client = TestClient(server.app)

    resp = client.get("/health")

    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["x-xss-protection"] == "1; mode=block"
    assert resp.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert resp.headers["cache-control"] == "no-store"
    assert "default-src 'none'" in resp.headers["content-security-policy"]


def test_hsts_header_only_when_enabled(monkeypatch):
    enabled = _load_server(monkeypatch, SCHOLARDEVCLAW_ENABLE_HSTS="true")
    enabled_client = TestClient(enabled.app)
    enabled_resp = enabled_client.get("/health")
    assert "strict-transport-security" in enabled_resp.headers

    disabled = _load_server(monkeypatch, SCHOLARDEVCLAW_ENABLE_HSTS="false")
    disabled_client = TestClient(disabled.app)
    disabled_resp = disabled_client.get("/health")
    assert "strict-transport-security" not in disabled_resp.headers


def test_repo_path_confinement_blocks_outside_paths(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()

    server = _load_server(monkeypatch, SCHOLARDEVCLAW_ALLOWED_REPO_DIRS=str(allowed))
    client = TestClient(server.app)

    resp = client.post("/repo/analyze", json={"repoPath": str(outside)})

    assert resp.status_code == 403
    assert "outside the allowed directories" in resp.json()["detail"]


def test_repo_path_not_found_and_not_directory(monkeypatch, tmp_path):
    server = _load_server(monkeypatch)
    client = TestClient(server.app)

    missing = tmp_path / "does-not-exist"
    missing_resp = client.post("/repo/analyze", json={"repoPath": str(missing)})
    assert missing_resp.status_code == 404

    file_path = tmp_path / "file.txt"
    file_path.write_text("x")
    file_resp = client.post("/repo/analyze", json={"repoPath": str(file_path)})
    assert file_resp.status_code == 400


def test_request_models_forbid_extra_fields(monkeypatch):
    server = _load_server(monkeypatch)
    client = TestClient(server.app)

    resp = client.post(
        "/research/extract",
        json={"source": "1910.07467", "sourceType": "arxiv", "extra": "boom"},
    )

    assert resp.status_code == 422
