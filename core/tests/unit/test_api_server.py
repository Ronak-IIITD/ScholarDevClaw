from __future__ import annotations

import importlib
from types import SimpleNamespace

from fastapi.testclient import TestClient


def _load_server(monkeypatch, **env):
    managed = {
        "SCHOLARDEVCLAW_API_AUTH_KEY",
        "SCHOLARDEVCLAW_ALLOWED_REPO_DIRS",
        "SCHOLARDEVCLAW_ENABLE_HSTS",
        "SCHOLARDEVCLAW_CORS_ORIGINS",
        "SCHOLARDEVCLAW_DEV_MODE",
    }
    for key in managed:
        monkeypatch.delenv(key, raising=False)
    # Default to dev mode so tests don't need to set both auth+confinement
    monkeypatch.setenv("SCHOLARDEVCLAW_DEV_MODE", "true")
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
    assert "x-request-id" in resp.headers


def test_auth_exempt_paths_still_accessible(monkeypatch):
    server = _load_server(monkeypatch, SCHOLARDEVCLAW_API_AUTH_KEY="secret")
    client = TestClient(server.app)

    for path in ("/health", "/health/live", "/health/ready", "/docs", "/openapi.json", "/metrics"):
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
    server = _load_server(monkeypatch, SCHOLARDEVCLAW_DEV_MODE="true")
    client = TestClient(server.app)

    resp = client.get("/health")

    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["x-xss-protection"] == "1; mode=block"
    assert resp.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert resp.headers["cache-control"] == "no-store"
    assert "default-src 'none'" in resp.headers["content-security-policy"]


def test_request_id_generated_when_missing(monkeypatch):
    server = _load_server(monkeypatch, SCHOLARDEVCLAW_DEV_MODE="true")
    client = TestClient(server.app)

    resp = client.get("/health")

    assert resp.status_code == 200
    assert "x-request-id" in resp.headers
    assert resp.headers["x-request-id"]


def test_request_id_echoes_incoming_header(monkeypatch):
    server = _load_server(monkeypatch, SCHOLARDEVCLAW_DEV_MODE="true")
    client = TestClient(server.app)

    resp = client.get("/health", headers={"X-Request-ID": "req-123"})

    assert resp.status_code == 200
    assert resp.headers["x-request-id"] == "req-123"


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

    server = _load_server(
        monkeypatch,
        SCHOLARDEVCLAW_ALLOWED_REPO_DIRS=str(allowed),
        SCHOLARDEVCLAW_API_AUTH_KEY="secret",
    )
    client = TestClient(server.app)

    resp = client.post(
        "/repo/analyze",
        json={"repoPath": str(outside)},
        headers={"Authorization": "Bearer secret"},
    )

    assert resp.status_code == 403
    assert "outside" in resp.json()["detail"] and "allowed" in resp.json()["detail"]


def test_repo_path_not_found_and_not_directory(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, SCHOLARDEVCLAW_DEV_MODE="true")
    client = TestClient(server.app)

    missing = tmp_path / "does-not-exist"
    missing_resp = client.post("/repo/analyze", json={"repoPath": str(missing)})
    assert missing_resp.status_code == 404

    file_path = tmp_path / "file.txt"
    file_path.write_text("x")
    file_resp = client.post("/repo/analyze", json={"repoPath": str(file_path)})
    assert file_resp.status_code == 400


def test_patch_generate_requires_repo_path(monkeypatch):
    server = _load_server(monkeypatch, SCHOLARDEVCLAW_DEV_MODE="true")
    client = TestClient(server.app)

    resp = client.post("/patch/generate", json={"mapping": {}})

    assert resp.status_code == 422


def test_patch_generate_repo_path_confinement(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()

    server = _load_server(
        monkeypatch,
        SCHOLARDEVCLAW_ALLOWED_REPO_DIRS=str(allowed),
        SCHOLARDEVCLAW_API_AUTH_KEY="secret",
    )
    client = TestClient(server.app)

    resp = client.post(
        "/patch/generate",
        json={"mapping": {}, "repoPath": str(outside)},
        headers={"Authorization": "Bearer secret"},
    )

    assert resp.status_code == 403


def test_request_models_forbid_extra_fields(monkeypatch):
    server = _load_server(monkeypatch, SCHOLARDEVCLAW_DEV_MODE="true")
    client = TestClient(server.app)

    resp = client.post(
        "/research/extract",
        json={"source": "1910.07467", "sourceType": "arxiv", "extra": "boom"},
    )

    assert resp.status_code == 422


def test_readiness_returns_ready_shape(monkeypatch):
    server = _load_server(monkeypatch, SCHOLARDEVCLAW_DEV_MODE="true")
    client = TestClient(server.app)

    resp = client.get("/health/ready")

    assert resp.status_code == 200
    payload = resp.json()
    assert "ready" in payload
    assert "reasons" in payload


def test_liveness_returns_alive_shape(monkeypatch):
    server = _load_server(monkeypatch, SCHOLARDEVCLAW_DEV_MODE="true")
    client = TestClient(server.app)

    resp = client.get("/health/live")

    assert resp.status_code in (200, 503)
    payload = resp.json()
    assert "alive" in payload
    assert "last_heartbeat" in payload


def test_repo_analyze_uses_multilang_analyzer(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, SCHOLARDEVCLAW_DEV_MODE="true")
    client = TestClient(server.app)

    repo = tmp_path / "demo_repo"
    repo.mkdir()
    (repo / "model.py").write_text(
        "class DemoModel:\n    pass\n\ndef train_model():\n    return 1\n"
    )

    resp = client.post("/repo/analyze", json={"repoPath": str(repo)})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["repoName"] == "demo_repo"
    assert payload["architecture"]["models"]
    assert payload["architecture"]["models"][0]["name"] == "DemoModel"
    assert payload["architecture"]["trainingLoop"]["file"] == "model.py"


def test_mapping_response_contains_patch_contract_fields(monkeypatch):
    server = _load_server(monkeypatch, SCHOLARDEVCLAW_DEV_MODE="true")
    client = TestClient(server.app)

    class FakeMappingEngine:
        seen_llm_assistant = "unset"

        def __init__(self, repo_analysis, research_spec, llm_assistant=None):
            FakeMappingEngine.seen_llm_assistant = llm_assistant
            self.research_spec = research_spec

        def map(self):
            return SimpleNamespace(
                targets=[
                    SimpleNamespace(
                        file="model.py",
                        line=12,
                        current_code="LayerNorm",
                        replacement_required=True,
                        context={"match_tier": "exact"},
                    )
                ],
                strategy="replace",
                confidence=88,
                confidence_breakdown={"version": "1", "total": 88},
                research_spec=self.research_spec,
            )

    monkeypatch.setattr(server, "MappingEngine", FakeMappingEngine)

    resp = client.post(
        "/mapping/map",
        json={
            "repoAnalysis": {"elements": [], "imports": []},
            "researchSpec": {
                "paper": {"title": "P", "authors": ["A"], "year": 2024},
                "algorithm": {"name": "RMSNorm", "description": "d"},
                "implementation": {"moduleName": "RMSNorm", "parentClass": "nn.Module"},
                "changes": {
                    "type": "replace",
                    "targetPattern": "LayerNorm",
                    "insertionPoints": ["Block"],
                    "replacement": "RMSNorm",
                    "expectedBenefits": ["speed"],
                },
            },
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["research_spec"]["changes"]["target_patterns"] == ["LayerNorm"]
    assert payload["researchSpec"]["changes"]["targetPattern"] == "LayerNorm"
    assert payload["targets"][0]["context"]["original"] == "LayerNorm"
    assert payload["targets"][0]["context"]["replacement"] == "RMSNorm"
    assert payload["targets"][0]["original"] == "LayerNorm"
    assert payload["targets"][0]["replacement"] == "RMSNorm"
    assert payload["confidence_breakdown"]["total"] == payload["confidence"]
    assert FakeMappingEngine.seen_llm_assistant is None


def test_mapping_to_patch_contract_continuity(monkeypatch, tmp_path):
    server = _load_server(monkeypatch, SCHOLARDEVCLAW_DEV_MODE="true")
    client = TestClient(server.app)

    class FakeMappingEngine:
        def __init__(self, repo_analysis, research_spec, llm_assistant=None):
            self.research_spec = research_spec

        def map(self):
            return SimpleNamespace(
                targets=[
                    SimpleNamespace(
                        file="model.py",
                        line=5,
                        current_code="LayerNorm",
                        replacement_required=True,
                        context={},
                    )
                ],
                strategy="replace",
                confidence=90,
                confidence_breakdown={"version": "1", "total": 90},
                research_spec=self.research_spec,
            )

    class FakePatchGenerator:
        received_mapping = None

        def __init__(self, repo_path, llm_assistant=None):
            self.repo_path = repo_path

        def generate(self, mapping):
            FakePatchGenerator.received_mapping = mapping
            return SimpleNamespace(
                new_files=[SimpleNamespace(path="rmsnorm.py", content="class RMSNorm: ...\n")],
                transformations=[],
                branch_name="integration/rmsnorm",
            )

    monkeypatch.setattr(server, "MappingEngine", FakeMappingEngine)
    monkeypatch.setattr(server, "PatchGenerator", FakePatchGenerator)

    map_resp = client.post(
        "/mapping/map",
        json={
            "repoAnalysis": {"elements": [], "imports": []},
            "researchSpec": {
                "paper": {"title": "P", "authors": ["A"], "year": 2024},
                "algorithm": {"name": "RMSNorm", "description": "d"},
                "implementation": {"moduleName": "RMSNorm", "parentClass": "nn.Module"},
                "changes": {
                    "type": "replace",
                    "targetPattern": "LayerNorm",
                    "insertionPoints": ["Block"],
                    "replacement": "RMSNorm",
                },
            },
        },
    )
    assert map_resp.status_code == 200

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "model.py").write_text("class LayerNorm:\n    pass\n")

    gen_resp = client.post(
        "/patch/generate",
        json={"mapping": map_resp.json(), "repoPath": str(repo)},
    )
    assert gen_resp.status_code == 200
    assert FakePatchGenerator.received_mapping is not None
    assert FakePatchGenerator.received_mapping["research_spec"]["changes"]["target_patterns"] == [
        "LayerNorm"
    ]
    assert map_resp.json()["confidence_breakdown"]["total"] == map_resp.json()["confidence"]
    assert FakePatchGenerator.received_mapping["targets"][0]["context"]["original"] == "LayerNorm"
    assert FakePatchGenerator.received_mapping["targets"][0]["context"]["replacement"] == "RMSNorm"


def test_research_extract_unknown_arxiv_returns_structured_422(monkeypatch):
    server = _load_server(monkeypatch, SCHOLARDEVCLAW_DEV_MODE="true")
    client = TestClient(server.app)

    resp = client.post(
        "/research/extract",
        json={"source": "9999.99999", "sourceType": "arxiv"},
    )

    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "extraction_failed"
    assert detail["source_type"] == "arxiv"
    assert detail["reason"] in {
        "llm_unavailable",
        "arxiv_abstract_unavailable",
        "llm_extraction_failed",
    }
