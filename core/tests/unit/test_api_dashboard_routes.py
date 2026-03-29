from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def _load_server(monkeypatch):
    for key in (
        "SCHOLARDEVCLAW_API_AUTH_KEY",
        "SCHOLARDEVCLAW_ALLOWED_REPO_DIRS",
        "SCHOLARDEVCLAW_ENABLE_HSTS",
        "SCHOLARDEVCLAW_CORS_ORIGINS",
    ):
        monkeypatch.delenv(key, raising=False)

    import scholardevclaw.api.routes.dashboard as dashboard
    import scholardevclaw.api.server as server

    reloaded_server = importlib.reload(server)
    reloaded_dashboard = importlib.reload(dashboard)
    setattr(reloaded_dashboard, "_current_run", None)
    getattr(reloaded_dashboard, "_ws_clients").clear()
    return reloaded_server, reloaded_dashboard


def test_specs_list_and_missing_spec_route(monkeypatch):
    server, _ = _load_server(monkeypatch)
    client = TestClient(server.app)

    specs = client.get("/api/specs")
    assert specs.status_code == 200
    data = specs.json()
    assert isinstance(data, list)
    assert data
    assert {"name", "title", "algorithm", "category", "replaces"}.issubset(data[0].keys())

    missing = client.get("/api/specs/__missing_spec__")
    assert missing.status_code == 404


def test_pipeline_run_conflict_when_already_running(monkeypatch):
    server, dashboard = _load_server(monkeypatch)
    client = TestClient(server.app)

    setattr(dashboard, "_current_run", dashboard.PipelineRunStatus(run_id="x", status="running"))
    try:
        resp = client.post("/api/pipeline/run", json={"repo_path": ".", "spec_names": []})
        assert resp.status_code == 409
        assert "already in progress" in resp.json()["detail"]
    finally:
        setattr(dashboard, "_current_run", None)


def test_websocket_ping_pong_contract(monkeypatch):
    server, _ = _load_server(monkeypatch)
    client = TestClient(server.app)

    with client.websocket_connect("/api/ws/pipeline") as ws:
        ws.send_text('{"type":"ping"}')
        message = ws.receive_text()
        assert '"type": "pong"' in message
