from __future__ import annotations

import asyncio
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

    # Set dev mode so server.py doesn't fail-closed on missing auth/confinement
    monkeypatch.setenv("SCHOLARDEVCLAW_DEV_MODE", "true")

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


def test_pipeline_async_uses_shared_pipeline_functions(monkeypatch):
    _, dashboard = _load_server(monkeypatch)

    analyze_calls: list[str] = []
    suggest_calls: list[str] = []
    map_calls: list[tuple[str, str]] = []
    generate_calls: list[tuple[str, str, str | None]] = []
    validate_calls: list[str] = []

    monkeypatch.setattr(
        dashboard,
        "run_analyze",
        lambda repo_path: analyze_calls.append(repo_path)
        or type(
            "Result",
            (),
            {
                "ok": True,
                "payload": {
                    "languages": ["python"],
                    "language_stats": [{"file_count": 1}],
                    "frameworks": ["pytorch"],
                    "entry_points": ["main.py"],
                    "test_files": ["tests/test_demo.py"],
                    "patterns": {"rmsnorm": ["model.py"]},
                },
                "error": None,
            },
        )(),
    )
    monkeypatch.setattr(
        dashboard,
        "run_suggest",
        lambda repo_path: suggest_calls.append(repo_path)
        or type(
            "Result",
            (),
            {
                "ok": True,
                "payload": {
                    "suggestions": [
                        {"pattern": "layernorm", "paper": {"name": "rmsnorm", "title": "RMSNorm"}}
                    ]
                },
                "error": None,
            },
        )(),
    )
    monkeypatch.setattr(
        dashboard,
        "run_map",
        lambda repo_path, spec_name: map_calls.append((repo_path, spec_name))
        or type(
            "Result",
            (),
            {
                "ok": True,
                "payload": {
                    "target_count": 1,
                    "strategy": "exact",
                    "confidence": 92,
                    "targets": [{"file": "model.py", "line": 3}],
                },
                "error": None,
            },
        )(),
    )
    monkeypatch.setattr(
        dashboard,
        "run_generate",
        lambda repo_path, spec_name, output_dir=None: generate_calls.append(
            (repo_path, spec_name, output_dir)
        )
        or type(
            "Result",
            (),
            {
                "ok": True,
                "payload": {
                    "branch_name": "integration/rmsnorm",
                    "new_files": [{"path": "rmsnorm.py"}],
                    "transformations": [{"file": "model.py"}],
                    "output_dir": output_dir,
                },
                "error": None,
            },
        )(),
    )
    monkeypatch.setattr(
        dashboard,
        "run_validate",
        lambda repo_path: validate_calls.append(repo_path)
        or type(
            "Result",
            (),
            {
                "ok": True,
                "payload": {
                    "passed": True,
                    "stage": "benchmark",
                    "scorecard": {"summary": "pass"},
                },
                "error": None,
            },
        )(),
    )

    dashboard._current_run = dashboard.PipelineRunStatus(
        run_id="run1234",
        status="running",
        repo_path="/tmp/repo",
        spec_names=[],
        started_at=0.0,
    )
    dashboard._ws_clients.clear()

    asyncio.run(
        dashboard._run_pipeline_async(
            run_id="run1234",
            repo_path="/tmp/repo",
            spec_names=[],
            skip_validate=False,
            output_dir="/tmp/out",
        )
    )

    assert analyze_calls == ["/tmp/repo"]
    assert suggest_calls == ["/tmp/repo"]
    assert map_calls == [("/tmp/repo", "rmsnorm")]
    assert generate_calls == [("/tmp/repo", "rmsnorm", "/tmp/out")]
    assert validate_calls == ["/tmp/repo"]
    assert dashboard._current_run is not None
    assert dashboard._current_run.status == "completed"
    assert [step.step for step in dashboard._current_run.steps] == [
        "analyze",
        "suggest",
        "map:rmsnorm",
        "generate:rmsnorm",
        "validate:rmsnorm",
    ]
