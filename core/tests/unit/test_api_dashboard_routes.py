from __future__ import annotations

import asyncio
import importlib

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


def _load_server(
    monkeypatch,
    *,
    api_auth_key: str | None = None,
    ws_query_token_compat: bool | None = None,
):
    for key in (
        "SCHOLARDEVCLAW_API_AUTH_KEY",
        "SCHOLARDEVCLAW_ALLOWED_REPO_DIRS",
        "SCHOLARDEVCLAW_ENABLE_HSTS",
        "SCHOLARDEVCLAW_CORS_ORIGINS",
        "SCHOLARDEVCLAW_WS_QUERY_TOKEN_COMPAT",
    ):
        monkeypatch.delenv(key, raising=False)

    # Set dev mode so server.py doesn't fail-closed on missing auth/confinement
    monkeypatch.setenv("SCHOLARDEVCLAW_DEV_MODE", "true")
    if api_auth_key is not None:
        monkeypatch.setenv("SCHOLARDEVCLAW_API_AUTH_KEY", api_auth_key)
    if ws_query_token_compat is not None:
        monkeypatch.setenv(
            "SCHOLARDEVCLAW_WS_QUERY_TOKEN_COMPAT",
            "true" if ws_query_token_compat else "false",
        )

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


def test_validate_output_dir_requires_repo_subpath(monkeypatch, tmp_path):
    _, dashboard = _load_server(monkeypatch)
    repo = tmp_path / "repo"
    repo.mkdir()

    inside = dashboard._validate_output_dir(str(repo / "out"), repo)
    assert inside == (repo / "out").resolve()

    with pytest.raises(HTTPException) as exc:
        dashboard._validate_output_dir(str(tmp_path / "sibling-out"), repo)

    assert exc.value.status_code == 403
    assert "inside the repository" in str(exc.value.detail)


def test_websocket_ping_pong_contract(monkeypatch):
    server, _ = _load_server(monkeypatch)
    client = TestClient(server.app)

    with client.websocket_connect("/api/ws/pipeline") as ws:
        ws.send_text('{"type":"ping"}')
        message = ws.receive_text()
        assert '"type": "pong"' in message


def test_websocket_auth_handshake_required_when_api_key_set(monkeypatch):
    server, _ = _load_server(monkeypatch, api_auth_key="secret")
    client = TestClient(server.app)

    with client.websocket_connect("/api/ws/pipeline") as ws:
        ws.send_text('{"type":"auth","token":"secret"}')
        auth_ok = ws.receive_text()
        assert '"type": "auth_ok"' in auth_ok

        ws.send_text('{"type":"ping"}')
        message = ws.receive_text()
        assert '"type": "pong"' in message


def test_websocket_auth_query_token_compat(monkeypatch):
    server, _ = _load_server(monkeypatch, api_auth_key="secret", ws_query_token_compat=True)
    client = TestClient(server.app)

    with client.websocket_connect("/api/ws/pipeline?token=secret") as ws:
        ws.send_text('{"type":"ping"}')
        message = ws.receive_text()
        assert '"type": "pong"' in message


def test_websocket_auth_fails_with_bad_handshake(monkeypatch):
    server, _ = _load_server(monkeypatch, api_auth_key="secret")
    client = TestClient(server.app)

    with client.websocket_connect("/api/ws/pipeline") as ws:
        ws.send_text('{"type":"auth","token":"wrong"}')
        # server closes on unauthorized; subsequent receive should fail
        try:
            ws.receive_text()
            assert False, "expected websocket auth failure"
        except Exception:
            pass


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
        lambda repo_path: (
            analyze_calls.append(repo_path)
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
            )()
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "run_suggest",
        lambda repo_path: (
            suggest_calls.append(repo_path)
            or type(
                "Result",
                (),
                {
                    "ok": True,
                    "payload": {
                        "suggestions": [
                            {
                                "pattern": "layernorm",
                                "paper": {"name": "rmsnorm", "title": "RMSNorm"},
                            }
                        ]
                    },
                    "error": None,
                },
            )()
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "run_map",
        lambda repo_path, spec_name: (
            map_calls.append((repo_path, spec_name))
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
            )()
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "run_generate",
        lambda repo_path, spec_name, output_dir=None: (
            generate_calls.append((repo_path, spec_name, output_dir))
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
            )()
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "run_validate",
        lambda repo_path, _patch=None: (
            validate_calls.append(repo_path)
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
            )()
        ),
    )

    setattr(
        dashboard,
        "_current_run",
        dashboard.PipelineRunStatus(
            run_id="run1234",
            status="running",
            repo_path="/tmp/repo",
            spec_names=[],
            started_at=0.0,
        ),
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
    current_run = getattr(dashboard, "_current_run")
    assert current_run is not None
    assert current_run.status == "completed"
    assert [step.step for step in current_run.steps] == [
        "analyze",
        "suggest",
        "map:rmsnorm",
        "generate:rmsnorm",
        "validate:rmsnorm",
    ]
