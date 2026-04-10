from __future__ import annotations

from scholardevclaw.deploy.preflight import run_deploy_preflight


def test_preflight_fails_with_missing_required_vars():
    report = run_deploy_preflight({})
    assert report.ok is False
    assert any("SCHOLARDEVCLAW_API_AUTH_KEY" in e for e in report.errors)


def test_preflight_passes_with_valid_env():
    report = run_deploy_preflight(
        {
            "SCHOLARDEVCLAW_API_AUTH_KEY": "secret",
            "SCHOLARDEVCLAW_ALLOWED_REPO_DIRS": "/repos",
            "SCHOLARDEVCLAW_CORS_ORIGINS": "https://app.example.com",
            "OPENCLAW_TOKEN": "token",
            "OPENCLAW_API_URL": "https://openclaw.example.com",
            "GRAFANA_ADMIN_USER": "admin",
            "GRAFANA_ADMIN_PASSWORD": "super-secret",
            "CORE_BRIDGE_MODE": "http",
            "SCHOLARDEVCLAW_API_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "sk-ant-123",
        }
    )
    assert report.ok is True
    assert report.errors == []
