from __future__ import annotations

import os

from scholardevclaw.utils.health import HealthChecker


def test_environment_check_requires_core_vars_in_prod(monkeypatch):
    monkeypatch.delenv("SCHOLARDEVCLAW_DEV_MODE", raising=False)
    monkeypatch.delenv("SCHOLARDEVCLAW_API_AUTH_KEY", raising=False)
    monkeypatch.delenv("SCHOLARDEVCLAW_ALLOWED_REPO_DIRS", raising=False)

    checker = HealthChecker()
    result = checker.run_check("environment")

    assert result.healthy is False
    assert "SCHOLARDEVCLAW_API_AUTH_KEY" in result.details["missing_required"]
    assert "SCHOLARDEVCLAW_ALLOWED_REPO_DIRS" in result.details["missing_required"]


def test_environment_check_relaxes_required_vars_in_dev_mode(monkeypatch):
    monkeypatch.setenv("SCHOLARDEVCLAW_DEV_MODE", "true")
    monkeypatch.delenv("SCHOLARDEVCLAW_API_AUTH_KEY", raising=False)
    monkeypatch.delenv("SCHOLARDEVCLAW_ALLOWED_REPO_DIRS", raising=False)

    checker = HealthChecker()
    result = checker.run_check("environment")

    assert result.healthy is True
    assert result.details["dev_mode"] is True
    assert result.details["missing_required"] == []

    # Avoid cross-test contamination for environments that reuse process state.
    os.environ.pop("SCHOLARDEVCLAW_DEV_MODE", None)


def test_production_preflight_fails_with_missing_vars(monkeypatch):
    monkeypatch.delenv("SCHOLARDEVCLAW_DEV_MODE", raising=False)
    monkeypatch.delenv("SCHOLARDEVCLAW_API_AUTH_KEY", raising=False)
    monkeypatch.delenv("SCHOLARDEVCLAW_ALLOWED_REPO_DIRS", raising=False)
    monkeypatch.delenv("SCHOLARDEVCLAW_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("OPENCLAW_TOKEN", raising=False)
    monkeypatch.delenv("OPENCLAW_API_URL", raising=False)
    monkeypatch.delenv("GRAFANA_ADMIN_USER", raising=False)
    monkeypatch.delenv("GRAFANA_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("CORE_BRIDGE_MODE", raising=False)

    checker = HealthChecker()
    result = checker.run_check("production")

    assert result.healthy is False
    assert result.name == "production"
    assert any("SCHOLARDEVCLAW_API_AUTH_KEY" in issue for issue in result.details["issues"])
    assert any("CORE_BRIDGE_MODE" in issue for issue in result.details["issues"])


def test_production_preflight_passes_with_required_vars(monkeypatch):
    monkeypatch.delenv("SCHOLARDEVCLAW_DEV_MODE", raising=False)
    monkeypatch.setenv("SCHOLARDEVCLAW_API_AUTH_KEY", "real-api-key")
    monkeypatch.setenv("SCHOLARDEVCLAW_ALLOWED_REPO_DIRS", "/repos")
    monkeypatch.setenv("SCHOLARDEVCLAW_CORS_ORIGINS", "https://scholardevclaw.ai")
    monkeypatch.setenv("OPENCLAW_TOKEN", "real-openclaw-token")
    monkeypatch.setenv("OPENCLAW_API_URL", "https://api.openclaw.example")
    monkeypatch.setenv("GRAFANA_ADMIN_USER", "admin")
    monkeypatch.setenv("GRAFANA_ADMIN_PASSWORD", "super-secure-password")
    monkeypatch.setenv("CORE_BRIDGE_MODE", "http")
    monkeypatch.setenv("CORE_API_URL", "http://core-api:8000")

    checker = HealthChecker()
    result = checker.run_check("production")

    assert result.healthy is True
    assert result.details["issues"] == []


def test_production_preflight_skips_in_dev_mode(monkeypatch):
    monkeypatch.setenv("SCHOLARDEVCLAW_DEV_MODE", "true")

    checker = HealthChecker()
    result = checker.run_check("production")

    assert result.healthy is True
    assert result.details["skipped"] is True
    assert result.details["dev_mode"] is True
