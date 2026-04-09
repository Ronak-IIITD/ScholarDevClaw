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


def test_production_preflight_flags_missing_requirements(monkeypatch):
    monkeypatch.delenv("SCHOLARDEVCLAW_DEV_MODE", raising=False)
    for var in (
        "SCHOLARDEVCLAW_API_AUTH_KEY",
        "SCHOLARDEVCLAW_ALLOWED_REPO_DIRS",
        "SCHOLARDEVCLAW_CORS_ORIGINS",
        "OPENCLAW_TOKEN",
        "OPENCLAW_API_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CORE_BRIDGE_MODE", "http")

    checker = HealthChecker()
    result = checker.run_check("production")

    assert result.healthy is False
    assert result.details["issues"]
    assert any("SCHOLARDEVCLAW_API_AUTH_KEY" in i for i in result.details["issues"])


def test_production_preflight_passes_when_required_env_is_set(monkeypatch):
    monkeypatch.delenv("SCHOLARDEVCLAW_DEV_MODE", raising=False)
    monkeypatch.setenv("SCHOLARDEVCLAW_API_AUTH_KEY", "secret")
    monkeypatch.setenv("SCHOLARDEVCLAW_ALLOWED_REPO_DIRS", "/repos")
    monkeypatch.setenv("SCHOLARDEVCLAW_CORS_ORIGINS", "https://example.com")
    monkeypatch.setenv("OPENCLAW_TOKEN", "token")
    monkeypatch.setenv("OPENCLAW_API_URL", "https://openclaw.example.com")
    monkeypatch.setenv("CORE_BRIDGE_MODE", "http")

    checker = HealthChecker()
    result = checker.run_check("production")

    assert result.healthy is True
    assert result.details["issues"] == []
