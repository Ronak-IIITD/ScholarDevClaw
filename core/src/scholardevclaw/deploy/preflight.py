from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PLACEHOLDER_MARKERS = (
    "replace_with",
    "your_",
    "change_me",
    "...",
)

PROVIDER_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
}


@dataclass(slots=True)
class DeployPreflightReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, str] = field(default_factory=dict)


def parse_env_file(env_file: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values


def _is_placeholder(value: str) -> bool:
    lower = value.strip().lower()
    return any(marker in lower for marker in PLACEHOLDER_MARKERS)


def run_deploy_preflight(env: dict[str, str]) -> DeployPreflightReport:
    report = DeployPreflightReport(ok=True)

    required = [
        "SCHOLARDEVCLAW_API_AUTH_KEY",
        "SCHOLARDEVCLAW_ALLOWED_REPO_DIRS",
        "SCHOLARDEVCLAW_CORS_ORIGINS",
        "OPENCLAW_TOKEN",
        "OPENCLAW_API_URL",
        "GRAFANA_ADMIN_USER",
        "GRAFANA_ADMIN_PASSWORD",
    ]
    for key in required:
        value = env.get(key, "").strip()
        if not value:
            report.errors.append(f"Missing required variable: {key}")
            report.checks[key] = "missing"
            continue
        if _is_placeholder(value):
            report.errors.append(f"{key} is using a placeholder value.")
            report.checks[key] = "placeholder"
        else:
            report.checks[key] = "ok"

    bridge_mode = (env.get("CORE_BRIDGE_MODE") or "http").strip().lower()
    if bridge_mode not in {"http", "subprocess"}:
        report.errors.append(f"CORE_BRIDGE_MODE must be 'http' or 'subprocess', got: {bridge_mode}")
        report.checks["CORE_BRIDGE_MODE"] = "invalid"
    else:
        report.checks["CORE_BRIDGE_MODE"] = "ok"

    cors = env.get("SCHOLARDEVCLAW_CORS_ORIGINS", "")
    if "localhost" in cors or "127.0.0.1" in cors:
        report.warnings.append(
            "SCHOLARDEVCLAW_CORS_ORIGINS contains localhost entries. Remove for internet-facing production."
        )

    provider = (env.get("SCHOLARDEVCLAW_API_PROVIDER") or "anthropic").strip().lower()
    provider_key = PROVIDER_KEY_ENV.get(provider)
    if provider_key:
        pvalue = env.get(provider_key, "").strip()
        if not pvalue:
            report.errors.append(
                f"Provider '{provider}' selected but {provider_key} is not set."
            )
            report.checks[provider_key] = "missing"
        elif _is_placeholder(pvalue):
            report.errors.append(
                f"Provider '{provider}' selected but {provider_key} looks like a placeholder."
            )
            report.checks[provider_key] = "placeholder"
        else:
            report.checks[provider_key] = "ok"

    if env.get("GRAFANA_ADMIN_PASSWORD", "").strip().lower() == "change_me_in_production":
        report.errors.append("GRAFANA_ADMIN_PASSWORD must be changed from default placeholder.")
        report.checks["GRAFANA_ADMIN_PASSWORD"] = "placeholder"

    report.ok = len(report.errors) == 0
    return report
