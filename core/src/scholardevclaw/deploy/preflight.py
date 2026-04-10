from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REQUIRED_ENV_KEYS: tuple[str, ...] = (
    "SCHOLARDEVCLAW_API_AUTH_KEY",
    "SCHOLARDEVCLAW_ALLOWED_REPO_DIRS",
    "SCHOLARDEVCLAW_CORS_ORIGINS",
    "OPENCLAW_TOKEN",
    "OPENCLAW_API_URL",
    "GRAFANA_ADMIN_USER",
    "GRAFANA_ADMIN_PASSWORD",
)

VALID_BRIDGE_MODES: set[str] = {"http", "subprocess"}

PROVIDER_ENV_KEY_MAP: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "ollama": "OLLAMA_HOST",
}

_PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "replace_with",
    "replace",
    "your_",
    "change_me",
    "placeholder",
    "todo",
)
_DOUBLE_UNDERSCORE_PLACEHOLDER_RE = re.compile(r"__[_A-Za-z0-9]+__")


@dataclass
class PreflightResult:
    env_file: str
    ok: bool
    errors: list[str]
    warnings: list[str]
    recommendations: list[str]


def parse_env_file(env_file: str | Path) -> dict[str, str]:
    path = Path(env_file)
    if not path.exists():
        raise FileNotFoundError(f"Environment file not found: {path}")

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        values[key] = value

    return values


def is_placeholder_value(value: str) -> bool:
    normalized = value.strip()
    if not normalized:
        return True

    lowered = normalized.lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        return True

    return _DOUBLE_UNDERSCORE_PLACEHOLDER_RE.search(normalized) is not None


def run_preflight(env_file: str | Path) -> PreflightResult:
    env_values = parse_env_file(env_file)
    errors: list[str] = []
    warnings: list[str] = []
    recommendations: list[str] = []

    for key in REQUIRED_ENV_KEYS:
        value = env_values.get(key, "").strip()
        if not value:
            errors.append(f"Missing required key: {key}")
            continue
        if is_placeholder_value(value):
            errors.append(f"{key} appears to use a placeholder value")

    bridge_mode = env_values.get("CORE_BRIDGE_MODE", "").strip().lower()
    if bridge_mode not in VALID_BRIDGE_MODES:
        errors.append(f"CORE_BRIDGE_MODE must be one of: {', '.join(sorted(VALID_BRIDGE_MODES))}")

    provider = env_values.get("SCHOLARDEVCLAW_API_PROVIDER", "").strip().lower()
    if provider:
        provider_key = PROVIDER_ENV_KEY_MAP.get(provider)
        if provider_key is None:
            errors.append(
                "SCHOLARDEVCLAW_API_PROVIDER is unsupported: "
                f"{provider} (supported: {', '.join(sorted(PROVIDER_ENV_KEY_MAP))})"
            )
        else:
            provider_value = env_values.get(provider_key, "").strip()
            if not provider_value:
                errors.append(
                    f"{provider_key} is required when SCHOLARDEVCLAW_API_PROVIDER={provider}"
                )
            elif is_placeholder_value(provider_value):
                errors.append(
                    f"{provider_key} appears to use a placeholder value for provider {provider}"
                )

    cors_origins = env_values.get("SCHOLARDEVCLAW_CORS_ORIGINS", "")
    if "localhost" in cors_origins.lower() or "127.0.0.1" in cors_origins:
        warnings.append(
            "SCHOLARDEVCLAW_CORS_ORIGINS includes localhost/127.0.0.1; "
            "remove local origins for production"
        )

    grafana_password = env_values.get("GRAFANA_ADMIN_PASSWORD", "").strip()
    if grafana_password == "change_me_in_production":
        errors.append("GRAFANA_ADMIN_PASSWORD must not be change_me_in_production")

    if bridge_mode == "http" and not env_values.get("CORE_API_URL", "").strip():
        recommendations.append("Set CORE_API_URL when CORE_BRIDGE_MODE=http")

    return PreflightResult(
        env_file=str(Path(env_file)),
        ok=not errors,
        errors=errors,
        warnings=warnings,
        recommendations=recommendations,
    )
