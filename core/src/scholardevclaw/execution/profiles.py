"""Cloud-ready execution profiles for sandboxed test runs.

Provides named profiles with different resource configurations
(CPU, memory, timeout, image, network) so users can pick the right
environment for their integration tests.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExecutionProfile:
    """A named configuration for sandbox execution."""

    name: str
    description: str = ""
    image: str = "sdc-sandbox:latest"
    timeout_seconds: int = 300
    memory_limit_mb: int = 4096
    cpu_limit: float = 2.0
    network_enabled: bool = False
    gpu_enabled: bool = False
    gpu_count: int = 0
    extra_volumes: dict[str, dict[str, str]] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    pytest_args: str = "tests/ -v --json-report --json-report-file=/tmp/report.json"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionProfile:
        return cls(**{k: v for k, v in data.items() if k in cls.__slots__})


# ── Preset profiles ─────────────────────────────────────────────────────

PRESET_PROFILES: dict[str, ExecutionProfile] = {
    "local": ExecutionProfile(
        name="local",
        description="Local development — minimal resources, fast iteration",
        image="sdc-sandbox:latest",
        timeout_seconds=120,
        memory_limit_mb=2048,
        cpu_limit=1.0,
        network_enabled=False,
        gpu_enabled=False,
    ),
    "cloud-cpu": ExecutionProfile(
        name="cloud-cpu",
        description="Cloud CPU instance — balanced resources for standard CI",
        image="sdc-sandbox:latest",
        timeout_seconds=300,
        memory_limit_mb=4096,
        cpu_limit=2.0,
        network_enabled=False,
        gpu_enabled=False,
    ),
    "cloud-gpu": ExecutionProfile(
        name="cloud-gpu",
        description="Cloud GPU instance — for ML model validation",
        image="sdc-sandbox-gpu:latest",
        timeout_seconds=600,
        memory_limit_mb=8192,
        cpu_limit=4.0,
        network_enabled=False,
        gpu_enabled=True,
        gpu_count=1,
    ),
    "heavy": ExecutionProfile(
        name="heavy",
        description="Heavy workload — large repos, full test suites",
        image="sdc-sandbox:latest",
        timeout_seconds=900,
        memory_limit_mb=16384,
        cpu_limit=8.0,
        network_enabled=False,
        gpu_enabled=False,
    ),
    "networked": ExecutionProfile(
        name="networked",
        description="Network-enabled — for tests that need external access",
        image="sdc-sandbox:latest",
        timeout_seconds=300,
        memory_limit_mb=4096,
        cpu_limit=2.0,
        network_enabled=True,
        gpu_enabled=False,
    ),
}


# ── Profile manager ─────────────────────────────────────────────────────


class ExecutionProfileManager:
    """Manage execution profiles with custom overrides on top of presets."""

    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or Path.home() / ".scholardevclaw" / "execution"
        self._custom_profiles: dict[str, ExecutionProfile] = {}
        self._active_profile: str = "local"
        self._load()

    @property
    def active_name(self) -> str:
        return self._active_profile

    def get_active(self) -> ExecutionProfile:
        return self.get(self._active_profile) or PRESET_PROFILES["local"]

    def get(self, name: str) -> ExecutionProfile | None:
        if name in self._custom_profiles:
            return self._custom_profiles[name]
        return PRESET_PROFILES.get(name)

    def list_profiles(self, include_presets: bool = True) -> list[ExecutionProfile]:
        profiles: list[ExecutionProfile] = []
        if include_presets:
            profiles.extend(PRESET_PROFILES.values())
        profiles.extend(self._custom_profiles.values())
        return profiles

    def save(self, profile: ExecutionProfile) -> None:
        self._custom_profiles[profile.name] = profile
        self._persist()

    def delete(self, name: str) -> bool:
        if name in PRESET_PROFILES:
            return False  # Cannot delete presets
        if name in self._custom_profiles:
            del self._custom_profiles[name]
            self._persist()
            return True
        return False

    def set_active(self, name: str) -> bool:
        if self.get(name) is not None:
            self._active_profile = name
            self._persist()
            return True
        return False

    def _config_file(self) -> Path:
        return self.config_dir / "profiles.json"

    def _persist(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "active_profile": self._active_profile,
            "custom_profiles": {k: v.to_dict() for k, v in self._custom_profiles.items()},
        }
        self._config_file().write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        config_file = self._config_file()
        if not config_file.exists():
            return
        try:
            data = json.loads(config_file.read_text())
            self._active_profile = data.get("active_profile", "local")
            for name, profile_data in data.get("custom_profiles", {}).items():
                self._custom_profiles[name] = ExecutionProfile.from_dict(profile_data)
        except (json.JSONDecodeError, TypeError):
            pass
