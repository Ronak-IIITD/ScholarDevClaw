"""Tests for execution profiles and profile-aware SandboxRunner."""

from pathlib import Path
from unittest.mock import patch

from scholardevclaw.execution.profiles import (
    PRESET_PROFILES,
    ExecutionProfile,
    ExecutionProfileManager,
)
from scholardevclaw.execution.sandbox import SandboxRunner

# =========================================================================
# ExecutionProfile
# =========================================================================


class TestExecutionProfile:
    def test_construction_defaults(self):
        p = ExecutionProfile(name="test")
        assert p.name == "test"
        assert p.image == "sdc-sandbox:latest"
        assert p.timeout_seconds == 300
        assert p.memory_limit_mb == 4096
        assert p.cpu_limit == 2.0
        assert p.network_enabled is False
        assert p.gpu_enabled is False

    def test_to_dict_roundtrip(self):
        p = ExecutionProfile(name="custom", gpu_enabled=True, gpu_count=2, timeout_seconds=600)
        data = p.to_dict()
        restored = ExecutionProfile.from_dict(data)
        assert restored.name == "custom"
        assert restored.gpu_enabled is True
        assert restored.gpu_count == 2
        assert restored.timeout_seconds == 600

    def test_from_dict_ignores_unknown_keys(self):
        data = {"name": "x", "unknown_key": 42}
        p = ExecutionProfile.from_dict(data)
        assert p.name == "x"

    def test_to_dict_contains_all_fields(self):
        p = ExecutionProfile(name="full")
        d = p.to_dict()
        assert "name" in d
        assert "image" in d
        assert "timeout_seconds" in d
        assert "memory_limit_mb" in d
        assert "cpu_limit" in d
        assert "network_enabled" in d
        assert "gpu_enabled" in d
        assert "gpu_count" in d
        assert "extra_volumes" in d
        assert "environment" in d
        assert "pytest_args" in d


# =========================================================================
# PRESET_PROFILES
# =========================================================================


class TestPresetProfiles:
    def test_all_presets_present(self):
        expected = {"local", "cloud-cpu", "cloud-gpu", "heavy", "networked"}
        assert set(PRESET_PROFILES.keys()) == expected

    def test_local_is_minimal(self):
        local = PRESET_PROFILES["local"]
        assert local.memory_limit_mb == 2048
        assert local.timeout_seconds == 120
        assert local.gpu_enabled is False

    def test_cloud_gpu_has_gpu(self):
        gpu = PRESET_PROFILES["cloud-gpu"]
        assert gpu.gpu_enabled is True
        assert gpu.gpu_count == 1

    def test_networked_has_network(self):
        net = PRESET_PROFILES["networked"]
        assert net.network_enabled is True

    def test_heaviest_is_heavy(self):
        heavy = PRESET_PROFILES["heavy"]
        assert heavy.memory_limit_mb >= 16384
        assert heavy.cpu_limit >= 8.0

    def test_presets_are_distinct(self):
        names = [p.name for p in PRESET_PROFILES.values()]
        assert len(names) == len(set(names))


# =========================================================================
# ExecutionProfileManager
# =========================================================================


class TestExecutionProfileManager:
    def test_default_active_is_local(self, tmp_path: Path):
        mgr = ExecutionProfileManager(config_dir=tmp_path)
        assert mgr.active_name == "local"

    def test_get_preset(self, tmp_path: Path):
        mgr = ExecutionProfileManager(config_dir=tmp_path)
        profile = mgr.get("local")
        assert profile is not None
        assert profile.name == "local"

    def test_get_unknown_returns_none(self, tmp_path: Path):
        mgr = ExecutionProfileManager(config_dir=tmp_path)
        assert mgr.get("nonexistent") is None

    def test_save_and_get_custom(self, tmp_path: Path):
        mgr = ExecutionProfileManager(config_dir=tmp_path)
        custom = ExecutionProfile(name="my-custom", memory_limit_mb=1024)
        mgr.save(custom)
        loaded = mgr.get("my-custom")
        assert loaded is not None
        assert loaded.memory_limit_mb == 1024

    def test_list_includes_presets_and_custom(self, tmp_path: Path):
        mgr = ExecutionProfileManager(config_dir=tmp_path)
        mgr.save(ExecutionProfile(name="custom1"))
        profiles = mgr.list_profiles()
        assert len(profiles) >= len(PRESET_PROFILES) + 1
        names = [p.name for p in profiles]
        assert "local" in names
        assert "custom1" in names

    def test_list_presets_only(self, tmp_path: Path):
        mgr = ExecutionProfileManager(config_dir=tmp_path)
        mgr.save(ExecutionProfile(name="custom1"))
        profiles = mgr.list_profiles(include_presets=False)
        names = [p.name for p in profiles]
        assert "local" not in names
        assert "custom1" in names

    def test_delete_custom(self, tmp_path: Path):
        mgr = ExecutionProfileManager(config_dir=tmp_path)
        mgr.save(ExecutionProfile(name="to-delete"))
        assert mgr.delete("to-delete") is True
        assert mgr.get("to-delete") is None

    def test_cannot_delete_preset(self, tmp_path: Path):
        mgr = ExecutionProfileManager(config_dir=tmp_path)
        assert mgr.delete("local") is False
        assert mgr.get("local") is not None

    def test_delete_nonexistent(self, tmp_path: Path):
        mgr = ExecutionProfileManager(config_dir=tmp_path)
        assert mgr.delete("ghost") is False

    def test_set_active(self, tmp_path: Path):
        mgr = ExecutionProfileManager(config_dir=tmp_path)
        assert mgr.set_active("cloud-cpu") is True
        assert mgr.active_name == "cloud-cpu"

    def test_set_active_unknown_fails(self, tmp_path: Path):
        mgr = ExecutionProfileManager(config_dir=tmp_path)
        assert mgr.set_active("nonexistent") is False
        assert mgr.active_name == "local"

    def test_persistence(self, tmp_path: Path):
        mgr1 = ExecutionProfileManager(config_dir=tmp_path)
        mgr1.save(ExecutionProfile(name="persist-me", timeout_seconds=999))
        mgr1.set_active("cloud-gpu")

        mgr2 = ExecutionProfileManager(config_dir=tmp_path)
        assert mgr2.active_name == "cloud-gpu"
        loaded = mgr2.get("persist-me")
        assert loaded is not None
        assert loaded.timeout_seconds == 999

    def test_corrupt_config_file(self, tmp_path: Path):
        config_dir = tmp_path / "exec"
        config_dir.mkdir()
        (config_dir / "profiles.json").write_text("NOT JSON {{{")
        mgr = ExecutionProfileManager(config_dir=config_dir)
        # Should fall back to defaults without crashing
        assert mgr.active_name == "local"

    def test_empty_config_dir(self, tmp_path: Path):
        mgr = ExecutionProfileManager(config_dir=tmp_path / "nonexistent")
        assert mgr.active_name == "local"
        assert mgr.list_profiles(include_presets=False) == []


# =========================================================================
# SandboxRunner with profiles
# =========================================================================


class TestSandboxRunnerProfiles:
    def test_runner_from_profile(self, tmp_path: Path):
        profile = ExecutionProfile(
            name="test",
            image="my-image:latest",
            timeout_seconds=123,
            memory_limit_mb=512,
            cpu_limit=1.5,
            network_enabled=True,
            gpu_enabled=True,
            gpu_count=1,
        )
        runner = SandboxRunner(profile=profile)
        assert runner.image_name == "my-image:latest"
        assert runner.timeout == 123
        assert runner.memory_limit == 512
        assert runner.cpu_limit == 1.5
        assert runner.network_enabled is True
        assert runner.gpu_enabled is True
        assert runner.gpu_count == 1

    def test_runner_without_profile_uses_defaults(self):
        runner = SandboxRunner()
        assert runner.timeout == 300
        assert runner.memory_limit == 4096
        assert runner.network_enabled is False
        assert runner.gpu_enabled is False

    def test_runner_custom_pytest_args(self):
        profile = ExecutionProfile(
            name="test",
            pytest_args="tests/unit -v --json-report --json-report-file=/tmp/report.json",
        )
        runner = SandboxRunner(profile=profile)
        assert (
            runner.pytest_args == "tests/unit -v --json-report --json-report-file=/tmp/report.json"
        )

    def test_runner_from_preset(self):
        profile = PRESET_PROFILES["cloud-gpu"]
        runner = SandboxRunner(profile=profile)
        assert runner.gpu_enabled is True
        assert runner.memory_limit == 8192
        assert runner.timeout == 600

    def test_runner_failed_report_no_docker(self):
        """SandboxRunner without docker should return a failed report."""
        with patch("scholardevclaw.execution.sandbox.docker", None):
            runner = SandboxRunner()
            report = runner.run_tests(Path("/nonexistent"))
            assert report.success is False

    def test_runner_failed_report_missing_dir(self):
        """SandboxRunner with nonexistent dir should return a failed report."""
        with patch("scholardevclaw.execution.sandbox.docker", None):
            runner = SandboxRunner()
            report = runner.run_tests(Path("/definitely/does/not/exist"))
            assert report.success is False
            assert "not found" in report.stderr.lower()
