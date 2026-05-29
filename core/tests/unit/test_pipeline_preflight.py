"""Tests for pipeline preflight, dependency checks, and framework detection."""

import subprocess
from pathlib import Path
from unittest.mock import patch

from scholardevclaw.application.pipeline import (
    _check_dependencies,
    _detect_frameworks,
    run_preflight,
)

# =========================================================================
# _check_dependencies
# =========================================================================


class TestCheckDependencies:
    def test_returns_dict(self):
        result = _check_dependencies()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_all_values_are_bool(self):
        result = _check_dependencies()
        for val in result.values():
            assert isinstance(val, bool)

    def test_includes_critical_packages(self):
        result = _check_dependencies()
        assert "numpy" in result
        assert "httpx" in result

    def test_present_packages_are_true(self):
        result = _check_dependencies()
        assert result.get("numpy") is True
        assert result.get("httpx") is True


# =========================================================================
# _detect_frameworks
# =========================================================================


class TestDetectFrameworks:
    def test_detects_pytorch(self, tmp_path: Path):
        (tmp_path / "model.py").write_text("import torch\nx = torch.tensor(1)\n")
        frameworks = _detect_frameworks(tmp_path)
        assert "pytorch" in frameworks

    def test_detects_transformers(self, tmp_path: Path):
        (tmp_path / "model.py").write_text("from transformers import BertModel\n")
        frameworks = _detect_frameworks(tmp_path)
        assert "transformers" in frameworks

    def test_detects_tensorflow(self, tmp_path: Path):
        (tmp_path / "model.py").write_text("import tensorflow as tf\n")
        frameworks = _detect_frameworks(tmp_path)
        assert "tensorflow" in frameworks

    def test_detects_jax(self, tmp_path: Path):
        (tmp_path / "model.py").write_text("import jax.numpy as jnp\n")
        frameworks = _detect_frameworks(tmp_path)
        assert "jax" in frameworks

    def test_empty_repo(self, tmp_path: Path):
        frameworks = _detect_frameworks(tmp_path)
        assert frameworks == []

    def test_no_frameworks(self, tmp_path: Path):
        (tmp_path / "utils.py").write_text("x = 1\ny = 2\n")
        frameworks = _detect_frameworks(tmp_path)
        assert frameworks == []

    def test_multiple_frameworks(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("import torch\nimport tensorflow\n")
        frameworks = _detect_frameworks(tmp_path)
        assert "pytorch" in frameworks
        assert "tensorflow" in frameworks

    def test_syntax_error_file_no_crash(self, tmp_path: Path):
        (tmp_path / "bad.py").write_text("this is not valid {{{\n")
        frameworks = _detect_frameworks(tmp_path)
        assert isinstance(frameworks, list)

    def test_subdirectory_files(self, tmp_path: Path):
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "model.py").write_text("import torch\n")
        frameworks = _detect_frameworks(tmp_path)
        assert "pytorch" in frameworks


# =========================================================================
# run_preflight
# =========================================================================


class TestRunPreflight:
    def test_empty_path_returns_error(self):
        result = run_preflight("/nonexistent/repo/path")
        assert result.ok is False
        assert result.error is not None

    def test_existing_repo_ok(self, tmp_path: Path):
        (tmp_path / "model.py").write_text("x = 1\n")
        result = run_preflight(str(tmp_path))
        assert result.ok is True
        assert result.payload["repo_exists"] is True

    def test_clean_repo(self, tmp_path: Path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=False, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=str(tmp_path),
            check=False,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=str(tmp_path),
            check=False,
            capture_output=True,
        )
        (tmp_path / "model.py").write_text("x = 1\n")
        subprocess.run(
            ["git", "add", "."],
            cwd=str(tmp_path),
            check=False,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=str(tmp_path),
            check=False,
            capture_output=True,
        )
        result = run_preflight(str(tmp_path))
        assert result.ok is True
        assert result.payload["is_clean"] is True

    def test_dirty_repo(self, tmp_path: Path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=False, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=str(tmp_path),
            check=False,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=str(tmp_path),
            check=False,
            capture_output=True,
        )
        (tmp_path / "model.py").write_text("x = 1\n")
        # Untracked file makes it dirty
        result = run_preflight(str(tmp_path))
        assert result.payload["is_clean"] is False

    def test_no_git_dir(self, tmp_path: Path):
        (tmp_path / "model.py").write_text("x = 1\n")
        result = run_preflight(str(tmp_path))
        assert result.ok is True
        assert result.payload["has_git_dir"] is False

    def test_no_python_files(self, tmp_path: Path):
        (tmp_path / "readme.md").write_text("# Hello\n")
        result = run_preflight(str(tmp_path))
        assert result.ok is True
        assert result.payload["python_file_count"] == 0
        assert any("No Python files" in w for w in result.payload["warnings"])

    def test_require_clean_no_git(self, tmp_path: Path):
        (tmp_path / "model.py").write_text("x = 1\n")
        result = run_preflight(str(tmp_path), require_clean=True)
        assert result.ok is False
        assert result.error is not None and "require_clean" in result.error

    def test_python_file_count(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")
        result = run_preflight(str(tmp_path))
        assert result.payload["python_file_count"] == 2

    def test_dependency_status_in_payload(self, tmp_path: Path):
        (tmp_path / "model.py").write_text("x = 1\n")
        result = run_preflight(str(tmp_path))
        assert "dependency_status" in result.payload
        assert isinstance(result.payload["dependency_status"], dict)

    def test_detected_frameworks_in_payload(self, tmp_path: Path):
        (tmp_path / "model.py").write_text("import torch\n")
        result = run_preflight(str(tmp_path))
        assert "detected_frameworks" in result.payload
        assert "pytorch" in result.payload["detected_frameworks"]

    def test_no_frameworks_warning(self, tmp_path: Path):
        (tmp_path / "utils.py").write_text("x = 1\n")
        result = run_preflight(str(tmp_path))
        assert any("No ML frameworks" in w for w in result.payload["warnings"])

    def test_frameworks_no_warning(self, tmp_path: Path):
        (tmp_path / "model.py").write_text("import torch\n")
        result = run_preflight(str(tmp_path))
        assert not any("No ML frameworks" in w for w in result.payload["warnings"])

    def test_missing_deps_warning(self, tmp_path: Path):
        (tmp_path / "model.py").write_text("x = 1\n")
        with patch(
            "scholardevclaw.application.pipeline._check_dependencies",
            return_value={"torch": True, "numpy": False},
        ):
            result = run_preflight(str(tmp_path))
            assert any("Missing runtime" in w for w in result.payload["warnings"])
            assert any("numpy" in r for r in result.payload["recommendations"])

    def test_all_deps_ok_no_warning(self, tmp_path: Path):
        (tmp_path / "model.py").write_text("x = 1\n")
        with patch(
            "scholardevclaw.application.pipeline._check_dependencies",
            return_value={"torch": True, "numpy": True},
        ):
            result = run_preflight(str(tmp_path))
            assert not any("Missing runtime" in w for w in result.payload["warnings"])

    def test_payload_has_all_expected_keys(self, tmp_path: Path):
        (tmp_path / "model.py").write_text("x = 1\n")
        result = run_preflight(str(tmp_path))
        expected_keys = {
            "repo_exists",
            "repo_is_writable",
            "python_file_count",
            "has_git_dir",
            "git_available",
            "is_clean",
            "changed_file_entries",
            "git_error",
            "dependency_status",
            "detected_frameworks",
            "warnings",
            "recommendations",
        }
        assert expected_keys <= set(result.payload.keys())

    def test_log_callback_receives_messages(self, tmp_path: Path):
        (tmp_path / "model.py").write_text("x = 1\n")
        messages = []
        run_preflight(str(tmp_path), log_callback=messages.append)
        assert len(messages) > 0
        assert any("preflight" in m.lower() for m in messages)
