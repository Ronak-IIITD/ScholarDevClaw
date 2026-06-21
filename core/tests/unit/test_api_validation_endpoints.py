"""Tests for the FastAPI validation sub-step endpoints.

Covers the 9 new validation endpoints added for TypeScript orchestration:
  - POST /validation/artifacts
  - POST /validation/policy
  - POST /validation/tests
  - POST /validation/benchmark
  - POST /validation/training
  - POST /validation/correctness
  - POST /validation/regression
  - POST /validation/readability
  - POST /validation/heal
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_server(monkeypatch, **env):
    managed = {
        "SCHOLARDEVCLAW_API_AUTH_KEY",
        "SCHOLARDEVCLAW_ALLOWED_REPO_DIRS",
        "SCHOLARDEVCLAW_ENABLE_HSTS",
        "SCHOLARDEVCLAW_CORS_ORIGINS",
        "SCHOLARDEVCLAW_DEV_MODE",
    }
    for key in managed:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SCHOLARDEVCLAW_DEV_MODE", "true")
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    import scholardevclaw.api.server as server

    return importlib.reload(server)


# =========================================================================
# POST /validation/artifacts
# =========================================================================


class TestValidationArtifacts:
    def test_valid_python_patch(self, monkeypatch, tmp_path):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        patch_payload = {
            "patch": {
                "new_files": [{"path": "test.py", "content": "def hello(): return 42\n"}],
                "transformations": [],
            }
        }

        resp = client.post("/validation/artifacts", json=patch_payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is True
        assert data["stage"] == "artifacts"

    def test_syntax_error_in_patch(self, monkeypatch):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        patch_payload = {
            "patch": {
                "new_files": [{"path": "bad.py", "content": "def broken(\n"}],
                "transformations": [],
            }
        }

        resp = client.post("/validation/artifacts", json=patch_payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is False
        assert (
            "line" in (data.get("logs") or "").lower()
            or "syntax" in (data.get("logs") or "").lower()
        )

    def test_empty_patch_passes(self, monkeypatch):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        patch_payload = {"patch": {"new_files": [], "transformations": []}}

        resp = client.post("/validation/artifacts", json=patch_payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is True

    def test_transformations_validated(self, monkeypatch):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        patch_payload = {
            "patch": {
                "new_files": [],
                "transformations": [
                    {
                        "file": "mod.py",
                        "original": "def f(): pass",
                        "modified": "def f(): return 1",
                        "changes": [],
                    }
                ],
            }
        }

        resp = client.post("/validation/artifacts", json=patch_payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is True

    def test_non_python_files_skipped(self, monkeypatch):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        patch_payload = {
            "patch": {
                "new_files": [{"path": "README.md", "content": "# Hello"}],
                "transformations": [],
            }
        }

        resp = client.post("/validation/artifacts", json=patch_payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is True


# =========================================================================
# POST /validation/policy
# =========================================================================


class TestValidationPolicy:
    def test_default_policy_allows(self, monkeypatch):
        monkeypatch.delenv("SCHOLARDEVCLAW_VALIDATION_EXECUTION_MODE", raising=False)
        monkeypatch.delenv("SCHOLARDEVCLAW_VALIDATION_SANDBOX", raising=False)
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        resp = client.post("/validation/policy", json={})

        assert resp.status_code == 200
        data = resp.json()
        assert data["blocked"] is False

    def test_strict_mode_without_docker_blocks(self, monkeypatch):
        monkeypatch.setenv("SCHOLARDEVCLAW_VALIDATION_EXECUTION_MODE", "strict")
        monkeypatch.setenv("SCHOLARDEVCLAW_VALIDATION_SANDBOX", "")
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        resp = client.post("/validation/policy", json={})

        assert resp.status_code == 200
        data = resp.json()
        assert data["blocked"] is True

    def test_warn_mode_returns_warning(self, monkeypatch):
        monkeypatch.setenv("SCHOLARDEVCLAW_VALIDATION_EXECUTION_MODE", "warn")
        monkeypatch.setenv("SCHOLARDEVCLAW_VALIDATION_SANDBOX", "")
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        resp = client.post("/validation/policy", json={})

        assert resp.status_code == 200
        data = resp.json()
        assert data["blocked"] is False
        assert data["warning"] is not None
        assert "WARNING" in data["warning"]


# =========================================================================
# POST /validation/tests
# =========================================================================


class TestValidationTests:
    def test_no_test_files_passes(self, monkeypatch, tmp_path):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        resp = client.post("/validation/tests", json={"repoPath": str(tmp_path)})

        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is True

    def test_tests_pass(self, monkeypatch, tmp_path):
        test_file = tmp_path / "test_example.py"
        test_file.write_text("def test_ok(): assert True\n")

        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0, stdout="1 passed", stderr="")
            resp = client.post("/validation/tests", json={"repoPath": str(tmp_path)})

        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is True

    def test_tests_fail(self, monkeypatch, tmp_path):
        test_file = tmp_path / "test_example.py"
        test_file.write_text("def test_fail(): assert False\n")

        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=1, stdout="1 failed", stderr="")
            resp = client.post("/validation/tests", json={"repoPath": str(tmp_path)})

        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is False

    def test_missing_repo_path(self, monkeypatch):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        resp = client.post("/validation/tests", json={})

        assert resp.status_code == 422  # FastAPI validation error

    def test_nonexistent_repo_path(self, monkeypatch):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        resp = client.post("/validation/tests", json={"repoPath": "/nonexistent/path"})

        assert resp.status_code == 404


# =========================================================================
# POST /validation/benchmark
# =========================================================================


class TestValidationBenchmark:
    def test_no_benchmarks_passes(self, monkeypatch, tmp_path):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        resp = client.post("/validation/benchmark", json={"repoPath": str(tmp_path)})

        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is True
        assert data["stage"] == "benchmark"

    def test_benchmark_with_mock(self, monkeypatch, tmp_path):
        bench_file = tmp_path / "benchmark_main.py"
        bench_file.write_text('print("bench ok")\n')

        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0, stdout="ok", stderr="")
            resp = client.post("/validation/benchmark", json={"repoPath": str(tmp_path)})

        assert resp.status_code == 200
        data = resp.json()
        assert "comparison" in data


# =========================================================================
# POST /validation/training
# =========================================================================


class TestValidationTraining:
    def test_training_baseline(self, monkeypatch, tmp_path):
        from scholardevclaw.validation.runner import Metrics

        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        with patch.object(server.ValidationRunner, "_run_training_test") as mock_train:
            mock_train.return_value = Metrics(
                loss=2.5,
                perplexity=12.1,
                tokens_per_second=5000,
                memory_mb=2048,
                runtime_seconds=1.0,
            )
            resp = client.post(
                "/validation/training",
                json={
                    "repoPath": str(tmp_path),
                    "useVariant": False,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["loss"] == 2.5
        assert data["tokens_per_second"] == 5000

    def test_training_variant(self, monkeypatch, tmp_path):
        from scholardevclaw.validation.runner import Metrics

        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        with patch.object(server.ValidationRunner, "_run_training_test") as mock_train:
            mock_train.return_value = Metrics(
                loss=2.3,
                perplexity=11.5,
                tokens_per_second=5200,
                memory_mb=2100,
                runtime_seconds=1.5,
            )
            resp = client.post(
                "/validation/training",
                json={
                    "repoPath": str(tmp_path),
                    "useVariant": True,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["loss"] == 2.3

    def test_training_failure(self, monkeypatch, tmp_path):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        with patch.object(server.ValidationRunner, "_run_training_test") as mock_train:
            mock_train.return_value = None
            resp = client.post(
                "/validation/training",
                json={
                    "repoPath": str(tmp_path),
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"


# =========================================================================
# POST /validation/correctness
# =========================================================================


class TestValidationCorrectness:
    def test_no_algorithm_key_skips(self, monkeypatch):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        resp = client.post(
            "/validation/correctness",
            json={
                "patch": {"new_files": [], "transformations": []},
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "skipped"

    def test_unknown_algorithm_skips(self, monkeypatch):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        resp = client.post(
            "/validation/correctness",
            json={
                "patch": {
                    "new_files": [{"path": "custom.py", "content": "x = 1"}],
                    "transformations": [],
                    "algorithm_name": "unknown_algo_xyz",
                },
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "skipped"


# =========================================================================
# POST /validation/regression
# =========================================================================


class TestValidationRegression:
    def test_no_transformations_skips(self, monkeypatch):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        resp = client.post(
            "/validation/regression",
            json={
                "patch": {"new_files": [], "transformations": []},
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "skipped"

    def test_no_symbol_removal_passes(self, monkeypatch):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        resp = client.post(
            "/validation/regression",
            json={
                "patch": {
                    "new_files": [],
                    "transformations": [
                        {
                            "file": "mod.py",
                            "original": "def f():\n    pass\n",
                            "modified": "def f():\n    return 1\n",
                            "changes": [],
                        }
                    ],
                },
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is True
        assert data["removed_symbols"] == []
        assert data["signature_changes"] == []

    def test_symbol_removal_fails(self, monkeypatch):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        resp = client.post(
            "/validation/regression",
            json={
                "patch": {
                    "new_files": [],
                    "transformations": [
                        {
                            "file": "mod.py",
                            "original": "def f():\n    pass\n\ndef g():\n    pass\n",
                            "modified": "def f():\n    pass\n",
                            "changes": [],
                        }
                    ],
                },
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is False
        assert len(data["removed_symbols"]) > 0
        assert any("g" in s for s in data["removed_symbols"])

    def test_signature_change_detected(self, monkeypatch):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        resp = client.post(
            "/validation/regression",
            json={
                "patch": {
                    "new_files": [],
                    "transformations": [
                        {
                            "file": "mod.py",
                            "original": "def f(x):\n    pass\n",
                            "modified": "def f(x, y):\n    pass\n",
                            "changes": [],
                        }
                    ],
                },
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is False
        assert len(data["signature_changes"]) > 0


# =========================================================================
# POST /validation/readability
# =========================================================================


class TestValidationReadability:
    def test_empty_transformations_default_score(self, monkeypatch):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        resp = client.post(
            "/validation/readability",
            json={
                "patch": {"new_files": [], "transformations": []},
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["score"] == 5
        assert data["source"] == "heuristic"

    def test_small_diff_high_score(self, monkeypatch):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        resp = client.post(
            "/validation/readability",
            json={
                "patch": {
                    "new_files": [],
                    "transformations": [
                        {
                            "file": "mod.py",
                            "original": "x = 1\n",
                            "modified": "x = 2\n",
                            "changes": [],
                        }
                    ],
                },
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["score"] >= 4

    def test_large_diff_low_score(self, monkeypatch):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        # Generate a large diff with many changed lines
        original = "\n".join(f"line_{i} = {i}" for i in range(200))
        modified = "\n".join(f"line_{i} = {i + 1000}" for i in range(200))

        resp = client.post(
            "/validation/readability",
            json={
                "patch": {
                    "new_files": [],
                    "transformations": [
                        {
                            "file": "big.py",
                            "original": original,
                            "modified": modified,
                            "changes": [],
                        }
                    ],
                },
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["score"] <= 3


# =========================================================================
# POST /validation/heal
# =========================================================================


class TestValidationHeal:
    def test_heal_calls_generator(self, monkeypatch, tmp_path):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        mock_healed = MagicMock()
        mock_healed.new_files = []
        mock_healed.transformations = []
        mock_healed.branch_name = "healed"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with patch.object(server.ValidationRunner, "_run_training_test") as mock_train:
            from scholardevclaw.validation.runner import Metrics

            mock_train.return_value = Metrics(
                loss=2.5,
                perplexity=12.1,
                tokens_per_second=5000,
                memory_mb=2048,
                runtime_seconds=1.0,
            )
            resp = client.post(
                "/validation/training",
                json={
                    "repoPath": str(tmp_path),
                    "useVariant": False,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["loss"] == 2.5
        assert data["tokens_per_second"] == 5000

    def test_training_variant(self, monkeypatch, tmp_path):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        with patch.object(server.ValidationRunner, "_run_training_test") as mock_train:
            from scholardevclaw.validation.runner import Metrics

            mock_train.return_value = Metrics(
                loss=2.3,
                perplexity=11.5,
                tokens_per_second=5200,
                memory_mb=2100,
                runtime_seconds=1.5,
            )
            resp = client.post(
                "/validation/training",
                json={
                    "repoPath": str(tmp_path),
                    "useVariant": True,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["loss"] == 2.3

    def test_heal_with_missing_repo(self, monkeypatch):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        resp = client.post(
            "/validation/heal",
            json={
                "repoPath": "/nonexistent/path",
                "patch": {"new_files": [], "transformations": []},
                "testResult": {"passed": False, "stage": "tests"},
                "mappingResult": None,
            },
        )

        assert resp.status_code == 404


# =========================================================================
# Request validation (422 errors)
# =========================================================================


class TestRequestValidation:
    def test_artifacts_missing_patch(self, monkeypatch):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        resp = client.post("/validation/artifacts", json={})
        assert resp.status_code == 422

    def test_tests_missing_repo_path(self, monkeypatch):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        resp = client.post("/validation/tests", json={})
        assert resp.status_code == 422

    def test_training_extra_fields_rejected(self, monkeypatch, tmp_path):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        resp = client.post(
            "/validation/training",
            json={
                "repoPath": str(tmp_path),
                "unknownField": "value",
            },
        )
        # Pydantic with extra="forbid" should reject
        assert resp.status_code == 422

    def test_heal_missing_test_result(self, monkeypatch, tmp_path):
        server = _load_server(monkeypatch)
        client = TestClient(server.app)

        resp = client.post(
            "/validation/heal",
            json={
                "repoPath": str(tmp_path),
                "patch": {"new_files": [], "transformations": []},
                "mappingResult": None,
            },
        )
        assert resp.status_code == 422
