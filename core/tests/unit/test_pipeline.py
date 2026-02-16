from __future__ import annotations

import sys
import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _pipeline_module():
    return importlib.import_module("scholardevclaw.application.pipeline")


def _install_fake_tree_sitter(monkeypatch):
    module = ModuleType("scholardevclaw.repo_intelligence.tree_sitter_analyzer")

    class FakeAnalysis:
        def __init__(self):
            self.languages = ["python"]
            self.elements = [SimpleNamespace(name="x")]
            self.frameworks = ["pytorch"]
            self.entry_points = ["train.py"]
            self.patterns = {"normalization": ["model.py:10"]}
            self.architecture = {
                "models": [
                    {
                        "file": "model.py",
                        "components": {"normalization": "LayerNorm"},
                    }
                ]
            }

    class FakeAnalyzer:
        def __init__(self, repo_path: Path):
            self.repo_path = repo_path

        def analyze(self):
            return FakeAnalysis()

        def suggest_research_papers(self):
            return [{"paper": {"name": "rmsnorm"}, "confidence": 92.0}]

    module.TreeSitterAnalyzer = FakeAnalyzer
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _install_fake_extractor(monkeypatch):
    module = ModuleType("scholardevclaw.research_intelligence.extractor")

    class FakeExtractor:
        def get_spec(self, name: str):
            if name != "rmsnorm":
                return None
            return {
                "paper": {"title": "RMSNorm", "arxiv": "1910.07467"},
                "algorithm": {"name": "RMSNorm"},
                "changes": {
                    "type": "replace",
                    "target_patterns": ["LayerNorm"],
                    "replacement": "RMSNorm",
                },
            }

    module.ResearchExtractor = FakeExtractor
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _install_fake_patch_generator(monkeypatch):
    module = ModuleType("scholardevclaw.patch_generation.generator")

    class FakeGenerator:
        def __init__(self, repo_path: Path):
            self.repo_path = repo_path

        def generate(self, mapping):
            return SimpleNamespace(
                branch_name="integration/rmsnorm",
                new_files=[SimpleNamespace(path="rmsnorm.py", content="class RMSNorm: ...\n")],
                transformations=[],
            )

    module.PatchGenerator = FakeGenerator
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _install_fake_validation_runner(monkeypatch):
    module = ModuleType("scholardevclaw.validation.runner")

    class FakeRunner:
        def __init__(self, repo_path: Path):
            self.repo_path = repo_path

        def run(self, patch, repo_path: str):
            return SimpleNamespace(
                passed=True,
                stage="benchmark",
                comparison={"speedup": 1.1},
                logs="ok",
                error=None,
            )

    module.ValidationRunner = FakeRunner
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_run_map_success(monkeypatch, tmp_path):
    _install_fake_tree_sitter(monkeypatch)
    _install_fake_extractor(monkeypatch)

    result = _pipeline_module().run_map(str(tmp_path), "rmsnorm")

    assert result.ok is True
    assert result.payload["algorithm"] == "RMSNorm"
    assert result.payload["target_count"] >= 1


def test_run_generate_writes_files(monkeypatch, tmp_path):
    _install_fake_tree_sitter(monkeypatch)
    _install_fake_extractor(monkeypatch)
    _install_fake_patch_generator(monkeypatch)

    output_dir = tmp_path / "out"
    result = _pipeline_module().run_generate(str(tmp_path), "rmsnorm", output_dir=str(output_dir))

    assert result.ok is True
    assert result.payload["branch_name"] == "integration/rmsnorm"
    written = result.payload["written_files"]
    assert len(written) == 1
    assert (output_dir / "rmsnorm.py").exists()


def test_run_integrate_and_validate(monkeypatch, tmp_path):
    _install_fake_tree_sitter(monkeypatch)
    _install_fake_extractor(monkeypatch)
    _install_fake_patch_generator(monkeypatch)
    _install_fake_validation_runner(monkeypatch)

    pipeline = _pipeline_module()
    result = pipeline.run_integrate(str(tmp_path), None)

    assert result.ok is True
    assert result.payload["spec"] == "rmsnorm"
    assert result.payload["validation"]["passed"] is True
    assert result.payload["validation"]["scorecard"]["summary"] == "pass"
    assert result.payload["_meta"]["payload_type"] == "integration"
    assert result.payload["_meta"]["schema_version"]

    validation_only = pipeline.run_validate(str(tmp_path))
    assert validation_only.ok is True
    assert validation_only.payload["stage"] == "benchmark"
    assert validation_only.payload["scorecard"]["summary"] == "pass"
    assert isinstance(validation_only.payload["scorecard"].get("checks"), list)
    assert validation_only.payload["_meta"]["payload_type"] == "validation"
    assert validation_only.payload["_meta"]["schema_version"]


def test_run_preflight_require_clean_blocks_dirty_repo(monkeypatch, tmp_path):
    pipeline = _pipeline_module()
    (tmp_path / ".git").mkdir()

    def _fake_git_status(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=" M model.py\n", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", _fake_git_status)

    result = pipeline.run_preflight(str(tmp_path), require_clean=True)

    assert result.ok is False
    assert "require_clean=True" in (result.error or "")
    assert result.payload["is_clean"] is False
    assert result.payload["recommendations"]


def test_run_preflight_require_clean_blocks_non_git_repo(tmp_path):
    pipeline = _pipeline_module()

    result = pipeline.run_preflight(str(tmp_path), require_clean=True)

    assert result.ok is False
    assert "not a git checkout" in (result.error or "")
    assert result.payload["has_git_dir"] is False
    assert result.payload["recommendations"]


def test_run_integrate_dry_run_skips_generate_and_validate(monkeypatch, tmp_path):
    _install_fake_tree_sitter(monkeypatch)
    _install_fake_extractor(monkeypatch)

    pipeline = _pipeline_module()

    def _fake_mapping_result(repo_path, spec_name, *, log_callback=None):
        return {"algorithm": "RMSNorm", "targets": [{"file": "model.py"}]}, {"name": "RMSNorm"}

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("Generation/validation should not run during dry-run")

    monkeypatch.setattr(pipeline, "_build_mapping_result", _fake_mapping_result)
    monkeypatch.setattr(pipeline, "run_generate", _should_not_be_called)
    monkeypatch.setattr(pipeline, "run_validate", _should_not_be_called)

    result = pipeline.run_integrate(str(tmp_path), "rmsnorm", dry_run=True, output_dir="/tmp/out")

    assert result.ok is True
    assert result.payload["dry_run"] is True
    assert result.payload["generation"] is None
    assert result.payload["validation"] is None
    assert result.payload["output_dir"] == "/tmp/out"


def test_run_integrate_returns_preflight_guidance(monkeypatch, tmp_path):
    pipeline = _pipeline_module()

    result = pipeline.run_integrate(str(tmp_path), "rmsnorm", require_clean=True)

    assert result.ok is False
    assert result.payload["step"] == "preflight"
    assert isinstance(result.payload.get("guidance"), list)
    assert result.payload["guidance"]
    assert result.payload["_meta"]["payload_type"] == "integration"
