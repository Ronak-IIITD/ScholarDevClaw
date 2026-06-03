"""Tests for pipeline analysis caching optimization."""

from __future__ import annotations

import importlib
import sys
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
            self.root_path = "."
            self.languages = ["python"]
            self.elements = [SimpleNamespace(name="x")]
            self.frameworks = ["pytorch"]
            self.entry_points = ["train.py"]
            self.patterns = {"normalization": ["model.py:10"]}
            self.architecture = {
                "models": [{"file": "model.py", "components": {"normalization": "LayerNorm"}}]
            }
            self.language_stats = [SimpleNamespace(language="python", file_count=5, line_count=100)]
            self.test_files = ["test_model.py"]

    class FakeAnalyzer:
        def __init__(self, repo_path: Path):
            self.repo_path = repo_path
            self._analyze_called = False

        def analyze(self):
            self._analyze_called = True
            return FakeAnalysis()

        def suggest_research_papers(self):
            return [{"paper": {"name": "rmsnorm"}, "confidence": 92.0}]

    module.TreeSitterAnalyzer = FakeAnalyzer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def _install_fake_extractor(monkeypatch):
    module = ModuleType("scholardevclaw.research_intelligence.extractor")

    class FakeExtractor:
        def __init__(self, llm_assistant=None):
            pass

        def get_spec(self, name: str):
            return {
                "paper": {"title": "RMSNorm"},
                "algorithm": {"name": "RMSNorm"},
                "changes": {
                    "target_patterns": ["LayerNorm"],
                    "replacement": "RMSNorm",
                },
            }

    module.ResearchExtractor = FakeExtractor
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _install_fake_mapping(monkeypatch):
    module = ModuleType("scholardevclaw.mapping.engine")

    class FakeMappingEngine:
        def __init__(self, repo_analysis, research_spec, llm_assistant=None):
            self.research_spec = research_spec

        def map(self):
            return SimpleNamespace(
                targets=[
                    SimpleNamespace(
                        file="model.py",
                        line=10,
                        current_code="LayerNorm",
                        replacement_required="RMSNorm",
                        context="model.py:10",
                    )
                ],
                strategy="direct_replacement",
                confidence=85.0,
                research_spec=self.research_spec,
            )

    module.MappingEngine = FakeMappingEngine
    monkeypatch.setitem(sys.modules, module.__name__, module)


class TestBuildMappingResultAnalysisCaching:
    """Tests for _build_mapping_result analysis parameter."""

    def test_creates_analyzer_when_analysis_none(self, monkeypatch, tmp_path):
        """When analysis=None, TreeSitterAnalyzer is created and analyze() called."""
        ts_mod = _install_fake_tree_sitter(monkeypatch)
        _install_fake_extractor(monkeypatch)
        _install_fake_mapping(monkeypatch)

        pipeline = _pipeline_module()

        # Track whether TreeSitterAnalyzer was instantiated
        init_calls = []
        orig_init = ts_mod.TreeSitterAnalyzer.__init__

        def tracking_init(self, repo_path):
            init_calls.append(repo_path)
            orig_init(self, repo_path)

        monkeypatch.setattr(ts_mod.TreeSitterAnalyzer, "__init__", tracking_init)

        mapping_result, spec = pipeline._build_mapping_result(
            tmp_path, "rmsnorm", llm_assistant=None
        )

        assert len(init_calls) == 1
        assert spec["algorithm"]["name"] == "RMSNorm"

    def test_skips_analyzer_when_analysis_provided(self, monkeypatch, tmp_path):
        """When analysis dict is provided, TreeSitterAnalyzer is NOT created."""
        ts_mod = _install_fake_tree_sitter(monkeypatch)
        _install_fake_extractor(monkeypatch)
        _install_fake_mapping(monkeypatch)

        pipeline = _pipeline_module()

        init_calls = []
        orig_init = ts_mod.TreeSitterAnalyzer.__init__

        def tracking_init(self, repo_path):
            init_calls.append(repo_path)
            orig_init(self, repo_path)

        monkeypatch.setattr(ts_mod.TreeSitterAnalyzer, "__init__", tracking_init)

        # Provide pre-computed analysis
        analysis_dict = {
            "root_path": str(tmp_path),
            "languages": ["python"],
            "elements": [SimpleNamespace(name="x")],
            "frameworks": ["pytorch"],
            "entry_points": ["train.py"],
            "patterns": {},
            "architecture": {"models": []},
            "language_stats": [SimpleNamespace(language="python", file_count=1, line_count=10)],
            "test_files": [],
        }

        mapping_result, spec = pipeline._build_mapping_result(
            tmp_path, "rmsnorm", llm_assistant=None, analysis=analysis_dict
        )

        # TreeSitterAnalyzer should NOT have been created
        assert len(init_calls) == 0
        assert spec["algorithm"]["name"] == "RMSNorm"


class TestRunGenerateAnalysisParameter:
    """Tests for run_generate analysis passthrough."""

    def test_run_generate_accepts_analysis(self, monkeypatch, tmp_path):
        """run_generate accepts analysis kwarg without error."""
        _install_fake_tree_sitter(monkeypatch)
        _install_fake_extractor(monkeypatch)
        _install_fake_mapping(monkeypatch)

        pipeline = _pipeline_module()
        analysis = {
            "root_path": str(tmp_path),
            "languages": ["python"],
            "elements": [],
            "frameworks": [],
            "entry_points": [],
            "patterns": {},
            "architecture": {"models": []},
            "language_stats": [],
            "test_files": [],
        }
        # Should not raise
        result = pipeline.run_generate(str(tmp_path), "rmsnorm", analysis=analysis)
        assert result.ok is True

    def test_run_generate_works_without_analysis(self, monkeypatch, tmp_path):
        """run_generate without analysis still works (backward compat)."""
        _install_fake_tree_sitter(monkeypatch)
        _install_fake_extractor(monkeypatch)
        _install_fake_mapping(monkeypatch)

        pipeline = _pipeline_module()
        result = pipeline.run_generate(str(tmp_path), "rmsnorm")
        assert result.ok is True
