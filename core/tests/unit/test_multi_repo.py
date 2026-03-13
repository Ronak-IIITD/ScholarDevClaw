"""Comprehensive tests for Phase 14: Multi-Repo Support.

Covers:
  - RepoProfile dataclass (creation, serialization, deserialization)
  - RepoProfileStatus enum
  - MultiRepoManager (add/remove/list/analyze/persistence)
  - CrossRepoAnalyzer (patterns, frameworks, languages, pairwise similarity)
  - ComparisonResult / PatternOverlap / FrameworkComparison / LanguageOverlap
  - KnowledgeTransferEngine (discover, discover_for_pair, scoring)
  - TransferOpportunity / TransferPlan / TransferDirection
  - Pipeline functions (run_multi_repo_analyze, run_multi_repo_compare, run_multi_repo_transfer)
  - __init__.py exports
  - CLI multi-repo subcommand wiring
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# =========================================================================
# Helper: build fake RepoProfile
# =========================================================================


def _make_profile(
    *,
    repo_path: str = "/tmp/fake_repo",
    name: str = "fake",
    repo_id: str = "",
    languages: list[str] | None = None,
    language_stats: list[dict[str, Any]] | None = None,
    frameworks: list[str] | None = None,
    patterns: dict[str, list[str]] | None = None,
    suggestions: list[dict[str, Any]] | None = None,
    element_count: int = 10,
    status: str = "ready",
):
    from scholardevclaw.multi_repo.manager import RepoProfile, RepoProfileStatus

    return RepoProfile(
        repo_path=repo_path,
        name=name,
        repo_id=repo_id or name,
        languages=languages if languages is not None else ["python"],
        language_stats=language_stats
        if language_stats is not None
        else [{"language": "python", "file_count": 5, "line_count": 500}],
        frameworks=frameworks if frameworks is not None else ["pytorch"],
        patterns=patterns
        if patterns is not None
        else {"normalization": ["LayerNorm"], "attention": ["MultiHeadAttention"]},
        suggestions=suggestions
        if suggestions is not None
        else [
            {"spec": "rmsnorm", "name": "rmsnorm", "category": "normalization"},
            {"spec": "swiglu", "name": "swiglu", "category": "activation"},
        ],
        element_count=element_count,
        status=RepoProfileStatus(status),
    )


# =========================================================================
# RepoProfileStatus enum
# =========================================================================


class TestRepoProfileStatus:
    def test_all_statuses(self):
        from scholardevclaw.multi_repo.manager import RepoProfileStatus

        expected = {"pending", "analyzing", "ready", "error", "stale"}
        assert {s.value for s in RepoProfileStatus} == expected

    def test_str_subclass(self):
        from scholardevclaw.multi_repo.manager import RepoProfileStatus

        assert isinstance(RepoProfileStatus.READY, str)
        assert RepoProfileStatus.READY == "ready"


# =========================================================================
# RepoProfile dataclass
# =========================================================================


class TestRepoProfile:
    def test_default_construction(self):
        from scholardevclaw.multi_repo.manager import RepoProfile

        p = RepoProfile(repo_path="/tmp/test", name="test")
        assert p.name == "test"
        assert p.repo_id  # auto-generated hash
        assert len(p.repo_id) == 12

    def test_auto_name_from_path(self):
        from scholardevclaw.multi_repo.manager import RepoProfile

        p = RepoProfile(repo_path="/home/user/my_project", name="")
        assert p.name == "my_project"

    def test_repo_id_deterministic(self):
        from scholardevclaw.multi_repo.manager import RepoProfile

        p1 = RepoProfile(repo_path="/tmp/test", name="a")
        p2 = RepoProfile(repo_path="/tmp/test", name="b")
        assert p1.repo_id == p2.repo_id

    def test_to_dict(self):
        p = _make_profile(name="proj_a", repo_id="aaa111")
        d = p.to_dict()
        assert d["name"] == "proj_a"
        assert d["repo_id"] == "aaa111"
        assert d["status"] == "ready"
        assert "languages" in d
        assert "frameworks" in d
        assert "patterns" in d
        assert "suggestions" in d

    def test_from_dict_roundtrip(self):
        from scholardevclaw.multi_repo.manager import RepoProfile

        p1 = _make_profile(name="proj_a", repo_id="aaa111")
        d = p1.to_dict()
        p2 = RepoProfile.from_dict(d)
        assert p2.name == p1.name
        assert p2.repo_id == p1.repo_id
        assert p2.languages == p1.languages
        assert p2.frameworks == p1.frameworks
        assert p2.patterns == p1.patterns
        assert p2.status == p1.status

    def test_from_dict_bad_status_defaults_pending(self):
        from scholardevclaw.multi_repo.manager import RepoProfile, RepoProfileStatus

        d = {"repo_path": "/tmp/x", "status": "invalid_status"}
        p = RepoProfile.from_dict(d)
        assert p.status == RepoProfileStatus.PENDING


# =========================================================================
# MultiRepoManager
# =========================================================================


class TestMultiRepoManager:
    def test_add_repo(self, tmp_path):
        from scholardevclaw.multi_repo.manager import MultiRepoManager

        ws = tmp_path / "workspace.json"
        mgr = MultiRepoManager(workspace_path=ws)
        # Create a fake repo dir
        repo_dir = tmp_path / "repo_a"
        repo_dir.mkdir()
        profile = mgr.add_repo(str(repo_dir))
        assert profile.name == "repo_a"
        assert profile.status.value == "pending"

    def test_add_existing_repo_returns_same(self, tmp_path):
        from scholardevclaw.multi_repo.manager import MultiRepoManager

        ws = tmp_path / "workspace.json"
        mgr = MultiRepoManager(workspace_path=ws)
        repo_dir = tmp_path / "repo_a"
        repo_dir.mkdir()
        p1 = mgr.add_repo(str(repo_dir))
        p2 = mgr.add_repo(str(repo_dir))
        assert p1.repo_id == p2.repo_id

    def test_remove_repo(self, tmp_path):
        from scholardevclaw.multi_repo.manager import MultiRepoManager

        ws = tmp_path / "workspace.json"
        mgr = MultiRepoManager(workspace_path=ws)
        repo_dir = tmp_path / "repo_a"
        repo_dir.mkdir()
        p = mgr.add_repo(str(repo_dir))
        assert mgr.remove_repo(p.repo_id)
        assert mgr.list_profiles() == []

    def test_remove_nonexistent_returns_false(self, tmp_path):
        from scholardevclaw.multi_repo.manager import MultiRepoManager

        ws = tmp_path / "workspace.json"
        mgr = MultiRepoManager(workspace_path=ws)
        assert not mgr.remove_repo("nonexistent")

    def test_list_profiles(self, tmp_path):
        from scholardevclaw.multi_repo.manager import MultiRepoManager

        ws = tmp_path / "workspace.json"
        mgr = MultiRepoManager(workspace_path=ws)
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        mgr.add_repo(str(a))
        mgr.add_repo(str(b))
        assert len(mgr.list_profiles()) == 2

    def test_persistence(self, tmp_path):
        from scholardevclaw.multi_repo.manager import MultiRepoManager

        ws = tmp_path / "workspace.json"
        mgr = MultiRepoManager(workspace_path=ws)
        repo_dir = tmp_path / "repo_a"
        repo_dir.mkdir()
        mgr.add_repo(str(repo_dir))

        # Create new manager from same workspace file
        mgr2 = MultiRepoManager(workspace_path=ws)
        assert len(mgr2.list_profiles()) == 1
        assert mgr2.list_profiles()[0].name == "repo_a"

    def test_get_profile_by_id(self, tmp_path):
        from scholardevclaw.multi_repo.manager import MultiRepoManager

        ws = tmp_path / "workspace.json"
        mgr = MultiRepoManager(workspace_path=ws)
        repo_dir = tmp_path / "repo_a"
        repo_dir.mkdir()
        p = mgr.add_repo(str(repo_dir))
        found = mgr.get_profile(p.repo_id)
        assert found is not None
        assert found.repo_id == p.repo_id

    def test_get_profile_by_name(self, tmp_path):
        from scholardevclaw.multi_repo.manager import MultiRepoManager

        ws = tmp_path / "workspace.json"
        mgr = MultiRepoManager(workspace_path=ws)
        repo_dir = tmp_path / "repo_a"
        repo_dir.mkdir()
        mgr.add_repo(str(repo_dir))
        found = mgr.get_profile("repo_a")
        assert found is not None
        assert found.name == "repo_a"

    def test_get_profile_not_found(self, tmp_path):
        from scholardevclaw.multi_repo.manager import MultiRepoManager

        ws = tmp_path / "workspace.json"
        mgr = MultiRepoManager(workspace_path=ws)
        assert mgr.get_profile("nonexistent") is None

    def test_clear_workspace(self, tmp_path):
        from scholardevclaw.multi_repo.manager import MultiRepoManager

        ws = tmp_path / "workspace.json"
        mgr = MultiRepoManager(workspace_path=ws)
        repo_dir = tmp_path / "repo_a"
        repo_dir.mkdir()
        mgr.add_repo(str(repo_dir))
        assert len(mgr.list_profiles()) == 1
        mgr.clear_workspace()
        assert len(mgr.list_profiles()) == 0

    def test_get_ready_profiles(self, tmp_path):
        from scholardevclaw.multi_repo.manager import MultiRepoManager, RepoProfileStatus

        ws = tmp_path / "workspace.json"
        mgr = MultiRepoManager(workspace_path=ws)
        repo_dir = tmp_path / "repo_a"
        repo_dir.mkdir()
        p = mgr.add_repo(str(repo_dir))
        # Pending by default
        assert len(mgr.get_ready_profiles()) == 0
        # Manually mark ready
        p.status = RepoProfileStatus.READY
        assert len(mgr.get_ready_profiles()) == 1

    def test_resolve_by_path(self, tmp_path):
        from scholardevclaw.multi_repo.manager import MultiRepoManager

        ws = tmp_path / "workspace.json"
        mgr = MultiRepoManager(workspace_path=ws)
        repo_dir = tmp_path / "repo_a"
        repo_dir.mkdir()
        mgr.add_repo(str(repo_dir))
        found = mgr.get_profile(str(repo_dir))
        assert found is not None


# =========================================================================
# PatternOverlap
# =========================================================================


class TestPatternOverlap:
    def test_to_dict(self):
        from scholardevclaw.multi_repo.analysis import PatternOverlap

        po = PatternOverlap(
            pattern="normalization",
            repos=["a", "b"],
            details={"a": ["LayerNorm"], "b": ["RMSNorm"]},
        )
        d = po.to_dict()
        assert d["pattern"] == "normalization"
        assert d["repo_count"] == 2
        assert d["repos"] == ["a", "b"]
        assert d["details"]["a"] == ["LayerNorm"]


# =========================================================================
# FrameworkComparison
# =========================================================================


class TestFrameworkComparison:
    def test_to_dict(self):
        from scholardevclaw.multi_repo.analysis import FrameworkComparison

        fc = FrameworkComparison(framework="pytorch", repos=["a", "b"])
        d = fc.to_dict()
        assert d["framework"] == "pytorch"
        assert d["repo_count"] == 2


# =========================================================================
# LanguageOverlap
# =========================================================================


class TestLanguageOverlap:
    def test_to_dict(self):
        from scholardevclaw.multi_repo.analysis import LanguageOverlap

        lo = LanguageOverlap(language="python", repos=["a", "b"], total_lines=1000, total_files=20)
        d = lo.to_dict()
        assert d["language"] == "python"
        assert d["repo_count"] == 2
        assert d["total_lines"] == 1000
        assert d["total_files"] == 20


# =========================================================================
# ComparisonResult
# =========================================================================


class TestComparisonResult:
    def test_to_dict(self):
        from scholardevclaw.multi_repo.analysis import ComparisonResult

        cr = ComparisonResult(
            repo_ids=["a", "b"],
            repo_names={"a": "repo_a", "b": "repo_b"},
            shared_patterns=["normalization"],
            summary="Test summary",
        )
        d = cr.to_dict()
        assert d["repo_count"] == 2
        assert d["shared_patterns"] == ["normalization"]
        assert d["summary"] == "Test summary"

    def test_empty_result(self):
        from scholardevclaw.multi_repo.analysis import ComparisonResult

        cr = ComparisonResult()
        d = cr.to_dict()
        assert d["repo_count"] == 0
        assert d["pattern_overlaps"] == []


# =========================================================================
# CrossRepoAnalyzer
# =========================================================================


class TestCrossRepoAnalyzer:
    def _two_profiles(self):
        a = _make_profile(
            name="repo_a",
            repo_id="aaa",
            languages=["python", "rust"],
            frameworks=["pytorch", "numpy"],
            patterns={
                "normalization": ["LayerNorm"],
                "attention": ["MultiHeadAttention"],
                "dropout": ["Dropout"],
            },
            suggestions=[
                {"spec": "rmsnorm", "name": "rmsnorm", "category": "normalization"},
                {"spec": "swiglu", "name": "swiglu", "category": "activation"},
            ],
        )
        b = _make_profile(
            name="repo_b",
            repo_id="bbb",
            languages=["python"],
            frameworks=["pytorch", "tensorflow"],
            patterns={
                "normalization": ["BatchNorm"],
                "attention": ["SelfAttention"],
                "embedding": ["Embedding"],
            },
            suggestions=[
                {"spec": "rmsnorm", "name": "rmsnorm", "category": "normalization"},
                {"spec": "rope", "name": "rope", "category": "positional"},
            ],
        )
        return a, b

    def test_single_profile_returns_minimal_result(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        a = _make_profile(name="solo", repo_id="solo")
        analyzer = CrossRepoAnalyzer([a])
        result = analyzer.compare()
        assert "Need at least 2" in result.summary
        assert result.repo_ids == ["solo"]

    def test_compare_two_profiles(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        a, b = self._two_profiles()
        analyzer = CrossRepoAnalyzer([a, b])
        result = analyzer.compare()

        assert len(result.repo_ids) == 2
        assert "aaa" in result.repo_ids
        assert "bbb" in result.repo_ids

    def test_pattern_overlaps(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        a, b = self._two_profiles()
        analyzer = CrossRepoAnalyzer([a, b])
        result = analyzer.compare()

        # normalization and attention are shared; dropout and embedding are unique
        pattern_names = [po.pattern for po in result.pattern_overlaps]
        assert "normalization" in pattern_names
        assert "attention" in pattern_names
        assert "dropout" in pattern_names
        assert "embedding" in pattern_names

        # normalization should have 2 repos
        norm_po = next(po for po in result.pattern_overlaps if po.pattern == "normalization")
        assert len(norm_po.repos) == 2

    def test_framework_comparisons(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        a, b = self._two_profiles()
        analyzer = CrossRepoAnalyzer([a, b])
        result = analyzer.compare()

        fw_names = [fc.framework for fc in result.framework_comparisons]
        assert "pytorch" in fw_names
        assert "numpy" in fw_names
        assert "tensorflow" in fw_names

        # pytorch shared by both
        pytorch_fc = next(fc for fc in result.framework_comparisons if fc.framework == "pytorch")
        assert len(pytorch_fc.repos) == 2

    def test_language_overlaps(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        a, b = self._two_profiles()
        analyzer = CrossRepoAnalyzer([a, b])
        result = analyzer.compare()

        lang_names = [lo.language for lo in result.language_overlaps]
        assert "python" in lang_names

        python_lo = next(lo for lo in result.language_overlaps if lo.language == "python")
        assert len(python_lo.repos) == 2

    def test_pairwise_similarity(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        a, b = self._two_profiles()
        analyzer = CrossRepoAnalyzer([a, b])
        result = analyzer.compare()

        assert len(result.pairwise_similarity) == 1
        key = "aaa:bbb"
        assert key in result.pairwise_similarity
        sim = result.pairwise_similarity[key]
        assert 0.0 <= sim <= 1.0

    def test_shared_patterns(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        a, b = self._two_profiles()
        analyzer = CrossRepoAnalyzer([a, b])
        result = analyzer.compare()

        # normalization and attention are in both
        assert "normalization" in result.shared_patterns
        assert "attention" in result.shared_patterns
        # dropout only in a, embedding only in b
        assert "dropout" not in result.shared_patterns
        assert "embedding" not in result.shared_patterns

    def test_unique_patterns(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        a, b = self._two_profiles()
        analyzer = CrossRepoAnalyzer([a, b])
        result = analyzer.compare()

        assert "dropout" in result.unique_patterns["aaa"]
        assert "embedding" in result.unique_patterns["bbb"]

    def test_spec_relevance_matrix(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        a, b = self._two_profiles()
        analyzer = CrossRepoAnalyzer([a, b])
        matrix = analyzer.spec_relevance_matrix()

        # rmsnorm in both
        assert matrix["rmsnorm"]["aaa"] is True
        assert matrix["rmsnorm"]["bbb"] is True
        # swiglu only in a
        assert matrix["swiglu"]["aaa"] is True
        assert matrix["swiglu"]["bbb"] is False
        # rope only in b
        assert matrix["rope"]["bbb"] is True
        assert matrix["rope"]["aaa"] is False

    def test_summary_contains_repo_names(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        a, b = self._two_profiles()
        analyzer = CrossRepoAnalyzer([a, b])
        result = analyzer.compare()

        assert "repo_a" in result.summary
        assert "repo_b" in result.summary

    def test_jaccard_both_empty(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        assert CrossRepoAnalyzer._jaccard(set(), set()) == 1.0

    def test_jaccard_no_overlap(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        assert CrossRepoAnalyzer._jaccard({"a"}, {"b"}) == 0.0

    def test_jaccard_full_overlap(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        assert CrossRepoAnalyzer._jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_jaccard_partial_overlap(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        # {a,b} & {b,c} = {b}, union = {a,b,c}, jaccard = 1/3
        result = CrossRepoAnalyzer._jaccard({"a", "b"}, {"b", "c"})
        assert abs(result - 1 / 3) < 0.01


# =========================================================================
# TransferDirection enum
# =========================================================================


class TestTransferDirection:
    def test_values(self):
        from scholardevclaw.multi_repo.transfer import TransferDirection

        assert TransferDirection.SOURCE_TO_TARGET == "source_to_target"
        assert TransferDirection.BIDIRECTIONAL == "bidirectional"


# =========================================================================
# TransferOpportunity
# =========================================================================


class TestTransferOpportunity:
    def test_to_dict(self):
        from scholardevclaw.multi_repo.transfer import TransferDirection, TransferOpportunity

        opp = TransferOpportunity(
            spec_name="swiglu",
            source_repo_id="aaa",
            target_repo_id="bbb",
            direction=TransferDirection.SOURCE_TO_TARGET,
            confidence=75,
            rationale="shared frameworks",
            shared_patterns=["normalization"],
            shared_frameworks=["pytorch"],
            category="activation",
        )
        d = opp.to_dict()
        assert d["spec_name"] == "swiglu"
        assert d["confidence"] == 75
        assert d["direction"] == "source_to_target"
        assert d["shared_frameworks"] == ["pytorch"]


# =========================================================================
# TransferPlan
# =========================================================================


class TestTransferPlan:
    def test_to_dict(self):
        from scholardevclaw.multi_repo.transfer import TransferPlan

        plan = TransferPlan(
            source_repo_id="aaa",
            source_name="repo_a",
            target_repo_id="bbb",
            target_name="repo_b",
            overall_score=65,
            summary="Test plan",
        )
        d = plan.to_dict()
        assert d["source_repo_id"] == "aaa"
        assert d["target_repo_id"] == "bbb"
        assert d["overall_score"] == 65
        assert d["opportunity_count"] == 0

    def test_to_dict_with_opportunities(self):
        from scholardevclaw.multi_repo.transfer import TransferOpportunity, TransferPlan

        plan = TransferPlan(
            source_repo_id="aaa",
            source_name="repo_a",
            target_repo_id="bbb",
            target_name="repo_b",
            opportunities=[
                TransferOpportunity(
                    spec_name="swiglu",
                    source_repo_id="aaa",
                    target_repo_id="bbb",
                    confidence=80,
                )
            ],
            overall_score=80,
        )
        d = plan.to_dict()
        assert d["opportunity_count"] == 1
        assert d["opportunities"][0]["spec_name"] == "swiglu"


# =========================================================================
# KnowledgeTransferEngine
# =========================================================================


class TestKnowledgeTransferEngine:
    def _two_profiles(self):
        a = _make_profile(
            name="repo_a",
            repo_id="aaa",
            languages=["python"],
            frameworks=["pytorch"],
            patterns={"normalization": ["LayerNorm"], "attention": ["MHA"]},
            suggestions=[
                {"spec": "rmsnorm", "name": "rmsnorm", "category": "normalization"},
                {"spec": "swiglu", "name": "swiglu", "category": "activation"},
                {"spec": "rope", "name": "rope", "category": "positional"},
            ],
            element_count=50,
        )
        b = _make_profile(
            name="repo_b",
            repo_id="bbb",
            languages=["python"],
            frameworks=["pytorch"],
            patterns={"normalization": ["BatchNorm"], "embedding": ["Embedding"]},
            suggestions=[
                {"spec": "rmsnorm", "name": "rmsnorm", "category": "normalization"},
            ],
            element_count=40,
        )
        return a, b

    def test_single_profile_returns_empty(self):
        from scholardevclaw.multi_repo.transfer import KnowledgeTransferEngine

        a = _make_profile(name="solo", repo_id="solo")
        engine = KnowledgeTransferEngine([a])
        plans = engine.discover()
        assert plans == []

    def test_discover_finds_transferable_specs(self):
        from scholardevclaw.multi_repo.transfer import KnowledgeTransferEngine

        a, b = self._two_profiles()
        engine = KnowledgeTransferEngine([a, b])
        plans = engine.discover()

        # a has swiglu and rope that b doesn't have -> a->b plans
        a_to_b = [p for p in plans if p.source_repo_id == "aaa" and p.target_repo_id == "bbb"]
        assert len(a_to_b) >= 1
        spec_names = set()
        for plan in a_to_b:
            for opp in plan.opportunities:
                spec_names.add(opp.spec_name)
        assert "swiglu" in spec_names or "rope" in spec_names

    def test_discover_for_pair(self):
        from scholardevclaw.multi_repo.transfer import KnowledgeTransferEngine

        a, b = self._two_profiles()
        engine = KnowledgeTransferEngine([a, b])
        plan = engine.discover_for_pair("aaa", "bbb")

        assert plan is not None
        assert plan.source_repo_id == "aaa"
        assert plan.target_repo_id == "bbb"

    def test_discover_for_pair_invalid_id(self):
        from scholardevclaw.multi_repo.transfer import KnowledgeTransferEngine

        a = _make_profile(name="solo", repo_id="solo")
        engine = KnowledgeTransferEngine([a])
        plan = engine.discover_for_pair("solo", "nonexistent")
        assert plan is None

    def test_confidence_scoring(self):
        from scholardevclaw.multi_repo.transfer import KnowledgeTransferEngine

        a, b = self._two_profiles()
        engine = KnowledgeTransferEngine([a, b])
        plan = engine.discover_for_pair("aaa", "bbb")
        assert plan is not None

        for opp in plan.opportunities:
            assert 0 <= opp.confidence <= 100
            assert opp.rationale  # not empty

    def test_plans_sorted_by_score(self):
        from scholardevclaw.multi_repo.transfer import KnowledgeTransferEngine

        a, b = self._two_profiles()
        engine = KnowledgeTransferEngine([a, b])
        plans = engine.discover()

        scores = [p.overall_score for p in plans]
        assert scores == sorted(scores, reverse=True)

    def test_summary_included(self):
        from scholardevclaw.multi_repo.transfer import KnowledgeTransferEngine

        a, b = self._two_profiles()
        engine = KnowledgeTransferEngine([a, b])
        plan = engine.discover_for_pair("aaa", "bbb")
        assert plan is not None
        assert plan.summary  # non-empty

    def test_no_self_transfer(self):
        from scholardevclaw.multi_repo.transfer import KnowledgeTransferEngine

        a, b = self._two_profiles()
        engine = KnowledgeTransferEngine([a, b])
        plans = engine.discover()

        for plan in plans:
            assert plan.source_repo_id != plan.target_repo_id

    def test_shared_helpers(self):
        from scholardevclaw.multi_repo.transfer import KnowledgeTransferEngine

        a, b = self._two_profiles()
        shared_patterns = KnowledgeTransferEngine._shared_patterns(a, b)
        assert "normalization" in shared_patterns

        shared_fws = KnowledgeTransferEngine._shared_frameworks(a, b)
        assert "pytorch" in shared_fws

        shared_langs = KnowledgeTransferEngine._shared_languages(a, b)
        assert "python" in shared_langs

    def test_extract_spec_names(self):
        from scholardevclaw.multi_repo.transfer import KnowledgeTransferEngine

        a, _ = self._two_profiles()
        names = KnowledgeTransferEngine._extract_spec_names(a)
        assert "rmsnorm" in names
        assert "swiglu" in names
        assert "rope" in names


# =========================================================================
# Pipeline functions
# =========================================================================


class TestPipelineFunctions:
    def test_run_multi_repo_analyze_imports(self):
        from scholardevclaw.application.pipeline import run_multi_repo_analyze

        assert callable(run_multi_repo_analyze)

    def test_run_multi_repo_compare_imports(self):
        from scholardevclaw.application.pipeline import run_multi_repo_compare

        assert callable(run_multi_repo_compare)

    def test_run_multi_repo_transfer_imports(self):
        from scholardevclaw.application.pipeline import run_multi_repo_transfer

        assert callable(run_multi_repo_transfer)

    def test_run_multi_repo_analyze_no_repos(self, tmp_path):
        from scholardevclaw.application.pipeline import run_multi_repo_analyze

        ws = tmp_path / "ws.json"
        result = run_multi_repo_analyze(
            [],
            workspace_path=str(ws),
        )
        # Empty list -> no repos reach ready -> error
        assert result.title == "Multi-Repo Analyze"

    def test_run_multi_repo_compare_needs_two(self, tmp_path):
        from scholardevclaw.application.pipeline import run_multi_repo_compare

        ws = tmp_path / "ws.json"
        result = run_multi_repo_compare(
            workspace_path=str(ws),
        )
        assert not result.ok
        assert "at least 2" in (result.error or "")

    def test_run_multi_repo_transfer_needs_two(self, tmp_path):
        from scholardevclaw.application.pipeline import run_multi_repo_transfer

        ws = tmp_path / "ws.json"
        result = run_multi_repo_transfer(
            workspace_path=str(ws),
        )
        assert not result.ok
        assert "at least 2" in (result.error or "")


# =========================================================================
# __init__.py exports
# =========================================================================


class TestExports:
    def test_all_12_exports(self):
        import scholardevclaw.multi_repo as mr

        expected = {
            "MultiRepoManager",
            "RepoProfile",
            "RepoProfileStatus",
            "CrossRepoAnalyzer",
            "ComparisonResult",
            "PatternOverlap",
            "FrameworkComparison",
            "LanguageOverlap",
            "KnowledgeTransferEngine",
            "TransferOpportunity",
            "TransferPlan",
            "TransferDirection",
        }
        assert set(mr.__all__) == expected

    def test_manager_importable(self):
        from scholardevclaw.multi_repo import MultiRepoManager, RepoProfile, RepoProfileStatus

        assert MultiRepoManager is not None
        assert RepoProfile is not None
        assert RepoProfileStatus is not None

    def test_analysis_importable(self):
        from scholardevclaw.multi_repo import (
            ComparisonResult,
            CrossRepoAnalyzer,
            FrameworkComparison,
            LanguageOverlap,
            PatternOverlap,
        )

        assert CrossRepoAnalyzer is not None
        assert ComparisonResult is not None
        assert PatternOverlap is not None
        assert FrameworkComparison is not None
        assert LanguageOverlap is not None

    def test_transfer_importable(self):
        from scholardevclaw.multi_repo import (
            KnowledgeTransferEngine,
            TransferDirection,
            TransferOpportunity,
            TransferPlan,
        )

        assert KnowledgeTransferEngine is not None
        assert TransferOpportunity is not None
        assert TransferPlan is not None
        assert TransferDirection is not None


# =========================================================================
# CLI wiring
# =========================================================================


class TestCLIWiring:
    def test_multi_repo_in_dispatch_dict(self):
        """Verify 'multi-repo' is in the CLI command dispatch table."""
        import scholardevclaw.cli as cli_module

        source = Path(cli_module.__file__).read_text()
        assert '"multi-repo"' in source or "'multi-repo'" in source

    def test_cmd_multi_repo_callable(self):
        from scholardevclaw.cli import cmd_multi_repo

        assert callable(cmd_multi_repo)

    def test_multi_repo_subparser_exists(self):
        """Verify the argparse subparser is set up."""
        import scholardevclaw.cli as cli_module

        source = Path(cli_module.__file__).read_text()
        assert "multi-repo" in source
        assert "multi_repo_action" in source
        assert "compare" in source
        assert "transfer" in source


# =========================================================================
# Three-repo comparison
# =========================================================================


class TestThreeRepoComparison:
    """Test with 3 repos to exercise the full combinatorics."""

    def _three_profiles(self):
        a = _make_profile(
            name="alpha",
            repo_id="aaa",
            languages=["python", "rust"],
            frameworks=["pytorch"],
            patterns={"normalization": ["LN"], "attention": ["MHA"], "optimizer": ["Adam"]},
            suggestions=[
                {"spec": "rmsnorm", "name": "rmsnorm", "category": "normalization"},
                {"spec": "swiglu", "name": "swiglu", "category": "activation"},
            ],
        )
        b = _make_profile(
            name="beta",
            repo_id="bbb",
            languages=["python"],
            frameworks=["pytorch", "tensorflow"],
            patterns={"normalization": ["BN"], "attention": ["SA"]},
            suggestions=[
                {"spec": "rmsnorm", "name": "rmsnorm", "category": "normalization"},
                {"spec": "rope", "name": "rope", "category": "positional"},
            ],
        )
        c = _make_profile(
            name="gamma",
            repo_id="ccc",
            languages=["python", "javascript"],
            frameworks=["tensorflow"],
            patterns={"normalization": ["GN"], "embedding": ["Emb"]},
            suggestions=[
                {"spec": "preln", "name": "preln", "category": "normalization"},
            ],
        )
        return a, b, c

    def test_three_repo_comparison(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        a, b, c = self._three_profiles()
        analyzer = CrossRepoAnalyzer([a, b, c])
        result = analyzer.compare()

        assert len(result.repo_ids) == 3
        # normalization is in all 3
        assert "normalization" in result.shared_patterns

    def test_three_repo_pairwise(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        a, b, c = self._three_profiles()
        analyzer = CrossRepoAnalyzer([a, b, c])
        result = analyzer.compare()

        # 3 repos -> 3 pairs
        assert len(result.pairwise_similarity) == 3

    def test_three_repo_transfer(self):
        from scholardevclaw.multi_repo.transfer import KnowledgeTransferEngine

        a, b, c = self._three_profiles()
        engine = KnowledgeTransferEngine([a, b, c])
        plans = engine.discover()

        # Should find multiple transfer plans
        assert len(plans) >= 1
        # No self-transfers
        for plan in plans:
            assert plan.source_repo_id != plan.target_repo_id

    def test_spec_matrix_three_repos(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        a, b, c = self._three_profiles()
        analyzer = CrossRepoAnalyzer([a, b, c])
        matrix = analyzer.spec_relevance_matrix()

        # rmsnorm in a and b, not c
        assert matrix["rmsnorm"]["aaa"] is True
        assert matrix["rmsnorm"]["bbb"] is True
        assert matrix["rmsnorm"]["ccc"] is False

        # preln only in c
        assert matrix["preln"]["ccc"] is True
        assert matrix["preln"]["aaa"] is False


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    def test_empty_patterns(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        a = _make_profile(name="a", repo_id="a", patterns={})
        b = _make_profile(name="b", repo_id="b", patterns={})
        analyzer = CrossRepoAnalyzer([a, b])
        result = analyzer.compare()
        assert result.pattern_overlaps == []
        assert result.shared_patterns == []

    def test_empty_frameworks(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        a = _make_profile(name="a", repo_id="a", frameworks=[])
        b = _make_profile(name="b", repo_id="b", frameworks=[])
        analyzer = CrossRepoAnalyzer([a, b])
        result = analyzer.compare()
        assert result.framework_comparisons == []

    def test_no_suggestions_means_no_transfers(self):
        from scholardevclaw.multi_repo.transfer import KnowledgeTransferEngine

        a = _make_profile(name="a", repo_id="a", suggestions=[])
        b = _make_profile(name="b", repo_id="b", suggestions=[])
        engine = KnowledgeTransferEngine([a, b])
        plans = engine.discover()
        # No specs to transfer
        all_opps = sum(len(p.opportunities) for p in plans)
        assert all_opps == 0

    def test_identical_suggestions_no_transfer(self):
        from scholardevclaw.multi_repo.transfer import KnowledgeTransferEngine

        same_sugg = [
            {"spec": "rmsnorm", "name": "rmsnorm", "category": "normalization"},
        ]
        a = _make_profile(name="a", repo_id="a", suggestions=same_sugg)
        b = _make_profile(name="b", repo_id="b", suggestions=same_sugg)
        engine = KnowledgeTransferEngine([a, b])
        plans = engine.discover()
        all_opps = sum(len(p.opportunities) for p in plans)
        assert all_opps == 0

    def test_comparison_result_json_serializable(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        a = _make_profile(name="a", repo_id="a")
        b = _make_profile(name="b", repo_id="b")
        analyzer = CrossRepoAnalyzer([a, b])
        result = analyzer.compare()
        # Should be fully JSON-serializable
        serialized = json.dumps(result.to_dict())
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert parsed["repo_count"] == 2

    def test_transfer_plan_json_serializable(self):
        from scholardevclaw.multi_repo.transfer import KnowledgeTransferEngine

        a = _make_profile(
            name="a",
            repo_id="a",
            suggestions=[{"spec": "swiglu", "name": "swiglu", "category": "activation"}],
        )
        b = _make_profile(name="b", repo_id="b", suggestions=[])
        engine = KnowledgeTransferEngine([a, b])
        plans = engine.discover()
        for plan in plans:
            serialized = json.dumps(plan.to_dict())
            assert isinstance(serialized, str)

    def test_case_insensitive_frameworks(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        a = _make_profile(name="a", repo_id="a", frameworks=["PyTorch"])
        b = _make_profile(name="b", repo_id="b", frameworks=["pytorch"])
        analyzer = CrossRepoAnalyzer([a, b])
        result = analyzer.compare()
        # Should be treated as same framework
        assert len(result.framework_comparisons) == 1
        assert len(result.framework_comparisons[0].repos) == 2

    def test_case_insensitive_languages(self):
        from scholardevclaw.multi_repo.analysis import CrossRepoAnalyzer

        a = _make_profile(name="a", repo_id="a", languages=["Python"])
        b = _make_profile(name="b", repo_id="b", languages=["python"])
        analyzer = CrossRepoAnalyzer([a, b])
        result = analyzer.compare()
        python_lo = [lo for lo in result.language_overlaps if lo.language == "python"]
        assert len(python_lo) == 1
        assert len(python_lo[0].repos) == 2
