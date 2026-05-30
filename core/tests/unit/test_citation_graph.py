"""Tests for citation_graph and similarity modules."""

from pathlib import Path

import pytest

from scholardevclaw.research_intelligence.citation_graph import (
    CitationAnalyzer,
    CitationGraph,
)
from scholardevclaw.research_intelligence.similarity import (
    ResearchRecommendationEngine,
    ResearchSimilaritySearch,
)

# =========================================================================
# CitationGraph
# =========================================================================


class TestCitationGraph:
    def test_add_paper(self):
        g = CitationGraph()
        node = g.add_paper("p1", "Paper One", 2020)
        assert node.paper_id == "p1"
        assert "p1" in g.nodes

    def test_add_paper_with_citations(self):
        g = CitationGraph()
        g.add_paper("p1", "Paper One", 2020, citations=["p2", "p3"])
        assert g.get_citations("p1") == {"p2", "p3"}
        assert "p2" in g.nodes
        assert "p3" in g.nodes

    def test_add_paper_with_references(self):
        g = CitationGraph()
        g.add_paper("p1", "Paper One", 2020, references=["p2"])
        assert g.get_references("p1") == {"p2"}
        assert "p2" in g.nodes

    def test_update_existing_paper(self):
        g = CitationGraph()
        g.add_paper("p1", "Old Title", 2020)
        g.add_paper("p1", "New Title", 2022)
        assert g.nodes["p1"].title == "New Title"
        assert g.nodes["p1"].year == 2022

    def test_get_citations_unknown(self):
        g = CitationGraph()
        assert g.get_citations("nonexistent") == set()

    def test_get_references_unknown(self):
        g = CitationGraph()
        assert g.get_references("nonexistent") == set()

    def test_find_shortest_path_self(self):
        g = CitationGraph()
        g.add_paper("p1", "One", 2020)
        path = g.find_shortest_path("p1", "p1")
        assert path is not None
        assert path.length == 0
        assert path.path == ["p1"]

    def test_find_shortest_path_direct(self):
        g = CitationGraph()
        g.add_paper("p1", "One", 2020, references=["p2"])
        g.add_paper("p2", "Two", 2021)
        path = g.find_shortest_path("p1", "p2")
        assert path is not None
        assert path.length == 1

    def test_find_shortest_path_none(self):
        g = CitationGraph()
        g.add_paper("p1", "One", 2020)
        assert g.find_shortest_path("p1", "p2") is None

    def test_find_shortest_path_nonexistent_source(self):
        g = CitationGraph()
        g.add_paper("p1", "One", 2020)
        assert g.find_shortest_path("ghost", "p1") is None

    def test_find_shortest_path_max_depth(self):
        g = CitationGraph()
        g.add_paper("p1", "A", 2020, references=["p2"])
        g.add_paper("p2", "B", 2020, references=["p3"])
        g.add_paper("p3", "C", 2020, references=["p4"])
        g.add_paper("p4", "D", 2020)
        # max_depth=1 should not find p4 from p1
        assert g.find_shortest_path("p1", "p4", max_depth=1) is None
        # max_depth=3 should find it
        path = g.find_shortest_path("p1", "p4", max_depth=3)
        assert path is not None

    def test_find_all_paths(self):
        g = CitationGraph()
        g.add_paper("p1", "A", 2020, references=["p2", "p3"])
        g.add_paper("p2", "B", 2020, references=["p4"])
        g.add_paper("p3", "C", 2020, references=["p4"])
        g.add_paper("p4", "D", 2020)
        paths = g.find_all_paths("p1", "p4")
        assert len(paths) == 2  # p1->p2->p4 and p1->p3->p4

    def test_find_all_paths_none(self):
        g = CitationGraph()
        g.add_paper("p1", "A", 2020)
        assert g.find_all_paths("p1", "nonexistent") == []

    def test_get_common_ancestors(self):
        g = CitationGraph()
        g.add_paper("p1", "A", 2020, citations=["p3"])
        g.add_paper("p2", "B", 2020, citations=["p3", "p4"])
        common = g.get_common_ancestors("p1", "p2")
        assert "p3" in common

    def test_get_common_ancestors_unknown(self):
        g = CitationGraph()
        g.add_paper("p1", "A", 2020)
        assert g.get_common_ancestors("p1", "ghost") == set()

    def test_get_common_descendants(self):
        g = CitationGraph()
        g.add_paper("p1", "A", 2020, references=["p3"])
        g.add_paper("p2", "B", 2020, references=["p3", "p4"])
        common = g.get_common_descendants("p1", "p2")
        assert "p3" in common

    def test_get_common_descendants_unknown(self):
        g = CitationGraph()
        g.add_paper("p1", "A", 2020)
        assert g.get_common_descendants("p1", "ghost") == set()

    def test_pagerank_empty(self):
        g = CitationGraph()
        assert g.get_pagerank() == {}

    def test_pagerank_single_node(self):
        g = CitationGraph()
        g.add_paper("p1", "A", 2020)
        ranks = g.get_pagerank()
        assert "p1" in ranks
        assert ranks["p1"] == pytest.approx(0.15, abs=0.01)  # (1-d)/n with no incoming links

    def test_pagerank_cited_node_higher(self):
        g = CitationGraph()
        g.add_paper("p1", "A", 2020, references=["p2"])
        g.add_paper("p2", "B", 2020)
        g.add_paper("p3", "C", 2020, references=["p2"])
        ranks = g.get_pagerank()
        assert ranks["p2"] > ranks["p1"]

    def test_influence_score(self):
        g = CitationGraph()
        g.add_paper("p1", "A", 2020, citations=["p2"], references=["p3"])
        score = g.get_influence_score("p1")
        assert score > 0

    def test_influence_score_no_citations(self):
        g = CitationGraph()
        g.add_paper("p1", "A", 2020)
        assert g.get_influence_score("p1") == 0.0

    def test_influence_score_unknown(self):
        g = CitationGraph()
        assert g.get_influence_score("ghost") == 0.0

    def test_get_related_papers(self):
        g = CitationGraph()
        g.add_paper("p1", "A", 2020, citations=["p3"], references=["p4"])
        g.add_paper("p2", "B", 2020, citations=["p3"], references=["p4"])
        g.add_paper("p3", "C", 2020)
        g.add_paper("p4", "D", 2020)
        related = g.get_related_papers("p1")
        assert len(related) > 0
        assert related[0][0] == "p2"  # shares both p3 and p4

    def test_get_related_papers_unknown(self):
        g = CitationGraph()
        assert g.get_related_papers("ghost") == []

    def test_to_dict_and_load(self, tmp_path: Path):
        g = CitationGraph()
        g.add_paper("p1", "A", 2020, citations=["p2"])
        g.add_paper("p2", "B", 2021)
        data = g.to_dict()
        assert "p1" in data["nodes"]

        path = tmp_path / "graph.json"
        g.save(path)
        loaded = CitationGraph.load(path)
        assert "p1" in loaded.nodes
        assert loaded.nodes["p1"].title == "A"


# =========================================================================
# CitationAnalyzer
# =========================================================================


class TestCitationAnalyzer:
    def test_analyze_influence(self):
        g = CitationGraph()
        g.add_paper("p1", "A", 2020, citations=["p2"], references=["p3"])
        g.add_paper("p2", "B", 2021)
        g.add_paper("p3", "C", 2022)
        analyzer = CitationAnalyzer(g)
        result = analyzer.analyze_influence("p1")
        assert result["paper_id"] == "p1"
        assert result["citations_count"] == 1
        assert result["references_count"] == 1
        assert "pagerank" in result
        assert "influence_score" in result

    def test_analyze_influence_unknown(self):
        g = CitationGraph()
        analyzer = CitationAnalyzer(g)
        assert analyzer.analyze_influence("ghost") == {}

    def test_compare_papers(self):
        g = CitationGraph()
        g.add_paper("p1", "A", 2020, citations=["p3"])
        g.add_paper("p2", "B", 2021, citations=["p3"])
        g.add_paper("p3", "C", 2019)
        analyzer = CitationAnalyzer(g)
        result = analyzer.compare_papers("p1", "p2")
        assert "paper1" in result
        assert "paper2" in result
        assert "p3" in result["common_ancestors"]

    def test_compare_papers_unknown(self):
        g = CitationGraph()
        analyzer = CitationAnalyzer(g)
        assert analyzer.compare_papers("p1", "ghost") == {}

    def test_get_citation_trends(self):
        g = CitationGraph()
        g.add_paper("p1", "A", 2020, citations=["p2", "p3"])
        g.add_paper("p2", "B", 2021)
        g.add_paper("p3", "C", 2020)
        analyzer = CitationAnalyzer(g)
        trends = analyzer.get_citation_trends()
        assert 2020 in trends["papers_per_year"]
        assert trends["papers_per_year"][2020] == 2


# =========================================================================
# ResearchSimilaritySearch
# =========================================================================


class TestResearchSimilaritySearch:
    def test_tokenize(self):
        s = ResearchSimilaritySearch()
        tokens = s._tokenize("The quick brown fox jumps")
        assert "quick" in tokens
        assert "the" not in tokens  # stop word

    def test_compute_tf(self):
        s = ResearchSimilaritySearch()
        tf = s._compute_tf(["a", "a", "b"])
        assert tf["a"] == pytest.approx(2 / 3)
        assert tf["b"] == pytest.approx(1 / 3)

    def test_compute_tf_empty(self):
        s = ResearchSimilaritySearch()
        assert s._compute_tf([]) == {}

    def test_compute_idf(self):
        s = ResearchSimilaritySearch()
        docs = [["a", "b"], ["b", "c"]]
        idf = s._compute_idf(docs)
        assert idf["a"] > idf["b"]  # "a" appears in fewer docs

    def test_keyword_similarity_identical(self):
        s = ResearchSimilaritySearch()
        score = s.keyword_similarity("attention mechanism", "attention mechanism")
        assert score == pytest.approx(1.0)

    def test_keyword_similarity_partial(self):
        s = ResearchSimilaritySearch()
        score = s.keyword_similarity("attention transformer model", "attention mechanism")
        assert 0 < score < 1

    def test_keyword_similarity_empty(self):
        s = ResearchSimilaritySearch()
        assert s.keyword_similarity("", "test") == 0.0

    def test_find_similar(self):
        s = ResearchSimilaritySearch()
        papers = [
            {
                "paper_id": "p1",
                "title": "Attention Is All You Need",
                "abstract": "transformer model",
                "year": 2017,
            },
            {
                "paper_id": "p2",
                "title": "Image Classification with CNNs",
                "abstract": "computer vision",
                "year": 2020,
            },
        ]
        results = s.find_similar("attention transformer", papers)
        assert len(results) > 0
        assert results[0].paper_id == "p1"

    def test_find_similar_empty(self):
        s = ResearchSimilaritySearch()
        assert s.find_similar("test", []) == []

    def test_find_related_by_papers(self):
        s = ResearchSimilaritySearch()
        source = [{"title": "attention transformer model", "abstract": "NLP"}]
        candidates = [
            {
                "paper_id": "p1",
                "title": "attention mechanism for NLP",
                "abstract": "language model",
            },
            {"paper_id": "p2", "title": "image segmentation", "abstract": "computer vision"},
        ]
        results = s.find_related_by_papers(source, candidates)
        assert len(results) > 0
        assert results[0].paper_id == "p1"

    def test_find_related_by_papers_empty_source(self):
        s = ResearchSimilaritySearch()
        assert s.find_related_by_papers([], [{"title": "test"}]) == []


# =========================================================================
# ResearchRecommendationEngine
# =========================================================================


class TestResearchRecommendationEngine:
    def test_recommend(self):
        engine = ResearchRecommendationEngine()
        papers = [
            {"paper_id": "p1", "title": "attention transformer NLP", "abstract": "language model"},
            {"paper_id": "p2", "title": "attention mechanism", "abstract": "NLP"},
            {"paper_id": "p3", "title": "image segmentation", "abstract": "computer vision"},
        ]
        engine.index_papers(papers)
        results = engine.recommend(["p1"])
        assert len(results) > 0

    def test_recommend_empty(self):
        engine = ResearchRecommendationEngine()
        assert engine.recommend([]) == []

    def test_recommend_by_query(self):
        engine = ResearchRecommendationEngine()
        papers = [
            {"paper_id": "p1", "title": "attention transformer", "abstract": "NLP"},
            {"paper_id": "p2", "title": "image segmentation", "abstract": "vision"},
        ]
        engine.index_papers(papers)
        results = engine.recommend_by_query("attention")
        assert len(results) > 0

    def test_recommend_excludes_read(self):
        engine = ResearchRecommendationEngine()
        papers = [
            {"paper_id": "p1", "title": "attention transformer", "abstract": "NLP"},
            {"paper_id": "p2", "title": "attention mechanism", "abstract": "NLP"},
        ]
        engine.index_papers(papers)
        results = engine.recommend_by_query("attention", exclude_paper_ids=["p1"])
        ids = [r.paper_id for r in results]
        assert "p1" not in ids
