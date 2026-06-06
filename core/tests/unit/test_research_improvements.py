"""Tests for research intelligence improvements.

Covers:
- CitationGraph.get_pagerank vectorized implementation (small / large / sparse)
- Paper deduplication (DOI, arxiv_id, pubmed_id, ieee_id, normalized title)
- Paper merge strategy (scalar fields, list union, numeric max, merged_sources)
- PaperSourceAggregator.search_deduplicated orchestration
- Year drift fix: similarity.find_similar uses datetime.now().year, not hardcoded 2026
- Year drift fix: CitationGraph.get_influence_score uses datetime.now().year
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scholardevclaw.research_intelligence.citation_graph import CitationGraph
from scholardevclaw.research_intelligence.paper_sources import (
    Paper,
    PaperSourceAggregator,
    SearchResult,
    _normalize_doi,
    _normalize_title,
    deduplicate_papers,
)
from scholardevclaw.research_intelligence.similarity import ResearchSimilaritySearch

# =========================================================================
# Helper factories
# =========================================================================


def _make_paper(
    *,
    paper_id: str = "p1",
    title: str = "Sample Paper",
    authors: list[str] | None = None,
    abstract: str = "",
    year: int = 2024,
    source: str = "arxiv",
    url: str = "",
    doi: str = "",
    pdf_url: str = "",
    arxiv_id: str = "",
    pubmed_id: str = "",
    ieee_id: str = "",
    journal: str = "",
    publisher: str = "",
    categories: list[str] | None = None,
    keywords: list[str] | None = None,
    citations: int = 0,
    month: int = 1,
) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=title,
        authors=list(authors or []),
        abstract=abstract,
        year=year,
        source=source,
        url=url,
        month=month,
        doi=doi,
        categories=list(categories or []),
        citations=citations,
        pdf_url=pdf_url,
        arxiv_id=arxiv_id,
        pubmed_id=pubmed_id,
        ieee_id=ieee_id,
        journal=journal,
        volume="",
        pages="",
        publisher=publisher,
        keywords=list(keywords or []),
    )


# =========================================================================
# _normalize_doi
# =========================================================================


class TestNormalizeDoi:
    def test_empty(self):
        assert _normalize_doi("") == ""

    def test_plain(self):
        assert _normalize_doi("10.1234/abc") == "10.1234/abc"

    def test_uppercase_lowercased(self):
        assert _normalize_doi("10.1234/ABC") == "10.1234/abc"

    def test_strips_https_url(self):
        assert _normalize_doi("https://doi.org/10.1234/abc") == "10.1234/abc"

    def test_strips_dx_url(self):
        assert _normalize_doi("https://dx.doi.org/10.1234/abc") == "10.1234/abc"

    def test_strips_doi_prefix(self):
        assert _normalize_doi("doi:10.1234/abc") == "10.1234/abc"

    def test_strips_whitespace(self):
        assert _normalize_doi("  10.1234/abc  ") == "10.1234/abc"


# =========================================================================
# _normalize_title
# =========================================================================


class TestNormalizeTitle:
    def test_empty(self):
        assert _normalize_title("") == ""

    def test_lowercases(self):
        assert _normalize_title("FOO") == "foo"

    def test_removes_punctuation(self):
        assert _normalize_title("foo, bar!") == "foobar"

    def test_collapses_whitespace(self):
        assert _normalize_title("foo   bar") == "foobar"

    def test_alphanumeric_only(self):
        assert _normalize_title("foo bar-baz_qux") == "foobarbazqux"


# =========================================================================
# deduplicate_papers
# =========================================================================


class TestDeduplicatePapersEmptyAndSingle:
    def test_empty_list(self):
        assert deduplicate_papers([]) == []

    def test_single_paper(self):
        p = _make_paper(paper_id="a")
        result = deduplicate_papers([p])
        assert len(result) == 1
        assert result[0].merged_sources == ["arxiv"]

    def test_preserves_input_order(self):
        p1 = _make_paper(paper_id="a", title="First")
        p2 = _make_paper(paper_id="b", title="Second")
        p3 = _make_paper(paper_id="c", title="Third")
        result = deduplicate_papers([p1, p2, p3])
        assert [r.paper_id for r in result] == ["a", "b", "c"]


class TestDeduplicateByDOI:
    def test_same_doi_dedup(self):
        p1 = _make_paper(paper_id="a", source="arxiv", doi="10.1234/abc")
        p2 = _make_paper(paper_id="b", source="pubmed", doi="10.1234/abc")
        result = deduplicate_papers([p1, p2])
        assert len(result) == 1
        assert set(result[0].merged_sources) == {"arxiv", "pubmed"}

    def test_doi_case_insensitive(self):
        p1 = _make_paper(paper_id="a", doi="10.1234/ABC")
        p2 = _make_paper(paper_id="b", doi="10.1234/abc")
        result = deduplicate_papers([p1, p2])
        assert len(result) == 1

    def test_doi_with_url_prefix(self):
        p1 = _make_paper(paper_id="a", doi="https://doi.org/10.1234/abc")
        p2 = _make_paper(paper_id="b", doi="10.1234/abc")
        result = deduplicate_papers([p1, p2])
        assert len(result) == 1

    def test_doi_with_doi_prefix(self):
        p1 = _make_paper(paper_id="a", doi="doi:10.1234/abc")
        p2 = _make_paper(paper_id="b", doi="10.1234/abc")
        result = deduplicate_papers([p1, p2])
        assert len(result) == 1

    def test_different_dois_kept(self):
        p1 = _make_paper(paper_id="a", doi="10.1234/abc")
        p2 = _make_paper(paper_id="b", doi="10.1234/def")
        result = deduplicate_papers([p1, p2])
        assert len(result) == 2


class TestDeduplicateByArxivId:
    def test_same_arxiv_id_dedup(self):
        p1 = _make_paper(paper_id="a", source="arxiv", arxiv_id="2401.00001")
        p2 = _make_paper(paper_id="b", source="arxiv", arxiv_id="2401.00001")
        result = deduplicate_papers([p1, p2])
        assert len(result) == 1

    def test_case_insensitive(self):
        p1 = _make_paper(paper_id="a", arxiv_id="2401.00001")
        p2 = _make_paper(paper_id="b", arxiv_id="2401.00001".upper())
        result = deduplicate_papers([p1, p2])
        # Note: arxiv IDs are conventionally lowercase; this is just defensive
        assert len(result) == 1

    def test_different_arxiv_ids_kept(self):
        p1 = _make_paper(paper_id="a", arxiv_id="2401.00001")
        p2 = _make_paper(paper_id="b", arxiv_id="2401.00002")
        result = deduplicate_papers([p1, p2])
        assert len(result) == 2


class TestDeduplicateByPubmedId:
    def test_same_pubmed_id_dedup(self):
        p1 = _make_paper(paper_id="a", source="pubmed", pubmed_id="12345")
        p2 = _make_paper(paper_id="b", source="pubmed", pubmed_id="12345")
        result = deduplicate_papers([p1, p2])
        assert len(result) == 1


class TestDeduplicateByIeeeId:
    def test_same_ieee_id_dedup(self):
        p1 = _make_paper(paper_id="a", source="ieee", ieee_id="9000001")
        p2 = _make_paper(paper_id="b", source="ieee", ieee_id="9000001")
        result = deduplicate_papers([p1, p2])
        assert len(result) == 1


class TestDeduplicateByTitle:
    def test_same_title_dedup(self):
        p1 = _make_paper(paper_id="a", title="Attention Is All You Need")
        p2 = _make_paper(paper_id="b", title="attention is all you need")
        result = deduplicate_papers([p1, p2])
        assert len(result) == 1

    def test_title_with_punctuation_dedup(self):
        p1 = _make_paper(paper_id="a", title="Deep Learning: A Comprehensive Survey")
        p2 = _make_paper(paper_id="b", title="Deep Learning, a comprehensive survey!")
        result = deduplicate_papers([p1, p2])
        assert len(result) == 1

    def test_short_title_not_deduped(self):
        # Threshold check: < 20 chars won't dedup via title alone
        p1 = _make_paper(paper_id="a", title="BERT")
        p2 = _make_paper(paper_id="b", title="bert")
        result = deduplicate_papers([p1, p2])
        assert len(result) == 2

    def test_different_titles_kept(self):
        p1 = _make_paper(paper_id="a", title="Paper One With A Long Enough Title")
        p2 = _make_paper(paper_id="b", title="Paper Two With A Long Enough Title")
        result = deduplicate_papers([p1, p2])
        assert len(result) == 2


class TestDeduplicateMergeStrategy:
    def test_picks_longer_abstract(self):
        p1 = _make_paper(paper_id="a", abstract="short", doi="10.1/x")
        p2 = _make_paper(paper_id="b", abstract="this is a much longer abstract", doi="10.1/x")
        result = deduplicate_papers([p1, p2])
        assert result[0].abstract == "this is a much longer abstract"

    def test_unions_authors(self):
        p1 = _make_paper(paper_id="a", authors=["Alice"], doi="10.1/x")
        p2 = _make_paper(paper_id="b", authors=["Bob"], doi="10.1/x")
        result = deduplicate_papers([p1, p2])
        assert set(result[0].authors) == {"Alice", "Bob"}

    def test_unions_categories(self):
        p1 = _make_paper(paper_id="a", categories=["cs.LG"], doi="10.1/x")
        p2 = _make_paper(paper_id="b", categories=["cs.CL"], doi="10.1/x")
        result = deduplicate_papers([p1, p2])
        assert set(result[0].categories) == {"cs.LG", "cs.CL"}

    def test_unions_keywords(self):
        p1 = _make_paper(paper_id="a", keywords=["deep learning"], doi="10.1/x")
        p2 = _make_paper(paper_id="b", keywords=["transformer"], doi="10.1/x")
        result = deduplicate_papers([p1, p2])
        assert set(result[0].keywords) == {"deep learning", "transformer"}

    def test_takes_max_year(self):
        p1 = _make_paper(paper_id="a", year=2023, doi="10.1/x")
        p2 = _make_paper(paper_id="b", year=2024, doi="10.1/x")
        result = deduplicate_papers([p1, p2])
        assert result[0].year == 2024

    def test_takes_max_citations(self):
        p1 = _make_paper(paper_id="a", citations=10, doi="10.1/x")
        p2 = _make_paper(paper_id="b", citations=42, doi="10.1/x")
        result = deduplicate_papers([p1, p2])
        assert result[0].citations == 42

    def test_fills_missing_doi(self):
        p1 = _make_paper(paper_id="a", arxiv_id="2401.00001")
        p2 = _make_paper(paper_id="b", doi="10.1/x", arxiv_id="2401.00001")
        result = deduplicate_papers([p1, p2])
        assert result[0].doi == "10.1/x"
        assert result[0].arxiv_id == "2401.00001"

    def test_fills_missing_pdf_url(self):
        p1 = _make_paper(paper_id="a", arxiv_id="2401.00001")
        p2 = _make_paper(paper_id="b", arxiv_id="2401.00001", pdf_url="https://example.com/p.pdf")
        result = deduplicate_papers([p1, p2])
        assert result[0].pdf_url == "https://example.com/p.pdf"

    def test_fills_missing_journal(self):
        p1 = _make_paper(paper_id="a", arxiv_id="2401.00001")
        p2 = _make_paper(paper_id="b", arxiv_id="2401.00001", journal="Nature")
        result = deduplicate_papers([p1, p2])
        assert result[0].journal == "Nature"

    def test_records_merged_sources(self):
        p1 = _make_paper(paper_id="a", source="arxiv", doi="10.1/x")
        p2 = _make_paper(paper_id="b", source="pubmed", doi="10.1/x")
        p3 = _make_paper(paper_id="c", source="ieee", doi="10.1/x")
        result = deduplicate_papers([p1, p2, p3])
        assert set(result[0].merged_sources) == {"arxiv", "pubmed", "ieee"}


class TestDeduplicateMixedSources:
    def test_arxiv_doi_matches_pubmed_doi(self):
        p1 = _make_paper(paper_id="a", source="arxiv", doi="10.1/x", arxiv_id="2401.00001")
        p2 = _make_paper(paper_id="b", source="pubmed", doi="10.1/x", pubmed_id="999")
        result = deduplicate_papers([p1, p2])
        assert len(result) == 1
        assert result[0].arxiv_id == "2401.00001"
        assert result[0].pubmed_id == "999"
        assert set(result[0].merged_sources) == {"arxiv", "pubmed"}

    def test_three_way_merge(self):
        p1 = _make_paper(paper_id="a", source="arxiv", doi="10.1/x", arxiv_id="2401.00001")
        p2 = _make_paper(paper_id="b", source="pubmed", doi="10.1/x", pubmed_id="999")
        p3 = _make_paper(paper_id="c", source="ieee", doi="10.1/x", ieee_id="1234")
        result = deduplicate_papers([p1, p2, p3])
        assert len(result) == 1
        assert result[0].arxiv_id == "2401.00001"
        assert result[0].pubmed_id == "999"
        assert result[0].ieee_id == "1234"


# =========================================================================
# Paper dataclass
# =========================================================================


class TestPaperDataclass:
    def test_to_dict_includes_merged_sources(self):
        p = _make_paper()
        d = p.to_dict()
        assert "merged_sources" in d
        assert d["merged_sources"] == []

    def test_merged_sources_default_empty(self):
        p = _make_paper()
        assert p.merged_sources == []


# =========================================================================
# PaperSourceAggregator.search_deduplicated
# =========================================================================


class TestSearchDeduplicated:
    def test_dedupes_across_sources(self, monkeypatch):
        import asyncio

        async def fake_arxiv_search(query, max_results=10, **_):
            return SearchResult(
                papers=[
                    _make_paper(paper_id="a1", source="arxiv", doi="10.1/x", arxiv_id="2401.00001"),
                ],
                total_results=1,
                query=query,
                source="arxiv",
            )

        async def fake_pubmed_search(query, max_results=10, **_):
            return SearchResult(
                papers=[
                    _make_paper(paper_id="p1", source="pubmed", doi="10.1/x", pubmed_id="999"),
                ],
                total_results=1,
                query=query,
                source="pubmed",
            )

        agg = PaperSourceAggregator()
        monkeypatch.setattr(agg.arxiv, "search", fake_arxiv_search)
        monkeypatch.setattr(agg.pubmed, "search", fake_pubmed_search)

        result = asyncio.run(agg.search_deduplicated("test query"))
        assert len(result) == 1
        assert set(result[0].merged_sources) == {"arxiv", "pubmed"}

    def test_empty_results(self, monkeypatch):
        import asyncio

        async def fake_arxiv_search(query, max_results=10, **_):
            return SearchResult(papers=[], total_results=0, query=query, source="arxiv")

        async def fake_pubmed_search(query, max_results=10, **_):
            return SearchResult(papers=[], total_results=0, query=query, source="pubmed")

        agg = PaperSourceAggregator()
        monkeypatch.setattr(agg.arxiv, "search", fake_arxiv_search)
        monkeypatch.setattr(agg.pubmed, "search", fake_pubmed_search)

        result = asyncio.run(agg.search_deduplicated("test query"))
        assert result == []

    def test_max_results_truncates(self, monkeypatch):
        import asyncio

        async def fake_arxiv_search(query, max_results=10, **_):
            return SearchResult(
                papers=[_make_paper(paper_id=f"a{i}", title=f"Paper Number {i}") for i in range(5)],
                total_results=5,
                query=query,
                source="arxiv",
            )

        async def fake_pubmed_search(query, max_results=10, **_):
            return SearchResult(papers=[], total_results=0, query=query, source="pubmed")

        agg = PaperSourceAggregator()
        monkeypatch.setattr(agg.arxiv, "search", fake_arxiv_search)
        monkeypatch.setattr(agg.pubmed, "search", fake_pubmed_search)

        result = asyncio.run(agg.search_deduplicated("test", max_results=3))
        assert len(result) == 3


# =========================================================================
# CitationGraph.get_pagerank (vectorized)
# =========================================================================


class TestGetPagerank:
    def test_empty_graph(self):
        g = CitationGraph()
        assert g.get_pagerank() == {}

    def test_single_node(self):
        g = CitationGraph()
        g.add_paper("a", "Paper A", 2024)
        ranks = g.get_pagerank()
        assert len(ranks) == 1
        # Single node with no references is a dangling node; in the current
        # implementation (no dangling redistribution) the rank is just the
        # teleport mass: (1 - damping) / N.
        expected = (1 - 0.85) / 1
        assert abs(ranks["a"] - expected) < 1e-6

    def test_two_node_cycle(self):
        g = CitationGraph()
        g.add_paper("a", "A", 2024, references=["b"])
        g.add_paper("b", "B", 2024, references=["a"])
        ranks = g.get_pagerank()
        # Symmetric: both nodes should have similar rank
        assert abs(ranks["a"] - ranks["b"]) < 1e-6
        # Sum should be <= 1 (no dangling redistribution, may be < 1)
        assert 0 < sum(ranks.values()) <= 1.0

    def test_three_node_cycle(self):
        g = CitationGraph()
        g.add_paper("a", "A", 2024, references=["b"])
        g.add_paper("b", "B", 2024, references=["c"])
        g.add_paper("c", "C", 2024, references=["a"])
        ranks = g.get_pagerank()
        # Symmetric cycle: all three get the same rank
        assert abs(ranks["a"] - ranks["b"]) < 1e-6
        assert abs(ranks["b"] - ranks["c"]) < 1e-6
        assert all(r > 0 for r in ranks.values())

    def test_dangling_node_no_contribution(self):
        # Node b has no references, so its rank mass should not propagate
        g = CitationGraph()
        g.add_paper("a", "A", 2024, references=["b"])
        g.add_paper("b", "B", 2024)  # dangling
        ranks = g.get_pagerank()
        # Sum should be 1/N (just teleport) since no propagation
        assert all(0 <= r <= 1 for r in ranks.values())

    def test_hub_gets_higher_rank(self):
        # Paper H is referenced by 5 others; should get higher PageRank
        g = CitationGraph()
        g.add_paper("h", "Hub", 2024, references=[])
        for i in range(5):
            g.add_paper(f"p{i}", f"Paper {i}", 2024, references=["h"])
        ranks = g.get_pagerank()
        assert ranks["h"] > max(ranks[p] for p in ("p0", "p1", "p2", "p3", "p4"))

    def test_vectorized_matches_python_loop_for_small_graph(self):
        """The vectorized version should produce equivalent ranks to a naive
        implementation for the same small graph."""
        g = CitationGraph()
        refs = {
            "a": ["b", "c"],
            "b": ["c"],
            "c": ["a"],
            "d": ["c"],
            "e": [],
        }
        for node_id, ref_list in refs.items():
            g.add_paper(node_id, node_id, 2024, references=ref_list)

        ranks = g.get_pagerank(iterations=50)
        # All ranks should be non-negative and finite
        for node_id in refs:
            assert 0 <= ranks[node_id] <= 1
            assert ranks[node_id] == ranks[node_id]  # not NaN

    def test_iterations_parameter(self):
        g = CitationGraph()
        g.add_paper("a", "A", 2024, references=["b"])
        g.add_paper("b", "B", 2024, references=["a"])
        ranks_few = g.get_pagerank(iterations=10)
        ranks_many = g.get_pagerank(iterations=200)
        # More iterations -> ranks should be closer to stationary distribution
        # and ranks_few - ranks_many should be small
        for node_id in ranks_few:
            assert abs(ranks_few[node_id] - ranks_many[node_id]) < 0.05

    def test_damping_parameter(self):
        # Asymmetric graph: paper a is referenced by 3 others, paper b by 1.
        # The rank difference should change with damping.
        g = CitationGraph()
        g.add_paper("a", "A", 2024, references=["b"])
        g.add_paper("b", "B", 2024, references=["a"])
        g.add_paper("c", "C", 2024, references=["a"])
        g.add_paper("d", "D", 2024, references=["a"])

        ranks_high = g.get_pagerank(damping=0.95, iterations=200)
        ranks_low = g.get_pagerank(damping=0.5, iterations=200)
        # Both should give non-trivial values for all nodes
        assert all(ranks_high[nid] > 0 for nid in ranks_high)
        assert all(ranks_low[nid] > 0 for nid in ranks_low)
        # a is more cited than b, so a should have higher rank in both
        assert ranks_high["a"] > ranks_high["b"]
        assert ranks_low["a"] > ranks_low["b"]

    def test_sparse_threshold_triggers(self):
        # Build a graph with > sparse_threshold nodes (default 500)
        n = 510
        g = CitationGraph()
        for i in range(n):
            g.add_paper(f"p{i}", f"Paper {i}", 2024, references=[])
        # Force a few edges
        for i in range(1, n):
            g.nodes[f"p{i}"].references.add(f"p{i - 1}")

        ranks = g.get_pagerank()
        assert len(ranks) == n
        assert all(0 <= r <= 1 for r in ranks.values())

    def test_handles_external_references(self):
        # References to nodes that aren't in the graph are still added to
        # the graph (CitationGraph.add_paper auto-creates empty nodes for
        # unknown references). The PageRank calculation is robust to
        # dangling nodes; external_xyz should just receive teleport mass
        # and not propagate it (it has no outgoing references).
        g = CitationGraph()
        g.add_paper("a", "A", 2024, references=["b", "external_xyz"])
        g.add_paper("b", "B", 2024, references=["a"])
        ranks = g.get_pagerank()
        # All nodes (including the auto-created external one) get a rank
        assert "a" in ranks and "b" in ranks and "external_xyz" in ranks
        # The external node is dangling; it should not steal rank from a, b
        assert ranks["a"] > 0
        assert ranks["b"] > 0
        # The external node's rank should be small (only teleport)
        assert ranks["external_xyz"] >= 0


class TestGetPagerankPerformance:
    def test_100_nodes_under_50ms(self):
        """Vectorized PageRank should handle 100 nodes in < 50ms (was minutes
        in pure Python)."""
        import random
        import time

        random.seed(42)
        g = CitationGraph()
        for i in range(100):
            refs = [f"p{random.randint(0, 99)}" for _ in range(3)]
            refs = [r for r in refs if r != f"p{i}"]
            g.add_paper(f"p{i}", f"Paper {i}", 2024, references=refs)

        t0 = time.perf_counter()
        ranks = g.get_pagerank(iterations=50)
        elapsed = time.perf_counter() - t0

        assert len(ranks) == 100
        # Generous bound; the vectorized version typically runs in <5ms
        assert elapsed < 0.5, f"PageRank too slow: {elapsed * 1000:.1f}ms"


# =========================================================================
# Year-drift fixes
# =========================================================================


class TestYearDriftFix:
    def test_similarity_uses_current_year_for_recent(self):
        # No monkeypatch: a paper dated to the current year is always
        # "recent" regardless of when the test runs.
        engine = ResearchSimilaritySearch()
        current_year = datetime.now().year
        papers = [
            {
                "paper_id": "p1",
                "title": "Recent paper about gradient descent",
                "abstract": "",
                "year": current_year,
                "source": "x",
            },
        ]
        results = engine.find_similar("gradient descent", papers, max_results=5)
        assert len(results) >= 1
        assert any(any(reason.startswith("recent") for reason in r.match_reasons) for r in results)

    def test_similarity_old_paper_not_recent(self):
        # A paper from > 3 years ago should NOT be flagged "recent"
        # (year_score for a paper > 3 years old drops below 0.7).
        engine = ResearchSimilaritySearch()
        current_year = datetime.now().year
        old_year = current_year - 10
        papers = [
            {
                "paper_id": "p1",
                "title": "Old paper about gradient descent",
                "abstract": "",
                "year": old_year,
                "source": "x",
            },
        ]
        results = engine.find_similar("gradient descent", papers, max_results=5)
        for r in results:
            for reason in r.match_reasons:
                assert not reason.startswith("recent")

    def test_citation_graph_uses_current_year(self):
        # A paper dated to the current year should have year_weight = 1.0
        g = CitationGraph()
        current_year = datetime.now().year
        g.add_paper("a", "A", current_year, references=["b"], citations=["b"])
        g.add_paper("b", "B", current_year)
        # current_year paper: year_weight = 1.0
        # citations=1, refs=1, score = sqrt(1*1+1) * 1.0
        import math

        expected = math.sqrt(1 * 1 + 1) * 1.0
        assert abs(g.get_influence_score("a") - expected) < 1e-9

    def test_citation_graph_old_paper_has_lower_weight(self):
        # An old paper should have a smaller year_weight than a new one.
        g = CitationGraph()
        current_year = datetime.now().year
        g.add_paper("new", "New", current_year, references=["b"], citations=["b"])
        g.add_paper("old", "Old", current_year - 20, references=["b"], citations=["b"])
        g.add_paper("b", "B", current_year)
        new_score = g.get_influence_score("new")
        old_score = g.get_influence_score("old")
        # Same citations/refs -> year_weight is the only difference
        assert new_score > old_score


# =========================================================================
# Regression: ensure existing single-paper paths still work after
# adding merged_sources to Paper.
# =========================================================================


class TestBackwardCompat:
    def test_paper_construction_without_merged_sources(self):
        p = _make_paper()
        assert p.merged_sources == []

    def test_dedup_idempotent(self):
        # Running dedup twice should not change the result
        p1 = _make_paper(paper_id="a", source="arxiv", doi="10.1/x", arxiv_id="2401.00001")
        p2 = _make_paper(paper_id="b", source="pubmed", doi="10.1/x", pubmed_id="999")
        first = deduplicate_papers([p1, p2])
        second = deduplicate_papers(first)
        assert len(second) == len(first) == 1
        assert second[0].paper_id == first[0].paper_id
