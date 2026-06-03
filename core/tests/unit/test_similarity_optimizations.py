"""Tests for ResearchSimilaritySearch performance optimizations."""

from __future__ import annotations

import pytest

from scholardevclaw.research_intelligence.similarity import ResearchSimilaritySearch


class TestFindSimilarOptimized:
    """Tests verifying find_similar produces correct results after optimization."""

    @pytest.fixture
    def finder(self) -> ResearchSimilaritySearch:
        return ResearchSimilaritySearch()

    @pytest.fixture
    def sample_papers(self) -> list[dict]:
        return [
            {
                "paper_id": "arxiv:0001",
                "title": "Efficient Transformer for Deep Learning",
                "abstract": "We propose an efficient transformer architecture that reduces computation cost.",
                "year": 2024,
                "source": "arxiv",
            },
            {
                "paper_id": "arxiv:0002",
                "title": "Image Classification with CNNs",
                "abstract": "A convolutional approach to image recognition tasks.",
                "year": 2023,
                "source": "arxiv",
            },
            {
                "paper_id": "arxiv:0003",
                "title": "Transformer Optimization Techniques",
                "abstract": "Various methods to optimize transformer performance and efficiency.",
                "year": 2025,
                "source": "arxiv",
            },
            {
                "paper_id": "arxiv:0004",
                "title": "Quantum Computing Basics",
                "abstract": "An introduction to quantum bits and quantum gates.",
                "year": 2020,
                "source": "arxiv",
            },
        ]

    def test_returns_ranked_results(self, finder, sample_papers):
        results = finder.find_similar(
            "transformer deep learning optimization", sample_papers, max_results=10
        )
        assert len(results) > 0
        for i in range(len(results) - 1):
            assert results[i].similarity_score >= results[i + 1].similarity_score

    def test_higher_relevance_ranks_first(self, finder, sample_papers):
        results = finder.find_similar(
            "transformer deep learning optimization", sample_papers, max_results=10
        )
        top_ids = {r.paper_id for r in results[:2]}
        assert "arxiv:0001" in top_ids or "arxiv:0003" in top_ids

    def test_max_results_respected(self, finder, sample_papers):
        results = finder.find_similar("deep learning", sample_papers, max_results=2)
        assert len(results) <= 2

    def test_empty_papers_returns_empty(self, finder):
        results = finder.find_similar("transformer", [], max_results=10)
        assert results == []

    def test_empty_query_does_not_crash(self, finder, sample_papers):
        results = finder.find_similar("", sample_papers, max_results=10)
        assert isinstance(results, list)

    def test_keyword_score_in_reasons(self, finder, sample_papers):
        results = finder.find_similar("transformer", sample_papers, max_results=4)
        assert len(results) >= 1
        # Both transformer papers should have keyword_match in their reasons
        keyword_results = [
            r for r in results if any("keyword_match" in reason for reason in r.match_reasons)
        ]
        assert len(keyword_results) >= 2

    def test_tfidf_score_in_reasons(self, finder, sample_papers):
        results = finder.find_similar(
            "efficient transformer architecture", sample_papers, max_results=4
        )
        assert len(results) >= 1
        has_tfidf = any("tfidf" in reason for r in results for reason in r.match_reasons)
        assert has_tfidf

    def test_year_score_favors_recent(self, finder, sample_papers):
        results = finder.find_similar("transformer", sample_papers, max_results=4)
        recent_results = [
            r for r in results if any("recent" in reason for reason in r.match_reasons)
        ]
        assert len(recent_results) >= 1
        recent_years = {r.year for r in recent_results}
        assert 2025 in recent_years or 2024 in recent_years

    def test_use_tfidf_false_skips_tfidf(self, finder, sample_papers):
        results = finder.find_similar("transformer", sample_papers, max_results=1, use_tfidf=False)
        assert len(results) >= 1
        has_tfidf = any("tfidf" in r.match_reasons for r in results)
        assert not has_tfidf

    def test_paper_id_preserved(self, finder, sample_papers):
        results = finder.find_similar("transformer", sample_papers, max_results=4)
        paper_ids = {r.paper_id for r in results}
        assert "arxiv:0001" in paper_ids
        assert "arxiv:0003" in paper_ids

    def test_large_batch_no_crash(self, finder):
        papers = [
            {
                "paper_id": f"arxiv:{i:04d}",
                "title": f"Paper {i} about deep learning",
                "abstract": f"Deep learning methods for task {i}",
                "year": 2020 + (i % 6),
                "source": "arxiv",
            }
            for i in range(500)
        ]
        results = finder.find_similar("deep learning", papers, max_results=10)
        assert len(results) == 10

    def test_keyword_only_no_tfidf(self, finder):
        """Keyword overlap produces non-zero score even with use_tfidf=False."""
        papers = [
            {
                "paper_id": "p1",
                "title": "transformer optimization",
                "abstract": "we optimize transformers",
                "year": 2024,
                "source": "arxiv",
            },
        ]
        results = finder.find_similar("transformer", papers, max_results=1, use_tfidf=False)
        assert len(results) == 1
        assert results[0].similarity_score > 0
