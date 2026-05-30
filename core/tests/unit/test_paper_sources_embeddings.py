"""Tests for paper_sources and research embeddings modules."""

from pathlib import Path

import numpy as np
import pytest

from scholardevclaw.research_intelligence.embeddings import (
    EmbeddingIndex,
    _cosine_similarity,
    _spec_identifier,
    _spec_text,
    _tokenize,
)
from scholardevclaw.research_intelligence.paper_sources import (
    ArxivSource,
    IEEESource,
    Paper,
    PaperSourceAggregator,
    PubmedSource,
    SearchResult,
    _allowed_fixed_source_url,
    get_paper_source,
)

# =========================================================================
# paper_sources — Paper dataclass
# =========================================================================


class TestPaper:
    def test_construction(self):
        p = Paper(
            paper_id="1234",
            title="Test Paper",
            authors=["Alice"],
            abstract="An abstract",
            year=2024,
            source="arxiv",
            url="https://arxiv.org/abs/1234",
        )
        assert p.paper_id == "1234"
        assert p.title == "Test Paper"

    def test_to_dict(self):
        p = Paper(
            paper_id="1234",
            title="Test Paper",
            authors=["Alice"],
            abstract="An abstract",
            year=2024,
            source="arxiv",
            url="https://arxiv.org/abs/1234",
            doi="10.1234/test",
        )
        d = p.to_dict()
        assert d["paper_id"] == "1234"
        assert d["doi"] == "10.1234/test"
        assert d["keywords"] == []

    def test_defaults(self):
        p = Paper(
            paper_id="x",
            title="X",
            authors=[],
            abstract="",
            year=2024,
            source="arxiv",
            url="",
        )
        assert p.month == 1
        assert p.citations == 0
        assert p.categories == []


# =========================================================================
# paper_sources — SearchResult
# =========================================================================


class TestSearchResult:
    def test_construction(self):
        r = SearchResult(papers=[], total_results=0, query="test", source="arxiv")
        assert r.source == "arxiv"


# =========================================================================
# paper_sources — URL validation
# =========================================================================


class TestAllowedFixedSourceUrl:
    def test_arxiv(self):
        assert _allowed_fixed_source_url("https://export.arxiv.org/api/query?test=1") is True

    def test_pubmed(self):
        assert (
            _allowed_fixed_source_url("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi")
            is True
        )

    def test_ieee(self):
        assert (
            _allowed_fixed_source_url("https://ieeexploreapi.ieee.org/api/v1/search/articles")
            is True
        )

    def test_http_blocked(self):
        assert _allowed_fixed_source_url("http://export.arxiv.org/api/query") is False

    def test_unknown_host_blocked(self):
        assert _allowed_fixed_source_url("https://evil.com/steal") is False


# =========================================================================
# paper_sources — Factory
# =========================================================================


class TestGetPaperSource:
    def test_arxiv(self):
        assert isinstance(get_paper_source("arxiv"), ArxivSource)

    def test_pubmed(self):
        assert isinstance(get_paper_source("pubmed"), PubmedSource)

    def test_ieee(self):
        assert isinstance(get_paper_source("ieee"), IEEESource)

    def test_unknown(self):
        with pytest.raises(ValueError, match="Unknown source"):
            get_paper_source("scopus")


# =========================================================================
# paper_sources — ArxivSource parsing
# =========================================================================


class TestArxivSourceParsing:
    SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>arXiv:Attention</title>
      <entry>
        <id>http://arxiv.org/abs/2301.12345v1</id>
        <title>Attention Is All You Need</title>
        <summary>We propose a new architecture...</summary>
        <author><name>Vaswani</name></author>
        <author><name>Shazeer</name></author>
        <published>2017-06-12T00:00:00Z</published>
        <category term="cs.CL"/>
        <category term="cs.LG"/>
        <link title="pdf" href="https://arxiv.org/pdf/2301.12345v1"/>
      </entry>
    </feed>"""

    def test_parse_atom(self):
        src = ArxivSource()
        result = src._parse_atom(self.SAMPLE_ATOM, "attention")
        assert len(result.papers) == 1
        p = result.papers[0]
        assert p.title == "Attention Is All You Need"
        assert p.authors == ["Vaswani", "Shazeer"]
        assert p.year == 2017
        assert p.source == "arxiv"
        assert p.arxiv_id == "2301.12345"
        assert "cs.CL" in p.categories
        assert p.pdf_url == "https://arxiv.org/pdf/2301.12345v1"

    def test_parse_atom_empty(self):
        src = ArxivSource()
        result = src._parse_atom(
            '<feed xmlns="http://www.w3.org/2005/Atom"><title>x</title></feed>',
            "query",
        )
        assert result.papers == []

    def test_clean_text(self):
        src = ArxivSource()
        assert src._clean_text("  hello   world  ") == "hello world"


# =========================================================================
# paper_sources — PubmedSource parsing
# =========================================================================


class TestPubmedSourceParsing:
    SAMPLE_XML = """<?xml version="1.0"?>
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>12345</PMID>
          <Article>
            <ArticleTitle>Gene Expression Analysis</ArticleTitle>
            <Abstract><AbstractText>We analyzed gene expression...</AbstractText></Abstract>
            <AuthorList>
              <Author>
                <ForeName>John</ForeName>
                <LastName>Smith</LastName>
              </Author>
            </AuthorList>
            <Journal><Title>Nature Genetics</Title></Journal>
            <PubDate><Year>2023</Year><Month>Mar</Month></PubDate>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>"""

    def test_parse_pubmed_xml(self):
        src = PubmedSource()
        papers = src._parse_pubmed_xml(self.SAMPLE_XML, "gene expression")
        assert len(papers) == 1
        p = papers[0]
        assert p.paper_id == "12345"
        assert p.title == "Gene Expression Analysis"
        assert p.authors == ["John Smith"]
        assert p.year == 2023
        assert p.month == 3
        assert p.source == "pubmed"
        assert p.journal == "Nature Genetics"
        assert "pubmed.ncbi.nlm.nih.gov/12345" in p.url

    def test_parse_pubmed_xml_empty(self):
        src = PubmedSource()
        papers = src._parse_pubmed_xml("<PubmedArticleSet/>", "query")
        assert papers == []


# =========================================================================
# paper_sources — IEEESource
# =========================================================================


class TestIEEESource:
    def test_no_api_key_returns_empty(self):
        src = IEEESource(api_key=None)

        async def _run():
            result = await src.search("test", max_results=5)
            return result

        import asyncio

        result = asyncio.run(_run())
        assert result.papers == []
        assert result.source == "ieee"

    def test_factory_with_key(self):
        src = IEEESource(api_key="test123")
        assert src.api_key == "test123"


# =========================================================================
# paper_sources — PaperSourceAggregator
# =========================================================================


class TestPaperSourceAggregator:
    def test_construction(self):
        agg = PaperSourceAggregator()
        assert agg.arxiv is not None
        assert agg.pubmed is not None
        assert agg.ieee is None

    def test_set_ieee_key(self):
        agg = PaperSourceAggregator()
        agg.set_ieee_key("key123")
        assert agg.ieee is not None
        assert agg.ieee.api_key == "key123"


# =========================================================================
# embeddings — Utility functions
# =========================================================================


class TestEmbeddingUtils:
    def test_tokenize(self):
        tokens = _tokenize("Hello World 123")
        assert "hello" in tokens
        assert "world" in tokens
        assert "123" in tokens

    def test_tokenize_empty(self):
        assert _tokenize("") == []

    def test_cosine_similarity_identical(self):
        v = np.array([1.0, 0.0, 0.0])
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_empty(self):
        assert _cosine_similarity(np.array([]), np.array([])) == 0.0

    def test_cosine_similarity_different_shape(self):
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert _cosine_similarity(a, b) == 0.0

    def test_spec_identifier_arxiv(self):
        spec = {"paper": {"arxiv": "2301.12345"}}
        assert _spec_identifier(spec, 0) == "2301.12345"  # only "/" replaced with "_"

    def test_spec_identifier_title(self):
        spec = {"paper": {"title": "Attention Is All You Need"}}
        ident = _spec_identifier(spec, 0)
        assert ident.startswith("title-")

    def test_spec_identifier_ordinal(self):
        spec = {"paper": {}}
        assert _spec_identifier(spec, 5) == "spec-5"

    def test_spec_text(self):
        spec = {
            "paper": {"title": "Test", "abstract": "An abstract"},
            "algorithm": {"name": "test-algo", "category": "attention"},
            "changes": {"target_patterns": ["layer_norm"], "replacement": "rmsnorm"},
            "implementation": {"parameters": ["eps=1e-6"]},
        }
        text = _spec_text(spec)
        assert "Test" in text
        assert "attention" in text
        assert "rmsnorm" in text

    def test_spec_text_empty(self):
        assert _spec_text({}) == ""


# =========================================================================
# embeddings — EmbeddingIndex
# =========================================================================


class TestEmbeddingIndex:
    def test_construction(self, tmp_path: Path):
        idx = EmbeddingIndex(cache_dir=tmp_path)
        assert idx.backend in ("hash", "sentence-transformers")
        assert idx.dimension == 768

    def test_encode_returns_correct_shape(self, tmp_path: Path):
        idx = EmbeddingIndex(cache_dir=tmp_path)
        vec = idx.encode("attention transformer model")
        assert vec.shape == (768,)
        assert vec.dtype == np.float32

    def test_encode_empty(self, tmp_path: Path):
        idx = EmbeddingIndex(cache_dir=tmp_path)
        vec = idx.encode("")
        assert np.linalg.norm(vec) == 0.0

    def test_encode_same_text_same_vector(self, tmp_path: Path):
        idx = EmbeddingIndex(cache_dir=tmp_path)
        v1 = idx.encode("test text")
        v2 = idx.encode("test text")
        np.testing.assert_array_equal(v1, v2)

    def test_encode_different_text_different_vectors(self, tmp_path: Path):
        idx = EmbeddingIndex(cache_dir=tmp_path)
        v1 = idx.encode("attention mechanism")
        v2 = idx.encode("image segmentation")
        assert not np.array_equal(v1, v2)

    def test_index_and_search(self, tmp_path: Path):
        idx = EmbeddingIndex(cache_dir=tmp_path)
        specs = [
            {"paper": {"title": "Attention Is All You Need"}, "algorithm": {"name": "attention"}},
            {"paper": {"title": "Image Classification"}, "algorithm": {"name": "cnn"}},
        ]
        idx.index(specs)
        results = idx.search("attention transformer", top_k=2)
        assert len(results) > 0
        assert "_semantic_score" in results[0]

    def test_index_empty(self, tmp_path: Path):
        idx = EmbeddingIndex(cache_dir=tmp_path)
        idx.index([])
        assert idx.search("test") == []

    def test_search_empty_query(self, tmp_path: Path):
        idx = EmbeddingIndex(cache_dir=tmp_path)
        idx.index([{"paper": {"title": "test"}}])
        assert idx.search("") == []

    def test_similarity(self, tmp_path: Path):
        idx = EmbeddingIndex(cache_dir=tmp_path)
        score = idx.similarity("attention mechanism", "attention transformer")
        assert 0 < score <= 1.0

    def test_similarity_identical(self, tmp_path: Path):
        idx = EmbeddingIndex(cache_dir=tmp_path)
        score = idx.similarity("test text", "test text")
        assert score == pytest.approx(1.0)

    def test_cache_roundtrip(self, tmp_path: Path):
        idx = EmbeddingIndex(cache_dir=tmp_path)
        idx.index([{"paper": {"title": "Cached Paper"}}])
        # Rebuild index — should use cached vectors
        idx2 = EmbeddingIndex(cache_dir=tmp_path)
        idx2.index([{"paper": {"title": "Cached Paper"}}])
        r1 = idx.search("cached paper", top_k=1)
        r2 = idx2.search("cached paper", top_k=1)
        assert len(r1) > 0 and len(r2) > 0
