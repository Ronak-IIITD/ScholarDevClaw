from __future__ import annotations

import pytest

from scholardevclaw.research_intelligence.extractor import (
    ResearchExtractionError,
    ResearchExtractor,
)


def test_extract_unknown_pdf_no_longer_fabricates_rmsnorm():
    extractor = ResearchExtractor(llm_assistant=None)

    with pytest.raises(ResearchExtractionError) as exc_info:
        extractor.extract("/tmp/definitely_missing_unknown.pdf", "pdf")

    meta = exc_info.value.to_error_metadata()
    assert meta["error"] == "extraction_failed"
    assert meta["source_type"] == "pdf"
    assert "RMSNorm" not in meta["message"]


def test_extract_unknown_arxiv_no_longer_fabricates_rmsnorm():
    extractor = ResearchExtractor(llm_assistant=None)

    with pytest.raises(ResearchExtractionError) as exc_info:
        extractor.extract("9999.99999", "arxiv")

    meta = exc_info.value.to_error_metadata()
    assert meta["error"] == "extraction_failed"
    assert meta["source_type"] == "arxiv"
    assert meta["reason"] in {"arxiv_metadata_unavailable", "llm_unavailable"}


def test_extract_dynamic_arxiv_uses_heuristic_fallback_and_cache(monkeypatch, tmp_path):
    record = {
        "title": "Flash Widget Attention for Tiny Models",
        "abstract": "We introduce a new attention mechanism for compact transformers.",
        "authors": ["Ada Lovelace"],
        "arxiv_id": "2410.12345",
        "year": 2026,
        "pdf_url": "https://export.arxiv.org/pdf/2410.12345.pdf",
    }

    monkeypatch.setattr(
        "scholardevclaw.research_intelligence.extractor._fetch_arxiv_record",
        lambda _arxiv_id: record,
    )
    monkeypatch.setattr(
        "scholardevclaw.research_intelligence.extractor._dynamic_spec_cache_dir",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        ResearchExtractor,
        "_attach_reference_implementations",
        lambda self, spec: None,
    )

    extractor = ResearchExtractor(llm_assistant=None)
    spec = extractor.extract("arxiv:2410.12345", "arxiv")

    assert spec["paper"]["arxiv"] == "2410.12345"
    assert spec["algorithm"]["category"] == "attention"
    assert tmp_path.joinpath("2410.12345.json").exists()

    fresh_extractor = ResearchExtractor(llm_assistant=None)
    cached = fresh_extractor.get_spec("arxiv:2410.12345")
    assert cached is not None
    assert cached["paper"]["title"] == record["title"]


def test_search_by_keyword_uses_semantic_search_when_enabled(monkeypatch):
    class FakeIndex:
        def search(self, query: str, top_k: int = 5):
            return [
                {
                    "paper": {"title": "RoFormer", "authors": [], "arxiv": "2104.09864", "year": 2021},
                    "algorithm": {
                        "name": "RoPE",
                        "category": "position_encoding",
                        "replaces": "Positional Encoding",
                        "description": "Rotary Position Embedding",
                    },
                    "_semantic_score": 0.91,
                }
            ]

    monkeypatch.setenv("USE_SEMANTIC_SEARCH", "true")
    monkeypatch.setattr(ResearchExtractor, "_get_semantic_index", lambda self: FakeIndex())

    extractor = ResearchExtractor(llm_assistant=None)
    results = extractor.search_by_keyword("rotary embeddings", max_results=3)

    assert results
    assert results[0]["name"] == "rope"
    assert results[0]["source"] == "semantic_search"
