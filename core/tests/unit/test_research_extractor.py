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
    assert meta["reason"] == "llm_unavailable"
