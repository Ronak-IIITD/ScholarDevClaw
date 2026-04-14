from __future__ import annotations

import os
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from scholardevclaw.ingestion.models import (
    Algorithm,
    Equation,
    Figure,
    PaperDocument,
    Section,
)
from scholardevclaw.ingestion.paper_fetcher import (
    PaperFetcher,
    PaperIngester,
    PaperNotAccessibleError,
)


def _build_sample_document(pdf_path: Path) -> PaperDocument:
    return PaperDocument(
        title="Sample Paper",
        authors=["Alice", "Bob"],
        arxiv_id="1234.56789",
        doi="10.1000/sample",
        year=2024,
        abstract="A short abstract",
        sections=[Section(title="Intro", level=1, content="Hello", page_start=1)],
        equations=[Equation(latex="x = y + z", description="Example", page=2)],
        algorithms=[
            Algorithm(
                name="Algorithm 1: Train",
                pseudocode="for step in steps:\n    update()",
                page=3,
                language_hint="python-like",
            )
        ],
        figures=[Figure(caption="Figure caption", page=1, image_path=pdf_path.parent / "f.png")],
        full_text="Full text",
        pdf_path=pdf_path,
        references=["[1] A citation"],
        keywords=["transformer", "attention"],
        domain="nlp",
    )


def test_paper_document_round_trip_serialization(tmp_path: Path) -> None:
    doc = _build_sample_document(tmp_path / "paper.pdf")

    serialized = doc.to_dict()
    restored = PaperDocument.from_dict(serialized)

    assert restored == doc
    assert asdict(restored) == asdict(doc)


def test_pdf_parser_extracts_algorithms_and_equations(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    from scholardevclaw.ingestion.pdf_parser import PDFParser

    pdf_path = tmp_path / "synthetic.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Synthetic Transformers")
    page.insert_text((72, 110), "Abstract")
    page.insert_text((72, 130), "We study attention mechanisms in depth.")
    page.insert_text((72, 160), "Algorithm 1: Training Procedure")
    page.insert_text((72, 180), "for step in range(T):")
    page.insert_text((72, 200), "  x = x + y")
    page.insert_text((72, 230), "$a = b + c$")
    page.insert_text((72, 250), "x = y / z + 1")
    doc.save(str(pdf_path))
    doc.close()

    parser = PDFParser()
    parsed = parser.parse(pdf_path)

    assert parsed.title
    assert parsed.algorithms
    assert any("Algorithm" in algorithm.name for algorithm in parsed.algorithms)
    assert parsed.equations
    assert any("=" in equation.latex for equation in parsed.equations)


def test_fetch_by_doi_raises_when_no_open_access_pdf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fetcher = PaperFetcher()

    monkeypatch.setattr(
        fetcher,
        "_fetch_semantic_scholar_metadata",
        lambda _doi: {
            "title": "Closed Paper",
            "authors": [{"name": "A"}],
            "year": 2022,
            "abstract": "Closed abstract",
            "openAccessPdf": None,
        },
    )

    with pytest.raises(PaperNotAccessibleError):
        fetcher.fetch_by_doi("10.1000/closed", tmp_path)


def test_fetch_auto_prefers_local_pdf_path(tmp_path: Path) -> None:
    pdf_path = tmp_path / "local.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")

    expected_doc = _build_sample_document(pdf_path)

    class StubParser:
        def parse(self, path: Path) -> PaperDocument:
            assert path == pdf_path
            return expected_doc

    fetcher = PaperFetcher(parser=StubParser())

    result = fetcher.fetch_auto(str(pdf_path), tmp_path / "out")

    assert result == expected_doc


def test_fetch_by_url_uses_html_pdf_discovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected_doc = _build_sample_document(tmp_path / "downloaded.pdf")

    class StubParser:
        def parse(self, path: Path) -> PaperDocument:
            assert path.exists()
            return expected_doc

    fetcher = PaperFetcher(parser=StubParser())

    monkeypatch.setattr(fetcher, "_normalize_http_url", lambda value: value)
    monkeypatch.setattr(fetcher, "_head_content_type", lambda _url: "text/html")
    monkeypatch.setattr(
        fetcher,
        "_http_get_text",
        lambda _url: (
            '<html><head><meta name="citation_pdf_url" content="https://example.org/paper.pdf"/>'
            "</head><body></body></html>"
        ),
    )

    downloaded: dict[str, Path] = {}

    def _fake_download(url: str, destination: Path) -> None:
        downloaded["url"] = Path(url)
        destination.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(fetcher, "_download_file", _fake_download)

    result = fetcher.fetch_by_url("https://example.org/page", tmp_path)

    assert result == expected_doc
    assert downloaded["url"].name == "paper.pdf"


def test_paper_ingester_delegates_to_fetcher(tmp_path: Path) -> None:
    expected = _build_sample_document(tmp_path / "delegated.pdf")

    class StubFetcher:
        def fetch_auto(self, source: str, dest_dir: Path) -> PaperDocument:
            assert source == "arxiv:2005.14165"
            assert dest_dir == tmp_path
            return expected

    ingester = PaperIngester(fetcher=StubFetcher())
    result = ingester.ingest("arxiv:2005.14165", tmp_path)

    assert result == expected


def test_fetch_by_arxiv_with_stubbed_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("arxiv")

    class StubEntry:
        title = "Language Models are Few-Shot Learners"
        summary = "A large language model paper"
        authors = [types.SimpleNamespace(name="Tom Brown")]
        published = types.SimpleNamespace(year=2020)
        doi = None

        def download_pdf(self, dirpath: str, filename: str) -> str:
            target = Path(dirpath) / filename
            target.write_bytes(b"%PDF-1.4\n")
            return str(target)

    class StubClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def results(self, _search: Any) -> Any:
            yield StubEntry()

    class StubSearch:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    import importlib

    fetcher_module = importlib.import_module("scholardevclaw.ingestion.paper_fetcher")

    if fetcher_module.arxiv is None:
        pytest.skip("arxiv package unavailable")

    monkeypatch.setattr(fetcher_module.arxiv, "Client", StubClient)
    monkeypatch.setattr(fetcher_module.arxiv, "Search", StubSearch)

    expected_doc = _build_sample_document(tmp_path / "2005_14165.pdf")

    class StubParser:
        def parse(self, _pdf_path: Path) -> PaperDocument:
            return expected_doc

    fetcher = PaperFetcher(parser=StubParser())
    result = fetcher.fetch_by_arxiv_id("2005.14165", tmp_path)

    assert result.title == "Language Models are Few-Shot Learners"
    assert result.arxiv_id == "2005.14165"
    assert result.year == 2020


@pytest.mark.integration
def test_fetch_by_arxiv_integration_gpt3_real(tmp_path: Path) -> None:
    if os.getenv("SCHOLARDEVCLAW_RUN_INTEGRATION", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        pytest.skip("Integration tests disabled by default")

    pytest.importorskip("arxiv")

    fetcher = PaperFetcher()
    try:
        result = fetcher.fetch_by_arxiv_id("2005.14165", tmp_path)
    except Exception as exc:
        pytest.skip(f"arXiv integration unavailable in this environment: {exc}")

    assert result.arxiv_id == "2005.14165"
    assert bool(result.title.strip())
    assert bool(result.abstract.strip())
    assert result.pdf_path is not None and result.pdf_path.exists()
