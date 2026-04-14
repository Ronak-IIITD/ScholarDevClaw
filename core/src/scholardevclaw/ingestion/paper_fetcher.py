from __future__ import annotations

import ipaddress
import logging
import re
import socket
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from scholardevclaw.ingestion.models import PaperDocument
from scholardevclaw.ingestion.pdf_parser import PDFParser
from scholardevclaw.utils.retry import RetryPolicy

LOGGER = logging.getLogger(__name__)

_SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/DOI:"
_SEMANTIC_SCHOLAR_FIELDS = "title,authors,year,abstract,openAccessPdf,externalIds"


class PaperFetchError(RuntimeError):
    """Base class for paper-fetching failures."""


class PaperNotAccessibleError(PaperFetchError):
    """Raised when a paper cannot be downloaded due to access limitations."""


class PaperSourceResolutionError(PaperFetchError):
    """Raised when input source type cannot be resolved."""


try:
    import arxiv  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised only without optional dependency
    arxiv = None  # type: ignore[assignment]


class ParserLike(Protocol):
    """Protocol for objects that can parse a local PDF path."""

    def parse(self, pdf_path: Path, /) -> PaperDocument:
        """Parse path into a paper document."""
        raise NotImplementedError


class FetcherLike(Protocol):
    """Protocol for objects that auto-resolve a paper source."""

    def fetch_auto(self, source: str, dest_dir: Path, /) -> PaperDocument:
        """Resolve source into a paper document."""
        raise NotImplementedError


class PaperFetcher:
    """Resolve DOI / arXiv ID / URL → downloaded PDF + parsed metadata."""

    def __init__(
        self,
        parser: ParserLike | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.parser = parser or PDFParser()
        self.timeout_seconds = timeout_seconds
        self._retry_policy = RetryPolicy(max_attempts=3, base_delay=1.0, max_delay=5.0)

    def fetch_by_arxiv_id(self, arxiv_id: str, dest_dir: Path) -> PaperDocument:
        """Fetch paper PDF from arXiv and parse it into ``PaperDocument``."""

        if arxiv is None:
            raise ImportError(
                "arxiv package is required. Install with: pip install -e '.[ingestion]'"
            )

        normalized_id = arxiv_id.removeprefix("arxiv:").strip()
        if not normalized_id:
            raise PaperSourceResolutionError("arXiv ID is empty")

        target_dir = dest_dir.expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Fetching arXiv paper: %s", normalized_id)

        search = arxiv.Search(id_list=[normalized_id], max_results=1)
        client = arxiv.Client(page_size=1, delay_seconds=0, num_retries=3)

        results = list(client.results(search))
        if not results:
            raise PaperNotAccessibleError(f"No arXiv paper found for ID '{normalized_id}'")

        entry = results[0]
        filename = f"{self._safe_filename(normalized_id)}.pdf"
        pdf_path = target_dir / filename

        try:
            if hasattr(entry, "download_pdf"):
                downloaded_path = entry.download_pdf(dirpath=str(target_dir), filename=filename)
                pdf_path = Path(downloaded_path).resolve() if downloaded_path else pdf_path
            else:
                pdf_url = str(getattr(entry, "pdf_url", "") or "")
                if not pdf_url:
                    raise PaperNotAccessibleError(
                        f"arXiv entry '{normalized_id}' does not expose a downloadable PDF URL"
                    )
                self._download_file(pdf_url, pdf_path)
        except PaperFetchError:
            raise
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            AttributeError,
        ) as exc:  # pragma: no cover - network errors are environment-dependent
            LOGGER.error("Failed to download arXiv PDF for %s: %s", normalized_id, exc)
            raise PaperFetchError(
                f"Failed to download arXiv PDF for '{normalized_id}': {exc}"
            ) from exc

        document = self.parser.parse(pdf_path)
        document.arxiv_id = normalized_id
        document.title = str(getattr(entry, "title", "") or document.title)
        document.authors = [
            str(getattr(author, "name", "") or str(author)).strip()
            for author in list(getattr(entry, "authors", []))
            if str(getattr(author, "name", "") or str(author)).strip()
        ]
        document.abstract = str(getattr(entry, "summary", "") or document.abstract)
        published = getattr(entry, "published", None)
        if published is not None and hasattr(published, "year"):
            try:
                document.year = int(published.year)
            except (TypeError, ValueError):
                LOGGER.warning("Could not parse published year for %s", normalized_id)
        document.doi = str(getattr(entry, "doi", "") or "") or document.doi

        return document

    def fetch_by_doi(self, doi: str, dest_dir: Path) -> PaperDocument:
        """Fetch open-access DOI paper via Semantic Scholar, then parse the PDF."""

        normalized_doi = self._normalize_doi(doi)
        if not normalized_doi:
            raise PaperSourceResolutionError("DOI is empty")

        target_dir = dest_dir.expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        metadata = self._fetch_semantic_scholar_metadata(normalized_doi)
        open_access_pdf = metadata.get("openAccessPdf")
        pdf_url = ""
        if isinstance(open_access_pdf, dict):
            pdf_url = str(open_access_pdf.get("url", "") or "")

        if not pdf_url:
            raise PaperNotAccessibleError(
                "Paper is not available as open-access PDF via Semantic Scholar. "
                "Try providing an arXiv ID or direct PDF URL."
            )

        pdf_path = target_dir / f"doi_{self._safe_filename(normalized_doi)}.pdf"
        self._download_file(pdf_url, pdf_path)

        document = self.parser.parse(pdf_path)
        document.doi = normalized_doi
        document.title = str(metadata.get("title", "") or document.title)
        document.authors = self._extract_semantic_scholar_authors(metadata)
        document.abstract = str(metadata.get("abstract", "") or document.abstract)

        year = metadata.get("year")
        if year is not None:
            try:
                document.year = int(year)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "Could not parse year from Semantic Scholar metadata for DOI %s", doi
                )

        external_ids = metadata.get("externalIds")
        if isinstance(external_ids, dict):
            arxiv_id = str(external_ids.get("ArXiv", "") or "").strip()
            if arxiv_id:
                document.arxiv_id = arxiv_id

        return document

    def fetch_by_url(self, url: str, dest_dir: Path) -> PaperDocument:
        """Fetch PDF document from direct PDF URL or paper landing page."""

        normalized_url = self._normalize_http_url(url)
        target_dir = dest_dir.expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Fetching paper by URL: %s", normalized_url)

        content_type = self._head_content_type(normalized_url)
        if "application/pdf" in content_type or normalized_url.lower().endswith(".pdf"):
            pdf_url = normalized_url
        else:
            html = self._http_get_text(normalized_url)
            resolved_pdf_url = self._extract_pdf_url_from_html(normalized_url, html)
            if not resolved_pdf_url:
                raise PaperNotAccessibleError(
                    "Could not discover a PDF URL on the target page. "
                    "Provide a direct PDF link if available."
                )
            pdf_url = resolved_pdf_url

        parsed_url = urlparse(pdf_url)
        pdf_name = Path(parsed_url.path).stem or "paper"
        pdf_path = target_dir / f"{self._safe_filename(pdf_name)}.pdf"
        self._download_file(pdf_url, pdf_path)

        return self.parser.parse(pdf_path)

    def fetch_auto(self, source: str, dest_dir: Path) -> PaperDocument:
        """Auto-resolve ``source`` as local PDF path, arXiv ID, DOI, or URL."""

        normalized_source = source.strip()
        if not normalized_source:
            raise PaperSourceResolutionError("Paper source is empty")

        local_path = Path(normalized_source).expanduser()
        if local_path.exists():
            if local_path.is_dir():
                raise PaperSourceResolutionError(
                    f"Expected a PDF file, got directory: {local_path}"
                )
            if local_path.suffix.casefold() != ".pdf":
                raise PaperSourceResolutionError(
                    f"Unsupported local file type '{local_path.suffix}'. Expected a .pdf file."
                )
            return self.parser.parse(local_path)

        if normalized_source.casefold().startswith("arxiv:"):
            return self.fetch_by_arxiv_id(normalized_source.split(":", maxsplit=1)[1], dest_dir)

        if self._looks_like_arxiv_id(normalized_source):
            return self.fetch_by_arxiv_id(normalized_source, dest_dir)

        if self._looks_like_doi(normalized_source):
            return self.fetch_by_doi(normalized_source, dest_dir)

        parsed = urlparse(normalized_source)
        if parsed.scheme in {"http", "https"}:
            return self.fetch_by_url(normalized_source, dest_dir)

        raise PaperSourceResolutionError(
            "Could not determine paper source type. Provide a local PDF path, DOI, "
            "arXiv ID (or arxiv:<id>), or URL."
        )

    def _fetch_semantic_scholar_metadata(self, doi: str) -> dict:
        url = f"{_SEMANTIC_SCHOLAR_URL}{doi}"

        def _request() -> dict:
            response = httpx.get(
                url,
                params={"fields": _SEMANTIC_SCHOLAR_FIELDS},
                timeout=self.timeout_seconds,
                follow_redirects=False,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Unexpected Semantic Scholar response payload")
            return payload

        try:
            payload = cast(dict, self._retry_policy.execute(_request))
            return payload
        except httpx.HTTPStatusError as exc:
            LOGGER.error("Semantic Scholar lookup failed for DOI %s: %s", doi, exc)
            raise PaperFetchError(f"Semantic Scholar lookup failed for DOI '{doi}': {exc}") from exc
        except (httpx.RequestError, ValueError) as exc:
            LOGGER.error("Semantic Scholar request failed for DOI %s: %s", doi, exc)
            raise PaperFetchError(
                f"Failed to query Semantic Scholar for DOI '{doi}': {exc}"
            ) from exc

    def _download_file(self, url: str, destination: Path) -> None:
        normalized_url = self._normalize_http_url(url)

        def _request() -> bytes:
            response = httpx.get(
                normalized_url,
                timeout=self.timeout_seconds,
                follow_redirects=True,
            )
            response.raise_for_status()
            if "application/pdf" not in response.headers.get("content-type", "").casefold():
                LOGGER.warning(
                    "Downloading URL without explicit PDF content-type: %s",
                    normalized_url,
                )
            return response.content

        try:
            content = cast(bytes, self._retry_policy.execute(_request))
        except httpx.HTTPStatusError as exc:
            LOGGER.error("Failed to download PDF from %s: %s", normalized_url, exc)
            raise PaperFetchError(f"Failed to download PDF from '{normalized_url}': {exc}") from exc
        except httpx.RequestError as exc:
            LOGGER.error("Request error downloading PDF from %s: %s", normalized_url, exc)
            raise PaperFetchError(
                f"Network error downloading PDF from '{normalized_url}': {exc}"
            ) from exc

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

    def _head_content_type(self, url: str) -> str:
        normalized_url = self._normalize_http_url(url)
        try:
            response = httpx.head(
                normalized_url,
                timeout=self.timeout_seconds,
                follow_redirects=True,
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            return str(content_type).casefold()
        except httpx.HTTPError as exc:
            LOGGER.warning(
                "HEAD request failed for %s; falling back to GET logic (%s)", normalized_url, exc
            )
            return ""

    def _http_get_text(self, url: str) -> str:
        normalized_url = self._normalize_http_url(url)
        try:
            response = httpx.get(
                normalized_url,
                timeout=self.timeout_seconds,
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as exc:
            LOGGER.error("Failed to fetch HTML from %s: %s", normalized_url, exc)
            raise PaperFetchError(f"Failed to fetch URL '{normalized_url}': {exc}") from exc
        except httpx.RequestError as exc:
            LOGGER.error("Network error fetching HTML from %s: %s", normalized_url, exc)
            raise PaperFetchError(f"Network error fetching URL '{normalized_url}': {exc}") from exc

    def _extract_pdf_url_from_html(self, page_url: str, html: str) -> str | None:
        soup = BeautifulSoup(html, "html.parser")

        meta_candidates = soup.find_all("meta", attrs={"name": "citation_pdf_url"})
        for meta in meta_candidates:
            if not hasattr(meta, "get"):
                continue
            content_value = meta.get("content")
            if content_value:
                candidate = str(content_value)
                return self._normalize_http_url(urljoin(page_url, candidate))

        for link in soup.find_all("a", href=True):
            if not hasattr(link, "get"):
                continue
            raw_href = link.get("href")
            if raw_href is None:
                continue
            href = str(raw_href)
            if ".pdf" in href.casefold():
                return self._normalize_http_url(urljoin(page_url, href))

        return None

    def _extract_semantic_scholar_authors(self, metadata: dict) -> list[str]:
        raw_authors = metadata.get("authors")
        if not isinstance(raw_authors, list):
            return []

        authors: list[str] = []
        for item in raw_authors:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                if name:
                    authors.append(name)
        return authors

    def _normalize_doi(self, doi: str) -> str:
        normalized = doi.strip()
        if normalized.casefold().startswith("doi:"):
            normalized = normalized.split(":", maxsplit=1)[1]
        return normalized.strip()

    def _looks_like_doi(self, value: str) -> bool:
        candidate = self._normalize_doi(value)
        return bool(candidate) and "/" in candidate and candidate.casefold().startswith("10.")

    def _looks_like_arxiv_id(self, value: str) -> bool:
        candidate = value.strip()
        if not candidate:
            return False
        if "://" in candidate:
            return False
        if "/" in candidate:
            # legacy format (e.g., cs/9901001)
            return bool(re.match(r"^[A-Za-z\-]+/\d{7}$", candidate))
        parts = candidate.split(".")
        return len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()

    def _normalize_http_url(self, url: str) -> str:
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"}:
            raise PaperSourceResolutionError(f"Unsupported URL scheme in '{url}'")

        host = parsed.hostname
        if not host:
            raise PaperSourceResolutionError(f"Invalid URL host in '{url}'")

        self._assert_public_host(host)
        return parsed.geturl()

    def _assert_public_host(self, host: str) -> None:
        lowered = host.casefold()
        if lowered in {"localhost", "0.0.0.0"}:
            raise PaperSourceResolutionError(f"Blocked local host '{host}'")

        if lowered.endswith(".local"):
            raise PaperSourceResolutionError(f"Blocked local-domain host '{host}'")

        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror as exc:
            raise PaperSourceResolutionError(f"Could not resolve host '{host}'") from exc

        for info in infos:
            ip_text = info[4][0]
            ip_addr = ipaddress.ip_address(ip_text)
            if (
                ip_addr.is_private
                or ip_addr.is_loopback
                or ip_addr.is_link_local
                or ip_addr.is_multicast
                or ip_addr.is_reserved
            ):
                raise PaperSourceResolutionError(f"Blocked non-public host '{host}'")

    def _safe_filename(self, value: str) -> str:
        return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


class PaperIngester:
    """High-level ingestion façade for local PDFs, arXiv IDs, DOI values, and URLs."""

    def __init__(
        self,
        fetcher: FetcherLike | None = None,
        parser: ParserLike | None = None,
    ) -> None:
        parser_instance = parser or PDFParser()
        self.fetcher = fetcher or PaperFetcher(parser=parser_instance)

    def ingest(self, source: str, output_dir: Path) -> PaperDocument:
        """Ingest source into a structured :class:`PaperDocument`."""

        return self.fetcher.fetch_auto(source, output_dir)
