from __future__ import annotations

import ipaddress
import json
import logging
import re
import socket
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from scholardevclaw.exceptions import (
    PaperFetchError,
    PaperNotAccessibleError,
    PaperSourceResolutionError,
)
from scholardevclaw.ingestion.models import PaperDocument
from scholardevclaw.ingestion.pdf_parser import PDFParser
from scholardevclaw.utils.retry import RetryPolicy

LOGGER = logging.getLogger(__name__)

_SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/DOI:"
_SEMANTIC_SCHOLAR_FIELDS = "title,authors,year,abstract,openAccessPdf,venue,externalIds"
_SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_DEFAULT_CACHE_DIR = Path.home() / ".scholardevclaw" / "cache"

__all__ = [
    "PaperFetchError",
    "PaperNotAccessibleError",
    "PaperSourceResolutionError",
    "PaperFetcher",
    "PaperIngester",
]

try:
    import arxiv  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    arxiv = None  # type: ignore[assignment]


class ParserLike(Protocol):
    def parse(self, pdf_path: Path, /) -> PaperDocument:
        raise NotImplementedError


class FetcherLike(Protocol):
    def fetch_auto(self, source: str, dest_dir: Path, /, no_cache: bool = False) -> PaperDocument:
        raise NotImplementedError


class PaperFetcher:
    """Resolve DOI / arXiv ID / URL / title → downloaded PDF + parsed metadata."""

    def __init__(
        self,
        parser: ParserLike | None = None,
        timeout_seconds: float = 30.0,
        cache_dir: Path | None = None,
    ) -> None:
        self.parser = parser or PDFParser()
        self.timeout_seconds = timeout_seconds
        self.cache_dir = (cache_dir or _DEFAULT_CACHE_DIR).expanduser().resolve()
        self._retry_policy = RetryPolicy(max_attempts=3, base_delay=1.0, max_delay=30.0)

    def fetch_auto(self, source: str, dest_dir: Path, no_cache: bool = False) -> PaperDocument:
        normalized_source = source.strip()
        if not normalized_source:
            raise PaperSourceResolutionError("Paper source is empty")

        local_path = Path(normalized_source).expanduser()
        if local_path.exists():
            if local_path.is_dir():
                raise PaperSourceResolutionError(f"Expected a PDF file, got directory: {local_path}")
            if local_path.suffix.casefold() != ".pdf":
                raise PaperSourceResolutionError(
                    f"Unsupported local file type '{local_path.suffix}'. Expected a .pdf file."
                )
            return self.parser.parse(local_path)

        if normalized_source.casefold().startswith("arxiv:"):
            return self.fetch_by_arxiv_id(
                normalized_source.split(":", maxsplit=1)[1],
                dest_dir,
                no_cache=no_cache,
            )
        if self._looks_like_arxiv_id(normalized_source):
            return self.fetch_by_arxiv_id(normalized_source, dest_dir, no_cache=no_cache)
        if self._looks_like_doi(normalized_source):
            return self.fetch_by_doi(normalized_source, dest_dir, no_cache=no_cache)

        parsed = urlparse(normalized_source)
        if parsed.scheme in {"http", "https"}:
            return self.fetch_by_url(normalized_source, dest_dir, no_cache=no_cache)

        return self.search_by_title(normalized_source, dest_dir, no_cache=no_cache)

    def fetch_by_arxiv_id(
        self,
        arxiv_id: str,
        dest_dir: Path,
        no_cache: bool = False,
    ) -> PaperDocument:
        normalized_id = arxiv_id.removeprefix("arxiv:").strip()
        if not normalized_id:
            raise PaperSourceResolutionError("arXiv ID is empty")

        cache_path = self._cache_path_for_key(normalized_id)
        if not no_cache:
            cached = self._load_cached_document(cache_path)
            if cached is not None:
                return cached

        if arxiv is None:
            raise ImportError("arxiv package is required. Install with: pip install -e '.[ingestion]'")

        target_dir = self._prepare_dest_dir(dest_dir)
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
                downloaded = entry.download_pdf(dirpath=str(target_dir), filename=filename)
                pdf_path = Path(downloaded).resolve() if downloaded else pdf_path
            else:
                pdf_url = str(getattr(entry, "pdf_url", "") or "")
                if not pdf_url:
                    raise PaperNotAccessibleError(
                        f"arXiv entry '{normalized_id}' does not expose a downloadable PDF URL"
                    )
                self._download_file(pdf_url, pdf_path)
        except PaperFetchError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError, AttributeError) as exc:
            raise PaperFetchError(f"Failed to download arXiv PDF for '{normalized_id}': {exc}") from exc

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
                LOGGER.warning("Could not parse published year for arXiv ID %s", normalized_id)
        document.doi = str(getattr(entry, "doi", "") or "") or document.doi
        document.venue = str(getattr(entry, "journal_ref", "") or "") or document.venue
        document.source_url = str(getattr(entry, "entry_id", "") or "") or document.source_url

        self._store_cached_document(cache_path, document)
        return document

    def fetch_by_doi(self, doi: str, dest_dir: Path, no_cache: bool = False) -> PaperDocument:
        normalized_doi = self._normalize_doi(doi)
        if not normalized_doi:
            raise PaperSourceResolutionError("DOI is empty")

        cache_path = self._cache_path_for_key(f"doi_{normalized_doi}")
        if not no_cache:
            cached = self._load_cached_document(cache_path)
            if cached is not None:
                return cached

        metadata = self._fetch_semantic_scholar_metadata(normalized_doi)
        target_dir = self._prepare_dest_dir(dest_dir)
        pdf_url = ""
        open_access_pdf = metadata.get("openAccessPdf")
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
        self._merge_semantic_scholar_metadata(document, normalized_doi, metadata)
        self._store_cached_document(cache_path, document)
        return document

    def fetch_by_url(self, url: str, dest_dir: Path, no_cache: bool = False) -> PaperDocument:
        normalized_url = self._normalize_http_url(url)
        cache_path = self._cache_path_for_key(f"url_{normalized_url}")
        if not no_cache:
            cached = self._load_cached_document(cache_path)
            if cached is not None:
                return cached

        target_dir = self._prepare_dest_dir(dest_dir)
        content_type = self._head_content_type(normalized_url)
        if "application/pdf" in content_type or normalized_url.lower().endswith(".pdf"):
            pdf_url = normalized_url
        else:
            html = self._http_get_text(normalized_url)
            resolved_pdf_url = self._extract_pdf_url_from_html(normalized_url, html)
            if not resolved_pdf_url:
                raise PaperNotAccessibleError(
                    "Could not discover a PDF URL on the target page. Provide a direct PDF link if available."
                )
            pdf_url = resolved_pdf_url

        pdf_name = Path(urlparse(pdf_url).path).stem or "paper"
        pdf_path = target_dir / f"{self._safe_filename(pdf_name)}.pdf"
        self._download_file(pdf_url, pdf_path)

        document = self.parser.parse(pdf_path)
        document.source_url = normalized_url
        self._store_cached_document(cache_path, document)
        return document

    def search_by_title(self, title: str, dest_dir: Path, no_cache: bool = False) -> PaperDocument:
        normalized_title = self._normalize_whitespace(title)
        if not normalized_title:
            raise PaperSourceResolutionError("Paper title is empty")

        cache_path = self._cache_path_for_key(f"title_{normalized_title}")
        if not no_cache:
            cached = self._load_cached_document(cache_path)
            if cached is not None:
                return cached

        arxiv_candidate = self._search_arxiv_by_title(normalized_title)
        semantic_candidate = self._search_semantic_scholar_by_title(normalized_title)

        candidates: list[tuple[str, float, dict[str, Any]]] = []
        if arxiv_candidate is not None:
            candidates.append(("arxiv", arxiv_candidate["similarity"], arxiv_candidate))
        if semantic_candidate is not None:
            candidates.append(("semantic", semantic_candidate["similarity"], semantic_candidate))
        if not candidates:
            raise PaperNotAccessibleError(f"No paper search result found for title '{normalized_title}'")

        source_kind, similarity, candidate = max(candidates, key=lambda item: item[1])
        if similarity < 0.85:
            raise PaperNotAccessibleError(
                f"No title match met confidence threshold for '{normalized_title}' (best={similarity:.2f})"
            )

        if source_kind == "arxiv":
            document = self.fetch_by_arxiv_id(str(candidate["arxiv_id"]), dest_dir, no_cache=no_cache)
        else:
            doi = str(candidate.get("doi", "") or "")
            if doi:
                document = self.fetch_by_doi(doi, dest_dir, no_cache=no_cache)
            else:
                pdf_url = str(candidate.get("pdf_url", "") or "")
                if not pdf_url:
                    raise PaperNotAccessibleError(
                        f"Semantic Scholar match for '{normalized_title}' has no DOI or open PDF"
                    )
                document = self.fetch_by_url(pdf_url, dest_dir, no_cache=no_cache)
            document.title = str(candidate.get("title", document.title))
            document.venue = str(candidate.get("venue", "") or document.venue or "") or document.venue

        self._store_cached_document(cache_path, document)
        return document

    def _fetch_semantic_scholar_metadata(self, doi: str) -> dict[str, Any]:
        url = f"{_SEMANTIC_SCHOLAR_URL}{doi}"

        def _request() -> dict[str, Any]:
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
            return cast(dict[str, Any], self._retry_policy.execute(_request))
        except httpx.HTTPStatusError as exc:
            raise PaperFetchError(f"Semantic Scholar lookup failed for DOI '{doi}': {exc}") from exc
        except (httpx.RequestError, ValueError) as exc:
            raise PaperFetchError(f"Failed to query Semantic Scholar for DOI '{doi}': {exc}") from exc

    def _search_semantic_scholar_by_title(self, title: str) -> dict[str, Any] | None:
        def _request() -> dict[str, Any]:
            response = httpx.get(
                _SEMANTIC_SCHOLAR_SEARCH_URL,
                params={
                    "query": title,
                    "limit": 5,
                    "fields": "title,authors,year,abstract,openAccessPdf,venue,externalIds",
                },
                timeout=self.timeout_seconds,
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Unexpected Semantic Scholar search payload")
            return payload

        try:
            payload = cast(dict[str, Any], self._retry_policy.execute(_request))
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
            LOGGER.warning("Semantic Scholar title search failed for '%s': %s", title, exc)
            return None

        raw_results = payload.get("data", [])
        if not isinstance(raw_results, list):
            return None

        best: dict[str, Any] | None = None
        best_score = 0.0
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            candidate_title = str(item.get("title", "") or "")
            score = SequenceMatcher(None, title.casefold(), candidate_title.casefold()).ratio()
            if score > best_score:
                open_access_pdf = item.get("openAccessPdf")
                pdf_url = str(open_access_pdf.get("url", "") or "") if isinstance(open_access_pdf, dict) else ""
                external_ids = item.get("externalIds")
                doi = str(external_ids.get("DOI", "") or "") if isinstance(external_ids, dict) else ""
                best_score = score
                best = {
                    "title": candidate_title,
                    "similarity": score,
                    "doi": doi,
                    "pdf_url": pdf_url,
                    "venue": str(item.get("venue", "") or ""),
                }
        return best

    def _search_arxiv_by_title(self, title: str) -> dict[str, Any] | None:
        if arxiv is None:
            return None
        search = arxiv.Search(query=f'ti:"{title}"', max_results=5)
        client = arxiv.Client(page_size=5, delay_seconds=0, num_retries=3)
        try:
            results = list(client.results(search))
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("arXiv title search failed for '%s': %s", title, exc)
            return None

        best: dict[str, Any] | None = None
        best_score = 0.0
        for entry in results:
            candidate_title = str(getattr(entry, "title", "") or "")
            score = SequenceMatcher(None, title.casefold(), candidate_title.casefold()).ratio()
            if score > best_score:
                best_score = score
                best = {
                    "title": candidate_title,
                    "similarity": score,
                    "arxiv_id": str(getattr(entry, "entry_id", "") or "").rstrip("/").split("/")[-1]
                    or str(getattr(entry, "get_short_id", lambda: "")()),
                }
        return best

    def _merge_semantic_scholar_metadata(
        self,
        document: PaperDocument,
        doi: str,
        metadata: dict[str, Any],
    ) -> None:
        document.doi = doi
        document.title = str(metadata.get("title", "") or document.title)
        document.authors = self._extract_semantic_scholar_authors(metadata)
        document.abstract = str(metadata.get("abstract", "") or document.abstract)
        document.venue = str(metadata.get("venue", "") or document.venue or "") or document.venue
        open_access_pdf = metadata.get("openAccessPdf")
        if isinstance(open_access_pdf, dict):
            document.source_url = str(open_access_pdf.get("url", "") or document.source_url or "") or document.source_url
        year = metadata.get("year")
        if year is not None:
            try:
                document.year = int(year)
            except (TypeError, ValueError):
                LOGGER.warning("Could not parse year from Semantic Scholar metadata for DOI %s", doi)
        external_ids = metadata.get("externalIds")
        if isinstance(external_ids, dict):
            arxiv_id = str(external_ids.get("ArXiv", "") or "").strip()
            if arxiv_id:
                document.arxiv_id = arxiv_id

    def _download_file(self, url: str, destination: Path) -> None:
        normalized_url = self._normalize_http_url(url)

        def _request() -> bytes:
            response = httpx.get(
                normalized_url,
                timeout=self.timeout_seconds,
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.content

        try:
            content = cast(bytes, self._retry_policy.execute(_request))
        except httpx.HTTPStatusError as exc:
            raise PaperFetchError(f"Failed to download PDF from '{normalized_url}': {exc}") from exc
        except httpx.RequestError as exc:
            raise PaperFetchError(f"Network error downloading PDF from '{normalized_url}': {exc}") from exc

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
            return str(response.headers.get("content-type", "")).casefold()
        except httpx.HTTPError as exc:
            LOGGER.warning("HEAD request failed for %s; falling back to GET logic (%s)", normalized_url, exc)
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
            raise PaperFetchError(f"Failed to fetch URL '{normalized_url}': {exc}") from exc
        except httpx.RequestError as exc:
            raise PaperFetchError(f"Network error fetching URL '{normalized_url}': {exc}") from exc

    def _extract_pdf_url_from_html(self, page_url: str, html: str) -> str | None:
        soup = BeautifulSoup(html, "html.parser")
        for meta in soup.find_all("meta", attrs={"name": "citation_pdf_url"}):
            if hasattr(meta, "get"):
                content = meta.get("content")
                if content:
                    return self._normalize_http_url(urljoin(page_url, str(content)))
        for link in soup.find_all("a", href=True):
            if hasattr(link, "get"):
                href = link.get("href")
                if href and ".pdf" in str(href).casefold():
                    return self._normalize_http_url(urljoin(page_url, str(href)))
        return None

    def _extract_semantic_scholar_authors(self, metadata: dict[str, Any]) -> list[str]:
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

    def _cache_path_for_key(self, key: str) -> Path:
        return self.cache_dir / f"{self._safe_filename(key)}.json"

    def _load_cached_document(self, cache_path: Path) -> PaperDocument | None:
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Failed to read cached paper document %s: %s", cache_path, exc)
            return None
        if not isinstance(payload, dict):
            return None
        LOGGER.info("Using cached paper metadata from %s", cache_path)
        return PaperDocument.from_dict(payload)

    def _store_cached_document(self, cache_path: Path, document: PaperDocument) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(document.to_dict(), indent=2), encoding="utf-8")

    def _prepare_dest_dir(self, dest_dir: Path) -> Path:
        target_dir = dest_dir.expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

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
        if not candidate or "://" in candidate:
            return False
        if "/" in candidate:
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

    def _normalize_whitespace(self, text: str) -> str:
        return " ".join(text.split()).strip()

    def _safe_filename(self, value: str) -> str:
        return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


class PaperIngester:
    """High-level ingestion façade for local PDFs, arXiv IDs, DOI values, URLs, and titles."""

    def __init__(
        self,
        fetcher: FetcherLike | None = None,
        parser: ParserLike | None = None,
    ) -> None:
        parser_instance = parser or PDFParser()
        self.fetcher = fetcher or PaperFetcher(parser=parser_instance)

    def ingest(self, source: str, output_dir: Path, no_cache: bool = False) -> PaperDocument:
        return self.fetcher.fetch_auto(source, output_dir, no_cache=no_cache)
