"""
Multi-source paper extraction from academic databases.

Supports:
- arXiv: https://arxiv.org
- PubMed: https://pubmed.ncbi.nlm.nih.gov
- IEEE Xplore: https://ieeexplore.ieee.org
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from urllib.parse import urlencode, quote


try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


@dataclass
class Paper:
    """Represents an academic paper"""

    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    year: int
    source: str  # arxiv, pubmed, ieee
    url: str
    month: int = 1
    doi: str = ""
    categories: list[str] = field(default_factory=list)
    citations: int = 0
    pdf_url: str = ""
    arxiv_id: str = ""
    pubmed_id: str = ""
    ieee_id: str = ""
    journal: str = ""
    volume: str = ""
    pages: str = ""
    publisher: str = ""
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "year": self.year,
            "month": self.month,
            "source": self.source,
            "url": self.url,
            "doi": self.doi,
            "categories": self.categories,
            "citations": self.citations,
            "pdf_url": self.pdf_url,
            "arxiv_id": self.arxiv_id,
            "pubmed_id": self.pubmed_id,
            "ieee_id": self.ieee_id,
            "journal": self.journal,
            "volume": self.volume,
            "pages": self.pages,
            "publisher": self.publisher,
            "keywords": self.keywords,
        }


@dataclass
class SearchResult:
    """Results from a paper search"""

    papers: list[Paper]
    total_results: int
    query: str
    source: str


class ArxivSource:
    """arXiv API integration"""

    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0) if HAS_HTTPX else None

    async def search(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        categories: list[str] | None = None,
    ) -> SearchResult:
        """Search arXiv for papers"""
        if not HAS_HTTPX:
            raise ImportError("httpx is required. Install with: pip install httpx")

        search_query = query
        if categories:
            cat_query = " OR ".join(f"cat:{c}" for c in categories)
            search_query = f"({query}) AND ({cat_query})"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        url = f"{self.BASE_URL}?{urlencode(params)}"
        response = await self.client.get(url)
        response.raise_for_status()

        return self._parse_atom(response.text, query)

    def _parse_atom(self, xml_content: str, query: str) -> SearchResult:
        root = ET.fromstring(xml_content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        papers = []
        total_results = int(root.get("totalResults", 0))

        for entry in root.findall("atom:entry", ns):
            arxiv_id = ""
            id_text = entry.find("atom:id", ns)
            if id_text is not None:
                id_url = id_text.text or ""
                match = re.search(r"(\d+\.\d+)", id_url)
                if match:
                    arxiv_id = match.group(1)

            title = ""
            title_elem = entry.find("atom:title", ns)
            if title_elem is not None:
                title = self._clean_text(title_elem.text or "")

            abstract = ""
            summary_elem = entry.find("atom:summary", ns)
            if summary_elem is not None:
                abstract = self._clean_text(summary_elem.text or "")

            authors = []
            for author in entry.findall("atom:author", ns):
                name_elem = author.find("atom:name", ns)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text)

            published = entry.find("atom:published", ns)
            year, month = 2024, 1
            if published is not None and published.text:
                try:
                    dt = datetime.fromisoformat(published.text[:10])
                    year, month = dt.year, dt.month
                except Exception:
                    pass

            categories = []
            for cat in entry.findall("atom:category", ns):
                term = cat.get("term", "")
                if term:
                    categories.append(term)

            pdf_url = ""
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                    break

            paper_id = arxiv_id or id_text.text or ""
            papers.append(
                Paper(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    year=year,
                    month=month,
                    source="arxiv",
                    url=id_text.text or "",
                    arxiv_id=arxiv_id,
                    pdf_url=pdf_url,
                    categories=categories,
                )
            )

        return SearchResult(papers=papers, total_results=total_results, query=query, source="arxiv")

    def _clean_text(self, text: str) -> str:
        return " ".join(text.split())


class PubmedSource:
    """PubMed API integration"""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0) if HAS_HTTPX else None

    async def search(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
    ) -> SearchResult:
        """Search PubMed for papers"""
        if not HAS_HTTPX:
            raise ImportError("httpx is required. Install with: pip install httpx")

        esearch_url = f"{self.BASE_URL}/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": sort_by,
            "format": "json",
        }

        response = await self.client.get(esearch_url, params=params)
        response.raise_for_status()
        data = response.json()

        id_list = data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return SearchResult(papers=[], total_results=0, query=query, source="pubmed")

        total = int(data.get("esearchresult", {}).get("count", 0))

        efetch_url = f"{self.BASE_URL}/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml",
            "rettype": "abstract",
        }

        fetch_response = await self.client.get(efetch_url, params=fetch_params)
        fetch_response.raise_for_status()

        papers = self._parse_pubmed_xml(fetch_response.text, query)
        return SearchResult(papers=papers, total_results=total, query=query, source="pubmed")

    def _parse_pubmed_xml(self, xml_content: str, query: str) -> list[Paper]:
        root = ET.fromstring(xml_content)
        papers = []

        for article in root.findall(".//PubmedArticle"):
            pmid = ""
            pmid_elem = article.find(".//PMID")
            if pmid_elem is not None and pmid_elem.text:
                pmid = pmid_elem.text

            title = ""
            title_elem = article.find(".//ArticleTitle")
            if title_elem is not None and title_elem.text:
                title = title_elem.text

            abstract = ""
            abstract_elem = article.find(".//AbstractText")
            if abstract_elem is not None and abstract_elem.text:
                abstract = abstract_elem.text

            authors = []
            for author in article.findall(".//Author"):
                last_name = author.find("LastName")
                fore_name = author.find("ForeName")
                name_parts = []
                if fore_name is not None and fore_name.text:
                    name_parts.append(fore_name.text)
                if last_name is not None and last_name.text:
                    name_parts.append(last_name.text)
                if name_parts:
                    authors.append(" ".join(name_parts))

            year, month = 2024, 1
            pub_date = article.find(".//PubDate")
            if pub_date is not None:
                year_elem = pub_date.find("Year")
                month_elem = pub_date.find("Month")
                if year_elem is not None and year_elem.text:
                    year = int(year_elem.text)
                if month_elem is not None and month_elem.text:
                    month_map = {
                        "Jan": 1,
                        "Feb": 2,
                        "Mar": 3,
                        "Apr": 4,
                        "May": 5,
                        "Jun": 6,
                        "Jul": 7,
                        "Aug": 8,
                        "Sep": 9,
                        "Oct": 10,
                        "Nov": 11,
                        "Dec": 12,
                    }
                    month = month_map.get(month_elem.text, 1)

            journal = ""
            journal_elem = article.find(".//Journal/Title")
            if journal_elem is not None and journal_elem.text:
                journal = journal_elem.text

            doi = ""
            doi_elem = article.find(".//ELocationID[@doi]")
            if doi_elem is not None and doi_elem.text:
                doi = doi_elem.text

            citations = 0
            cited_elem = article.find(".//CitedMedium")
            if cited_elem is not None:
                citations = 1

            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            papers.append(
                Paper(
                    paper_id=pmid,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    year=year,
                    month=month,
                    source="pubmed",
                    url=pubmed_url,
                    pubmed_id=pmid,
                    doi=doi,
                    journal=journal,
                    citations=citations,
                )
            )

        return papers


class IEEESource:
    """IEEE Xplore API integration"""

    BASE_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or ""
        self.client = httpx.AsyncClient(timeout=30.0) if HAS_HTTPX else None

    async def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> SearchResult:
        """Search IEEE Xplore for papers"""
        if not HAS_HTTPX:
            raise ImportError("httpx is required. Install with: pip install httpx")

        params = {
            "apikey": self.api_key,
            "article_number": max_results,
            "querytext": query,
        }

        if self.api_key:
            url = f"{self.BASE_URL}?{urlencode(params)}"
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()
        else:
            return SearchResult(papers=[], total_results=0, query=query, source="ieee")

        papers = []
        total = data.get("total_records", 0)

        for item in data.get("articles", []):
            ieee_id = item.get("article_number", "")
            title = item.get("title", "")

            authors = []
            for author in item.get("authors", []):
                name = author.get("full_name", "")
                if name:
                    authors.append(name)

            abstract = item.get("abstract", "")

            year, month = 2024, 1
            pub_date = item.get("publication_date", "")
            if pub_date:
                try:
                    dt = datetime.strptime(pub_date, "%Y-%m-%d")
                    year, month = dt.year, dt.month
                except Exception:
                    pass

            journal = item.get("journal_title", "")
            doi = item.get("doi", "")
            pdf_url = item.get("pdf_url", "")

            ieee_url = f"https://ieeexplore.ieee.org/document/{ieee_id}/"

            papers.append(
                Paper(
                    paper_id=ieee_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    year=year,
                    month=month,
                    source="ieee",
                    url=ieee_url,
                    ieee_id=ieee_id,
                    doi=doi,
                    journal=journal,
                    pdf_url=pdf_url,
                )
            )

        return SearchResult(papers=papers, total_results=total, query=query, source="ieee")


class PaperSourceAggregator:
    """Aggregates paper search across multiple sources"""

    def __init__(self):
        self.arxiv = ArxivSource()
        self.pubmed = PubmedSource()
        self.ieee: IEEESource | None = None

    def set_ieee_key(self, api_key: str):
        """Set IEEE API key"""
        self.ieee = IEEESource(api_key)

    async def search_all(
        self,
        query: str,
        max_results_per_source: int = 5,
        sources: list[str] | None = None,
    ) -> dict[str, SearchResult]:
        """Search across all or specified sources"""
        sources = sources or ["arxiv", "pubmed"]
        results = {}

        if "arxiv" in sources:
            try:
                results["arxiv"] = await self.arxiv.search(query, max_results_per_source)
            except Exception:
                pass

        if "pubmed" in sources:
            try:
                results["pubmed"] = await self.pubmed.search(query, max_results_per_source)
            except Exception:
                pass

        if "ieee" in sources and self.ieee:
            try:
                results["ieee"] = await self.ieee.search(query, max_results_per_source)
            except Exception:
                pass

        return results

    async def get_paper_by_id(self, paper_id: str, source: str) -> Paper | None:
        """Fetch a specific paper by ID"""
        if source == "arxiv":
            result = await self.arxiv.search(f"id:{paper_id}", max_results=1)
            return result.papers[0] if result.papers else None
        elif source == "pubmed":
            result = await self.pubmed.search(f"{paper_id}[PMID]", max_results=1)
            return result.papers[0] if result.papers else None
        elif source == "ieee" and self.ieee:
            result = await self.ieee.search(f"{paper_id}", max_results=1)
            return result.papers[0] if result.papers else None
        return None


def get_paper_source(source: str, **kwargs) -> ArxivSource | PubmedSource | IEEESource:
    """Factory function to get a paper source"""
    if source == "arxiv":
        return ArxivSource()
    elif source == "pubmed":
        return PubmedSource()
    elif source == "ieee":
        return IEEESource(**kwargs)
    else:
        raise ValueError(f"Unknown source: {source}")
