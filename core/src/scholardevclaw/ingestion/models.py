from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


@dataclass(slots=True)
class Equation:
    """A mathematical expression extracted from the paper."""

    latex: str
    description: str
    page: int
    equation_type: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "latex": self.latex,
            "description": self.description,
            "page": self.page,
            "equation_type": self.equation_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Equation:
        return cls(
            latex=str(data.get("latex", "")),
            description=str(data.get("description", "")),
            page=int(data.get("page", 0)),
            equation_type=str(data.get("equation_type", "unknown")),
        )


@dataclass(slots=True)
class Algorithm:
    """A pseudocode block extracted from the paper."""

    name: str
    pseudocode: str
    page: int
    language_hint: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pseudocode": self.pseudocode,
            "page": self.page,
            "language_hint": self.language_hint,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Algorithm:
        return cls(
            name=str(data.get("name", "")),
            pseudocode=str(data.get("pseudocode", "")),
            page=int(data.get("page", 0)),
            language_hint=str(data.get("language_hint", "unknown")),
            inputs=_as_str_list(data.get("inputs", [])),
            outputs=_as_str_list(data.get("outputs", [])),
        )


@dataclass(slots=True)
class Figure:
    """A figure reference with optional extracted image artifact path."""

    caption: str
    page: int
    figure_type: str = "diagram"
    image_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "caption": self.caption,
            "page": self.page,
            "figure_type": self.figure_type,
            "image_path": str(self.image_path) if self.image_path is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Figure:
        raw_path = data.get("image_path")
        return cls(
            caption=str(data.get("caption", "")),
            page=int(data.get("page", 0)),
            figure_type=str(data.get("figure_type", "diagram")),
            image_path=Path(raw_path) if isinstance(raw_path, str) and raw_path else None,
        )


@dataclass(slots=True)
class Section:
    """A hierarchical section extracted from the paper."""

    title: str
    level: int
    content: str
    page_start: int
    section_type: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "level": self.level,
            "content": self.content,
            "page_start": self.page_start,
            "section_type": self.section_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Section:
        return cls(
            title=str(data.get("title", "")),
            level=int(data.get("level", 1)),
            content=str(data.get("content", "")),
            page_start=int(data.get("page_start", 1)),
            section_type=str(data.get("section_type", "unknown")),
        )


@dataclass(slots=True)
class PaperDocument:
    """Structured paper representation consumed by downstream pipeline stages."""

    title: str
    authors: list[str]
    arxiv_id: str | None
    doi: str | None
    year: int | None
    abstract: str

    sections: list[Section]
    equations: list[Equation]
    algorithms: list[Algorithm]
    figures: list[Figure]
    tables: list[dict[str, Any]] = field(default_factory=list)

    full_text: str = ""
    pdf_path: Path | None = None
    source_url: str | None = None

    references: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    domain: str = "unknown"
    subdomain: str = "unknown"
    venue: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize PaperDocument into JSON-safe dictionary."""

        return {
            "title": self.title,
            "authors": list(self.authors),
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "year": self.year,
            "abstract": self.abstract,
            "venue": self.venue,
            "sections": [section.to_dict() for section in self.sections],
            "equations": [equation.to_dict() for equation in self.equations],
            "algorithms": [algorithm.to_dict() for algorithm in self.algorithms],
            "figures": [figure.to_dict() for figure in self.figures],
            "tables": list(self.tables),
            "full_text": self.full_text,
            "pdf_path": str(self.pdf_path) if self.pdf_path is not None else None,
            "source_url": self.source_url,
            "references": list(self.references),
            "keywords": list(self.keywords),
            "domain": self.domain,
            "subdomain": self.subdomain,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperDocument:
        """Deserialize PaperDocument from dictionary generated by ``to_dict``."""

        raw_pdf_path = data.get("pdf_path")
        return cls(
            title=str(data.get("title", "")),
            authors=_as_str_list(data.get("authors", [])),
            arxiv_id=str(data["arxiv_id"]) if data.get("arxiv_id") is not None else None,
            doi=str(data["doi"]) if data.get("doi") is not None else None,
            year=int(data["year"]) if data.get("year") is not None else None,
            abstract=str(data.get("abstract", "")),
            sections=[Section.from_dict(item) for item in _as_dict_list(data.get("sections", []))],
            equations=[
                Equation.from_dict(item) for item in _as_dict_list(data.get("equations", []))
            ],
            algorithms=[
                Algorithm.from_dict(item) for item in _as_dict_list(data.get("algorithms", []))
            ],
            figures=[Figure.from_dict(item) for item in _as_dict_list(data.get("figures", []))],
            tables=_as_dict_list(data.get("tables", [])),
            full_text=str(data.get("full_text", "")),
            pdf_path=Path(raw_pdf_path) if isinstance(raw_pdf_path, str) and raw_pdf_path else None,
            source_url=str(data["source_url"]) if data.get("source_url") is not None else None,
            references=_as_str_list(data.get("references", [])),
            keywords=_as_str_list(data.get("keywords", [])),
            domain=str(data.get("domain", "unknown")),
            subdomain=str(data.get("subdomain", "unknown")),
            venue=str(data["venue"]) if data.get("venue") is not None else None,
        )
