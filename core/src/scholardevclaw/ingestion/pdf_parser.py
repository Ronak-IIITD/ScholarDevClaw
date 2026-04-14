from __future__ import annotations

import io
import logging
import re
import statistics
from pathlib import Path
from typing import Any

from scholardevclaw.ingestion.models import Algorithm, Equation, Figure, PaperDocument, Section

try:
    import fitz  # type: ignore[import-not-found,import-untyped]
except ImportError:  # pragma: no cover - exercised only when optional dependency missing
    fitz = None  # type: ignore[assignment]

try:
    import pdfplumber  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised only when optional dependency missing
    pdfplumber = None  # type: ignore[assignment]

try:
    from PIL import Image  # type: ignore[import-not-found,import-untyped]
except ImportError:  # pragma: no cover - exercised only when optional dependency missing
    Image = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

_SECTION_KEYWORDS = {
    "abstract",
    "introduction",
    "related work",
    "method",
    "methods",
    "approach",
    "experiment",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "references",
    "appendix",
}


class PDFParser:
    """Parse PDF files into a structured :class:`PaperDocument`."""

    def __init__(self, figure_output_root: Path | None = None) -> None:
        self.figure_output_root = figure_output_root

    def parse(self, pdf_path: Path) -> PaperDocument:
        """Parse a local PDF into a structured paper representation."""

        if fitz is None:
            raise ImportError(
                "pymupdf is required for PDF parsing. Install with: pip install -e '.[ingestion]'"
            )

        resolved_path = pdf_path.expanduser().resolve()
        if not resolved_path.exists() or not resolved_path.is_file():
            raise FileNotFoundError(f"PDF not found: {resolved_path}")

        LOGGER.info("Parsing PDF: %s", resolved_path)

        doc = fitz.open(str(resolved_path))
        try:
            full_text_pages: list[str] = []
            for page in doc:
                raw_page_text = page.get_text("text")
                full_text_pages.append(raw_page_text if isinstance(raw_page_text, str) else "")
            full_text = "\n".join(full_text_pages)

            sections = self._extract_text_by_section(doc)
            equations = self._extract_equations(doc)
            algorithms = self._extract_algorithms(doc)
            figures_output_dir = self.figure_output_root or resolved_path.parent
            figures = self._extract_figures(doc, figures_output_dir)

            metadata = dict(doc.metadata or {})
            title = str(metadata.get("title") or "").strip() or self._infer_title(
                full_text, resolved_path
            )
            authors = self._parse_authors(str(metadata.get("author") or ""))
            abstract = self._extract_abstract(sections, full_text)
            references = self._extract_references(sections, full_text)
            keywords = self._extract_keywords(str(metadata.get("keywords") or ""), full_text)
            year = self._extract_year(metadata)

            return PaperDocument(
                title=title,
                authors=authors,
                arxiv_id=None,
                doi=None,
                year=year,
                abstract=abstract,
                sections=sections,
                equations=equations,
                algorithms=algorithms,
                figures=figures,
                full_text=full_text,
                pdf_path=resolved_path,
                references=references,
                keywords=keywords,
                domain=self._detect_domain(full_text),
            )
        finally:
            doc.close()

    def _extract_text_by_section(self, doc: Any) -> list[Section]:
        """
        Extract sections using font-size heuristics and heading patterns.

        Heuristic: larger fonts and canonical heading patterns are interpreted as section titles.
        """

        page_texts: list[str] = []
        for page in doc:
            raw_page_text = page.get_text("text")
            page_texts.append(raw_page_text if isinstance(raw_page_text, str) else "")
        spans: list[tuple[int, float, str]] = []
        font_sizes: list[float] = []

        for page_index, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict") or {}
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = self._normalize_whitespace(str(span.get("text", "")))
                        if not text:
                            continue
                        size = float(span.get("size", 0.0) or 0.0)
                        spans.append((page_index, size, text))
                        if size > 0:
                            font_sizes.append(size)

        median_size = statistics.median(font_sizes) if font_sizes else 10.0
        heading_size_threshold = max(median_size + 1.0, median_size * 1.15)

        headers: list[tuple[int, str, int]] = []
        seen_headers: set[tuple[int, str]] = set()
        for page_number, font_size, text in spans:
            if not self._is_section_header(text, font_size, heading_size_threshold):
                continue
            key = (page_number, text.casefold())
            if key in seen_headers:
                continue
            seen_headers.add(key)
            level = self._estimate_section_level(text, font_size, heading_size_threshold)
            headers.append((page_number, text, level))

        if not headers:
            content = "\n\n".join(page_texts).strip()
            if not content:
                return []
            return [Section(title="Full Text", level=1, content=content, page_start=1)]

        sections: list[Section] = []
        for index, (page_number, title, level) in enumerate(headers):
            next_page = headers[index + 1][0] if index + 1 < len(headers) else len(page_texts) + 1
            chunks: list[str] = []
            for current_page in range(page_number, next_page):
                raw_page_text = page_texts[current_page - 1]
                if current_page == page_number:
                    chunks.append(self._text_after_header(raw_page_text, title))
                else:
                    chunks.append(raw_page_text)
            section_content = self._normalize_whitespace("\n".join(chunks), preserve_newlines=True)
            if section_content:
                sections.append(
                    Section(
                        title=title,
                        level=level,
                        content=section_content,
                        page_start=page_number,
                    )
                )

        if pdfplumber is not None:
            try:
                tables = self._extract_tables_via_pdfplumber(doc)
                sections.extend(tables)
            except (OSError, ValueError) as exc:
                LOGGER.warning("pdfplumber table fallback failed: %s", exc)

        return sections

    def _extract_equations(self, doc: Any) -> list[Equation]:
        """Extract equations using LaTeX delimiters and symbolic-line heuristics."""

        equations: list[Equation] = []
        seen: set[tuple[str, int]] = set()

        equation_patterns = [
            re.compile(r"\$(.+?)\$", re.DOTALL),
            re.compile(r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}", re.DOTALL),
            re.compile(r"\\\[(.*?)\\\]", re.DOTALL),
            re.compile(r"\\\((.*?)\\\)", re.DOTALL),
        ]

        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            normalized_text = text.replace("\r\n", "\n")

            for pattern in equation_patterns:
                for match in pattern.finditer(normalized_text):
                    latex = self._normalize_whitespace(match.group(1))
                    if len(latex) < 2:
                        continue
                    context = self._build_context_window(
                        normalized_text, match.start(), match.end()
                    )
                    key = (latex, page_number)
                    if key in seen:
                        continue
                    seen.add(key)
                    equations.append(
                        Equation(
                            latex=latex,
                            description=context[:200],
                            page=page_number,
                        )
                    )

            lines = [line.strip() for line in normalized_text.splitlines()]
            for line_index, line in enumerate(lines):
                if not self._looks_symbolic_equation(line):
                    continue
                context_lines = [
                    lines[line_index - 1] if line_index > 0 else "",
                    line,
                    lines[line_index + 1] if line_index + 1 < len(lines) else "",
                ]
                context = self._normalize_whitespace(" ".join(filter(None, context_lines)))[:200]
                key = (line, page_number)
                if key in seen:
                    continue
                seen.add(key)
                equations.append(Equation(latex=line, description=context, page=page_number))

        return equations

    def _extract_algorithms(self, doc: Any) -> list[Algorithm]:
        """Extract algorithm blocks beginning with ``Algorithm N`` style headers."""

        algorithms: list[Algorithm] = []
        header_pattern = re.compile(r"^\s*Algorithm\s+\d+.*", re.IGNORECASE)

        for page_number, page in enumerate(doc, start=1):
            lines = [
                self._normalize_whitespace(line)
                for line in (page.get_text("text") or "").splitlines()
            ]
            idx = 0
            while idx < len(lines):
                line = lines[idx]
                if not header_pattern.match(line):
                    idx += 1
                    continue

                name = line.strip()
                idx += 1
                block_lines: list[str] = []
                while idx < len(lines):
                    current = lines[idx]
                    if header_pattern.match(current):
                        break
                    if block_lines and self._looks_like_major_heading(current):
                        break
                    block_lines.append(current)
                    idx += 1

                pseudocode = "\n".join([line for line in block_lines if line]).strip()
                if not pseudocode:
                    pseudocode = name

                algorithms.append(
                    Algorithm(
                        name=name,
                        pseudocode=pseudocode,
                        page=page_number,
                        language_hint=self._detect_algorithm_language_hint(pseudocode),
                    )
                )

        if not algorithms:
            algorithms = self._extract_inferred_algorithms(doc)

        return algorithms

    def _extract_inferred_algorithms(self, doc: Any) -> list[Algorithm]:
        """
        Infer algorithm-like blocks from procedural section headings.

        This is a fallback for papers that describe procedures narratively without
        explicit ``Algorithm N`` headers.
        """

        trigger_terms = (
            "training",
            "optimization",
            "decoding",
            "inference",
            "procedure",
            "method",
        )

        inferred: list[Algorithm] = []
        seen_blocks: set[str] = set()

        for page_number, page in enumerate(doc, start=1):
            lines = [
                self._normalize_whitespace(line)
                for line in (page.get_text("text") or "").splitlines()
            ]
            for line_index, line in enumerate(lines):
                normalized_line = line.casefold()
                if not normalized_line:
                    continue
                if not any(term in normalized_line for term in trigger_terms):
                    continue
                if len(line) > 120:
                    continue

                block_lines: list[str] = []
                start_index = line_index + 1
                for idx in range(start_index, min(len(lines), start_index + 45)):
                    current = lines[idx]
                    if not current:
                        if block_lines:
                            break
                        continue
                    if block_lines and self._looks_like_major_heading(current):
                        break
                    block_lines.append(current)

                if len(block_lines) < 3:
                    continue

                pseudocode = "\n".join(block_lines).strip()
                if len(pseudocode) < 140:
                    continue

                signature = pseudocode[:500]
                if signature in seen_blocks:
                    continue
                seen_blocks.add(signature)

                name = f"Algorithm (inferred): {line[:70]}"
                inferred.append(
                    Algorithm(
                        name=name,
                        pseudocode=pseudocode[:4000],
                        page=page_number,
                        language_hint=self._detect_algorithm_language_hint(pseudocode),
                    )
                )

                if len(inferred) >= 5:
                    return inferred

        return inferred

    def _extract_figures(self, doc: Any, output_dir: Path) -> list[Figure]:
        """Extract embedded images using PyMuPDF and save into ``output_dir/figures``."""

        figures: list[Figure] = []
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        for page_number, page in enumerate(doc, start=1):
            page_text = page.get_text("text") or ""
            captions = self._extract_figure_captions(page_text)
            images = page.get_images(full=True)

            for image_index, image in enumerate(images, start=1):
                caption = captions[image_index - 1] if image_index - 1 < len(captions) else ""
                image_path: Path | None = None
                try:
                    xref = int(image[0])
                    image_info = doc.extract_image(xref)
                    raw_image_bytes = image_info.get("image")
                    image_ext = str(image_info.get("ext") or "png").lower()
                    if raw_image_bytes is None:
                        raise ValueError("Image payload missing")

                    target = figures_dir / f"page_{page_number}_img_{image_index}.png"
                    if image_ext == "png":
                        target.write_bytes(raw_image_bytes)
                        image_path = target
                    elif Image is not None:
                        with io.BytesIO(raw_image_bytes) as buffer:
                            with Image.open(buffer) as pil_image:
                                pil_image.save(target, format="PNG")
                        image_path = target
                    else:
                        LOGGER.warning(
                            "Skipping non-PNG figure extraction without Pillow support on page %d",
                            page_number,
                        )
                except (OSError, RuntimeError, ValueError) as exc:
                    LOGGER.warning("Figure extraction failed on page %d: %s", page_number, exc)

                figures.append(
                    Figure(
                        caption=caption,
                        page=page_number,
                        image_path=image_path,
                    )
                )

        return figures

    def _detect_domain(self, text: str) -> str:
        """Detect rough paper domain from keywords in the paper text."""

        normalized = text.casefold()

        keyword_map = [
            (
                "nlp",
                {"transformer", "bert", "gpt", "attention", "token", "language model"},
            ),
            (
                "cv",
                {
                    "convolution",
                    "resnet",
                    "yolo",
                    "segmentation",
                    "detection",
                    "image classification",
                },
            ),
            (
                "rl",
                {"reward", "policy", "q-learning", "environment", "actor-critic"},
            ),
            (
                "systems",
                {"kernel", "mutex", "scheduler", "memory", "throughput", "latency"},
            ),
        ]

        for domain, keywords in keyword_map:
            if any(keyword in normalized for keyword in keywords):
                return domain

        return "theory"

    def _extract_tables_via_pdfplumber(self, doc: Any) -> list[Section]:
        """Fallback extraction for table-like content using pdfplumber."""

        if pdfplumber is None:
            return []

        doc_name = str(getattr(doc, "name", "") or "")
        if not doc_name:
            return []

        sections: list[Section] = []
        with pdfplumber.open(doc_name) as plumber_doc:
            for page_number, page in enumerate(plumber_doc.pages, start=1):
                tables = page.extract_tables() or []
                for table_index, table in enumerate(tables, start=1):
                    rows: list[str] = []
                    for row in table:
                        if not row:
                            continue
                        row_cells = [self._normalize_whitespace(str(cell or "")) for cell in row]
                        rows.append(" | ".join(row_cells))
                    if not rows:
                        continue
                    sections.append(
                        Section(
                            title=f"Table {page_number}.{table_index}",
                            level=3,
                            content="\n".join(rows),
                            page_start=page_number,
                        )
                    )
        return sections

    def _infer_title(self, full_text: str, pdf_path: Path) -> str:
        lines = [self._normalize_whitespace(line) for line in full_text.splitlines()]
        for line in lines:
            if 8 <= len(line) <= 180 and not line.casefold().startswith("arxiv"):
                return line
        return pdf_path.stem.replace("_", " ").strip()

    def _parse_authors(self, raw_authors: str) -> list[str]:
        if not raw_authors.strip():
            return []
        chunks = re.split(r"[;,]", raw_authors)
        return [
            self._normalize_whitespace(chunk)
            for chunk in chunks
            if self._normalize_whitespace(chunk)
        ]

    def _extract_abstract(self, sections: list[Section], full_text: str) -> str:
        for section in sections:
            if "abstract" in section.title.casefold():
                return section.content[:4000]

        abstract_match = re.search(
            r"\babstract\b\s*[:\-]?\s*(.{80,3000}?)(?:\n\s*\n|\n\s*1\.?\s+[A-Z]|\n\s*Introduction\b)",
            full_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if abstract_match:
            return self._normalize_whitespace(abstract_match.group(1))[:4000]

        return self._normalize_whitespace(full_text)[:1200]

    def _extract_references(self, sections: list[Section], full_text: str) -> list[str]:
        reference_blocks = [s.content for s in sections if "reference" in s.title.casefold()]
        raw_references: list[str] = []

        for block in reference_blocks:
            for line in block.splitlines():
                normalized = self._normalize_whitespace(line)
                if normalized:
                    raw_references.append(normalized)

        if raw_references:
            return raw_references

        candidate_lines = [self._normalize_whitespace(line) for line in full_text.splitlines()]
        return [line for line in candidate_lines if re.match(r"^\[?\d+\]?", line)]

    def _extract_keywords(self, metadata_keywords: str, full_text: str) -> list[str]:
        if metadata_keywords.strip():
            parts = re.split(r"[;,]", metadata_keywords)
            return [self._normalize_whitespace(p) for p in parts if self._normalize_whitespace(p)]

        keyword_match = re.search(
            r"\bkeywords?\b\s*[:\-]\s*([^\n]{3,300})",
            full_text,
            flags=re.IGNORECASE,
        )
        if not keyword_match:
            return []
        raw_keywords = keyword_match.group(1)
        parts = re.split(r"[;,]", raw_keywords)
        return [self._normalize_whitespace(p) for p in parts if self._normalize_whitespace(p)]

    def _extract_year(self, metadata: dict[str, Any]) -> int | None:
        date_candidates = [
            str(metadata.get("creationDate") or ""),
            str(metadata.get("modDate") or ""),
        ]
        for candidate in date_candidates:
            match = re.search(r"(19|20)\d{2}", candidate)
            if match:
                return int(match.group(0))
        return None

    def _is_section_header(self, text: str, font_size: float, threshold: float) -> bool:
        cleaned = text.strip()
        if len(cleaned) < 2 or len(cleaned) > 120:
            return False
        if cleaned.casefold() in _SECTION_KEYWORDS:
            return True
        if re.match(r"^\d+(\.\d+)*\s+[A-Z].*$", cleaned):
            return True
        if font_size >= threshold and not cleaned.endswith("."):
            return True
        if cleaned.isupper() and len(cleaned.split()) <= 8:
            return True
        return False

    def _estimate_section_level(self, title: str, font_size: float, threshold: float) -> int:
        if re.match(r"^\d+\.\d+\.\d+\s+", title):
            return 3
        if re.match(r"^\d+\.\d+\s+", title):
            return 2
        if font_size >= threshold + 1.5:
            return 1
        return 2

    def _text_after_header(self, page_text: str, title: str) -> str:
        lines = page_text.splitlines()
        title_folded = title.casefold().strip()
        found = False
        kept_lines: list[str] = []
        for line in lines:
            normalized = self._normalize_whitespace(line)
            if not found and normalized.casefold() == title_folded:
                found = True
                continue
            kept_lines.append(line)
        return "\n".join(kept_lines).strip()

    def _build_context_window(self, text: str, start: int, end: int, radius: int = 110) -> str:
        left = max(0, start - radius)
        right = min(len(text), end + radius)
        context = text[left:right]
        return self._normalize_whitespace(context)[:200]

    def _looks_symbolic_equation(self, line: str) -> bool:
        stripped = line.strip()
        if len(stripped) < 6 or len(stripped) > 220:
            return False
        if stripped.casefold().startswith("algorithm"):
            return False

        non_space_chars = [ch for ch in stripped if not ch.isspace()]
        if not non_space_chars:
            return False

        alpha = sum(char.isalpha() for char in non_space_chars)
        non_alpha_ratio = (len(non_space_chars) - alpha) / len(non_space_chars)
        has_math_symbol = any(symbol in stripped for symbol in "=_^{}[]\\/*+-<>|≈∑∏")

        if has_math_symbol and non_alpha_ratio >= 0.45:
            return True
        if re.match(r"^[A-Za-z]\s*=\s*.+", stripped):
            return True
        return False

    def _looks_like_major_heading(self, line: str) -> bool:
        cleaned = line.strip()
        if not cleaned:
            return False
        if re.match(r"^\d+(\.\d+)*\s+[A-Z]", cleaned):
            return True
        if cleaned.casefold() in _SECTION_KEYWORDS:
            return True
        if cleaned.isupper() and len(cleaned.split()) <= 8:
            return True
        return False

    def _detect_algorithm_language_hint(self, pseudocode: str) -> str:
        lowered = pseudocode.casefold()
        if any(token in lowered for token in ["for ", "while ", "if ", "return", "def "]):
            return "python-like"
        symbolic_chars = sum(c in "=_^{}[]\\/*+-<>|" for c in pseudocode)
        if pseudocode and symbolic_chars / max(len(pseudocode), 1) > 0.08:
            return "math"
        return "unknown"

    def _extract_figure_captions(self, page_text: str) -> list[str]:
        captions: list[str] = []
        for line in page_text.splitlines():
            normalized = self._normalize_whitespace(line)
            if not normalized:
                continue
            figure_match = re.match(
                r"^(Figure|Fig\.)\s*\d+[:\.]?\s*(.*)$", normalized, re.IGNORECASE
            )
            if figure_match:
                caption = figure_match.group(2).strip() or normalized
                captions.append(caption)
        return captions

    def _normalize_whitespace(self, text: str, *, preserve_newlines: bool = False) -> str:
        if preserve_newlines:
            lines = [" ".join(line.split()) for line in text.splitlines()]
            return "\n".join([line for line in lines if line]).strip()
        return " ".join(text.split()).strip()
