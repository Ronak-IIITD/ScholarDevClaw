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
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore[assignment]

try:
    import pdfplumber  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    pdfplumber = None  # type: ignore[assignment]

try:
    from PIL import Image  # type: ignore[import-not-found,import-untyped]
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

_SECTION_TYPE_PATTERNS = [
    ("abstract", re.compile(r"^abstract$", re.IGNORECASE)),
    ("introduction", re.compile(r"^(?:\d+\.?\s+)?introduction$", re.IGNORECASE)),
    ("related", re.compile(r"related work|background|prior work", re.IGNORECASE)),
    ("method", re.compile(r"method|approach|model|architecture|algorithm", re.IGNORECASE)),
    ("experiments", re.compile(r"experiment|evaluation|results|benchmark", re.IGNORECASE)),
    ("conclusion", re.compile(r"conclusion|discussion|future work", re.IGNORECASE)),
]

_DOMAIN_KEYWORDS: list[tuple[str, dict[str, str]]] = [
    (
        "nlp",
        {
            "language-modeling": r"language model|causal lm|masked lm|perplexity|tokeniz",
            "machine-translation": r"translation|bleu|encoder-decoder",
            "question-answering": r"question answering|qa benchmark|squad",
            "general": r"transformer|bert|gpt|attention|tokeniz|embedding",
        },
    ),
    (
        "cv",
        {
            "object-detection": r"object detection|yolo|faster r-cnn|mAP",
            "segmentation": r"segmentation|mask r-cnn|iou",
            "classification": r"image classification|resnet|vit|imagenet",
            "general": r"convolution|resnet|yolo|detection|segmentation|vision transformer|vit",
        },
    ),
    (
        "rl",
        {
            "policy-gradient": r"policy gradient|ppo|actor-critic",
            "value-learning": r"q-learning|dqn|td error",
            "general": r"reward|policy|environment|agent|markov|reinforcement learning",
        },
    ),
    (
        "systems",
        {
            "distributed-systems": r"distributed|consensus|replication|fault tolerance",
            "memory-systems": r"cache|memory allocator|paging|tlb",
            "general": r"kernel|mutex|scheduler|memory|cache|network throughput|latency",
        },
    ),
    (
        "theory",
        {
            "optimization-theory": r"convex|convergence|regret",
            "general": r"theorem|proof|lemma|corollary|bound|complexity",
        },
    ),
    (
        "biology",
        {
            "protein-modeling": r"protein|amino acid|folding",
            "genomics": r"genom|rna|dna|sequence alignment|cell",
            "general": r"protein|sequence|genomics|rna|dna|cell|biopython|rdkit",
        },
    ),
    (
        "multimodal",
        {
            "vision-language": r"vision-language|image-text|captioning|clip",
            "audio-visual": r"audio-visual|speech-image|multimodal fusion",
            "general": r"vision-language|clip|image-text|audio-visual|multimodal",
        },
    ),
]


class PDFParser:
    """Parse PDF files into a structured :class:`PaperDocument`."""

    def __init__(self, figure_output_root: Path | None = None) -> None:
        self.figure_output_root = figure_output_root

    def parse(self, pdf_path: Path) -> PaperDocument:
        if fitz is None:
            raise ImportError(
                "pymupdf is required for PDF parsing. Install with: pip install -e '.[ingestion]'"
            )

        resolved_path = pdf_path.expanduser().resolve()
        if not resolved_path.exists() or not resolved_path.is_file():
            raise FileNotFoundError(f"PDF not found: {resolved_path}")

        LOGGER.info("Parsing PDF: %s", resolved_path)
        document = fitz.open(str(resolved_path))
        try:
            page_texts = self._extract_page_texts(document)
            full_text = "\n".join(page_texts).strip()
            sections = self._extract_sections(document, page_texts)
            equations = self._extract_equations(page_texts)
            algorithms = self._extract_algorithms(page_texts)
            figures = self._extract_figures(document, page_texts, resolved_path.parent)
            tables = self._extract_tables(resolved_path)
            metadata = dict(document.metadata or {})
        finally:
            document.close()

        title = str(metadata.get("title") or "").strip() or self._infer_title(full_text, resolved_path)
        abstract = self._extract_abstract(sections, full_text)
        domain, subdomain = self._detect_domain_and_subdomain(full_text)

        return PaperDocument(
            title=title,
            authors=self._parse_authors(str(metadata.get("author") or "")),
            arxiv_id=None,
            doi=None,
            year=self._extract_year(metadata),
            abstract=abstract,
            venue=self._extract_venue(metadata, full_text),
            sections=sections,
            equations=equations,
            algorithms=algorithms,
            figures=figures,
            tables=tables,
            full_text=full_text,
            pdf_path=resolved_path,
            source_url=None,
            references=self._extract_references(sections, full_text),
            keywords=self._extract_keywords(str(metadata.get("keywords") or ""), full_text),
            domain=domain,
            subdomain=subdomain,
        )

    def _extract_page_texts(self, document: Any) -> list[str]:
        page_texts: list[str] = []
        for page in document:
            raw = page.get_text("text")
            page_texts.append(raw if isinstance(raw, str) else "")
        return page_texts

    def _extract_sections(self, document: Any, page_texts: list[str]) -> list[Section]:
        spans: list[tuple[int, float, str]] = []
        font_sizes: list[float] = []

        for page_index, page in enumerate(document, start=1):
            page_dict = page.get_text("dict") or {}
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = self._normalize_whitespace(str(span.get("text", "")))
                        if not text:
                            continue
                        size = float(span.get("size", 0.0) or 0.0)
                        spans.append((page_index, size, text))
                        if size > 0.0:
                            font_sizes.append(size)

        if not spans:
            content = "\n\n".join(page_texts).strip()
            if not content:
                return []
            return [
                Section(
                    title="Full Text",
                    level=1,
                    content=content,
                    page_start=1,
                    section_type="unknown",
                )
            ]

        median_size = statistics.median(font_sizes) if font_sizes else 10.0
        heading_threshold = max(median_size + 1.0, median_size * 1.15)

        headers: list[tuple[int, str, int]] = []
        seen_headers: set[tuple[int, str]] = set()
        for page_number, font_size, text in spans:
            if not self._is_section_header(text, font_size, heading_threshold):
                continue
            key = (page_number, text.casefold())
            if key in seen_headers:
                continue
            seen_headers.add(key)
            headers.append(
                (page_number, text, self._estimate_section_level(text, font_size, heading_threshold))
            )

        if not headers:
            return [
                Section(
                    title="Full Text",
                    level=1,
                    content="\n\n".join(page_texts).strip(),
                    page_start=1,
                    section_type="unknown",
                )
            ]

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

            content = self._normalize_whitespace("\n".join(chunks), preserve_newlines=True)
            if not content:
                continue
            sections.append(
                Section(
                    title=title,
                    level=level,
                    content=content,
                    page_start=page_number,
                    section_type=self._classify_section_type(title),
                )
            )

        return sections

    def _extract_equations(self, page_texts: list[str]) -> list[Equation]:
        equations: list[Equation] = []
        seen: set[tuple[str, int]] = set()
        patterns = [
            re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
            re.compile(r"\$(.+?)\$", re.DOTALL),
            re.compile(r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}", re.DOTALL),
            re.compile(r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}", re.DOTALL),
            re.compile(r"\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}", re.DOTALL),
            re.compile(r"\\\[(.*?)\\\]", re.DOTALL),
            re.compile(r"\\\((.*?)\\\)", re.DOTALL),
        ]

        for page_number, page_text in enumerate(page_texts, start=1):
            normalized_text = page_text.replace("\r\n", "\n")
            for pattern in patterns:
                for match in pattern.finditer(normalized_text):
                    latex = self._normalize_whitespace(match.group(1))
                    if len(latex) < 2:
                        continue
                    key = (latex, page_number)
                    if key in seen:
                        continue
                    seen.add(key)
                    context = self._build_context_window(normalized_text, match.start(), match.end())
                    equations.append(
                        Equation(
                            latex=latex,
                            description=context[:200],
                            page=page_number,
                            equation_type=self._classify_equation_type(latex, context),
                        )
                    )

            lines = [line.strip() for line in normalized_text.splitlines()]
            for index, line in enumerate(lines):
                if not self._looks_symbolic_equation(line):
                    continue
                key = (line, page_number)
                if key in seen:
                    continue
                seen.add(key)
                context = " ".join(
                    filter(
                        None,
                        [
                            lines[index - 1] if index > 0 else "",
                            line,
                            lines[index + 1] if index + 1 < len(lines) else "",
                        ],
                    )
                )
                equations.append(
                    Equation(
                        latex=line,
                        description=self._normalize_whitespace(context)[:200],
                        page=page_number,
                        equation_type=self._classify_equation_type(line, context),
                    )
                )

        return equations

    def _extract_algorithms(self, page_texts: list[str]) -> list[Algorithm]:
        algorithms: list[Algorithm] = []
        seen_blocks: set[tuple[int, str]] = set()
        header_pattern = re.compile(r"^\s*(Algorithm|Procedure)\s+\d+[:.]?.*", re.IGNORECASE)

        for page_number, page_text in enumerate(page_texts, start=1):
            lines = page_text.splitlines()
            idx = 0
            while idx < len(lines):
                current = lines[idx].rstrip()
                normalized = self._normalize_whitespace(current)
                if not normalized:
                    idx += 1
                    continue

                starts_block = bool(header_pattern.match(normalized)) or self._looks_like_algorithm_line(
                    normalized
                )
                if not starts_block:
                    idx += 1
                    continue

                name = normalized
                block_lines = [current.strip()] if header_pattern.match(normalized) else []
                idx += 1
                while idx < len(lines):
                    candidate = lines[idx].rstrip("\n")
                    candidate_norm = self._normalize_whitespace(candidate)
                    if not candidate_norm and block_lines:
                        break
                    if block_lines and header_pattern.match(candidate_norm):
                        break
                    if block_lines and self._looks_like_major_heading(candidate_norm):
                        break
                    if not block_lines and not candidate_norm:
                        idx += 1
                        continue
                    if candidate_norm:
                        block_lines.append(candidate.rstrip())
                    idx += 1

                if not block_lines:
                    continue
                pseudocode = "\n".join(block_lines).strip()
                signature = (page_number, pseudocode[:250])
                if signature in seen_blocks:
                    continue
                seen_blocks.add(signature)
                inputs, outputs = self._extract_algorithm_io(block_lines)
                algorithms.append(
                    Algorithm(
                        name=name,
                        pseudocode=pseudocode[:6000],
                        page=page_number,
                        language_hint=self._detect_algorithm_language_hint(pseudocode),
                        inputs=inputs,
                        outputs=outputs,
                    )
                )

        return algorithms

    def _extract_figures(self, document: Any, page_texts: list[str], base_dir: Path) -> list[Figure]:
        figures: list[Figure] = []
        figures_dir = (self.figure_output_root or base_dir) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        for page_number, page in enumerate(document, start=1):
            captions = self._extract_figure_captions(page_texts[page_number - 1])
            images = page.get_images(full=True)
            for image_index, image in enumerate(images, start=1):
                image_path: Path | None = None
                try:
                    xref = int(image[0])
                    image_info = document.extract_image(xref)
                    payload = image_info.get("image")
                    image_ext = str(image_info.get("ext") or "png").lower()
                    if payload is None:
                        raise ValueError("Missing image payload")
                    target = figures_dir / f"fig_{page_number}_{image_index}.png"
                    if image_ext == "png":
                        target.write_bytes(payload)
                        image_path = target
                    elif Image is not None:
                        with io.BytesIO(payload) as buffer:
                            with Image.open(buffer) as extracted:
                                extracted.save(target, format="PNG")
                        image_path = target
                except (OSError, RuntimeError, ValueError) as exc:
                    LOGGER.warning("Figure extraction failed on page %s image %s: %s", page_number, image_index, exc)

                caption = captions[image_index - 1] if image_index - 1 < len(captions) else ""
                figures.append(
                    Figure(
                        caption=caption,
                        page=page_number,
                        figure_type=self._classify_figure_type(caption),
                        image_path=image_path,
                    )
                )
        return figures

    def _extract_tables(self, pdf_path: Path) -> list[dict[str, Any]]:
        if pdfplumber is None:
            return []

        tables: list[dict[str, Any]] = []
        with pdfplumber.open(str(pdf_path)) as plumber_doc:
            for page_number, page in enumerate(plumber_doc.pages, start=1):
                extracted_tables = page.extract_tables() or []
                for table_index, table in enumerate(extracted_tables, start=1):
                    rows: list[list[str]] = []
                    for row in table:
                        if row is None:
                            continue
                        rows.append([self._normalize_whitespace(str(cell or "")) for cell in row])
                    if not rows:
                        continue
                    tables.append(
                        {
                            "page": page_number,
                            "table_index": table_index,
                            "rows": rows,
                        }
                    )
        return tables

    def _infer_title(self, full_text: str, pdf_path: Path) -> str:
        for line in full_text.splitlines():
            normalized = self._normalize_whitespace(line)
            if 8 <= len(normalized) <= 180 and not normalized.casefold().startswith("arxiv"):
                return normalized
        return pdf_path.stem.replace("_", " ").strip()

    def _parse_authors(self, raw_authors: str) -> list[str]:
        if not raw_authors.strip():
            return []
        parts = re.split(r"[;,]", raw_authors)
        return [self._normalize_whitespace(part) for part in parts if self._normalize_whitespace(part)]

    def _extract_abstract(self, sections: list[Section], full_text: str) -> str:
        for section in sections:
            if section.section_type == "abstract" or "abstract" in section.title.casefold():
                return section.content[:4000]

        match = re.search(
            r"\babstract\b\s*[:\-]?\s*(.{80,3000}?)(?:\n\s*\n|\n\s*1\.?\s+[A-Z]|\n\s*Introduction\b)",
            full_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return self._normalize_whitespace(match.group(1))[:4000]
        return self._normalize_whitespace(full_text)[:1200]

    def _extract_references(self, sections: list[Section], full_text: str) -> list[str]:
        references: list[str] = []
        for section in sections:
            if "reference" in section.title.casefold():
                references.extend(
                    [self._normalize_whitespace(line) for line in section.content.splitlines() if self._normalize_whitespace(line)]
                )
        if references:
            return references
        return [
            self._normalize_whitespace(line)
            for line in full_text.splitlines()
            if re.match(r"^\[?\d+\]?\s+\S+", self._normalize_whitespace(line))
        ]

    def _extract_keywords(self, metadata_keywords: str, full_text: str) -> list[str]:
        if metadata_keywords.strip():
            parts = re.split(r"[;,]", metadata_keywords)
            return [self._normalize_whitespace(part) for part in parts if self._normalize_whitespace(part)]
        match = re.search(r"\bkeywords?\b\s*[:\-]\s*([^\n]{3,300})", full_text, re.IGNORECASE)
        if match is None:
            return []
        parts = re.split(r"[;,]", match.group(1))
        return [self._normalize_whitespace(part) for part in parts if self._normalize_whitespace(part)]

    def _extract_year(self, metadata: dict[str, Any]) -> int | None:
        for candidate in (str(metadata.get("creationDate") or ""), str(metadata.get("modDate") or "")):
            match = re.search(r"(19|20)\d{2}", candidate)
            if match:
                return int(match.group(0))
        return None

    def _extract_venue(self, metadata: dict[str, Any], full_text: str) -> str | None:
        subject = str(metadata.get("subject") or "").strip()
        if subject:
            return subject
        match = re.search(r"(NeurIPS|ICML|ICLR|ACL|EMNLP|CVPR|ECCV|ICCV)\s+\d{4}", full_text)
        if match:
            return match.group(0)
        return None

    def _detect_domain_and_subdomain(self, full_text: str) -> tuple[str, str]:
        normalized = full_text.casefold()
        for domain, subdomains in _DOMAIN_KEYWORDS:
            for subdomain, pattern in subdomains.items():
                if re.search(pattern, normalized):
                    if subdomain == "general":
                        return domain, "general"
                    return domain, subdomain
        return "theory", "general"

    def _classify_section_type(self, title: str) -> str:
        normalized = self._normalize_whitespace(title)
        for section_type, pattern in _SECTION_TYPE_PATTERNS:
            if pattern.search(normalized):
                return section_type
        return "unknown"

    def _classify_equation_type(self, latex: str, context: str) -> str:
        combined = f"{latex} {context}".casefold()
        if any(token in combined for token in ("loss", "objective", "cross entropy", "regularization")):
            return "loss"
        if any(token in combined for token in ("accuracy", "bleu", "f1", "precision", "recall", "perplexity", "metric")):
            return "metric"
        if any(token in combined for token in ("embedding", "attention", "transformer", "decoder", "encoder", "model")):
            return "model"
        if any(token in combined for token in ("notation", "where", "let", "denote")):
            return "notation"
        return "unknown"

    def _classify_figure_type(self, caption: str) -> str:
        normalized = caption.casefold()
        if any(term in normalized for term in ("architecture", "model overview", "pipeline")):
            return "architecture"
        if any(term in normalized for term in ("result", "bleu", "accuracy", "ablation")):
            return "results"
        if any(term in normalized for term in ("plot", "curve", "histogram")):
            return "plot"
        return "diagram"

    def _extract_algorithm_io(self, block_lines: list[str]) -> tuple[list[str], list[str]]:
        inputs: list[str] = []
        outputs: list[str] = []
        for line in block_lines:
            normalized = self._normalize_whitespace(line)
            lower = normalized.casefold()
            if lower.startswith("input:") or lower.startswith("inputs:"):
                inputs.extend(self._split_io_values(normalized.split(":", 1)[1]))
            if lower.startswith("output:") or lower.startswith("outputs:"):
                outputs.extend(self._split_io_values(normalized.split(":", 1)[1]))
        return inputs, outputs

    def _split_io_values(self, value: str) -> list[str]:
        parts = re.split(r"[,;/]", value)
        cleaned = [self._normalize_whitespace(part) for part in parts if self._normalize_whitespace(part)]
        return cleaned

    def _is_section_header(self, text: str, font_size: float, threshold: float) -> bool:
        cleaned = text.strip()
        if len(cleaned) < 2 or len(cleaned) > 120:
            return False
        if re.match(r"^\d+(\.\d+)*\s+[A-Z].*$", cleaned):
            return True
        if font_size >= threshold and not cleaned.endswith("."):
            return True
        if cleaned.isupper() and len(cleaned.split()) <= 8:
            return True
        return any(pattern.search(cleaned) for _, pattern in _SECTION_TYPE_PATTERNS)

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
        header = self._normalize_whitespace(title).casefold()
        found = False
        kept: list[str] = []
        for line in lines:
            normalized = self._normalize_whitespace(line).casefold()
            if not found and normalized == header:
                found = True
                continue
            kept.append(line)
        return "\n".join(kept).strip()

    def _build_context_window(self, text: str, start: int, end: int, radius: int = 110) -> str:
        left = max(0, start - radius)
        right = min(len(text), end + radius)
        return self._normalize_whitespace(text[left:right])[:200]

    def _looks_symbolic_equation(self, line: str) -> bool:
        stripped = line.strip()
        if len(stripped) < 6 or len(stripped) > 220:
            return False
        if stripped.casefold().startswith(("algorithm", "procedure", "figure")):
            return False
        non_space = [char for char in stripped if not char.isspace()]
        if not non_space:
            return False
        alpha = sum(char.isalpha() for char in non_space)
        non_alpha_ratio = (len(non_space) - alpha) / len(non_space)
        has_math_symbol = any(symbol in stripped for symbol in "=_^{}[]\\/*+-<>|≈∑∏")
        return (has_math_symbol and non_alpha_ratio >= 0.45) or bool(
            re.match(r"^[A-Za-z]\s*=\s*.+", stripped)
        )

    def _looks_like_major_heading(self, line: str) -> bool:
        cleaned = line.strip()
        if not cleaned:
            return False
        if re.match(r"^\d+(\.\d+)*\s+[A-Z]", cleaned):
            return True
        if cleaned.isupper() and len(cleaned.split()) <= 8:
            return True
        return any(pattern.search(cleaned) for _, pattern in _SECTION_TYPE_PATTERNS)

    def _looks_like_algorithm_line(self, line: str) -> bool:
        lowered = line.casefold()
        return lowered.startswith(("input:", "inputs:", "output:", "outputs:", "for ", "while ", "repeat ", "return "))

    def _detect_algorithm_language_hint(self, pseudocode: str) -> str:
        lowered = pseudocode.casefold()
        if any(token in lowered for token in ("for ", "while ", "if ", "return", "def ", "range(")):
            return "python-like"
        if any(token in lowered for token in ("{", "}", "->", "++", "--", "printf")):
            return "c-like"
        symbolic_chars = sum(char in "=_^{}[]\\/*+-<>|" for char in pseudocode)
        if pseudocode and symbolic_chars / max(len(pseudocode), 1) > 0.08:
            return "math"
        return "unknown"

    def _extract_figure_captions(self, page_text: str) -> list[str]:
        captions: list[str] = []
        for line in page_text.splitlines():
            normalized = self._normalize_whitespace(line)
            if not normalized:
                continue
            match = re.match(r"^(Figure|Fig\.)\s*\d+[:.]?\s*(.*)$", normalized, re.IGNORECASE)
            if match:
                captions.append(match.group(2).strip() or normalized)
        return captions

    def _normalize_whitespace(self, text: str, *, preserve_newlines: bool = False) -> str:
        if preserve_newlines:
            lines = [" ".join(line.split()) for line in text.splitlines()]
            return "\n".join([line for line in lines if line]).strip()
        return " ".join(text.split()).strip()
