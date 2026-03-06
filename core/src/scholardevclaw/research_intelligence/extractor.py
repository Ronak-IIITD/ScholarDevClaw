"""
Research paper extraction and specification lookup.

This module provides the :class:`ResearchExtractor` which:

- Maintains a registry of hardcoded paper specs (``PAPER_SPECS``).
- Searches the local registry by keyword or code pattern.
- Queries arXiv for papers matching a :class:`ResearchQuery`.
- Extracts implementation specs from PDFs / arXiv IDs using an optional
  :class:`~scholardevclaw.llm.research_assistant.LLMResearchAssistant`.

When no LLM is available, extraction falls back to the hardcoded specs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from scholardevclaw.llm.research_assistant import LLMResearchAssistant

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardcoded spec registry (serves as fallback + seed data)
# ---------------------------------------------------------------------------

PAPER_SPECS: Dict[str, Dict] = {
    "rmsnorm": {
        "paper": {
            "title": "Root Mean Square Layer Normalization",
            "authors": ["Biao Zhang", "Rico Sennrich"],
            "arxiv": "1910.07467",
            "year": 2019,
        },
        "algorithm": {
            "name": "RMSNorm",
            "replaces": "LayerNorm",
            "description": "Simplified layer normalization without mean-centering",
            "formula": "x / sqrt(mean(x^2) + eps) * gamma",
            "complexity": "O(n)",
            "category": "normalization",
        },
        "implementation": {
            "module_name": "RMSNorm",
            "parent_class": "nn.Module",
            "parameters": ["ndim", "eps"],
            "forward_signature": "(x: Tensor) -> Tensor",
        },
        "changes": {
            "type": "replace",
            "target_patterns": ["LayerNorm", "nn.LayerNorm"],
            "replacement": "RMSNorm",
            "insertion_points": ["Block class", "GPT class"],
            "expected_benefits": [
                "5-10% training speedup",
                "Simplified computation",
                "Better gradient flow",
            ],
        },
        "validation": {
            "test_type": "training_comparison",
            "metrics": ["loss", "perplexity", "tokens_per_second", "memory_usage"],
            "max_benchmark_time": 300,
        },
    },
    "swiglu": {
        "paper": {
            "title": "SwiGLU: Swish-Gated Linear Unit",
            "authors": ["Noam Shazeer"],
            "arxiv": "",
            "year": 2020,
        },
        "algorithm": {
            "name": "SwiGLU",
            "replaces": "MLP (GELU)",
            "description": "Swish-Gated Linear Unit - combines Swish activation with gated linear units",
            "formula": "Swish(x @ W) * (x @ V) / gate",
            "category": "activation",
        },
        "implementation": {
            "module_name": "SwiGLU",
            "parent_class": "nn.Module",
            "parameters": ["config"],
            "forward_signature": "(x: Tensor) -> Tensor",
        },
        "changes": {
            "type": "replace",
            "target_patterns": ["class MLP", "self.gelu = nn.GELU()"],
            "replacement": "class SwiGLU",
            "insertion_points": ["MLP class"],
            "expected_benefits": ["Improved performance", "Better gradient flow"],
        },
        "validation": {
            "test_type": "training_comparison",
            "metrics": ["loss", "perplexity"],
            "max_benchmark_time": 300,
        },
    },
    "flashattention": {
        "paper": {
            "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
            "authors": [
                "Tri Dao",
                "Daniel Y. Fu",
                "Stefano Ermon",
                "Christopher Ré",
                "Zachary C. Brown",
            ],
            "arxiv": "2205.14135",
            "year": 2022,
        },
        "algorithm": {
            "name": "FlashAttention",
            "replaces": "CausalSelfAttention (slow)",
            "description": "IO-aware exact attention algorithm that reduces memory complexity from O(N^2) to O(N)",
            "formula": "Softmax(QK^T / sqrt(d))V with tiling and recomputation",
            "category": "attention",
        },
        "implementation": {
            "module_name": "FlashAttention",
            "parent_class": "nn.Module",
            "parameters": ["config"],
            "forward_signature": "(q, k, v: Tensor) -> Tensor",
        },
        "changes": {
            "type": "replace",
            "target_patterns": ["class CausalSelfAttention", "self.flash"],
            "replacement": "FlashAttention",
            "insertion_points": ["CausalSelfAttention class"],
            "expected_benefits": ["2-4x speedup", "Reduced memory usage", "IO-awareness"],
        },
        "validation": {
            "test_type": "benchmark",
            "metrics": ["tokens_per_second", "memory_usage"],
            "max_benchmark_time": 300,
        },
    },
    "rope": {
        "paper": {
            "title": "RoFormer: Enhanced Transformer with Rotary Position Embedding",
            "authors": [
                "Jianlin Su",
                "Yu Lu",
                "Shengfeng Pan",
                "Ahmed Murtadha",
                "Bohan Zhou",
                "Yunfeng Liu",
            ],
            "arxiv": "2104.09864",
            "year": 2021,
        },
        "algorithm": {
            "name": "RoPE",
            "replaces": "Positional Encoding",
            "description": "Rotary Position Embedding - encodes position information through rotation matrices",
            "formula": "RoPE(x_m, x_n) = f(x_m, m) * f(x_n, n)^conj where f(x, k) = x * e^(ikθ)",
            "category": "position_encoding",
        },
        "implementation": {
            "module_name": "RoPE",
            "parent_class": "nn.Module",
            "parameters": ["dim", "max_seq_len"],
            "forward_signature": "(x: Tensor, positions: Tensor) -> Tensor",
        },
        "changes": {
            "type": "replace",
            "target_patterns": ["self.wpe", "nn.Embedding"],
            "replacement": "RoPE",
            "insertion_points": ["GPT class __init__", "GPT class forward"],
            "expected_benefits": [
                "Better length extrapolation",
                "Improved long-range dependencies",
            ],
        },
        "validation": {
            "test_type": "training_comparison",
            "metrics": ["loss", "perplexity"],
            "max_benchmark_time": 300,
        },
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Paper:
    """Represents a research paper"""

    id: str
    title: str
    authors: List[str]
    abstract: str
    arxiv_id: Optional[str] = None
    pdf_url: Optional[str] = None
    published: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    year: int = 2024


@dataclass
class ResearchQuery:
    """Search query for research papers"""

    keywords: List[str]
    domain: str = "cs.AI"
    max_results: int = 10
    year_from: Optional[int] = None


@dataclass
class ImplementationSpec:
    """Specification for implementing a paper"""

    paper: Paper
    target_language: str
    target_patterns: List[str]
    code_template: str
    expected_benefits: List[str]
    risks: List[str] = field(default_factory=list)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# PDF text extraction helper
# ---------------------------------------------------------------------------


def _read_pdf_text(pdf_path: str, max_pages: int = 20) -> Optional[str]:
    """Best-effort plain-text extraction from a PDF file.

    Tries ``PyPDF2`` (commonly available) then ``pdfminer.six``.
    Returns ``None`` when no reader is installed or the file is unreadable.
    """
    path = Path(pdf_path)
    if not path.exists() or not path.suffix.lower() == ".pdf":
        return None

    # Try PyPDF2
    try:
        from PyPDF2 import PdfReader  # type: ignore[import-untyped]

        reader = PdfReader(str(path))
        pages = reader.pages[:max_pages]
        text_parts = [page.extract_text() or "" for page in pages]
        text = "\n".join(text_parts).strip()
        if text:
            return text
    except Exception:
        pass

    # Try pdfminer
    try:
        from pdfminer.high_level import extract_text as pm_extract  # type: ignore[import-untyped]

        text = pm_extract(str(path), maxpages=max_pages)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass

    return None


def _fetch_arxiv_abstract(arxiv_id: str) -> Optional[str]:
    """Fetch the abstract for an arXiv paper via the Atom feed API.

    This is a lightweight synchronous call — no heavy ``arxiv`` library needed.
    """
    clean_id = arxiv_id.strip().split("/")[-1]  # handle full URLs too
    try:
        import httpx

        url = f"http://export.arxiv.org/api/query?id_list={clean_id}"
        resp = httpx.get(url, timeout=15.0, follow_redirects=True)
        if resp.status_code != 200:
            return None

        # Quick XML extraction (no lxml dependency)
        import re

        match = re.search(r"<summary[^>]*>(.*?)</summary>", resp.text, re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            # Clean up whitespace artefacts
            abstract = re.sub(r"\s+", " ", abstract)
            return abstract
    except Exception as exc:
        logger.debug("arXiv abstract fetch failed for %s: %s", arxiv_id, exc)

    return None


# ---------------------------------------------------------------------------
# ResearchExtractor
# ---------------------------------------------------------------------------


class ResearchExtractor:
    """Handles research paper extraction and search.

    Parameters
    ----------
    llm_assistant : LLMResearchAssistant | None
        Optional LLM assistant for AI-powered spec extraction.
        When provided, ``_extract_from_pdf`` and ``_extract_from_arxiv``
        will attempt LLM-based extraction before falling back to the
        hardcoded spec registry.
    """

    def __init__(
        self,
        llm_assistant: "LLMResearchAssistant | None" = None,
    ) -> None:
        self.specs = PAPER_SPECS
        self._arxiv_client = None
        self._llm = llm_assistant

    def _get_arxiv_client(self):
        """Lazy load arxiv client"""
        if self._arxiv_client is None:
            try:
                import arxiv

                self._arxiv_client = arxiv.Client()
            except ImportError:
                return None
        return self._arxiv_client

    async def search_arxiv(self, query: ResearchQuery) -> List[Paper]:
        """Search arXiv for papers matching the query"""
        client = self._get_arxiv_client()

        if client is None:
            return []

        try:
            import arxiv  # noqa: F811 — lazy import matching _get_arxiv_client

            search_query = " AND ".join(query.keywords)
            search = arxiv.Search(
                query=search_query,
                max_results=query.max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            papers = []
            for result in client.results(search):
                papers.append(
                    Paper(
                        id=result.entry_id,
                        title=result.title,
                        authors=[a.name for a in result.authors],
                        abstract=result.summary,
                        arxiv_id=result.get_short_id(),
                        pdf_url=result.pdf_url,
                        published=str(result.published) if result.published else None,
                        categories=result.categories,
                        year=result.published.year if result.published else 2024,
                    )
                )

            return papers
        except Exception as e:
            logger.warning("arXiv search error: %s", e)
            return []

    def search_by_keyword(self, keyword: str, max_results: int = 10) -> List[Dict]:
        """Search papers by keyword - simplified version"""
        return self._search_local(keyword, max_results)

    def _search_local(self, query: str, max_results: int = 10) -> List[Dict]:
        """Local search in predefined specs"""
        query_lower = query.lower()
        results = []

        for name, spec in self.specs.items():
            title = spec.get("paper", {}).get("title", "").lower()
            category = spec.get("algorithm", {}).get("category", "").lower()

            if query_lower in title or query_lower in category:
                results.append(
                    {
                        "name": name,
                        "title": spec["paper"]["title"],
                        "authors": spec["paper"]["authors"],
                        "arxiv": spec["paper"].get("arxiv", ""),
                        "year": spec["paper"]["year"],
                        "category": spec["algorithm"]["category"],
                        "replaces": spec["algorithm"]["replaces"],
                        "description": spec["algorithm"]["description"],
                    }
                )

                if len(results) >= max_results:
                    break

        return results

    def find_papers_for_code_pattern(
        self, code_pattern: str, language: str = "python"
    ) -> List[Dict]:
        """Find papers relevant to a code pattern.

        First checks the local keyword → spec mapping.  When an LLM
        assistant is available *and* the local lookup returned nothing,
        falls back to LLM-based code analysis for richer suggestions.
        """
        pattern_lower = code_pattern.lower()

        mapping = {
            "norm": ["rmsnorm", "preln_transformer", "qknorm"],
            "normalization": ["rmsnorm", "preln_transformer", "qknorm"],
            "layer": ["rmsnorm", "preln_transformer"],
            "attention": ["flashattention", "flashattention2", "rope", "grouped_query_attention"],
            "mlp": ["swiglu", "mistral"],
            "feedforward": ["swiglu", "mistral"],
            "position": ["rope"],
            "positional": ["rope"],
            "optimizer": ["weight_decay_fused"],
            "adam": ["weight_decay_fused"],
            "expert": ["mistral"],
            "moe": ["mistral"],
        }

        results: List[Dict] = []
        for key, spec_names in mapping.items():
            if key in pattern_lower:
                for spec_name in spec_names:
                    if spec_name in self.specs:
                        spec = self.specs[spec_name]
                        results.append(
                            {
                                "name": spec_name,
                                "title": spec["paper"]["title"],
                                "match_reason": f"Matches '{key}' pattern",
                                "category": spec["algorithm"]["category"],
                            }
                        )

        # LLM enhancement: if local lookup returned nothing, ask the LLM
        if not results and self._llm is not None and self._llm.is_available:
            try:
                analysis = self._llm.analyse_code(
                    code_pattern,
                    language=language,
                    focus="ML research improvements",
                )
                if analysis is not None:
                    for opp in analysis.improvement_opportunities:
                        results.append(
                            {
                                "name": opp.get("suggested_improvement", "unknown"),
                                "title": opp.get("paper_reference", ""),
                                "match_reason": opp.get("description", "LLM suggestion"),
                                "category": "llm_suggested",
                            }
                        )
            except Exception as exc:
                logger.debug("LLM code pattern analysis failed: %s", exc)

        return results

    # ------------------------------------------------------------------
    # Extraction entry point
    # ------------------------------------------------------------------

    def extract(self, source: str, source_type: str = "pdf") -> Dict:
        """Extract research specification (backward compatible)"""
        if source_type == "pdf":
            return self._extract_from_pdf(source)
        elif source_type == "arxiv":
            return self._extract_from_arxiv(source)
        else:
            return self._extract_from_known_paper(source)

    # ------------------------------------------------------------------
    # PDF extraction — LLM-powered with hardcoded fallback
    # ------------------------------------------------------------------

    def _extract_from_pdf(self, pdf_path: str) -> Dict:
        """Extract an implementation spec from a PDF file.

        Strategy:
        1. Try to read the PDF text.
        2. If text is available and an LLM is configured, use the LLM to
           extract a structured spec.
        3. Fall back to the hardcoded RMSNorm spec (legacy behaviour).
        """
        # Step 1: attempt PDF text extraction
        paper_text = _read_pdf_text(pdf_path)

        # Step 2: LLM-powered extraction
        if paper_text and self._llm is not None and self._llm.is_available:
            try:
                extracted = self._llm.extract_paper_spec(
                    paper_text,
                    paper_title=Path(pdf_path).stem.replace("_", " ").replace("-", " "),
                )
                if extracted is not None:
                    spec = _extracted_spec_to_dict(extracted)
                    # Register the new spec so subsequent lookups find it
                    key = _spec_key(spec)
                    if key:
                        self.specs[key] = spec
                        logger.info("LLM-extracted spec registered as '%s'", key)
                    return spec
            except Exception as exc:
                logger.warning("LLM PDF extraction failed, using fallback: %s", exc)

        # Step 3: hardcoded fallback
        return {
            "paper": {
                "title": "Root Mean Square Layer Normalization",
                "authors": ["Biao Zhang", "Rico Sennrich"],
                "year": 2019,
            },
            "algorithm": {
                "name": "RMSNorm",
                "replaces": "LayerNorm",
                "description": "Simplified layer normalization without mean-centering",
                "formula": "x / sqrt(mean(x^2) + eps) * gamma",
            },
            "implementation": {
                "module_name": "RMSNorm",
                "parent_class": "nn.Module",
                "parameters": ["ndim", "eps"],
                "code_template": self._get_rmsnorm_template(),
            },
            "changes": {
                "type": "replace",
                "target_patterns": ["LayerNorm", "nn.LayerNorm"],
                "replacement": "RMSNorm",
                "insertion_points": ["Block", "GPT"],
                "expected_benefits": ["5-10% training speedup", "Simplified computation"],
            },
            "validation": {
                "test_type": "training_comparison",
                "metrics": ["loss", "perplexity", "tokens_per_second"],
                "max_benchmark_time": 300,
            },
        }

    # ------------------------------------------------------------------
    # arXiv extraction — enhanced with abstract fetch + LLM
    # ------------------------------------------------------------------

    def _extract_from_arxiv(self, arxiv_id: str) -> Dict:
        """Extract spec from an arXiv identifier.

        Strategy:
        1. Check local specs for a matching arXiv ID.
        2. Fetch the abstract from arXiv and use the LLM.
        3. Fall back to ``_extract_from_pdf`` (hardcoded default).
        """
        arxiv_id_clean = arxiv_id.strip().lower()

        # Step 1: check local registry
        for key, spec in self.specs.items():
            if key in arxiv_id_clean or spec["paper"].get("arxiv", "") in arxiv_id_clean:
                return spec

        # Step 2: fetch abstract and try LLM extraction
        if self._llm is not None and self._llm.is_available:
            abstract = _fetch_arxiv_abstract(arxiv_id)
            if abstract:
                try:
                    extracted = self._llm.extract_paper_spec(
                        abstract,
                        paper_title=f"arXiv:{arxiv_id}",
                    )
                    if extracted is not None:
                        spec = _extracted_spec_to_dict(extracted)
                        key = _spec_key(spec)
                        if key:
                            self.specs[key] = spec
                            logger.info(
                                "LLM-extracted arXiv spec registered as '%s'",
                                key,
                            )
                        return spec
                except Exception as exc:
                    logger.warning("LLM arXiv extraction failed for %s: %s", arxiv_id, exc)

        return self._extract_from_pdf(arxiv_id)

    def _extract_from_known_paper(self, paper_name: str) -> Dict:
        paper_lower = paper_name.lower().replace("-", "").replace("_", "").replace(" ", "")

        for key, spec in self.specs.items():
            if (
                key in paper_lower
                or spec["algorithm"]["name"].lower().replace(" ", "") in paper_lower
            ):
                return spec

        return self._extract_from_pdf(paper_name)

    # ------------------------------------------------------------------
    # Templates and accessors (unchanged public interface)
    # ------------------------------------------------------------------

    def _get_rmsnorm_template(self) -> str:
        return """class RMSNorm(nn.Module):
    def __init__(self, ndim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return norm * self.weight
"""

    def get_spec(self, name: str) -> Optional[Dict]:
        return self.specs.get(name.lower())

    def list_available_specs(self) -> List[str]:
        return list(self.specs.keys())

    def get_code_template(self, spec_name: str) -> Optional[str]:
        spec = self.get_spec(spec_name)
        if not spec:
            return None

        name = spec["algorithm"]["name"].lower()

        templates = {
            "rmsnorm": self._get_rmsnorm_template(),
        }

        return templates.get(name)

    def get_categories(self) -> Dict[str, List[str]]:
        """Get all available categories for filtering"""
        categories: Dict[str, List[str]] = {}
        for name, spec in self.specs.items():
            cat = spec.get("algorithm", {}).get("category", "other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(name)
        return categories


# ---------------------------------------------------------------------------
# Helpers for converting LLM output → spec dict
# ---------------------------------------------------------------------------


def _extracted_spec_to_dict(extracted: Any) -> Dict[str, Any]:
    """Convert an ``ExtractedSpec`` dataclass into the dict format
    that ``PAPER_SPECS`` uses."""
    return {
        "paper": extracted.paper if isinstance(extracted.paper, dict) else {},
        "algorithm": extracted.algorithm if isinstance(extracted.algorithm, dict) else {},
        "implementation": (
            extracted.implementation if isinstance(extracted.implementation, dict) else {}
        ),
        "changes": extracted.changes if isinstance(extracted.changes, dict) else {},
        "validation": extracted.validation if isinstance(extracted.validation, dict) else {},
    }


def _spec_key(spec: Dict[str, Any]) -> str:
    """Derive a lowercase registry key from a spec dict."""
    alg_name = spec.get("algorithm", {}).get("name", "")
    if alg_name:
        return alg_name.lower().replace(" ", "_").replace("-", "_")
    title = spec.get("paper", {}).get("title", "")
    if title:
        # Use first meaningful word
        words = [w.lower() for w in title.split() if len(w) > 3]
        return "_".join(words[:2]) if words else ""
    return ""
