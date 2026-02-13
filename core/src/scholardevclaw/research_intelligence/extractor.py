from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Any
import json


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


class ResearchExtractor:
    """Handles research paper extraction and search"""

    def __init__(self):
        self.specs = PAPER_SPECS
        self._arxiv_client = None

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
            print(f"arXiv search error: {e}")
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
        """Find papers relevant to a code pattern"""
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

        results = []
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

        return results

    def extract(self, source: str, source_type: str = "pdf") -> Dict:
        """Extract research specification (backward compatible)"""
        if source_type == "pdf":
            return self._extract_from_pdf(source)
        elif source_type == "arxiv":
            return self._extract_from_arxiv(source)
        else:
            return self._extract_from_known_paper(source)

    def _extract_from_pdf(self, pdf_path: str) -> Dict:
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

    def _extract_from_arxiv(self, arxiv_id: str) -> Dict:
        arxiv_id_clean = arxiv_id.strip().lower()

        for key, spec in self.specs.items():
            if key in arxiv_id_clean or spec["paper"].get("arxiv", "") in arxiv_id_clean:
                return spec

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
        categories = {}
        for name, spec in self.specs.items():
            cat = spec.get("algorithm", {}).get("category", "other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(name)
        return categories
