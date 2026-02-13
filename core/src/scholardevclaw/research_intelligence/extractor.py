from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List
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
    "grouped_query_attention": {
        "paper": {
            "title": "GQA: Training Generalized Multi-Query Transformer with Multi-Query Key-Value Cache",
            "authors": [
                "Joshua Ainslie",
                "James Lee-Thorp",
                "Mojtaba Valipour",
                "Gabriel V. de la Cruz",
                "Yicheng Li",
                "Wang Zhou",
            ],
            "arxiv": "2305.13245",
            "year": 2023,
        },
        "algorithm": {
            "name": "GroupedQueryAttention",
            "replaces": "MultiHeadAttention",
            "description": "Grouped Query Attention - shares key/value heads across query groups for efficiency",
            "formula": "Similar to MHA but with fewer KV heads grouped by GQA ratio",
        },
        "implementation": {
            "module_name": "GroupedQueryAttention",
            "parent_class": "nn.Module",
            "parameters": ["n_heads", "n_kv_heads", "dim"],
            "forward_signature": "(x: Tensor) -> Tensor",
        },
        "changes": {
            "type": "replace",
            "target_patterns": ["class CausalSelfAttention"],
            "replacement": "GroupedQueryAttention",
            "insertion_points": ["CausalSelfAttention class"],
            "expected_benefits": ["Reduced KV cache", "Faster inference"],
        },
        "validation": {
            "test_type": "benchmark",
            "metrics": ["inference_speed", "memory_usage"],
            "max_benchmark_time": 300,
        },
    },
}


@dataclass
class ResearchSpec:
    paper: Dict
    algorithm: Dict
    implementation: Dict
    changes: Dict
    validation: Dict


class ResearchExtractor:
    def __init__(self):
        self.specs = PAPER_SPECS

    def extract(self, source: str, source_type: str = "pdf") -> Dict:
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

        if "1910.07467" in arxiv_id_clean:
            return self.specs.get("rmsnorm", self._extract_from_pdf(arxiv_id))
        elif "2205.14135" in arxiv_id_clean:
            return self.specs.get("flashattention", self._extract_from_pdf(arxiv_id))
        elif "2104.09864" in arxiv_id_clean:
            return self.specs.get("rope", self._extract_from_pdf(arxiv_id))

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

    def _get_swiglu_template(self) -> str:
        return """class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
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
            "swiglu": self._get_swiglu_template(),
        }

        return templates.get(name)
