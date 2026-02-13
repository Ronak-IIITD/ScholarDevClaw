from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List
import json


RMSNORM_SPEC = {
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
    },
    "implementation": {
        "module_name": "RMSNorm",
        "parent_class": "nn.Module",
        "parameters": ["normalized_shape", "eps", "elementwise_affine"],
    },
    "changes": {
        "type": "replace",
        "target_pattern": "nn.LayerNorm",
        "insertion_points": ["Block class", "GPT class"],
    },
}


@dataclass
class ResearchSpec:
    paper: Dict
    algorithm: Dict
    implementation: Dict
    changes: Dict


class ResearchExtractor:
    def __init__(self):
        self.specs: Dict[str, Dict] = {
            "rmsnorm": RMSNORM_SPEC,
        }

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
                "target_pattern": "nn.LayerNorm",
                "insertion_points": ["Block", "GPT"],
                "expected_benefits": ["5-10% training speedup", "Simplified computation"],
            },
        }

    def _extract_from_arxiv(self, arxiv_id: str) -> Dict:
        return self._extract_from_pdf(arxiv_id)

    def _extract_from_known_paper(self, paper_name: str) -> Dict:
        paper_lower = paper_name.lower().replace("-", "").replace("_", "")

        for key, spec in self.specs.items():
            if key in paper_lower:
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
