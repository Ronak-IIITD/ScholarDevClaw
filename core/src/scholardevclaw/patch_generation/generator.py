from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict


@dataclass
class NewFile:
    path: str
    content: str


@dataclass
class Transformation:
    file: str
    original: str
    modified: str
    changes: List[Dict] = field(default_factory=list)


@dataclass
class Patch:
    new_files: List[NewFile]
    transformations: List[Transformation]
    branch_name: str


class PatchGenerator:
    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)

    def generate(self, mapping_result: Dict) -> Patch:
        spec = mapping_result.get("research_spec", {})

        new_files = self._create_new_files(spec)
        transformations = self._create_transformations(mapping_result)

        algorithm_name = spec.get("algorithm", {}).get("name", "research").lower()
        branch_name = f"integration/{algorithm_name}"

        return Patch(
            new_files=new_files,
            transformations=transformations,
            branch_name=branch_name,
        )

    def _create_new_files(self, spec: Dict) -> List[NewFile]:
        new_files = []

        algorithm_name = spec.get("algorithm", {}).get("name", "").lower()

        if algorithm_name == "rmsnorm":
            new_files.append(
                NewFile(
                    path="rmsnorm.py",
                    content=self._get_rmsnorm_code(spec),
                )
            )

        return new_files

    def _get_rmsnorm_code(self, spec: Dict) -> str:
        paper = spec.get("paper", {})
        authors = ", ".join(paper.get("authors", []))
        year = paper.get("year", 2019)

        return f'''"""
RMSNorm: Root Mean Square Layer Normalization

Integrated from "{paper.get("title", "Root Mean Square Layer Normalization")}"
by {authors} ({year})
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    Simplified layer normalization without mean-centering.
    Formula: x / sqrt(mean(x^2) + eps) * gamma
    """
    
    def __init__(self, ndim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return norm * self.weight
'''

    def _create_transformations(self, mapping_result: Dict) -> List[Transformation]:
        transformations = []

        targets = mapping_result.get("targets", [])

        for target in targets:
            file_path = self.repo_path / target.get("file", "model.py")

            if file_path.exists():
                original = file_path.read_text()

                replacement = self._create_replacement(target, mapping_result)

                transformations.append(
                    Transformation(
                        file=str(target.get("file", "model.py")),
                        original=original,
                        modified=replacement,
                        changes=[{"type": "replace", "target": target.get("current_code")}],
                    )
                )

        return transformations

    def _create_replacement(self, target: Dict, mapping_result: Dict) -> str:
        spec = mapping_result.get("research_spec", {})
        algorithm_name = spec.get("algorithm", {}).get("name", "").lower()

        original = target.get("current_code", "")

        if algorithm_name == "rmsnorm":
            return original.replace("nn.LayerNorm", "RMSNorm").replace("LayerNorm", "RMSNorm")

        return original
