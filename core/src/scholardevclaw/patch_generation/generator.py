from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
import libcst as cst
from libcst import parse_module, CSTTransformer


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
    algorithm_name: str
    paper_reference: str


class RMSNormTransformer(CSTTransformer):
    def __init__(self, original_name: str = "LayerNorm", replacement_name: str = "RMSNorm"):
        self.original_name = original_name
        self.replacement_name = replacement_name
        self.changes = []

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        if original_node.name.value == self.original_name:
            new_name = cst.Name(self.replacement_name)
            self.changes.append(
                {
                    "type": "rename_class",
                    "from": self.original_name,
                    "to": self.replacement_name,
                    "line": original_node.lineno,
                }
            )
            return updated_node.with_changes(name=new_name)
        return updated_node

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:
        if original_node.value == self.original_name:
            self.changes.append(
                {
                    "type": "rename_name",
                    "from": self.original_name,
                    "to": self.replacement_name,
                }
            )
            return cst.Name(self.replacement_name)
        return updated_node


class SwiGLUTransformer(CSTTransformer):
    def __init__(self):
        self.changes = []
        self.in_mlp_class = False

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[cst.ClassDef]:
        if node.name.value == "MLP":
            self.in_mlp_class = True
        return node

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        if original_node.name.value == "MLP":
            new_name = cst.Name("SwiGLU")
            self.changes.append(
                {
                    "type": "rename_class",
                    "from": "MLP",
                    "to": "SwiGLU",
                }
            )
            self.in_mlp_class = False
            return updated_node.with_changes(name=new_name)
        return updated_node

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        if self.in_mlp_class:
            func = original_node.func
            if isinstance(func, cst.Attribute) and func.attr.value == "GELU":
                self.changes.append({"type": "replace_activation", "from": "GELU", "to": "SiLU"})
        return updated_node


class PatchGenerator:
    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)

    def generate(self, mapping_result: Dict) -> Patch:
        spec = mapping_result.get("research_spec", {})

        new_files = self._create_new_files(spec)
        transformations = self._create_transformations(mapping_result)

        algorithm_name = spec.get("algorithm", {}).get("name", "research")
        paper_info = spec.get("paper", {})

        paper_reference = ""
        if paper_info.get("arxiv"):
            paper_reference = f"arXiv:{paper_info['arxiv']}"
        elif paper_info.get("title"):
            paper_reference = paper_info["title"]

        branch_name = f"integration/{algorithm_name.lower().replace(' ', '-')}"

        return Patch(
            new_files=new_files,
            transformations=transformations,
            branch_name=branch_name,
            algorithm_name=algorithm_name,
            paper_reference=paper_reference,
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
        elif algorithm_name == "swiglu":
            new_files.append(
                NewFile(
                    path="swiglu.py",
                    content=self._get_swiglu_code(spec),
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

Paper: {paper.get("arxiv", "N/A")}
Description: {spec.get("algorithm", {}).get("description", "")}
Formula: {spec.get("algorithm", {}).get("formula", "N/A")}
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    Simplified layer normalization without mean-centering.
    Formula: x / sqrt(mean(x^2) + eps) * gamma
    
    Benefits:
    - Faster computation than LayerNorm
    - Simplified forward pass
    - Often achieves similar or better results
    """
    
    def __init__(self, ndim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return norm * self.weight
'''

    def _get_swiglu_code(self, spec: Dict) -> str:
        paper = spec.get("paper", {})

        return f'''"""
SwiGLU: Swish-Gated Linear Unit

Integrated from "{paper.get("title", "SwiGLU")}"
by {", ".join(paper.get("authors", []))} ({paper.get("year", 2020)})

Description: {spec.get("algorithm", {}).get("description", "")}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU)
    
    Combines Swish activation with gated linear units for improved performance.
    """
    
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
'''

    def _create_transformations(self, mapping_result: Dict) -> List[Transformation]:
        transformations = []

        targets = mapping_result.get("targets", [])
        spec = mapping_result.get("research_spec", {})

        for target in targets:
            file_path = self.repo_path / target.get("file", "model.py")

            # SECURITY: Prevent path traversal — ensure target stays within repo
            try:
                resolved = file_path.resolve()
                if not resolved.is_relative_to(self.repo_path.resolve()):
                    continue
            except (ValueError, OSError):
                continue

            if not file_path.exists():
                continue

            try:
                original = file_path.read_text()

                context = target.get("context", {})
                replacement = context.get("replacement", "") if isinstance(context, dict) else ""
                original_name = context.get("original", "") if isinstance(context, dict) else ""

                if replacement and original_name:
                    modified = self._apply_transformation(
                        original,
                        original_name,
                        replacement,
                        spec.get("algorithm", {}).get("name", ""),
                    )

                    if modified != original:
                        transformations.append(
                            Transformation(
                                file=str(target.get("file", "model.py")),
                                original=original[:500],
                                modified=modified[:500],
                                changes=[
                                    {
                                        "type": "replace",
                                        "from": original_name,
                                        "to": replacement,
                                    }
                                ],
                            )
                        )
            except Exception as e:
                print(f"Error transforming {file_path}: {e}")

        return transformations

    def _apply_transformation(
        self, source: str, original: str, replacement: str, algorithm: str
    ) -> str:
        try:
            tree = parse_module(source)

            if algorithm.lower() == "rmsnorm":
                transformer = RMSNormTransformer(original, replacement)
            elif algorithm.lower() == "swiglu":
                transformer = SwiGLUTransformer()
            else:
                return source.replace(original, replacement)

            modified_tree = tree.visit(transformer)
            return modified_tree.code

        except Exception as e:
            print(f"AST transformation failed: {e}, using string replacement")
            return source.replace(original, replacement)
