"""
Enhanced spec extraction from papers using AI and multiple sources.

Features:
- Parse papers from arXiv, PubMed, IEEE
- Extract algorithm specifications
- Generate implementation hints
- Cross-reference with existing specs
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .paper_sources import Paper, PaperSourceAggregator


@dataclass
class ExtractedAlgorithm:
    """Algorithm extracted from a paper"""

    name: str
    category: str
    replaces: str
    description: str
    formula: str = ""
    complexity: str = ""
    parameters: list[str] = field(default_factory=list)
    implementation_hints: list[str] = field(default_factory=list)
    expected_benefits: list[str] = field(default_factory=list)
    validation_metrics: list[str] = field(default_factory=list)


@dataclass
class PaperSpec:
    """Complete specification extracted from a paper"""

    paper: Paper
    algorithms: list[ExtractedAlgorithm]
    code_patterns: list[str] = field(default_factory=list)
    insertion_points: list[str] = field(default_factory=list)
    related_papers: list[str] = field(default_factory=list)


class EnhancedSpecExtractor:
    """Extract specifications from academic papers"""

    CATEGORY_PATTERNS = {
        "normalization": [
            r"layer\s*normalization",
            r"rmsnorm",
            r"batch\s*normalization",
            r"group\s*normalization",
            r"instance\s*normalization",
        ],
        "activation": [
            r"swiglu",
            r"swish",
            r"gelu",
            r"relu",
            r"silu",
            r"gated\s*linear",
            r"glu",
        ],
        "attention": [
            r"attention",
            r"self.?attention",
            r"multi.?head",
            r"flash.?attention",
            r"linear.?attention",
        ],
        "optimization": [
            r"optimizer",
            r"adam",
            r"sgd",
            r"lion",
            r"adafactor",
            r"learning\s*rate",
            r"scheduler",
        ],
        "architecture": [
            r"transformer",
            r"encoder",
            r"decoder",
            r"mlp",
            r"feed.?forward",
            r"residual",
            r"skip.?connection",
        ],
        "tokenizer": [
            r"tokenizer",
            r"tokenization",
            r"bpe",
            r"byte.?pair",
            r"sentencepiece",
            r"wordpiece",
        ],
    }

    def __init__(self):
        self.paper_source = PaperSourceAggregator()

    async def extract_from_paper(self, paper: Paper) -> PaperSpec:
        """Extract specification from a paper"""
        algorithms = self._extract_algorithms(paper)
        code_patterns = self._extract_code_patterns(paper)
        insertion_points = self._find_insertion_points(paper, algorithms)

        return PaperSpec(
            paper=paper,
            algorithms=algorithms,
            code_patterns=code_patterns,
            insertion_points=insertion_points,
        )

    def _extract_algorithms(self, paper: Paper) -> list[ExtractedAlgorithm]:
        """Extract algorithms mentioned in the paper"""
        text = (paper.title + " " + paper.abstract).lower()
        algorithms = []

        for category, patterns in self.CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    algorithm = ExtractedAlgorithm(
                        name=self._extract_algorithm_name(text, pattern),
                        category=category,
                        replaces=self._find_replacement(text, category),
                        description=self._generate_description(paper, category),
                        formula=self._extract_formula(paper),
                    )
                    algorithms.append(algorithm)
                    break

        if not algorithms:
            algorithms.append(
                ExtractedAlgorithm(
                    name="Unknown",
                    category="general",
                    replaces="",
                    description=paper.abstract[:200],
                )
            )

        return algorithms

    def _extract_algorithm_name(self, text: str, pattern: str) -> str:
        """Extract algorithm name from text"""
        pattern_clean = pattern.replace(r"\s*", " ").strip()
        words = pattern_clean.split()
        if words:
            return "".join(w.capitalize() for w in words if w.isalpha())
        return pattern_clean.capitalize()

    def _find_replacement(self, text: str, category: str) -> str:
        """Find what the algorithm replaces"""
        replacements = {
            "normalization": "LayerNorm",
            "activation": "GELU",
            "attention": "Standard Attention",
            "optimization": "SGD",
            "architecture": "Standard MLP",
        }
        return replacements.get(category, "")

    def _generate_description(self, paper: Paper, category: str) -> str:
        """Generate algorithm description"""
        category_descriptions = {
            "normalization": "A normalization technique that normalizes activations",
            "activation": "An activation function that introduces non-linearity",
            "attention": "A mechanism that allows models to focus on relevant parts",
            "optimization": "An optimization algorithm for training",
            "architecture": "A neural network architecture component",
        }
        base = category_descriptions.get(category, "")
        if paper.year:
            base += f" introduced in {paper.year}."
        return base

    def _extract_formula(self, paper: Paper) -> str:
        """Extract mathematical formula from paper abstract"""
        formula_pattern = r"([A-Za-z]\s*/\s*[A-Za-z]|[A-Za-z]\s*\*\s*[A-Za-z]|\w+\s*\(\s*\w+\s*\)|\w+\s*\[\s*\w+\s*\])"
        matches = re.findall(formula_pattern, paper.abstract)
        if matches:
            return " ".join(matches[:3])
        return ""

    def _extract_code_patterns(self, paper: Paper) -> list[str]:
        """Extract code patterns from paper"""
        patterns = []

        text = paper.abstract.lower()

        if "python" in text or "pytorch" in text or "tensorflow" in text:
            patterns.append("import torch.nn as nn")

        if "class" in text:
            patterns.append("class NewAlgorithm(nn.Module):")

        if "parameter" in text:
            patterns.append("def __init__(self, config):")

        if "forward" in text or "call" in text:
            patterns.append("def forward(self, x):")

        return patterns

    def _find_insertion_points(
        self,
        paper: Paper,
        algorithms: list[ExtractedAlgorithm],
    ) -> list[str]:
        """Find where to insert the implementation"""
        points = []

        for algo in algorithms:
            if algo.category == "normalization":
                points.extend(["Block class", "GPT class", "Model class"])
            elif algo.category == "activation":
                points.extend(["MLP class", "FeedForward class", "Block class"])
            elif algo.category == "attention":
                points.extend(["Attention class", "Block class"])
            elif algo.category == "architecture":
                points.extend(["Model class", "Transformer class"])

        return list(set(points)) if points else ["Model class"]

    async def search_and_extract(
        self,
        query: str,
        max_papers: int = 5,
    ) -> list[PaperSpec]:
        """Search for papers and extract specifications"""
        results = await self.paper_source.search_all(query, max_papers)

        specs = []
        for source, result in results.items():
            for paper in result.papers:
                spec = await self.extract_from_paper(paper)
                specs.append(spec)

        return specs


async def extract_spec_from_arxiv(arxiv_id: str) -> PaperSpec | None:
    """Convenience function to extract spec from arXiv ID"""
    extractor = EnhancedSpecExtractor()
    paper = await extractor.paper_source.get_paper_by_id(arxiv_id, "arxiv")
    if paper:
        return await extractor.extract_from_paper(paper)
    return None
