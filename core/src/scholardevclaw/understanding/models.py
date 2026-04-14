from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_ALLOWED_COMPLEXITY = {"low", "medium", "high", "research-only"}


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n", ""}:
            return False
    return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class Contribution:
    """A key paper contribution and its implementation feasibility."""

    claim: str
    novelty: str
    is_implementable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "novelty": self.novelty,
            "is_implementable": self.is_implementable,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Contribution:
        return cls(
            claim=str(data.get("claim", "")),
            novelty=str(data.get("novelty", "")),
            is_implementable=_as_bool(data.get("is_implementable", False)),
        )


@dataclass(slots=True)
class Requirement:
    """An implementation requirement extracted from paper context."""

    name: str
    type: str
    is_optional: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "is_optional": self.is_optional,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Requirement:
        return cls(
            name=str(data.get("name", "")),
            type=str(data.get("type", "")),
            is_optional=_as_bool(data.get("is_optional", False)),
            notes=str(data.get("notes", "")),
        )


@dataclass(slots=True)
class ConceptNode:
    """A concept node used to construct the paper concept graph."""

    id: str
    label: str
    type: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConceptNode:
        return cls(
            id=str(data.get("id", "")),
            label=str(data.get("label", "")),
            type=str(data.get("type", "")),
            description=str(data.get("description", "")),
        )


@dataclass(slots=True)
class ConceptEdge:
    """A directed relationship between concept nodes."""

    source_id: str
    target_id: str
    relation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConceptEdge:
        return cls(
            source_id=str(data.get("source_id", "")),
            target_id=str(data.get("target_id", "")),
            relation=str(data.get("relation", "")),
        )


@dataclass(slots=True)
class PaperUnderstanding:
    """Structured understanding payload produced from a paper document."""

    paper_title: str = ""
    one_line_summary: str = ""
    problem_statement: str = ""
    key_insight: str = ""

    contributions: list[Contribution] = field(default_factory=list)
    requirements: list[Requirement] = field(default_factory=list)

    concept_nodes: list[ConceptNode] = field(default_factory=list)
    concept_edges: list[ConceptEdge] = field(default_factory=list)

    core_algorithm_description: str = ""
    input_output_spec: str = ""
    evaluation_protocol: str = ""

    complexity: str = "research-only"
    estimated_impl_hours: int = 0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.complexity not in _ALLOWED_COMPLEXITY:
            self.complexity = "research-only"
        if self.estimated_impl_hours < 0:
            self.estimated_impl_hours = 0
        self.confidence = min(max(self.confidence, 0.0), 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_title": self.paper_title,
            "one_line_summary": self.one_line_summary,
            "problem_statement": self.problem_statement,
            "key_insight": self.key_insight,
            "contributions": [item.to_dict() for item in self.contributions],
            "requirements": [item.to_dict() for item in self.requirements],
            "concept_nodes": [item.to_dict() for item in self.concept_nodes],
            "concept_edges": [item.to_dict() for item in self.concept_edges],
            "core_algorithm_description": self.core_algorithm_description,
            "input_output_spec": self.input_output_spec,
            "evaluation_protocol": self.evaluation_protocol,
            "complexity": self.complexity,
            "estimated_impl_hours": self.estimated_impl_hours,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperUnderstanding:
        raw_contributions = data.get("contributions", [])
        raw_requirements = data.get("requirements", [])
        raw_nodes = data.get("concept_nodes", [])
        raw_edges = data.get("concept_edges", [])

        contributions = [
            Contribution.from_dict(item) for item in raw_contributions if isinstance(item, dict)
        ]
        requirements = [
            Requirement.from_dict(item) for item in raw_requirements if isinstance(item, dict)
        ]
        concept_nodes = [
            ConceptNode.from_dict(item) for item in raw_nodes if isinstance(item, dict)
        ]
        concept_edges = [
            ConceptEdge.from_dict(item) for item in raw_edges if isinstance(item, dict)
        ]

        return cls(
            paper_title=str(data.get("paper_title", "")),
            one_line_summary=str(data.get("one_line_summary", "")),
            problem_statement=str(data.get("problem_statement", "")),
            key_insight=str(data.get("key_insight", "")),
            contributions=contributions,
            requirements=requirements,
            concept_nodes=concept_nodes,
            concept_edges=concept_edges,
            core_algorithm_description=str(data.get("core_algorithm_description", "")),
            input_output_spec=str(data.get("input_output_spec", "")),
            evaluation_protocol=str(data.get("evaluation_protocol", "")),
            complexity=str(data.get("complexity", "research-only")),
            estimated_impl_hours=_as_int(data.get("estimated_impl_hours", 0), default=0),
            confidence=_as_float(data.get("confidence", 0.0), default=0.0),
        )
