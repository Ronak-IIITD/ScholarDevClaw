from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_ALLOWED_COMPLEXITY = {"trivial", "low", "medium", "high", "frontier-only"}
_COMPLEXITY_ALIASES = {"research-only": "frontier-only"}


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


def _as_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): nested for key, nested in value.items()}


@dataclass(slots=True)
class Contribution:
    claim: str
    novelty: str
    is_implementable: bool
    implementation_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "novelty": self.novelty,
            "is_implementable": self.is_implementable,
            "implementation_notes": self.implementation_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Contribution:
        return cls(
            claim=str(data.get("claim", "")),
            novelty=str(data.get("novelty", "")),
            is_implementable=_as_bool(data.get("is_implementable", False)),
            implementation_notes=str(data.get("implementation_notes", "")),
        )


@dataclass(slots=True)
class Requirement:
    name: str
    requirement_type: str
    is_optional: bool
    notes: str
    version_constraint: str | None = None
    acquisition_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requirement_type": self.requirement_type,
            "is_optional": self.is_optional,
            "version_constraint": self.version_constraint,
            "acquisition_url": self.acquisition_url,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Requirement:
        requirement_type = data.get("requirement_type", data.get("type", ""))
        return cls(
            name=str(data.get("name", "")),
            requirement_type=str(requirement_type),
            is_optional=_as_bool(data.get("is_optional", False)),
            version_constraint=(
                str(data["version_constraint"])
                if data.get("version_constraint") is not None
                else None
            ),
            acquisition_url=(
                str(data["acquisition_url"]) if data.get("acquisition_url") is not None else None
            ),
            notes=str(data.get("notes", "")),
        )

    @property
    def type(self) -> str:
        return self.requirement_type


@dataclass(slots=True)
class ConceptNode:
    id: str
    label: str
    concept_type: str
    description: str
    paper_section: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "concept_type": self.concept_type,
            "description": self.description,
            "paper_section": self.paper_section,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConceptNode:
        concept_type = data.get("concept_type", data.get("type", ""))
        return cls(
            id=str(data.get("id", "")),
            label=str(data.get("label", "")),
            concept_type=str(concept_type),
            description=str(data.get("description", "")),
            paper_section=str(data.get("paper_section", "")),
        )

    @property
    def type(self) -> str:
        return self.concept_type


@dataclass(slots=True)
class ConceptEdge:
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        self.weight = min(max(self.weight, 0.0), 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConceptEdge:
        return cls(
            source_id=str(data.get("source_id", "")),
            target_id=str(data.get("target_id", "")),
            relation=str(data.get("relation", "")),
            weight=_as_float(data.get("weight", 1.0), default=1.0),
        )


@dataclass(slots=True)
class PaperUnderstanding:
    paper_title: str = ""
    one_line_summary: str = ""
    problem_statement: str = ""
    prior_state_of_art: str = ""
    key_insight: str = ""
    why_it_works: str = ""

    contributions: list[Contribution] = field(default_factory=list)
    requirements: list[Requirement] = field(default_factory=list)

    concept_nodes: list[ConceptNode] = field(default_factory=list)
    concept_edges: list[ConceptEdge] = field(default_factory=list)

    core_algorithm_description: str = ""
    input_output_spec: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    evaluation_protocol: str = ""
    known_limitations: str = ""

    complexity: str = "frontier-only"
    estimated_impl_hours: int = 0
    can_reproduce_without_compute: bool = False
    confidence: float = 0.0
    confidence_notes: str = ""

    def __post_init__(self) -> None:
        normalized_complexity = _COMPLEXITY_ALIASES.get(self.complexity, self.complexity)
        if normalized_complexity not in _ALLOWED_COMPLEXITY:
            normalized_complexity = "frontier-only"
        self.complexity = normalized_complexity
        if self.estimated_impl_hours < 0:
            self.estimated_impl_hours = 0
        self.confidence = min(max(self.confidence, 0.0), 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_title": self.paper_title,
            "one_line_summary": self.one_line_summary,
            "problem_statement": self.problem_statement,
            "prior_state_of_art": self.prior_state_of_art,
            "key_insight": self.key_insight,
            "why_it_works": self.why_it_works,
            "contributions": [item.to_dict() for item in self.contributions],
            "requirements": [item.to_dict() for item in self.requirements],
            "concept_nodes": [item.to_dict() for item in self.concept_nodes],
            "concept_edges": [item.to_dict() for item in self.concept_edges],
            "core_algorithm_description": self.core_algorithm_description,
            "input_output_spec": self.input_output_spec,
            "hyperparameters": dict(self.hyperparameters),
            "evaluation_protocol": self.evaluation_protocol,
            "known_limitations": self.known_limitations,
            "complexity": self.complexity,
            "estimated_impl_hours": self.estimated_impl_hours,
            "can_reproduce_without_compute": self.can_reproduce_without_compute,
            "confidence": self.confidence,
            "confidence_notes": self.confidence_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperUnderstanding:
        contributions = [
            Contribution.from_dict(item)
            for item in data.get("contributions", [])
            if isinstance(item, dict)
        ]
        requirements = [
            Requirement.from_dict(item) for item in data.get("requirements", []) if isinstance(item, dict)
        ]
        concept_nodes = [
            ConceptNode.from_dict(item) for item in data.get("concept_nodes", []) if isinstance(item, dict)
        ]
        concept_edges = [
            ConceptEdge.from_dict(item) for item in data.get("concept_edges", []) if isinstance(item, dict)
        ]

        return cls(
            paper_title=str(data.get("paper_title", "")),
            one_line_summary=str(data.get("one_line_summary", "")),
            problem_statement=str(data.get("problem_statement", "")),
            prior_state_of_art=str(data.get("prior_state_of_art", "")),
            key_insight=str(data.get("key_insight", "")),
            why_it_works=str(data.get("why_it_works", "")),
            contributions=contributions,
            requirements=requirements,
            concept_nodes=concept_nodes,
            concept_edges=concept_edges,
            core_algorithm_description=str(data.get("core_algorithm_description", "")),
            input_output_spec=str(data.get("input_output_spec", "")),
            hyperparameters=_as_dict(data.get("hyperparameters", {})),
            evaluation_protocol=str(data.get("evaluation_protocol", "")),
            known_limitations=str(data.get("known_limitations", "")),
            complexity=str(data.get("complexity", "frontier-only")),
            estimated_impl_hours=_as_int(data.get("estimated_impl_hours", 0), default=0),
            can_reproduce_without_compute=_as_bool(
                data.get("can_reproduce_without_compute", False)
            ),
            confidence=_as_float(data.get("confidence", 0.0), default=0.0),
            confidence_notes=str(data.get("confidence_notes", "")),
        )
