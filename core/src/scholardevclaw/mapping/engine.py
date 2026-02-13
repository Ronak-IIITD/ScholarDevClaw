from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class InsertionPoint:
    file: str
    line: int
    current_code: str
    replacement_required: bool
    context: Dict = field(default_factory=dict)


@dataclass
class CompatibilityIssue:
    location: str
    issue: str
    severity: str


@dataclass
class ValidationResult:
    passed: bool
    issues: List[CompatibilityIssue] = field(default_factory=list)


@dataclass
class MappingResult:
    targets: List[InsertionPoint]
    strategy: str
    confidence: int


class MappingEngine:
    def __init__(self, repo_analysis: Dict, research_spec: Dict):
        self.repo_analysis = repo_analysis
        self.research_spec = research_spec

    def map(self) -> MappingResult:
        targets = self._find_target_locations()
        validation = self._validate_compatibility(targets)
        strategy = self._select_strategy(targets, validation)

        confidence = self._calculate_confidence(validation, targets)

        return MappingResult(
            targets=targets,
            strategy=strategy,
            confidence=confidence,
        )

    def _find_target_locations(self) -> List[InsertionPoint]:
        targets = []

        changes = self.research_spec.get("changes", {})
        target_pattern = changes.get("target_pattern", "")
        insertion_points = changes.get("insertion_points", [])

        models = self.repo_analysis.get("architecture", {}).get("models", [])

        for model in models:
            model_file = model.get("file", "")
            components = model.get("components", {})

            if target_pattern in ["nn.LayerNorm", "LayerNorm"]:
                for key, value in components.items():
                    if "norm" in key.lower() or "layer" in key.lower():
                        targets.append(
                            InsertionPoint(
                                file=model_file,
                                line=1,
                                current_code=value,
                                replacement_required=True,
                                context={"component": key, "value": value},
                            )
                        )

        if not targets and insertion_points:
            for point in insertion_points:
                targets.append(
                    InsertionPoint(
                        file="model.py",
                        line=1,
                        current_code="nn.LayerNorm",
                        replacement_required=True,
                        context={"insertion_point": point},
                    )
                )

        return targets

    def _validate_compatibility(self, targets: List[InsertionPoint]) -> ValidationResult:
        issues = []

        for target in targets:
            if target.replacement_required:
                changes = self.research_spec.get("changes", {})
                if changes.get("type") == "replace":
                    pass

        return ValidationResult(
            passed=len(issues) == 0,
            issues=issues,
        )

    def _select_strategy(self, targets: List[InsertionPoint], validation: ValidationResult) -> str:
        if not targets:
            return "none"

        if validation.passed:
            return "replace"

        return "extend"

    def _calculate_confidence(
        self, validation: ValidationResult, targets: List[InsertionPoint]
    ) -> int:
        confidence = 50

        if len(targets) > 0:
            confidence += 20

        if validation.passed:
            confidence += 20

        if self.research_spec.get("implementation", {}).get("code_template"):
            confidence += 10

        return min(confidence, 100)
