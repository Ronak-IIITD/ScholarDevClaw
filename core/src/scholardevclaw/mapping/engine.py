from dataclasses import dataclass, field
from pathlib import Path
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
    research_spec: Dict


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
            research_spec=self.research_spec,
        )

    def _find_target_locations(self) -> List[InsertionPoint]:
        targets = []

        changes = self.research_spec.get("changes", {})
        target_patterns = changes.get("target_patterns", [])
        replacement = changes.get("replacement", "")

        models = self.repo_analysis.get("architecture", {}).get("models", [])

        for model in models:
            model_file = model.get("file", "")
            components = model.get("components", {})

            if "custom_norms" in components:
                for norm in components["custom_norms"]:
                    for pattern in target_patterns:
                        if pattern.lower() in norm.lower():
                            targets.append(
                                InsertionPoint(
                                    file=model_file,
                                    line=1,
                                    current_code=norm,
                                    replacement_required=True,
                                    context={
                                        "component_type": "custom_norm",
                                        "replacement": replacement,
                                        "original": norm,
                                    },
                                )
                            )

            if "normalization" in components:
                norm = components["normalization"]
                for pattern in target_patterns:
                    if pattern.lower() in norm.lower():
                        targets.append(
                            InsertionPoint(
                                file=model_file,
                                line=1,
                                current_code=norm,
                                replacement_required=True,
                                context={
                                    "component_type": "normalization",
                                    "replacement": replacement,
                                    "original": norm,
                                },
                            )
                        )

        if not targets:
            modules = self.repo_analysis.get("modules", [])
            for module in modules:
                for cls in module.get("classes", []):
                    for pattern in target_patterns:
                        if pattern in cls.get("name", ""):
                            targets.append(
                                InsertionPoint(
                                    file=module.get("file", "model.py"),
                                    line=cls.get("line", 1),
                                    current_code=cls.get("name", ""),
                                    replacement_required=True,
                                    context={
                                        "component_type": "class",
                                        "replacement": replacement,
                                        "original": cls.get("name", ""),
                                    },
                                )
                            )

        if not targets and target_patterns:
            for pattern in target_patterns:
                targets.append(
                    InsertionPoint(
                        file="model.py",
                        line=1,
                        current_code=pattern,
                        replacement_required=True,
                        context={
                            "component_type": "import",
                            "replacement": replacement,
                            "original": pattern,
                        },
                    )
                )

        return targets

    def _validate_compatibility(self, targets: List[InsertionPoint]) -> ValidationResult:
        issues = []

        changes = self.research_spec.get("changes", {})

        for target in targets:
            if target.replacement_required:
                original = target.context.get("original", "")
                replacement = target.context.get("replacement", "")

                if original == replacement:
                    issues.append(
                        CompatibilityIssue(
                            location=f"{target.file}:{target.line}",
                            issue=f"Same replacement: {original} -> {replacement}",
                            severity="error",
                        )
                    )

        return ValidationResult(
            passed=len(issues) == 0,
            issues=issues,
        )

    def _select_strategy(self, targets: List[InsertionPoint], validation: ValidationResult) -> str:
        if not targets:
            return "none"

        if not validation.passed:
            return "manual_review"

        changes = self.research_spec.get("changes", {})
        change_type = changes.get("type", "replace")

        return change_type

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


def analyze_repo_for_pattern(repo_path: str, pattern: str) -> List[Dict]:
    results = []
    path = Path(repo_path)

    for py_file in path.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        try:
            content = py_file.read_text()
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                if pattern.lower() in line.lower():
                    results.append(
                        {
                            "file": str(py_file.relative_to(path)),
                            "line": i,
                            "content": line.strip(),
                        }
                    )
        except Exception:
            continue

    return results
