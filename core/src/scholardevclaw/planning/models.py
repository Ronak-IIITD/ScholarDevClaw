from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _as_dict_any(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): nested for key, nested in value.items()}


def _as_dict_str_str(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(raw_value) for key, raw_value in value.items()}


@dataclass(slots=True)
class CodeModule:
    """A single implementation module in a generated project plan."""

    id: str = ""
    name: str = ""
    description: str = ""
    file_path: str = ""
    depends_on: list[str] = field(default_factory=list)
    priority: int = 0
    estimated_lines: int = 0
    test_file_path: str = ""
    tech_stack: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "file_path": self.file_path,
            "depends_on": list(self.depends_on),
            "priority": self.priority,
            "estimated_lines": self.estimated_lines,
            "test_file_path": self.test_file_path,
            "tech_stack": self.tech_stack,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeModule:
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            file_path=str(data.get("file_path", "")),
            depends_on=_as_str_list(data.get("depends_on", [])),
            priority=_as_int(data.get("priority", 0), default=0),
            estimated_lines=_as_int(data.get("estimated_lines", 0), default=0),
            test_file_path=str(data.get("test_file_path", "")),
            tech_stack=str(data.get("tech_stack", "")),
        )


@dataclass(slots=True)
class ImplementationPlan:
    """Structured project-level plan derived from paper understanding."""

    project_name: str = ""
    target_language: str = "python"
    tech_stack: str = ""
    modules: list[CodeModule] = field(default_factory=list)
    directory_structure: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    entry_points: list[str] = field(default_factory=list)
    estimated_total_lines: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "target_language": self.target_language,
            "tech_stack": self.tech_stack,
            "modules": [module.to_dict() for module in self.modules],
            "directory_structure": dict(self.directory_structure),
            "environment": dict(self.environment),
            "entry_points": list(self.entry_points),
            "estimated_total_lines": self.estimated_total_lines,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImplementationPlan:
        raw_modules = data.get("modules", [])
        modules = [CodeModule.from_dict(item) for item in raw_modules if isinstance(item, dict)]

        return cls(
            project_name=str(data.get("project_name", "")),
            target_language=str(data.get("target_language", "python")),
            tech_stack=str(data.get("tech_stack", "")),
            modules=modules,
            directory_structure=_as_dict_any(data.get("directory_structure", {})),
            environment=_as_dict_str_str(data.get("environment", {})),
            entry_points=_as_str_list(data.get("entry_points", [])),
            estimated_total_lines=_as_int(data.get("estimated_total_lines", 0), default=0),
        )
