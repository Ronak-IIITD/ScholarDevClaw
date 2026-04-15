from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scholardevclaw.planning.models import ImplementationPlan


@dataclass(slots=True)
class ModuleResult:
    module_id: str
    file_path: str
    test_file_path: str
    code: str
    test_code: str
    generation_attempts: int
    final_errors: list[str]
    tokens_used: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "file_path": self.file_path,
            "test_file_path": self.test_file_path,
            "code": self.code,
            "test_code": self.test_code,
            "generation_attempts": self.generation_attempts,
            "final_errors": list(self.final_errors),
            "tokens_used": self.tokens_used,
        }


@dataclass(slots=True)
class GenerationResult:
    plan: ImplementationPlan
    module_results: list[ModuleResult] = field(default_factory=list)
    output_dir: Path = Path()
    success_rate: float = 0.0
    total_tokens_used: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "module_results": [result.to_dict() for result in self.module_results],
            "output_dir": str(self.output_dir),
            "success_rate": self.success_rate,
            "total_tokens_used": self.total_tokens_used,
            "duration_seconds": self.duration_seconds,
        }
