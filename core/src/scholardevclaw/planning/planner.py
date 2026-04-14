from __future__ import annotations

import json
import re
from typing import Any

from scholardevclaw.ingestion.models import PaperDocument
from scholardevclaw.planning.models import ImplementationPlan
from scholardevclaw.understanding.models import PaperUnderstanding

try:
    import anthropic  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised only without optional dependency
    anthropic = None  # type: ignore[assignment]


def _is_topologically_ordered(plan: ImplementationPlan) -> bool:
    priorities = {module.id: module.priority for module in plan.modules}
    for module in plan.modules:
        for dependency in module.depends_on:
            dependency_priority = priorities.get(dependency)
            if dependency_priority is None:
                return False
            if dependency_priority >= module.priority:
                return False
    return True


class ImplementationPlanner:
    """LLM-backed implementation planner producing ``ImplementationPlan``."""

    def __init__(self, api_key: str, model: str = "claude-opus-4-5") -> None:
        if anthropic is None:
            raise ImportError(
                "anthropic SDK is required for ImplementationPlanner. "
                "Install with: pip install -e '.[understanding,execution]'"
            )
        if not api_key.strip():
            raise ValueError("api_key must be non-empty")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def plan(
        self,
        understanding: PaperUnderstanding,
        doc: PaperDocument,
        *,
        forced_stack: str | None = None,
    ) -> ImplementationPlan:
        stack = forced_stack or self._select_tech_stack(understanding, doc)
        plan_data = self._llm_plan(understanding, stack)
        plan = ImplementationPlan.from_dict(plan_data)
        if not _is_topologically_ordered(plan):
            raise ValueError(
                "Invalid implementation plan: module priorities are not topologically ordered "
                "against dependencies."
            )
        return plan

    def _select_tech_stack(self, understanding: PaperUnderstanding, doc: PaperDocument) -> str:
        """Rule-based stack selection. No LLM needed here."""
        domain = doc.domain
        requirements_text = " ".join(r.name.casefold() for r in understanding.requirements)
        text = understanding.core_algorithm_description.casefold()

        if "jax" in requirements_text or "tpu" in text:
            return "jax"
        if domain == "systems" and "pytorch" not in requirements_text:
            return "numpy-only"
        return "pytorch"

    def _llm_plan(self, understanding: PaperUnderstanding, stack: str) -> dict[str, Any]:
        prompt = f"""You are a senior ML engineer planning the implementation of a research paper.

Paper: {understanding.paper_title}
Summary: {understanding.one_line_summary}
Core algorithm: {understanding.core_algorithm_description}
Input/output: {understanding.input_output_spec}
Tech stack: {stack}
Requirements: {[r.name for r in understanding.requirements if not r.is_optional]}
Evaluation: {understanding.evaluation_protocol}
Complexity: {understanding.complexity}

Design a complete, production-quality Python project structure to implement this paper.
Return a JSON object with exactly these fields:
{{
  "project_name": str,             // snake_case
  "target_language": "python",
  "tech_stack": "{stack}",
  "modules": [
    {{
      "id": str,                   // snake_case unique identifier
      "name": str,
      "description": str,
      "file_path": str,            // e.g. "src/model/attention.py"
      "depends_on": [str],         // list of other module ids
      "priority": int,             // 1=first to implement
      "estimated_lines": int,
      "test_file_path": str,       // e.g. "tests/test_attention.py"
      "tech_stack": str
    }}
  ],
  "directory_structure": {{
    "src/": {{
      "model/": {{}},
      "data/": {{}}
    }},
    "tests/": {{}},
    "scripts/": {{}}
  }},
  "environment": {{"torch": ">=2.0.0", "numpy": ">=1.26.0"}},
  "entry_points": [str],
  "estimated_total_lines": int
}}

Rules:
- Every paper needs at minimum: a data loader, a model definition, a training loop,
  an evaluation script, and a README generator.
- tests/ must mirror src/ structure exactly.
- Modules must be ordered so that a module's dependencies are always lower priority numbers.
- Return only JSON. No markdown. No preamble.
"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        if not response.content:
            raise ValueError("Anthropic response had no content blocks.")

        first_block = response.content[0]
        raw = getattr(first_block, "text", None)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Anthropic response did not contain text in the first content block.")

        return self._parse_json_response(raw)

    def _parse_json_response(self, raw: str) -> dict[str, Any]:
        cleaned = self._strip_markdown_fences(raw.strip())

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            object_block = self._extract_first_json_object_block(cleaned)
            if object_block is None:
                preview = cleaned[:200].replace("\n", " ")
                raise ValueError(
                    "Failed to parse ImplementationPlanner response as JSON. "
                    "Ensure the model returns a JSON object only. "
                    f"Response preview: {preview!r}"
                )
            try:
                parsed = json.loads(object_block)
            except json.JSONDecodeError as exc:
                preview = object_block[:200].replace("\n", " ")
                raise ValueError(
                    "Failed to parse extracted JSON object from ImplementationPlanner response. "
                    "Ensure field names and quotes are valid JSON. "
                    f"Extracted preview: {preview!r}"
                ) from exc

        if not isinstance(parsed, dict):
            raise ValueError("ImplementationPlanner response must be a JSON object at top level.")
        return parsed

    def _strip_markdown_fences(self, text: str) -> str:
        fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        return text

    def _extract_first_json_object_block(self, text: str) -> str | None:
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escaped = False

        for index, char in enumerate(text[start:], start=start):
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue
            if char == "{":
                depth += 1
                continue
            if char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]

        return None
