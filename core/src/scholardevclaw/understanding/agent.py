from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from scholardevclaw.exceptions import UnderstandingError
from scholardevclaw.ingestion.models import PaperDocument, Section
from scholardevclaw.understanding.models import PaperUnderstanding

try:
    import anthropic  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a world-class AI researcher and senior software engineer with deep
expertise across machine learning, computer science, and scientific writing.
You read research papers with surgical precision and extract structured
information that allows a developer to implement the paper completely from
scratch — without reading the paper themselves.

Your job is ANALYSIS, not summarization. You are not writing an abstract.
You are reverse-engineering the paper into an implementation blueprint.

Critical rules:
1. Never hallucinate results, citations, or metrics not stated in the paper.
2. If you are unsure about something, say so explicitly in confidence_notes.
3. The core_algorithm_description must be step-by-step, implementation-ready,
   and understandable by someone who has never read the paper.
4. Requirements must be exhaustive — missing a required dataset or library
   wastes the user's time.
5. Respond with valid JSON only. No markdown. No explanation outside the JSON.
"""


class UnderstandingAgent:
    """LLM-backed paper understanding agent producing ``PaperUnderstanding``."""

    _MAX_PROMPT_CHARS: int = 120_000
    _MAX_SECTION_CHARS: int = 35_000
    _MAX_RESPONSE_TOKENS: int = 4096

    def __init__(self, api_key: str, model: str = "claude-opus-4-5") -> None:
        if anthropic is None:
            raise ImportError(
                "anthropic SDK is required for UnderstandingAgent. "
                "Install with: pip install -e '.[understanding]'"
            )
        if not api_key.strip():
            raise ValueError("api_key must be non-empty")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def understand(self, doc: PaperDocument) -> PaperUnderstanding:
        prompt = self._build_prompt(doc)
        if len(prompt) <= self._MAX_PROMPT_CHARS:
            return self._single_pass_understanding(prompt)

        architecture_prompt, experiments_prompt = self._build_split_prompts(doc)
        architecture_understanding = self._single_pass_understanding(architecture_prompt)
        experiments_understanding = self._single_pass_understanding(experiments_prompt)
        return self._merge_understandings(architecture_understanding, experiments_understanding)

    def _single_pass_understanding(self, prompt: str) -> PaperUnderstanding:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self._MAX_RESPONSE_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # pragma: no cover - SDK/runtime dependent
            raise UnderstandingError(f"Understanding model call failed: {exc}") from exc

        if not response.content:
            raise UnderstandingError("Anthropic response had no content blocks")

        raw = getattr(response.content[0], "text", None)
        if not isinstance(raw, str) or not raw.strip():
            raise UnderstandingError("Anthropic response did not contain text in the first content block")

        data = self._parse_json_response(raw)
        return PaperUnderstanding.from_dict(data)

    def _build_prompt(self, doc: PaperDocument) -> str:
        method_text = self._select_section_text(doc.sections, {"method", "model", "architecture", "approach"})
        experiments_text = self._select_section_text(doc.sections, {"experiments", "results", "evaluation"})
        conclusion_text = self._select_section_text(doc.sections, {"conclusion", "discussion", "future work"})

        algorithms_text = self._join_algorithm_blocks(doc)
        equations_text = self._join_equation_blocks(doc)

        prompt = self._render_prompt(
            doc=doc,
            method_text=method_text,
            experiments_text=experiments_text,
            conclusion_text=conclusion_text,
            algorithms_text=algorithms_text,
            equations_text=equations_text,
            pass_label="full paper analysis",
        )
        return self._truncate_prompt(prompt)

    def _build_split_prompts(self, doc: PaperDocument) -> tuple[str, str]:
        architecture_prompt = self._render_prompt(
            doc=doc,
            method_text=self._select_section_text(doc.sections, {"method", "model", "architecture", "approach"}),
            experiments_text="(omitted for architecture pass)",
            conclusion_text=self._select_section_text(doc.sections, {"conclusion", "discussion"}),
            algorithms_text=self._join_algorithm_blocks(doc),
            equations_text=self._join_equation_blocks(doc),
            pass_label="architecture pass",
        )
        experiments_prompt = self._render_prompt(
            doc=doc,
            method_text="(method already analyzed in architecture pass)",
            experiments_text=self._select_section_text(doc.sections, {"experiments", "results", "evaluation"}),
            conclusion_text=self._select_section_text(doc.sections, {"conclusion", "discussion"}),
            algorithms_text="(algorithm blocks already analyzed in architecture pass)",
            equations_text="(equations already analyzed in architecture pass)",
            pass_label="experiments pass",
        )
        return self._truncate_prompt(architecture_prompt), self._truncate_prompt(experiments_prompt)

    def _render_prompt(
        self,
        *,
        doc: PaperDocument,
        method_text: str,
        experiments_text: str,
        conclusion_text: str,
        algorithms_text: str,
        equations_text: str,
        pass_label: str,
    ) -> str:
        return f"""Pass: {pass_label}
Title: {doc.title}
Authors: {", ".join(doc.authors)}
Abstract:
{doc.abstract}

Method / Model Sections:
{method_text}

Experiments / Evaluation Sections:
{experiments_text}

Algorithm Blocks:
{algorithms_text}

Top Equations With Context:
{equations_text}

Conclusion:
{conclusion_text}

Return a JSON object with these fields:
{{
  "paper_title": str,
  "one_line_summary": str,
  "problem_statement": str,
  "prior_state_of_art": str,
  "key_insight": str,
  "why_it_works": str,
  "contributions": [
    {{"claim": str, "novelty": str, "is_implementable": bool, "implementation_notes": str}}
  ],
  "requirements": [
    {{"name": str, "requirement_type": "dataset|library|hardware|baseline|pretrained_model", "is_optional": bool, "version_constraint": str | null, "acquisition_url": str | null, "notes": str}}
  ],
  "concept_nodes": [
    {{"id": str, "label": str, "concept_type": "model|operation|loss|dataset|metric|technique", "description": str, "paper_section": str}}
  ],
  "concept_edges": [
    {{"source_id": str, "target_id": str, "relation": "uses|produces|replaces|compared_against|trained_on|evaluated_on", "weight": float}}
  ],
  "core_algorithm_description": str,
  "input_output_spec": str,
  "hyperparameters": dict,
  "evaluation_protocol": str,
  "known_limitations": str,
  "complexity": "trivial|low|medium|high|frontier-only",
  "estimated_impl_hours": int,
  "can_reproduce_without_compute": bool,
  "confidence": float,
  "confidence_notes": str
}}
Return only JSON.
"""

    def _truncate_prompt(self, prompt: str) -> str:
        if len(prompt) <= self._MAX_PROMPT_CHARS:
            return prompt
        truncated = prompt[: self._MAX_PROMPT_CHARS - len("\n[truncated due to prompt budget]")]
        return f"{truncated}\n[truncated due to prompt budget]"

    def _select_section_text(self, sections: list[Section], kinds: set[str]) -> str:
        selected: list[str] = []
        for section in sections:
            section_key = (section.section_type or "").casefold()
            title_key = section.title.casefold()
            if section_key in kinds or any(kind in title_key for kind in kinds):
                selected.append(f"## {section.title}\n{section.content}")
        if not selected:
            return "(none)"
        text = "\n\n".join(selected)
        if len(text) > self._MAX_SECTION_CHARS:
            return text[: self._MAX_SECTION_CHARS] + "\n[truncated due to prompt budget]"
        return text

    def _join_algorithm_blocks(self, doc: PaperDocument) -> str:
        blocks = []
        for algorithm in doc.algorithms:
            blocks.append(
                f"=== {algorithm.name} (p.{algorithm.page}) ===\n{algorithm.pseudocode}"
            )
        return "\n\n".join(blocks) if blocks else "(none)"

    def _join_equation_blocks(self, doc: PaperDocument) -> str:
        blocks = []
        for equation in doc.equations[:20]:
            blocks.append(
                f"[Eq p.{equation.page}] {equation.latex}\nContext: {equation.description}"
            )
        return "\n\n".join(blocks) if blocks else "(none)"

    def _parse_json_response(self, raw: str) -> dict[str, Any]:
        try:
            return self.clean_json_response(raw)
        except json.JSONDecodeError as exc:
            preview = raw[:200].replace("\n", " ")
            raise UnderstandingError(
                "Failed to parse UnderstandingAgent response as JSON. "
                f"Response preview: {preview!r}"
            ) from exc

    @staticmethod
    def clean_json_response(raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            fence_parts = cleaned.split("```")
            if len(fence_parts) >= 2:
                cleaned = fence_parts[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
        try:
            parsed = json.loads(cleaned.strip())
        except json.JSONDecodeError:
            extracted = UnderstandingAgent._extract_first_json_object_block(cleaned)
            if extracted is None:
                raise
            parsed = json.loads(extracted)
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError("Top-level JSON value is not an object", cleaned, 0)
        return parsed

    @staticmethod
    def _extract_first_json_object_block(text: str) -> Optional[str]:
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
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        return None

    def _merge_understandings(
        self,
        architecture: PaperUnderstanding,
        experiments: PaperUnderstanding,
    ) -> PaperUnderstanding:
        merged = architecture.to_dict()
        experiment_data = experiments.to_dict()

        for scalar_field in [
            "evaluation_protocol",
            "known_limitations",
            "confidence_notes",
            "prior_state_of_art",
            "why_it_works",
        ]:
            if experiment_data.get(scalar_field):
                merged[scalar_field] = experiment_data[scalar_field]

        merged["requirements"] = self._merge_list_dicts(
            architecture.requirements, experiments.requirements, key_attr="name"
        )
        merged["contributions"] = self._merge_list_dicts(
            architecture.contributions, experiments.contributions, key_attr="claim"
        )
        merged["concept_nodes"] = self._merge_list_dicts(
            architecture.concept_nodes, experiments.concept_nodes, key_attr="id"
        )
        merged["concept_edges"] = self._merge_edge_dicts(architecture, experiments)
        merged["hyperparameters"] = {
            **architecture.hyperparameters,
            **experiments.hyperparameters,
        }
        merged["confidence"] = max(architecture.confidence, experiments.confidence)
        merged["can_reproduce_without_compute"] = (
            architecture.can_reproduce_without_compute or experiments.can_reproduce_without_compute
        )
        return PaperUnderstanding.from_dict(merged)

    def _merge_list_dicts(
        self,
        primary: list[Any],
        secondary: list[Any],
        *,
        key_attr: str,
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for item in list(primary) + list(secondary):
            key = str(getattr(item, key_attr, "")).strip()
            if not key:
                continue
            merged[key] = item.to_dict()
        return list(merged.values())

    def _merge_edge_dicts(
        self,
        architecture: PaperUnderstanding,
        experiments: PaperUnderstanding,
    ) -> list[dict[str, Any]]:
        merged: dict[tuple[str, str, str], dict[str, Any]] = {}
        for edge in architecture.concept_edges + experiments.concept_edges:
            key = (edge.source_id, edge.target_id, edge.relation)
            payload = edge.to_dict()
            existing = merged.get(key)
            if existing is None or float(payload.get("weight", 0.0)) > float(
                existing.get("weight", 0.0)
            ):
                merged[key] = payload
        return list(merged.values())
