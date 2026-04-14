from __future__ import annotations

import json
import re
from typing import Any

from scholardevclaw.ingestion.models import PaperDocument
from scholardevclaw.understanding.models import PaperUnderstanding

try:
    import anthropic  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised only without optional dependency
    anthropic = None  # type: ignore[assignment]

SYSTEM_PROMPT = """You are an expert ML researcher and senior software engineer.
You read research papers with precision and extract structured information
that allows a developer to implement the paper from scratch.
Always respond with valid JSON matching the exact schema provided.
Never hallucinate citations or results. If unsure, lower confidence.
"""


class UnderstandingAgent:
    """LLM-backed paper understanding agent producing ``PaperUnderstanding``."""

    _MAX_PROMPT_CHARS: int = 120_000
    _MIN_EQUATION_CHARS: int = 1_000
    _MIN_ALGORITHM_CHARS: int = 2_000

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
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        if not response.content:
            raise ValueError("Anthropic response had no content blocks.")

        first_block = response.content[0]
        raw = getattr(first_block, "text", None)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Anthropic response did not contain text in the first content block.")

        data = self._parse_json_response(raw)
        return PaperUnderstanding.from_dict(data)

    def _build_prompt(self, doc: PaperDocument) -> str:
        authors = ", ".join(doc.authors)

        algorithm_blocks = [
            f"=== {algorithm.name} (p.{algorithm.page}) ===\n{algorithm.pseudocode}"
            for algorithm in doc.algorithms
        ]
        equation_blocks = [
            f"[Eq p.{equation.page}] {equation.latex} ({equation.description})"
            for equation in doc.equations[:20]
        ]

        conclusion_candidates = [
            section.content for section in doc.sections if "conclusion" in section.title.casefold()
        ]
        conclusion = "\n\n".join(conclusion_candidates)[:3000]

        static_prompt = self._render_prompt(
            title=doc.title,
            authors=authors,
            abstract=doc.abstract,
            algo_text="",
            eq_text="",
            conclusion=conclusion,
        )
        static_len = len(static_prompt)

        remaining_budget = max(self._MAX_PROMPT_CHARS - static_len, 0)
        eq_budget = 0
        algo_budget = remaining_budget

        if equation_blocks and algorithm_blocks:
            eq_budget = min(max(self._MIN_EQUATION_CHARS, remaining_budget // 4), remaining_budget)
            algo_budget = max(remaining_budget - eq_budget, 0)
            if algo_budget < self._MIN_ALGORITHM_CHARS:
                deficit = self._MIN_ALGORITHM_CHARS - algo_budget
                transfer = min(deficit, eq_budget)
                eq_budget -= transfer
                algo_budget += transfer
        elif equation_blocks:
            eq_budget = remaining_budget
            algo_budget = 0

        algo_text = self._join_blocks_with_budget(algorithm_blocks, algo_budget) or "(none)"
        eq_text = self._join_blocks_with_budget(equation_blocks, eq_budget) or "(none)"

        return self._render_prompt(
            title=doc.title,
            authors=authors,
            abstract=doc.abstract,
            algo_text=algo_text,
            eq_text=eq_text,
            conclusion=conclusion,
        )

    def _parse_json_response(self, raw: str) -> dict[str, Any]:
        cleaned = self._strip_markdown_fences(raw.strip())

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            object_block = self._extract_first_json_object_block(cleaned)
            if object_block is None:
                preview = cleaned[:200].replace("\n", " ")
                raise ValueError(
                    "Failed to parse UnderstandingAgent response as JSON. "
                    "Ensure the model returns a JSON object only. "
                    f"Response preview: {preview!r}"
                )
            try:
                parsed = json.loads(object_block)
            except json.JSONDecodeError as exc:
                preview = object_block[:200].replace("\n", " ")
                raise ValueError(
                    "Failed to parse extracted JSON object from UnderstandingAgent response. "
                    "Ensure field names and quotes are valid JSON. "
                    f"Extracted preview: {preview!r}"
                ) from exc

        if not isinstance(parsed, dict):
            raise ValueError("UnderstandingAgent response must be a JSON object at top level.")
        return parsed

    def _render_prompt(
        self,
        *,
        title: str,
        authors: str,
        abstract: str,
        algo_text: str,
        eq_text: str,
        conclusion: str,
    ) -> str:
        return f"""Paper: {title}
Authors: {authors}
Abstract: {abstract}

Algorithms:
{algo_text}

Key Equations (max 20):
{eq_text}

Conclusion:
{conclusion}

---
Analyze this paper and return a JSON object with exactly these fields:
{{
  "paper_title": str,
  "one_line_summary": str,
  "problem_statement": str,
  "key_insight": str,
  "contributions": [
    {{"claim": str, "novelty": str, "is_implementable": bool}}
  ],
  "requirements": [
    {{"name": str, "type": "dataset|library|hardware|baseline", "is_optional": bool, "notes": str}}
  ],
  "concept_nodes": [
    {{"id": str, "label": str, "type": "model|operation|loss|dataset|metric", "description": str}}
  ],
  "concept_edges": [
    {{"source_id": str, "target_id": str, "relation": "uses|produces|compared_against|trained_on"}}
  ],
  "core_algorithm_description": str,
  "input_output_spec": str,
  "evaluation_protocol": str,
  "complexity": "low|medium|high|research-only",
  "estimated_impl_hours": int,
  "confidence": float
}}
Return only the JSON object. No markdown fences. No preamble.
"""

    def _join_blocks_with_budget(self, blocks: list[str], budget: int) -> str:
        if not blocks:
            return ""

        if budget <= 0:
            return "...[truncated due to prompt budget]"

        separator = "\n\n"
        suffix = "\n...[truncated due to prompt budget]"

        chunks: list[str] = []
        used = 0

        for block in blocks:
            block_with_separator_len = len(block)
            if chunks:
                block_with_separator_len += len(separator)

            if used + block_with_separator_len <= budget:
                chunks.append(block)
                used += block_with_separator_len
                continue

            remaining = budget - used
            if chunks:
                remaining -= len(separator)

            if remaining <= 0:
                break

            if len(suffix) >= remaining:
                truncated_chunk = suffix[:remaining]
            else:
                truncated_chunk = block[: remaining - len(suffix)] + suffix
            chunks.append(truncated_chunk)
            break

        return separator.join(chunks)

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
