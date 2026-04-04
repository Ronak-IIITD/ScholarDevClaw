"""
LLM-powered research assistant for paper analysis and code understanding.

Wraps :class:`LLMClient` with structured prompts for:

- Extracting implementation specs from research paper text.
- Analysing code snippets and suggesting improvements.
- Generating implementation plans from paper+code context.
- Finding implementation references by summarising search results.

When no LLM key is configured the assistant is still constructable but
every method returns ``None``, allowing callers to fall back to hardcoded
data gracefully.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from .client import LLMAPIError, LLMClient, LLMConfigError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExtractedSpec:
    """Structured specification extracted from a research paper."""

    paper: dict[str, Any]
    algorithm: dict[str, Any]
    implementation: dict[str, Any]
    changes: dict[str, Any]
    validation: dict[str, Any]
    raw_response: str = ""


@dataclass
class CodeAnalysis:
    """Result of LLM-powered code analysis."""

    summary: str
    patterns_found: list[str]
    improvement_opportunities: list[dict[str, Any]]
    complexity_assessment: str = ""
    raw_response: str = ""


@dataclass
class ImplementationPlan:
    """LLM-generated plan for implementing a paper's ideas into code."""

    steps: list[dict[str, Any]]
    estimated_difficulty: str = "medium"
    target_files: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    expected_benefits: list[str] = field(default_factory=list)
    raw_response: str = ""


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from LLM text output.

    Handles:
    - Pure JSON responses
    - JSON wrapped in ```json ... ``` fences
    - JSON embedded in surrounding prose
    """
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try fenced code block
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try first { ... } block
    brace_start = text.find("{")
    if brace_start == -1:
        return None

    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ---------------------------------------------------------------------------
# LLMResearchAssistant
# ---------------------------------------------------------------------------


class LLMResearchAssistant:
    """High-level research assistant backed by an LLM provider.

    Parameters
    ----------
    client : LLMClient | None
        An already-constructed LLM client.  When *None* the assistant
        operates in **offline mode** — every method returns ``None`` so
        callers can gracefully fall back.
    max_tokens : int
        Default max tokens for LLM calls (per request).
    temperature : float
        Default temperature.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> None:
        self._client = client
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def is_available(self) -> bool:
        """Return True when an LLM client is configured."""
        return self._client is not None

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResearchAssistant:
        """Try to build an assistant from env vars / explicit params.

        Returns an *offline* assistant (``is_available == False``) when no
        credentials are found — never raises.
        """
        try:
            if provider:
                client = LLMClient.from_provider(provider, api_key=api_key, model=model)
            else:
                # Auto-detect from environment
                client = _auto_detect_client(model=model)

            if client is None:
                return cls(None, max_tokens=max_tokens, temperature=temperature)

            return cls(client, max_tokens=max_tokens, temperature=temperature)
        except (LLMConfigError, LLMAPIError, Exception) as exc:
            logger.debug("LLMResearchAssistant could not initialise: %s", exc)
            return cls(None, max_tokens=max_tokens, temperature=temperature)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def extract_paper_spec(
        self,
        paper_text: str,
        *,
        paper_title: str = "",
        target_language: str = "python",
    ) -> ExtractedSpec | None:
        """Extract a structured implementation spec from paper text.

        Returns ``None`` when the LLM is unavailable.
        """
        if not self.is_available:
            return None

        system = (
            "You are a research engineer specialising in extracting actionable "
            "implementation specifications from machine learning papers. "
            "Always respond with valid JSON — no extra prose."
        )

        prompt = f"""Analyse the following research paper text and extract a structured
implementation specification in JSON format.

Paper title: {paper_title or "(unknown)"}
Target language: {target_language}

---BEGIN PAPER TEXT---
{paper_text[:12000]}
---END PAPER TEXT---

Return a JSON object with exactly these top-level keys:

{{
  "paper": {{
    "title": "<string>",
    "authors": ["<string>", ...],
    "year": <int>,
    "arxiv": "<id or empty string>"
  }},
  "algorithm": {{
    "name": "<string>",
    "replaces": "<what it replaces or 'N/A'>",
    "description": "<1-2 sentence description>",
    "formula": "<core formula or 'N/A'>",
    "category": "<normalization|attention|activation|position_encoding|optimizer|architecture|other>"
  }},
  "implementation": {{
    "module_name": "<class/function name>",
    "parent_class": "<parent class or 'N/A'>",
    "parameters": ["<param1>", ...],
    "forward_signature": "<signature>"
  }},
  "changes": {{
    "type": "<replace|add|modify>",
    "target_patterns": ["<pattern1>", ...],
    "replacement": "<replacement name>",
    "insertion_points": ["<location1>", ...],
    "expected_benefits": ["<benefit1>", ...]
  }},
  "validation": {{
    "test_type": "<training_comparison|benchmark|unit_test>",
    "metrics": ["<metric1>", ...],
    "max_benchmark_time": <int seconds>
  }}
}}"""

        try:
            response = self._client.chat(
                prompt,
                system=system,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            parsed = _extract_json(response.content)
            if parsed is None:
                logger.warning("LLM returned non-JSON for paper spec extraction")
                return None

            return ExtractedSpec(
                paper=parsed.get("paper", {}),
                algorithm=parsed.get("algorithm", {}),
                implementation=parsed.get("implementation", {}),
                changes=parsed.get("changes", {}),
                validation=parsed.get("validation", {}),
                raw_response=response.content,
            )
        except (LLMAPIError, Exception) as exc:
            logger.warning("LLM paper spec extraction failed: %s", exc)
            return None

    def analyse_code(
        self,
        code: str,
        *,
        file_path: str = "",
        language: str = "python",
        focus: str = "",
    ) -> CodeAnalysis | None:
        """Analyse a code snippet for patterns and improvement opportunities.

        Returns ``None`` when the LLM is unavailable.
        """
        if not self.is_available:
            return None

        system = (
            "You are a senior ML engineer reviewing code for potential "
            "research-backed improvements. Respond with valid JSON only."
        )

        focus_instruction = ""
        if focus:
            focus_instruction = f"\nFocus area: {focus}"

        prompt = f"""Analyse the following {language} code and identify patterns
and potential improvements based on recent ML research.{focus_instruction}

File: {file_path or "(unknown)"}

```{language}
{code[:8000]}
```

Return a JSON object:
{{
  "summary": "<brief summary of what the code does>",
  "patterns_found": ["<pattern1>", "<pattern2>", ...],
  "improvement_opportunities": [
    {{
      "current_pattern": "<what exists>",
      "suggested_improvement": "<research-backed improvement>",
      "paper_reference": "<paper name/arxiv id if known>",
      "estimated_impact": "<low|medium|high>",
      "description": "<1-2 sentences>"
    }}
  ],
  "complexity_assessment": "<simple|moderate|complex>"
}}"""

        try:
            response = self._client.chat(
                prompt,
                system=system,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            parsed = _extract_json(response.content)
            if parsed is None:
                logger.warning("LLM returned non-JSON for code analysis")
                return None

            return CodeAnalysis(
                summary=parsed.get("summary", ""),
                patterns_found=parsed.get("patterns_found", []),
                improvement_opportunities=parsed.get("improvement_opportunities", []),
                complexity_assessment=parsed.get("complexity_assessment", ""),
                raw_response=response.content,
            )
        except (LLMAPIError, Exception) as exc:
            logger.warning("LLM code analysis failed: %s", exc)
            return None

    def generate_implementation_plan(
        self,
        paper_spec: dict[str, Any],
        code_context: str,
        *,
        language: str = "python",
    ) -> ImplementationPlan | None:
        """Create a step-by-step implementation plan.

        Combines the paper specification with existing code context to
        generate concrete integration steps.

        Returns ``None`` when the LLM is unavailable.
        """
        if not self.is_available:
            return None

        system = (
            "You are a principal ML engineer planning the integration of "
            "a research technique into production code. Respond with valid JSON only."
        )

        prompt = f"""Given the following paper specification and existing code,
create a detailed implementation plan.

Paper Spec:
```json
{json.dumps(paper_spec, indent=2)[:4000]}
```

Existing code context:
```{language}
{code_context[:6000]}
```

Return a JSON object:
{{
  "steps": [
    {{
      "order": <int>,
      "action": "<create|modify|replace|add_import>",
      "target": "<file or class name>",
      "description": "<what to do>",
      "code_hint": "<short code snippet or empty>"
    }}
  ],
  "estimated_difficulty": "<easy|medium|hard>",
  "target_files": ["<file1>", ...],
  "risks": ["<risk1>", ...],
  "expected_benefits": ["<benefit1>", ...]
}}"""

        try:
            response = self._client.chat(
                prompt,
                system=system,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            parsed = _extract_json(response.content)
            if parsed is None:
                logger.warning("LLM returned non-JSON for implementation plan")
                return None

            return ImplementationPlan(
                steps=parsed.get("steps", []),
                estimated_difficulty=parsed.get("estimated_difficulty", "medium"),
                target_files=parsed.get("target_files", []),
                risks=parsed.get("risks", []),
                expected_benefits=parsed.get("expected_benefits", []),
                raw_response=response.content,
            )
        except (LLMAPIError, Exception) as exc:
            logger.warning("LLM implementation plan generation failed: %s", exc)
            return None

    def summarise_search_results(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> str | None:
        """Produce a concise summary of web/GitHub search results.

        Returns ``None`` when the LLM is unavailable.
        """
        if not self.is_available:
            return None

        if not results:
            return None

        system = (
            "You are a research assistant. Summarise search results concisely, "
            "highlighting the most relevant implementations and resources."
        )

        results_text = json.dumps(results[:20], indent=2, default=str)[:6000]
        prompt = f"""Summarise these search results for the query: "{query}"

{results_text}

Provide a 2-4 sentence summary of the most relevant findings, including
any notable repositories, papers, or implementations."""

        try:
            response = self._client.chat(
                prompt,
                system=system,
                max_tokens=1024,
                temperature=0.2,
            )
            return response.content.strip()
        except (LLMAPIError, Exception) as exc:
            logger.warning("LLM search summarisation failed: %s", exc)
            return None

    def analyse_github_repo_content(
        self,
        repo_info: dict[str, Any],
        file_contents: dict[str, str],
    ) -> dict[str, Any] | None:
        """Analyse fetched GitHub repo files for relevant implementations.

        Parameters
        ----------
        repo_info : dict
            Basic info (owner, repo, url, description).
        file_contents : dict
            Mapping of file paths to their content.

        Returns ``None`` when the LLM is unavailable.
        """
        if not self.is_available:
            return None

        if not file_contents:
            return None

        system = (
            "You are analysing a GitHub repository's code for ML research "
            "implementations. Respond with valid JSON only."
        )

        # Build a truncated view of the files
        files_text = ""
        for path, content in list(file_contents.items())[:10]:
            files_text += f"\n--- {path} ---\n{content[:2000]}\n"

        prompt = f"""Analyse this GitHub repository's code.

Repository: {repo_info.get("owner", "")}/{repo_info.get("repo", "")}
Description: {repo_info.get("description", "N/A")}

Files:
{files_text[:8000]}

Return a JSON object:
{{
  "summary": "<what this repo implements>",
  "algorithms_found": ["<alg1>", ...],
  "quality_score": <0-100>,
  "key_files": ["<most important file paths>", ...],
  "implementation_notes": "<notable implementation details>"
}}"""

        try:
            response = self._client.chat(
                prompt,
                system=system,
                max_tokens=2048,
                temperature=0.1,
            )
            parsed = _extract_json(response.content)
            if parsed is None:
                return None

            parsed["raw_response"] = response.content
            return parsed
        except (LLMAPIError, Exception) as exc:
            logger.warning("LLM GitHub repo analysis failed: %s", exc)
            return None

    def close(self) -> None:
        """Close the underlying LLM client if present."""
        if self._client is not None:
            self._client.close()

    def __enter__(self) -> LLMResearchAssistant:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Auto-detection helper
# ---------------------------------------------------------------------------

# Priority order for auto-detecting an LLM provider from env vars
_AUTO_DETECT_ENV_VARS = [
    ("ANTHROPIC_API_KEY", "anthropic"),
    ("OPENAI_API_KEY", "openai"),
    ("GROQ_API_KEY", "groq"),
    ("DEEPSEEK_API_KEY", "deepseek"),
    ("MISTRAL_API_KEY", "mistral"),
    ("COHERE_API_KEY", "cohere"),
    ("TOGETHER_API_KEY", "together"),
    ("FIREWORKS_API_KEY", "fireworks"),
    ("OPENROUTER_API_KEY", "openrouter"),
]


def _auto_detect_client(*, model: str | None = None) -> LLMClient | None:
    """Try to create an LLMClient from the first available env var."""
    for env_var, provider_name in _AUTO_DETECT_ENV_VARS:
        key = os.environ.get(env_var, "").strip()
        if key:
            try:
                return LLMClient.from_provider(provider_name, api_key=key, model=model)
            except Exception:
                continue

    # Check for Ollama (no key needed)
    try:
        import httpx

        ollama_host = os.environ.get("OLLAMA_HOST", "").strip() or "http://localhost:11434"
        resp = httpx.get(f"{ollama_host.rstrip('/')}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            return LLMClient.from_provider(
                "ollama",
                api_key="",
                base_url=ollama_host,
                model=model,
            )
    except Exception:
        pass

    return None
