from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from scholardevclaw.auth.types import AuthProvider
from scholardevclaw.execution.sandbox import ExecutionReport
from scholardevclaw.llm.client import LLMClient
from scholardevclaw.understanding.models import PaperUnderstanding

try:
    import anthropic  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency path
    anthropic = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_METRIC_SYNONYMS = {
    "acc": "accuracy",
    "top1": "top_1_accuracy",
    "top_1": "top_1_accuracy",
    "top5": "top_5_accuracy",
    "top_5": "top_5_accuracy",
    "f1_score": "f1",
    "f1score": "f1",
    "rouge_l": "rouge_l",
    "bleu_score": "bleu",
    "eval_accuracy": "accuracy",
    "test_accuracy": "accuracy",
    "validation_accuracy": "accuracy",
}


@dataclass(slots=True)
class ReproducibilityReport:
    paper_title: str
    claimed_metrics: dict[str, float]
    achieved_metrics: dict[str, float]
    delta: dict[str, float]
    score: float
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_title": self.paper_title,
            "claimed_metrics": dict(self.claimed_metrics),
            "achieved_metrics": dict(self.achieved_metrics),
            "delta": dict(self.delta),
            "score": self.score,
            "verdict": self.verdict,
        }


class ReproducibilityScorer:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5",
        provider: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.client: Any | None = None

        if client is not None:
            self.client = client
            return

        provider_name = (provider or "anthropic").strip().lower()

        normalized_key = (api_key or "").strip()
        if provider_name != AuthProvider.ANTHROPIC.value:
            try:
                auth_provider = AuthProvider(provider_name)
            except ValueError:
                auth_provider = None
            if auth_provider is not None and (normalized_key or not auth_provider.requires_api_key):
                try:
                    self.client = LLMClient.from_provider(
                        auth_provider,
                        api_key=normalized_key,
                        model=model,
                    )
                except Exception as exc:  # pragma: no cover - runtime/client dependent
                    logger.warning("Failed to initialize scorer LLM client: %s", exc)
            return

        if normalized_key and anthropic is not None:
            try:
                self.client = anthropic.Anthropic(api_key=normalized_key)
            except Exception as exc:  # pragma: no cover - runtime-dependent SDK init failures
                logger.warning("Failed to initialize Anthropic client for scorer: %s", exc)

    def score(
        self,
        understanding: PaperUnderstanding,
        execution_report: ExecutionReport,
    ) -> ReproducibilityReport:
        achieved = self._extract_metrics_from_output(
            f"{execution_report.stdout}\n{execution_report.stderr}"
        )
        claimed = self._extract_claimed_metrics(understanding)

        delta = {key: achieved.get(key, 0.0) - value for key, value in claimed.items()}
        raw_score = self._compute_score(claimed=claimed, achieved=achieved)
        score = max(0.0, min(1.0, raw_score))
        verdict = "reproduced" if score > 0.9 else "partial" if score > 0.5 else "failed"

        return ReproducibilityReport(
            paper_title=understanding.paper_title,
            claimed_metrics=claimed,
            achieved_metrics=achieved,
            delta=delta,
            score=score,
            verdict=verdict,
        )

    def _extract_metrics_from_output(self, output: str) -> dict[str, float]:
        metrics = self._parse_metrics_regex(output)
        if metrics:
            return metrics

        fallback = self._llm_extract_metrics(output)
        return fallback if fallback else {}

    def _extract_claimed_metrics(self, understanding: PaperUnderstanding) -> dict[str, float]:
        text_blocks = [
            understanding.evaluation_protocol,
            understanding.key_insight,
            understanding.problem_statement,
            understanding.core_algorithm_description,
            "\n".join(contribution.claim for contribution in understanding.contributions),
            "\n".join(contribution.novelty for contribution in understanding.contributions),
        ]
        merged_text = "\n".join(block for block in text_blocks if block)

        metrics = self._parse_metrics_regex(merged_text)
        if metrics:
            return metrics

        fallback = self._llm_extract_metrics(merged_text)
        return fallback if fallback else {}

    def _parse_metrics_regex(self, text: str) -> dict[str, float]:
        if not text.strip():
            return {}

        metrics: dict[str, float] = {}

        pattern = re.compile(
            r"(?P<name>[A-Za-z][A-Za-z0-9_\-\s@]{1,40}?)\s*(?:[:=]|is)\s*"
            r"(?P<value>[-+]?\d*\.?\d+)\s*(?P<suffix>%?)",
            flags=re.IGNORECASE,
        )

        for match in pattern.finditer(text):
            raw_name = match.group("name")
            metric_name = self._normalize_metric_name(raw_name)
            if not metric_name:
                continue

            value = self._coerce_metric_value(
                metric_name=metric_name,
                raw_value=match.group("value"),
                percent_suffix=bool(match.group("suffix")),
            )
            metrics[metric_name] = value

        return metrics

    def _normalize_metric_name(self, raw_name: str) -> str:
        cleaned = raw_name.strip().casefold()
        cleaned = re.sub(r"[^a-z0-9@\s_\-]", "", cleaned)
        cleaned = re.sub(r"\s+", "_", cleaned)
        cleaned = cleaned.replace("-", "_")
        cleaned = cleaned.replace("@", "_")
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")

        if not cleaned:
            return ""

        if cleaned in _METRIC_SYNONYMS:
            return _METRIC_SYNONYMS[cleaned]

        for token in (
            "accuracy",
            "acc",
            "f1",
            "bleu",
            "rouge",
            "mrr",
            "map",
            "auc",
            "precision",
            "recall",
            "loss",
            "perplexity",
            "wer",
            "cer",
        ):
            if token in cleaned:
                if token == "acc" and "accuracy" not in cleaned:
                    return "accuracy"
                if token.startswith("rouge") and "rouge_l" in cleaned:
                    return "rouge_l"
                return _METRIC_SYNONYMS.get(cleaned, cleaned)

        if cleaned.startswith("top_1"):
            return "top_1_accuracy"
        if cleaned.startswith("top_5"):
            return "top_5_accuracy"

        return cleaned

    def _coerce_metric_value(self, metric_name: str, raw_value: str, percent_suffix: bool) -> float:
        try:
            value = float(raw_value)
        except ValueError:
            return 0.0

        if percent_suffix:
            return value / 100.0

        higher_is_better = not any(
            token in metric_name for token in ("loss", "error", "perplexity", "latency", "time")
        )
        if higher_is_better and 1.0 < value <= 100.0:
            return value / 100.0
        return value

    def _llm_extract_metrics(self, text: str) -> dict[str, float]:
        if self.client is None or not text.strip():
            return {}

        prompt = (
            "Extract metric/value pairs from the text below.\n"
            "Return strict JSON object mapping metric_name->float, and no other text.\n"
            "Normalize percentages to 0-1 scale (e.g. 93.5% => 0.935).\n\n"
            f"TEXT:\n{text[:8000]}"
        )

        try:
            chat_fn = getattr(self.client, "chat", None)
            if callable(chat_fn):
                response = chat_fn(prompt, model=self.model, max_tokens=512)
                raw = response.content if isinstance(response.content, str) else ""
            else:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                if not response.content:
                    return {}
                raw = self._extract_first_text_block(response.content)
                if not isinstance(raw, str):
                    return {}

            if not raw.strip():
                return {}
            loaded = json.loads(raw.strip())
            if not isinstance(loaded, dict):
                return {}

            parsed: dict[str, float] = {}
            for key, value in loaded.items():
                if not isinstance(key, str):
                    continue
                try:
                    parsed[self._normalize_metric_name(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            return {k: v for k, v in parsed.items() if k}
        except Exception as exc:  # pragma: no cover - runtime-dependent SDK failures
            logger.debug("LLM metric extraction fallback failed: %s", exc)
            return {}

    @staticmethod
    def _extract_first_text_block(content_blocks: Any) -> str | None:
        if not isinstance(content_blocks, list):
            return None
        for block in content_blocks:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                return text
        return None

    def _compute_score(self, claimed: dict[str, float], achieved: dict[str, float]) -> float:
        if not claimed:
            return 0.5

        scores: list[float] = []
        for metric_name, claimed_value in claimed.items():
            achieved_value = achieved.get(metric_name, 0.0)

            if claimed_value == 0:
                continue
            if achieved_value <= 0:
                scores.append(0.0)
                continue

            ratio = min(achieved_value / claimed_value, claimed_value / achieved_value)
            scores.append(max(0.0, min(1.0, ratio)))

        return sum(scores) / len(scores) if scores else 0.0
