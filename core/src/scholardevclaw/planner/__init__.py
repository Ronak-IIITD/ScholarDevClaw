from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

LogCallback = Callable[[str], None]


@dataclass(slots=True)
class PlannerResult:
    ok: bool
    title: str
    payload: dict[str, Any]
    logs: list[str]
    error: str | None = None


def _log(logs: list[str], message: str, log_callback: LogCallback | None = None) -> None:
    logs.append(message)
    if log_callback is not None:
        log_callback(message)


def run_planner(
    repo_path: str,
    *,
    max_specs: int = 5,
    target_categories: list[str] | None = None,
    log_callback: LogCallback | None = None,
) -> PlannerResult:
    from pathlib import Path

    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer
    from scholardevclaw.research_intelligence.extractor import ResearchExtractor

    logs: list[str] = []
    _log(logs, f"Planning multi-spec migration for: {repo_path}", log_callback)

    try:
        path = Path(repo_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Repository not found: {repo_path}")

        _log(logs, "Analyzing repository structure...", log_callback)
        analyzer = TreeSitterAnalyzer(path)
        analysis = analyzer.analyze()

        _log(
            logs,
            f"Detected languages: {', '.join(analysis.languages) if analysis.languages else 'None'}",
            log_callback,
        )
        _log(
            logs,
            f"Detected frameworks: {', '.join(analysis.frameworks) if analysis.frameworks else 'None'}",
            log_callback,
        )

        extractor = ResearchExtractor()
        suggestions = analyzer.suggest_research_papers()

        _log(logs, f"Found {len(suggestions)} improvement opportunities", log_callback)

        available_specs = extractor.list_available_specs()
        categories = extractor.get_categories()

        selected_specs: list[dict[str, Any]] = []
        seen_specs: set[str] = set()

        if suggestions:
            for suggestion in suggestions[:max_specs]:
                spec_name = suggestion.get("paper", {}).get("name")
                if spec_name and spec_name not in seen_specs:
                    seen_specs.add(spec_name)
                    spec = extractor.get_spec(spec_name)
                    if spec:
                        if target_categories:
                            cat = spec.get("algorithm", {}).get("category", "")
                            if cat not in target_categories:
                                continue
                        selected_specs.append(
                            {
                                "name": spec_name,
                                "confidence": suggestion.get("confidence", 0),
                                "category": spec.get("algorithm", {}).get("category", ""),
                                "replaces": spec.get("algorithm", {}).get("replaces", ""),
                                "expected_benefits": spec.get("changes", {}).get(
                                    "expected_benefits", []
                                ),
                            }
                        )

        if not selected_specs:
            for spec_name in available_specs[:max_specs]:
                if spec_name in seen_specs:
                    continue
                spec = extractor.get_spec(spec_name)
                if spec:
                    if target_categories:
                        cat = spec.get("algorithm", {}).get("category", "")
                        if cat not in target_categories:
                            continue
                    selected_specs.append(
                        {
                            "name": spec_name,
                            "confidence": 50,
                            "category": spec.get("algorithm", {}).get("category", ""),
                            "replaces": spec.get("algorithm", {}).get("replaces", ""),
                            "expected_benefits": spec.get("changes", {}).get(
                                "expected_benefits", []
                            ),
                        }
                    )

        ordered_specs, dependency_reasoning = _order_specs_by_dependency(selected_specs)

        combined_benefits = _estimate_combined_impact(ordered_specs)

        payload = {
            "repo_path": str(path),
            "languages": analysis.languages,
            "frameworks": analysis.frameworks,
            "opportunities_found": len(suggestions),
            "selected_specs": ordered_specs,
            "dependency_order": [s["name"] for s in ordered_specs],
            "dependency_reasoning": dependency_reasoning,
            "combined_benefits": combined_benefits,
            "total_expected_improvement": _summarize_improvement(combined_benefits),
            "available_categories": categories,
        }

        _log(logs, f"Planned migration with {len(ordered_specs)} specs", log_callback)
        _log(logs, f"Execution order: {' -> '.join(payload['dependency_order'])}", log_callback)

        return PlannerResult(
            ok=True,
            title="Migration Planner",
            payload=payload,
            logs=logs,
        )
    except Exception as exc:
        _log(logs, f"Failed: {exc}", log_callback)
        return PlannerResult(
            ok=False,
            title="Migration Planner",
            payload={},
            logs=logs,
            error=str(exc),
        )


def _order_specs_by_dependency(
    specs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    category_order = {
        "normalization": 1,
        "activation": 2,
        "attention": 3,
        "position_encoding": 4,
    }

    def get_priority(spec: dict[str, Any]) -> int:
        cat = spec.get("category", "")
        return category_order.get(cat, 99)

    ordered = sorted(specs, key=get_priority)

    reasoning = []
    if len(ordered) > 1:
        reasoning.append(
            f"Normalization ({ordered[0].get('name', '')}) applied first for stable gradients"
        )
        if len(ordered) > 2:
            reasoning.append(
                f"Activation ({ordered[1].get('name', '')}) applied after normalization"
            )
            reasoning.append(
                f"Attention ({ordered[2].get('name', '')}) benefits from prior optimizations"
            )
        reasoning.append(
            "Dependencies flow: stable gradients -> efficient activation -> better attention"
        )

    return ordered, reasoning


def _estimate_combined_impact(specs: list[dict[str, Any]]) -> dict[str, Any]:
    combined_benefits: dict[str, Any] = {
        "speedup_estimate": 0.0,
        "memory_estimate": 0.0,
        "benefit_categories": set(),
    }

    speedup_map = {
        "rmsnorm": 0.07,
        "swiglu": 0.1,
        "flashattention": 0.25,
        "rope": 0.05,
    }

    memory_map = {
        "rmsnorm": 0.02,
        "swiglu": 0.05,
        "flashattention": 0.3,
        "rope": 0.01,
    }

    for spec in specs:
        name = spec.get("name", "").lower()
        combined_benefits["speedup_estimate"] += speedup_map.get(name, 0.05)
        combined_benefits["memory_estimate"] += memory_map.get(name, 0.02)

        for benefit in spec.get("expected_benefits", []):
            combined_benefits["benefit_categories"].add(benefit)

    combined_benefits["speedup_estimate"] = min(combined_benefits["speedup_estimate"], 0.5)
    combined_benefits["memory_estimate"] = min(combined_benefits["memory_estimate"], 0.4)

    combined_benefits["benefit_categories"] = list(combined_benefits["benefit_categories"])

    return combined_benefits


def _summarize_improvement(combined_benefits: dict[str, Any]) -> str:
    speedup = combined_benefits.get("speedup_estimate", 0) * 100
    memory = combined_benefits.get("memory_estimate", 0) * 100

    return (
        f"Estimated {speedup:.0f}% training speedup, "
        f"{memory:.0f}% memory reduction, "
        f"{len(combined_benefits.get('benefit_categories', []))} benefit areas"
    )
