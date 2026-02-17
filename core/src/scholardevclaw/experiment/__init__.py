from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

LogCallback = Callable[[str], None]


@dataclass
class ExperimentVariant:
    name: str
    spec: str
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    expected_benefits: list[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    variant_name: str
    spec: str
    status: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    validation_passed: bool | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    rank: int = 0
    error: str | None = None


@dataclass
class Experiment:
    id: str
    repo_path: str
    name: str
    description: str
    variants: list[ExperimentVariant] = field(default_factory=list)
    results: list[ExperimentResult] = field(default_factory=list)
    status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None


def _log(logs: list[str], message: str, log_callback: LogCallback | None = None) -> None:
    logs.append(message)
    if log_callback is not None:
        log_callback(message)


def generate_variants(
    spec_name: str,
    repo_path: str,
    *,
    variant_count: int = 3,
    log_callback: LogCallback | None = None,
) -> list[ExperimentVariant]:
    from scholardevclaw.research_intelligence.extractor import ResearchExtractor

    logs: list[str] = []
    _log(logs, f"Generating {variant_count} variants for {spec_name}", log_callback)

    extractor = ResearchExtractor()
    spec = extractor.get_spec(spec_name)

    if not spec:
        return []

    variants = []
    algorithm_name = spec.get("algorithm", {}).get("name", spec_name)

    param_variations = _get_parameter_variations(spec_name, variant_count)

    for i, params in enumerate(param_variations):
        variant = ExperimentVariant(
            name=f"{algorithm_name}_variant_{i + 1}",
            spec=spec_name,
            parameters=params,
            confidence=0.6 + (0.1 * (variant_count - i - 1)),
            expected_benefits=spec.get("changes", {}).get("expected_benefits", []),
        )
        variants.append(variant)

    return variants


def _get_parameter_variations(spec_name: str, count: int) -> list[dict[str, Any]]:
    base_params = {
        "rmsnorm": [
            {"eps": 1e-5, "learnable": True},
            {"eps": 1e-6, "learnable": True},
            {"eps": 1e-8, "learnable": False},
        ],
        "swiglu": [
            {"hidden_dim_multiplier": 4.0, "activation": "swish"},
            {"hidden_dim_multiplier": 4.0, "activation": "gelu"},
            {"hidden_dim_multiplier": 2.0, "activation": "silu"},
        ],
        "flashattention": [
            {"block_size": 128, "dropout": 0.0},
            {"block_size": 256, "dropout": 0.0},
            {"block_size": 64, "dropout": 0.1},
        ],
        "rope": [
            {"base": 10000, "max_seq_len": 2048},
            {"base": 50000, "max_seq_len": 4096},
            {"base": 10000, "max_seq_len": 8192},
        ],
    }

    default_params = [{"variant_id": i} for i in range(count)]

    return base_params.get(spec_name, default_params)[:count]


def run_experiment(
    repo_path: str,
    spec_name: str,
    *,
    variant_count: int = 3,
    output_dir: str | None = None,
    log_callback: LogCallback | None = None,
) -> dict[str, Any]:
    from scholardevclaw.application.pipeline import run_generate, run_validate, run_preflight

    logs: list[str] = []
    experiment_id = hashlib.sha256(
        f"{repo_path}{spec_name}{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]

    _log(logs, f"Starting experiment {experiment_id}", log_callback)

    try:
        path = Path(repo_path).expanduser().resolve()

        preflight = run_preflight(str(path))
        logs.extend(preflight.logs)
        if not preflight.ok:
            return {
                "ok": False,
                "experiment_id": experiment_id,
                "error": "Preflight failed",
                "logs": logs,
            }

        variants = generate_variants(
            spec_name, repo_path, variant_count=variant_count, log_callback=log_callback
        )

        if not variants:
            return {
                "ok": False,
                "experiment_id": experiment_id,
                "error": "No variants generated",
                "logs": logs,
            }

        _log(logs, f"Running {len(variants)} variants", log_callback)

        results: list[dict[str, Any]] = []

        for i, variant in enumerate(variants, 1):
            _log(
                logs, f"\n--- Running variant {i}/{len(variants)}: {variant.name} ---", log_callback
            )

            try:
                mapping_result, _ = _build_mapping_result_with_params(path, variant, log_callback)

                generator_result = run_generate(
                    str(path),
                    variant.spec,
                    output_dir=f"{output_dir}/{variant.name}" if output_dir else None,
                    log_callback=log_callback,
                )

                if not generator_result.ok:
                    result = ExperimentResult(
                        variant_name=variant.name,
                        spec=variant.spec,
                        status="failed",
                        error=generator_result.error,
                    )
                    results.append(asdict(result))
                    continue

                validate_result = run_validate(str(path), log_callback=log_callback)

                metrics = {}
                if validate_result.payload:
                    metrics = {
                        "speedup": validate_result.payload.get("scorecard", {})
                        .get("deltas", {})
                        .get("speedup", 0),
                        "loss_change": validate_result.payload.get("scorecard", {})
                        .get("deltas", {})
                        .get("loss_change_pct", 0),
                    }

                score = _calculate_variant_score(metrics, variant.confidence)

                result = ExperimentResult(
                    variant_name=variant.name,
                    spec=variant.spec,
                    status="completed",
                    validation_passed=validate_result.ok,
                    metrics=metrics,
                    score=score,
                )
                results.append(asdict(result))

                _log(
                    logs,
                    f"Variant {variant.name}: score={score:.2f}, passed={validate_result.ok}",
                    log_callback,
                )

            except Exception as e:
                result = ExperimentResult(
                    variant_name=variant.name,
                    spec=variant.spec,
                    status="error",
                    error=str(e),
                )
                results.append(asdict(result))
                _log(logs, f"Error running {variant.name}: {e}", log_callback)

        ranked_results = _rank_results(results)

        summary = _generate_experiment_summary(ranked_results, logs)

        return {
            "ok": True,
            "experiment_id": experiment_id,
            "repo_path": repo_path,
            "spec": spec_name,
            "variant_count": len(variants),
            "variants": [asdict(v) for v in variants],
            "results": ranked_results,
            "summary": summary,
            "logs": logs,
        }

    except Exception as exc:
        _log(logs, f"Experiment failed: {exc}", log_callback)
        return {
            "ok": False,
            "experiment_id": experiment_id,
            "error": str(exc),
            "logs": logs,
        }


def _build_mapping_result_with_params(path: Path, variant: ExperimentVariant, log_callback):
    from scholardevclaw.mapping.engine import MappingEngine
    from scholardevclaw.repo_intelligence.tree_sitter_analyzer import TreeSitterAnalyzer
    from scholardevclaw.research_intelligence.extractor import ResearchExtractor

    analyzer = TreeSitterAnalyzer(path)
    analysis = analyzer.analyze()

    extractor = ResearchExtractor()
    spec = extractor.get_spec(variant.spec)
    if spec is None:
        raise ValueError(f"Unknown spec: {variant.spec}")

    engine = MappingEngine(analysis.__dict__, spec)
    mapping = engine.map()

    mapping_result = {
        "targets": [
            {
                "file": t.file,
                "line": t.line,
                "current_code": t.current_code,
                "replacement_required": t.replacement_required,
                "context": t.context,
            }
            for t in mapping.targets
        ],
        "strategy": mapping.strategy,
        "confidence": mapping.confidence,
        "variant_params": variant.parameters,
    }

    return mapping_result, spec


def _calculate_variant_score(metrics: dict[str, float], confidence: float) -> float:
    speedup = metrics.get("speedup", 0)
    loss_change = abs(metrics.get("loss_change", 0))

    speedup_score = speedup * 40
    stability_score = max(0, 30 - loss_change * 3)
    confidence_score = confidence * 30

    return speedup_score + stability_score + confidence_score


def _rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid_results = [r for r in results if r.get("status") == "completed"]
    valid_results.sort(key=lambda x: x.get("score", 0), reverse=True)

    for i, result in enumerate(valid_results, 1):
        result["rank"] = i

    failed_results = [r for r in results if r.get("status") != "completed"]
    for result in failed_results:
        result["rank"] = len(valid_results) + 1

    return valid_results + failed_results


def _generate_experiment_summary(
    ranked_results: list[dict[str, Any]], logs: list[str]
) -> dict[str, Any]:
    completed = [r for r in ranked_results if r.get("status") == "completed"]
    failed = [r for r in ranked_results if r.get("status") != "completed"]

    best = completed[0] if completed else None

    avg_speedup = 0
    if completed:
        speedups = [r.get("metrics", {}).get("speedup", 0) for r in completed]
        avg_speedup = sum(speedups) / len(speedups)

    return {
        "total_variants": len(ranked_results),
        "completed": len(completed),
        "failed": len(failed),
        "best_variant": best.get("variant_name") if best else None,
        "best_score": best.get("score", 0) if best else None,
        "average_speedup": avg_speedup,
        "recommendation": f"Use {best['variant_name']}" if best else "No recommendation",
    }
