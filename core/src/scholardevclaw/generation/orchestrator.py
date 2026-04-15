from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from itertools import groupby
from pathlib import Path
from typing import Any

from scholardevclaw.generation.models import GenerationResult, ModuleResult
from scholardevclaw.generation.module_agent import ModuleAgent
from scholardevclaw.planning.models import CodeModule, ImplementationPlan
from scholardevclaw.understanding.models import PaperUnderstanding

try:
    import anthropic  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised only without optional dependency
    anthropic = None  # type: ignore[assignment]


def _resolve_confined_destination(base_dir: Path, relative_path: str) -> Path | None:
    destination = (base_dir / relative_path).resolve()
    try:
        destination.relative_to(base_dir)
    except ValueError:
        return None
    return destination


class CodeOrchestrator:
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5",
        *,
        client: Any | None = None,
    ) -> None:
        self.model = model
        if client is not None:
            self.client = client
            return

        if anthropic is None:
            raise ImportError(
                "anthropic SDK is required for CodeOrchestrator. "
                "Install with: pip install -e '.[understanding,execution]'"
            )
        if not api_key.strip():
            raise ValueError("api_key must be non-empty")

        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def generate_all(
        self,
        plan: ImplementationPlan,
        understanding: PaperUnderstanding,
        output_dir: Path,
        max_parallel: int = 4,
        *,
        module_filter: Iterable[str] | None = None,
        error_context: str | None = None,
    ) -> GenerationResult:
        if max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        self._validate_plan_integrity(plan)

        start = time.perf_counter()
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        allowed_modules = (
            {module_id for module_id in module_filter} if module_filter is not None else None
        )

        sorted_modules = sorted(plan.modules, key=lambda module: module.priority)
        if allowed_modules is not None:
            sorted_modules = [module for module in sorted_modules if module.id in allowed_modules]

        results: dict[str, ModuleResult] = {}
        context_modules = self._load_existing_context(plan, output_dir)

        for _priority, group in groupby(sorted_modules, key=lambda module: module.priority):
            batch = list(group)
            batch_results = await self._generate_priority_batch(
                modules=batch,
                plan=plan,
                understanding=understanding,
                context_modules=context_modules,
                max_parallel=max_parallel,
                error_context=error_context,
            )

            for result in batch_results:
                self._write_module(result, output_dir)
                results[result.module_id] = result
                if not result.final_errors:
                    context_modules[result.module_id] = result.code

        ordered_results = [results[module.id] for module in sorted_modules if module.id in results]
        success_count = sum(1 for result in ordered_results if not result.final_errors)
        success_rate = success_count / len(ordered_results) if ordered_results else 0.0

        return GenerationResult(
            plan=plan,
            module_results=ordered_results,
            output_dir=output_dir,
            success_rate=success_rate,
            total_tokens_used=sum(result.tokens_used for result in ordered_results),
            duration_seconds=time.perf_counter() - start,
        )

    async def _generate_priority_batch(
        self,
        modules: list[CodeModule],
        plan: ImplementationPlan,
        understanding: PaperUnderstanding,
        context_modules: dict[str, str],
        max_parallel: int,
        error_context: str | None,
    ) -> list[ModuleResult]:
        semaphore = asyncio.Semaphore(max_parallel)
        agent = ModuleAgent(self.client, self.model)

        async def _bounded_generate(module: CodeModule) -> ModuleResult:
            async with semaphore:
                return await agent.generate(
                    module=module,
                    plan=plan,
                    understanding=understanding,
                    context_modules=context_modules,
                    error_context=error_context,
                )

        return await asyncio.gather(*(_bounded_generate(module) for module in modules))

    def _load_existing_context(
        self,
        plan: ImplementationPlan,
        output_dir: Path,
    ) -> dict[str, str]:
        context: dict[str, str] = {}
        for module in plan.modules:
            target = _resolve_confined_destination(output_dir, module.file_path)
            if target is None or not target.exists() or not target.is_file():
                continue
            try:
                context[module.id] = target.read_text(encoding="utf-8")
            except OSError:
                continue
        return context

    def _write_module(self, result: ModuleResult, output_dir: Path) -> None:
        target = _resolve_confined_destination(output_dir, result.file_path)
        test_target = _resolve_confined_destination(output_dir, result.test_file_path)

        if target is None:
            result.final_errors.append(f"Unsafe module output path: {result.file_path}")
            return
        if test_target is None:
            result.final_errors.append(f"Unsafe test output path: {result.test_file_path}")
            return

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(result.code, encoding="utf-8")
        except OSError as exc:
            result.final_errors.append(
                f"Failed to write generated module file '{result.file_path}': {exc}"
            )

        try:
            test_target.parent.mkdir(parents=True, exist_ok=True)
            test_target.write_text(result.test_code, encoding="utf-8")
        except OSError as exc:
            result.final_errors.append(
                f"Failed to write generated test file '{result.test_file_path}': {exc}"
            )

    def _validate_plan_integrity(self, plan: ImplementationPlan) -> None:
        errors: list[str] = []
        module_ids: list[str] = []

        for index, module in enumerate(plan.modules):
            module_id = module.id.strip()
            if not module_id:
                errors.append(f"Module at index {index} has empty id")
            module_ids.append(module_id)

            if not module.file_path.strip():
                label = module_id or f"index {index}"
                errors.append(f"Module '{label}' has empty file_path")

            if not module.test_file_path.strip():
                label = module_id or f"index {index}"
                errors.append(f"Module '{label}' has empty test_file_path")

        seen: set[str] = set()
        duplicate_ids: set[str] = set()
        for module_id in module_ids:
            if not module_id:
                continue
            if module_id in seen:
                duplicate_ids.add(module_id)
            seen.add(module_id)

        if duplicate_ids:
            duplicates_text = ", ".join(sorted(duplicate_ids))
            errors.append(f"Duplicate module ids found: {duplicates_text}")

        known_ids = {module_id for module_id in module_ids if module_id}
        for index, module in enumerate(plan.modules):
            label = module.id.strip() or f"index {index}"
            unknown_dependencies = sorted(
                dependency_id
                for dependency in module.depends_on
                if (dependency_id := dependency.strip()) and dependency_id not in known_ids
            )
            if unknown_dependencies:
                dependency_text = ", ".join(unknown_dependencies)
                errors.append(f"Module '{label}' has unknown dependencies: {dependency_text}")

        if errors:
            error_text = "; ".join(errors)
            raise ValueError(f"Invalid implementation plan for generation: {error_text}")

    def generate_sync(
        self,
        plan: ImplementationPlan,
        understanding: PaperUnderstanding,
        output_dir: Path,
        max_parallel: int = 4,
        *,
        module_filter: Iterable[str] | None = None,
        error_context: str | None = None,
    ) -> GenerationResult:
        return asyncio.run(
            self.generate_all(
                plan=plan,
                understanding=understanding,
                output_dir=output_dir,
                max_parallel=max_parallel,
                module_filter=module_filter,
                error_context=error_context,
            )
        )
