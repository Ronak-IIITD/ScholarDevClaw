from __future__ import annotations

import re
from typing import Any

from scholardevclaw.execution.sandbox import ExecutionReport, SandboxRunner
from scholardevclaw.generation.models import GenerationResult, ModuleResult
from scholardevclaw.generation.orchestrator import CodeOrchestrator
from scholardevclaw.planning.models import ImplementationPlan
from scholardevclaw.understanding.models import PaperUnderstanding


class SelfHealingLoop:
    """Run sandbox tests and regenerate only failing modules."""

    def __init__(
        self,
        orchestrator: CodeOrchestrator,
        runner: SandboxRunner,
        max_healing_rounds: int = 3,
    ) -> None:
        self.orchestrator = orchestrator
        self.runner = runner
        self.max_rounds = max_healing_rounds
        self.round_reports: list[dict[str, Any]] = []

    def heal(
        self,
        generation_result: GenerationResult,
        plan: ImplementationPlan,
        understanding: PaperUnderstanding,
    ) -> GenerationResult:
        current = generation_result
        self.round_reports = []

        for round_num in range(1, self.max_rounds + 1):
            report = self.runner.run_tests(current.output_dir)
            failing_module_ids = self._identify_failing_modules(report, plan)

            self.round_reports.append(
                {
                    "round": round_num,
                    "tests_passed": report.tests_passed,
                    "tests_failed": report.tests_failed,
                    "tests_errors": report.tests_errors,
                    "success": report.success,
                    "failing_modules": list(failing_module_ids),
                }
            )

            if report.success:
                break
            if not failing_module_ids:
                break

            partial_result = self.orchestrator.generate_sync(
                plan=plan,
                understanding=understanding,
                output_dir=current.output_dir,
                module_filter=failing_module_ids,
                error_context=f"{report.stderr}\n{report.stdout}"[:6000],
            )
            current = self._merge_generation_results(current, partial_result, plan)

        return current

    def _identify_failing_modules(
        self,
        report: ExecutionReport,
        plan: ImplementationPlan,
    ) -> list[str]:
        combined = f"{report.stderr}\n{report.stdout}"
        candidates: set[str] = set()

        for pattern in (
            r"(?:FAILED|ERROR)\s+(tests/[a-zA-Z0-9_./-]+)\.py",
            r"(tests/[a-zA-Z0-9_./-]+)\.py::",
            r"(src/[a-zA-Z0-9_./-]+)\.py",
            r"(?:FAILED|ERROR)\s+tests/test_([a-zA-Z0-9_]+)\.py",
            r"tests/test_([a-zA-Z0-9_]+)\.py::",
        ):
            for matched in re.findall(pattern, combined):
                if isinstance(matched, str) and matched:
                    candidates.add(matched.strip())

        known_module_ids = {module.id for module in plan.modules}
        module_by_test_path: dict[str, str] = {}
        module_by_test_stem: dict[str, str] = {}
        module_by_file_stem: dict[str, str] = {}
        module_by_file_path: dict[str, str] = {}

        for module in plan.modules:
            test_path_without_ext = module.test_file_path.removesuffix(".py").replace("\\", "/")
            module_by_test_path[test_path_without_ext] = module.id

            test_stem = module.test_file_path.rsplit("/", 1)[-1].removesuffix(".py")
            if test_stem.startswith("test_"):
                module_by_test_stem[test_stem.removeprefix("test_")] = module.id

            file_stem = module.file_path.rsplit("/", 1)[-1].removesuffix(".py")
            module_by_file_stem[file_stem] = module.id
            module_by_file_path[module.file_path.removesuffix(".py").replace("\\", "/")] = module.id

        resolved: set[str] = set()
        for candidate in candidates:
            if candidate in known_module_ids:
                resolved.add(candidate)
                continue

            normalized = candidate.replace("\\", "/").strip("/")
            normalized = normalized.removeprefix("workspace/")
            normalized = normalized.removeprefix("/workspace/")
            normalized = normalized.removeprefix("./")
            if normalized.endswith(".py"):
                normalized = normalized.removesuffix(".py")

            if normalized in module_by_test_path:
                resolved.add(module_by_test_path[normalized])
                continue

            test_stem = normalized.rsplit("/", 1)[-1]
            if test_stem.startswith("test_"):
                test_stem = test_stem.removeprefix("test_")
            if test_stem in module_by_test_stem:
                resolved.add(module_by_test_stem[test_stem])
                continue

            file_stem = normalized.rsplit("/", 1)[-1]
            if file_stem in module_by_file_stem:
                resolved.add(module_by_file_stem[file_stem])
                continue

            if normalized in module_by_file_path:
                resolved.add(module_by_file_path[normalized])
                continue

            for path_without_ext, module_id in module_by_file_path.items():
                if normalized.endswith(path_without_ext):
                    resolved.add(module_id)
                    break

        return sorted(resolved)

    def _merge_generation_results(
        self,
        current: GenerationResult,
        partial: GenerationResult,
        plan: ImplementationPlan,
    ) -> GenerationResult:
        merged: dict[str, ModuleResult] = {
            result.module_id: result for result in current.module_results
        }
        for result in partial.module_results:
            merged[result.module_id] = result

        ordered_results = [merged[module.id] for module in plan.modules if module.id in merged]
        success_count = sum(1 for result in ordered_results if not result.final_errors)
        success_rate = success_count / len(ordered_results) if ordered_results else 0.0

        return GenerationResult(
            plan=plan,
            module_results=ordered_results,
            output_dir=current.output_dir,
            success_rate=success_rate,
            total_tokens_used=(current.total_tokens_used + partial.total_tokens_used),
            duration_seconds=current.duration_seconds + partial.duration_seconds,
        )
