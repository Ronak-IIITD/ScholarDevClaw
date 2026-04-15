from __future__ import annotations

import ast
import asyncio

from scholardevclaw.generation.orchestrator import CodeOrchestrator
from scholardevclaw.planning.models import CodeModule, ImplementationPlan
from scholardevclaw.understanding.models import PaperUnderstanding


class _FakeUsage:
    def __init__(self, total_tokens: int) -> None:
        self.total_tokens = total_tokens


class _FakeContentBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResponse:
    def __init__(self, text: str, *, tokens: int) -> None:
        self.content = [_FakeContentBlock(text)]
        self.usage = _FakeUsage(tokens)


class _FakeMessages:
    def __init__(self, delay_seconds: float = 0.0) -> None:
        self.delay_seconds = delay_seconds

    async def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, str]],
    ) -> _FakeResponse:
        del model, system
        await asyncio.sleep(self.delay_seconds)

        prompt = messages[0]["content"]
        if max_tokens == 8192:
            module_name = self._extract_module_name(prompt)
            code = f"def build_{module_name}() -> str:\n    return {module_name!r}\n"
            return _FakeResponse(code, tokens=13)

        test_code = "def test_generated_module() -> None:\n    assert True\n"
        return _FakeResponse(test_code, tokens=7)

    @staticmethod
    def _extract_module_name(prompt: str) -> str:
        for line in prompt.splitlines():
            if line.startswith("Module: "):
                raw_name = line.removeprefix("Module: ").split("(", 1)[0].strip()
                return raw_name.casefold().replace(" ", "_").replace("-", "_")
        return "generated_module"


class _FakeAsyncClient:
    def __init__(self, delay_seconds: float = 0.0) -> None:
        self.messages = _FakeMessages(delay_seconds=delay_seconds)


def _build_plan(module_count: int, *, priority: int = 1) -> ImplementationPlan:
    modules = [
        CodeModule(
            id=f"module_{index}",
            name=f"Module {index}",
            description="Generated module",
            file_path=f"src/demo/module_{index}.py",
            depends_on=[],
            priority=priority,
            estimated_lines=20,
            test_file_path=f"tests/test_module_{index}.py",
            tech_stack="python",
        )
        for index in range(module_count)
    ]

    return ImplementationPlan(
        project_name="demo_project",
        target_language="python",
        tech_stack="python",
        modules=modules,
    )


def _build_understanding() -> PaperUnderstanding:
    return PaperUnderstanding(
        paper_title="Demo Paper",
        one_line_summary="A deterministic generation test",
        core_algorithm_description="Generate modules and tests from a plan.",
    )


def test_orchestrator_writes_expected_files_and_report_fields(tmp_path):
    plan = _build_plan(module_count=3)
    understanding = _build_understanding()
    orchestrator = CodeOrchestrator(api_key="fake-key", client=_FakeAsyncClient())

    result = orchestrator.generate_sync(
        plan=plan,
        understanding=understanding,
        output_dir=tmp_path,
        max_parallel=2,
    )

    for module in plan.modules:
        module_file = tmp_path / module.file_path
        test_file = tmp_path / module.test_file_path
        assert module_file.exists()
        assert test_file.exists()

        ast.parse(module_file.read_text(encoding="utf-8"))
        ast.parse(test_file.read_text(encoding="utf-8"))

    report = result.to_dict()
    assert report["success_rate"] == 1.0
    assert report["duration_seconds"] >= 0.0
    assert len(report["module_results"]) == len(plan.modules)

    for module_report in report["module_results"]:
        assert module_report["generation_attempts"] == 1
        assert module_report["final_errors"] == []


def test_orchestrator_parallel_generation_is_faster_for_large_plans(tmp_path):
    plan = _build_plan(module_count=6)
    understanding = _build_understanding()

    serial_orchestrator = CodeOrchestrator(
        api_key="fake-key",
        client=_FakeAsyncClient(delay_seconds=0.04),
    )
    parallel_orchestrator = CodeOrchestrator(
        api_key="fake-key",
        client=_FakeAsyncClient(delay_seconds=0.04),
    )

    serial_result = serial_orchestrator.generate_sync(
        plan=plan,
        understanding=understanding,
        output_dir=tmp_path / "serial",
        max_parallel=1,
    )
    parallel_result = parallel_orchestrator.generate_sync(
        plan=plan,
        understanding=understanding,
        output_dir=tmp_path / "parallel",
        max_parallel=4,
    )

    assert serial_result.success_rate == 1.0
    assert parallel_result.success_rate == 1.0
    assert parallel_result.duration_seconds < serial_result.duration_seconds * 0.7
    assert serial_result.duration_seconds - parallel_result.duration_seconds > 0.12
