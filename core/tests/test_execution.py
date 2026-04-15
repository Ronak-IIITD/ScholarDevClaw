from __future__ import annotations

import io
import json
import tarfile
from types import SimpleNamespace
from typing import Any, cast

import pytest

import scholardevclaw.execution.sandbox as sandbox_module
from scholardevclaw.execution.healer import SelfHealingLoop
from scholardevclaw.execution.sandbox import ExecutionReport, SandboxRunner
from scholardevclaw.execution.scorer import ReproducibilityScorer
from scholardevclaw.generation.models import GenerationResult, ModuleResult
from scholardevclaw.planning.models import CodeModule, ImplementationPlan
from scholardevclaw.understanding.models import PaperUnderstanding


def _build_report_tar(payload: dict[str, object]) -> bytes:
    report_bytes = json.dumps(payload).encode("utf-8")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        member = tarfile.TarInfo(name="report.json")
        member.size = len(report_bytes)
        archive.addfile(member, io.BytesIO(report_bytes))
    return buffer.getvalue()


def _build_plan() -> ImplementationPlan:
    return ImplementationPlan(
        project_name="demo",
        target_language="python",
        tech_stack="python",
        modules=[
            CodeModule(
                id="module_a",
                name="Module A",
                description="",
                file_path="src/module_a.py",
                depends_on=[],
                priority=1,
                estimated_lines=10,
                test_file_path="tests/test_module_a.py",
                tech_stack="python",
            ),
            CodeModule(
                id="module_b",
                name="Module B",
                description="",
                file_path="src/module_b.py",
                depends_on=[],
                priority=1,
                estimated_lines=10,
                test_file_path="tests/test_module_b.py",
                tech_stack="python",
            ),
        ],
    )


def test_sandbox_runner_success_parses_json_and_enforces_limits(monkeypatch, tmp_path):
    observed: dict[str, object] = {}
    report_tar = _build_report_tar({"summary": {"passed": 7, "failed": 0, "errors": 0}})

    class FakeContainer:
        def __init__(self) -> None:
            self.removed_force = False

        def wait(self, timeout: int) -> dict[str, int]:
            observed["wait_timeout"] = timeout
            return {"StatusCode": 0}

        def logs(self, *, stdout: bool, stderr: bool) -> bytes:
            if stdout and not stderr:
                return b"pytest finished"
            if stderr and not stdout:
                return b""
            return b""

        def get_archive(self, path: str):
            observed["archive_path"] = path
            return [report_tar], {}

        def stats(self, *, stream: bool) -> dict[str, dict[str, int]]:
            observed["stats_stream"] = stream
            return {"memory_stats": {"max_usage": 128 * 1024 * 1024}}

        def remove(self, *, force: bool) -> None:
            self.removed_force = force

    fake_container = FakeContainer()

    class FakeContainers:
        def run(self, **kwargs):
            observed["run_kwargs"] = kwargs
            return fake_container

    fake_client = SimpleNamespace(containers=FakeContainers())
    monkeypatch.setattr(sandbox_module, "docker", SimpleNamespace(from_env=lambda: fake_client))

    runner = SandboxRunner(timeout_seconds=55, memory_limit_mb=512)
    result = runner.run_tests(tmp_path)

    assert result.tests_passed == 7
    assert result.tests_failed == 0
    assert result.tests_errors == 0
    assert result.success is True

    run_kwargs = observed["run_kwargs"]
    assert isinstance(run_kwargs, dict)
    assert run_kwargs["network_disabled"] is True
    assert run_kwargs["mem_limit"] == "512m"
    assert fake_container.removed_force is True


def test_sandbox_runner_parses_mixed_pass_fail_counts(monkeypatch, tmp_path):
    report_tar = _build_report_tar({"summary": {"passed": 3, "failed": 1, "errors": 0}})

    class FakeContainer:
        def wait(self, timeout: int) -> dict[str, int]:
            del timeout
            return {"StatusCode": 1}

        def logs(self, *, stdout: bool, stderr: bool) -> bytes:
            if stdout and not stderr:
                return b"1 failed, 3 passed"
            if stderr and not stdout:
                return b""
            return b""

        def get_archive(self, path: str):
            del path
            return [report_tar], {}

        def stats(self, *, stream: bool) -> dict[str, dict[str, int]]:
            del stream
            return {"memory_stats": {"max_usage": 64 * 1024 * 1024}}

        def remove(self, *, force: bool) -> None:
            del force

    class FakeContainers:
        def run(self, **kwargs):
            del kwargs
            return FakeContainer()

    fake_client = SimpleNamespace(containers=FakeContainers())
    monkeypatch.setattr(sandbox_module, "docker", SimpleNamespace(from_env=lambda: fake_client))

    runner = SandboxRunner(timeout_seconds=30, memory_limit_mb=256)
    result = runner.run_tests(tmp_path)

    assert result.tests_passed == 3
    assert result.tests_failed == 1
    assert result.tests_errors == 0
    assert result.success is False


@pytest.mark.parametrize(
    ("docker_stub", "stderr_snippet"),
    [
        (None, "docker SDK is not installed"),
        (
            SimpleNamespace(
                from_env=lambda: (_ for _ in ()).throw(RuntimeError("daemon unavailable"))
            ),
            "Failed to initialize Docker client",
        ),
    ],
)
def test_sandbox_runner_unavailable_docker_returns_actionable_failure(
    monkeypatch,
    tmp_path,
    docker_stub,
    stderr_snippet,
):
    monkeypatch.setattr(sandbox_module, "docker", docker_stub)

    runner = SandboxRunner()
    result = runner.run_tests(tmp_path)

    assert result.success is False
    assert result.exit_code == 1
    assert stderr_snippet in result.stderr


def test_reproducibility_scorer_regex_extraction_and_scoring():
    understanding = PaperUnderstanding(
        paper_title="Deterministic Metrics",
        evaluation_protocol="Accuracy: 92%\nF1 score = 0.88",
    )
    execution_report = ExecutionReport(
        exit_code=0,
        stdout="accuracy: 90%",
        stderr="F1 score is 0.84",
        duration_seconds=1.0,
        peak_memory_mb=64.0,
        tests_passed=2,
        tests_failed=0,
        tests_errors=0,
        success=True,
    )

    scorer = ReproducibilityScorer(api_key=None)
    report = scorer.score(understanding, execution_report)

    assert report.claimed_metrics["accuracy"] == pytest.approx(0.92)
    assert report.claimed_metrics["f1"] == pytest.approx(0.88)
    assert report.achieved_metrics["accuracy"] == pytest.approx(0.90)
    assert report.achieved_metrics["f1"] == pytest.approx(0.84)

    expected_score = (min(0.90 / 0.92, 0.92 / 0.90) + min(0.84 / 0.88, 0.88 / 0.84)) / 2
    assert report.score == pytest.approx(expected_score)
    assert report.verdict == "reproduced"


def test_self_healing_loop_regenerates_only_failing_module_and_tracks_round_reports(tmp_path):
    plan = _build_plan()
    understanding = PaperUnderstanding(paper_title="Healing Demo")
    initial = GenerationResult(
        plan=plan,
        module_results=[
            ModuleResult(
                module_id="module_a",
                file_path="src/module_a.py",
                test_file_path="tests/test_module_a.py",
                code="def broken():\n    return 0\n",
                test_code="def test_broken():\n    assert False\n",
                generation_attempts=1,
                final_errors=["initial failure"],
                tokens_used=12,
            ),
            ModuleResult(
                module_id="module_b",
                file_path="src/module_b.py",
                test_file_path="tests/test_module_b.py",
                code="def stable():\n    return 1\n",
                test_code="def test_stable():\n    assert True\n",
                generation_attempts=1,
                final_errors=[],
                tokens_used=11,
            ),
        ],
        output_dir=tmp_path,
        success_rate=0.5,
        total_tokens_used=23,
        duration_seconds=0.2,
    )

    reports = [
        ExecutionReport(
            exit_code=1,
            stdout="",
            stderr="FAILED tests/test_module_a.py::test_broken - AssertionError",
            duration_seconds=0.1,
            peak_memory_mb=32.0,
            tests_passed=5,
            tests_failed=1,
            tests_errors=0,
            success=False,
        ),
        ExecutionReport(
            exit_code=0,
            stdout="all green",
            stderr="",
            duration_seconds=0.1,
            peak_memory_mb=32.0,
            tests_passed=6,
            tests_failed=0,
            tests_errors=0,
            success=True,
        ),
    ]

    class FakeRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run_tests(self, _project_dir):
            report = reports[min(self.calls, len(reports) - 1)]
            self.calls += 1
            return report

    class FakeOrchestrator:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate_sync(self, **kwargs):
            self.calls.append(dict(kwargs))
            return GenerationResult(
                plan=kwargs["plan"],
                module_results=[
                    ModuleResult(
                        module_id="module_a",
                        file_path="src/module_a.py",
                        test_file_path="tests/test_module_a.py",
                        code="def healed():\n    return 1\n",
                        test_code="def test_healed():\n    assert True\n",
                        generation_attempts=2,
                        final_errors=[],
                        tokens_used=8,
                    )
                ],
                output_dir=kwargs["output_dir"],
                success_rate=1.0,
                total_tokens_used=8,
                duration_seconds=0.1,
            )

    orchestrator = FakeOrchestrator()
    runner = FakeRunner()
    healer = SelfHealingLoop(
        orchestrator=cast(Any, orchestrator),
        runner=cast(Any, runner),
        max_healing_rounds=3,
    )

    healed = healer.heal(initial, plan, understanding)

    assert len(orchestrator.calls) == 1
    assert orchestrator.calls[0]["module_filter"] == ["module_a"]

    assert healed.success_rate > initial.success_rate
    assert {result.module_id for result in healed.module_results} == {"module_a", "module_b"}
    assert all(not result.final_errors for result in healed.module_results)

    assert len(healer.round_reports) == 2
    assert healer.round_reports[0]["failing_modules"] == ["module_a"]
    assert healer.round_reports[1]["success"] is True


def test_self_healing_loop_resolves_nested_test_path_failure(tmp_path):
    plan = ImplementationPlan(
        project_name="nested",
        target_language="python",
        tech_stack="python",
        modules=[
            CodeModule(
                id="attention",
                name="Attention",
                description="",
                file_path="src/model/attention.py",
                depends_on=[],
                priority=1,
                estimated_lines=10,
                test_file_path="tests/model/test_attention.py",
                tech_stack="python",
            )
        ],
    )
    understanding = PaperUnderstanding(paper_title="Nested Paths")
    initial = GenerationResult(
        plan=plan,
        module_results=[
            ModuleResult(
                module_id="attention",
                file_path="src/model/attention.py",
                test_file_path="tests/model/test_attention.py",
                code="def attention():\n    return 0\n",
                test_code="def test_attention():\n    assert False\n",
                generation_attempts=1,
                final_errors=["initial"],
                tokens_used=2,
            )
        ],
        output_dir=tmp_path,
        success_rate=0.0,
        total_tokens_used=2,
        duration_seconds=0.1,
    )

    reports = [
        ExecutionReport(
            exit_code=1,
            stdout="",
            stderr="FAILED tests/model/test_attention.py::test_attention - AssertionError",
            duration_seconds=0.1,
            peak_memory_mb=16.0,
            tests_passed=0,
            tests_failed=1,
            tests_errors=0,
            success=False,
        ),
        ExecutionReport(
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=0.1,
            peak_memory_mb=16.0,
            tests_passed=1,
            tests_failed=0,
            tests_errors=0,
            success=True,
        ),
    ]

    class FakeRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run_tests(self, _project_dir):
            report = reports[min(self.calls, len(reports) - 1)]
            self.calls += 1
            return report

    class FakeOrchestrator:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate_sync(self, **kwargs):
            self.calls.append(dict(kwargs))
            return GenerationResult(
                plan=kwargs["plan"],
                module_results=[
                    ModuleResult(
                        module_id="attention",
                        file_path="src/model/attention.py",
                        test_file_path="tests/model/test_attention.py",
                        code="def attention() -> int:\n    return 1\n",
                        test_code="def test_attention() -> None:\n    assert True\n",
                        generation_attempts=2,
                        final_errors=[],
                        tokens_used=3,
                    )
                ],
                output_dir=kwargs["output_dir"],
                success_rate=1.0,
                total_tokens_used=3,
                duration_seconds=0.1,
            )

    orchestrator = FakeOrchestrator()
    runner = FakeRunner()
    healer = SelfHealingLoop(
        orchestrator=cast(Any, orchestrator),
        runner=cast(Any, runner),
        max_healing_rounds=3,
    )

    healed = healer.heal(initial, plan, understanding)

    assert len(orchestrator.calls) == 1
    assert orchestrator.calls[0]["module_filter"] == ["attention"]
    assert healed.success_rate == 1.0
