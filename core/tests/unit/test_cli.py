from __future__ import annotations

import builtins
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import scholardevclaw.cli as cli


@pytest.mark.parametrize(
    ("command", "argv", "handler_name"),
    [
        ("analyze", ["scholardevclaw", "analyze", "/tmp/repo"], "cmd_analyze"),
        ("search", ["scholardevclaw", "search", "rmsnorm"], "cmd_search"),
        ("kb", ["scholardevclaw", "kb", "stats"], "cmd_kb"),
        ("ingest", ["scholardevclaw", "ingest", "arxiv:1706.03762"], "cmd_ingest"),
        (
            "understand",
            ["scholardevclaw", "understand", "/tmp/paper_document.json"],
            "cmd_understand",
        ),
        ("plan", ["scholardevclaw", "plan", "/tmp/understanding.json"], "cmd_plan"),
        ("suggest", ["scholardevclaw", "suggest", "/tmp/repo"], "cmd_suggest"),
        ("integrate", ["scholardevclaw", "integrate", "/tmp/repo"], "cmd_integrate"),
        ("map", ["scholardevclaw", "map", "/tmp/repo", "rmsnorm"], "cmd_map"),
        ("generate", ["scholardevclaw", "generate", "/tmp/repo", "rmsnorm"], "cmd_generate"),
        ("execute", ["scholardevclaw", "execute", "/tmp/repo"], "cmd_execute"),
        (
            "scaffold",
            [
                "scholardevclaw",
                "scaffold",
                "/tmp/repo",
                "/tmp/implementation_plan.json",
                "/tmp/understanding.json",
                "/tmp/reproducibility_report.json",
            ],
            "cmd_scaffold",
        ),
        (
            "from-paper",
            ["scholardevclaw", "from-paper", "arxiv:1706.03762"],
            "cmd_from_paper",
        ),
        ("validate", ["scholardevclaw", "validate", "/tmp/repo"], "cmd_validate"),
        ("specs", ["scholardevclaw", "specs"], "cmd_specs"),
        ("planner", ["scholardevclaw", "planner", "/tmp/repo"], "cmd_planner"),
        ("critic", ["scholardevclaw", "critic", "/tmp/repo"], "cmd_critic"),
        ("context", ["scholardevclaw", "context", "list"], "cmd_context"),
        ("experiment", ["scholardevclaw", "experiment", "/tmp/repo", "rmsnorm"], "cmd_experiment"),
        ("plugin", ["scholardevclaw", "plugin", "list"], "cmd_plugin"),
        ("rollback", ["scholardevclaw", "rollback", "list"], "cmd_rollback"),
        ("github-app", ["scholardevclaw", "github-app", "status"], "cmd_github_app"),
        ("security", ["scholardevclaw", "security", "/tmp/repo"], "cmd_security"),
        ("agent", ["scholardevclaw", "agent"], "cmd_agent"),
        ("auth", ["scholardevclaw", "auth", "status"], "cmd_auth"),
        ("demo", ["scholardevclaw", "demo"], "cmd_demo"),
        ("multi-repo", ["scholardevclaw", "multi-repo", "list"], "cmd_multi_repo"),
        ("workspace", ["scholardevclaw", "workspace", "list"], "cmd_workspace"),
        ("deploy-check", ["scholardevclaw", "deploy-check"], "cmd_deploy_check"),
        ("tui", ["scholardevclaw", "tui"], "cmd_tui"),
    ],
)
def test_main_dispatches_all_supported_commands(monkeypatch, command, argv, handler_name):
    called = {}

    def fake_cmd(args):
        called["command"] = args.command
        called["args"] = args

    monkeypatch.setattr(cli, handler_name, fake_cmd)
    monkeypatch.setattr(cli.sys, "argv", argv)

    cli.main()

    assert called["command"] == command


def test_main_no_command_exits_with_help(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "argv", ["scholardevclaw"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower()


def test_main_dispatches_to_selected_handler(monkeypatch):
    called = {}

    def fake_cmd(args):
        called["repo_path"] = args.repo_path

    monkeypatch.setattr(cli, "cmd_analyze", fake_cmd)
    monkeypatch.setattr(cli.sys, "argv", ["scholardevclaw", "analyze", "/tmp/repo"])

    cli.main()

    assert called["repo_path"] == "/tmp/repo"


def test_main_integrate_without_spec_sets_spec_to_none(monkeypatch):
    called = {}

    def fake_cmd(args):
        called["spec"] = args.spec
        called["dry_run"] = args.dry_run
        called["require_clean"] = args.require_clean

    monkeypatch.setattr(cli, "cmd_integrate", fake_cmd)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["scholardevclaw", "integrate", "/tmp/repo", "--dry-run", "--require-clean"],
    )

    cli.main()

    assert called["spec"] is None
    assert called["dry_run"] is True
    assert called["require_clean"] is True


def test_cmd_validate_exits_when_failed_and_payload_empty(monkeypatch, capsys):
    import scholardevclaw.application.pipeline as pipeline

    monkeypatch.setattr(
        pipeline,
        "run_validate",
        lambda *_a, **_k: SimpleNamespace(ok=False, payload={}, error="boom"),
    )

    args = SimpleNamespace(repo_path="/tmp/repo", output_json=False)
    with pytest.raises(SystemExit) as exc:
        cli.cmd_validate(args)

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Error: boom" in captured.err


def test_cmd_integrate_failure_prints_guidance_and_exits(monkeypatch, capsys):
    import scholardevclaw.application.pipeline as pipeline

    monkeypatch.setattr(
        pipeline,
        "run_integrate",
        lambda *_a, **_k: SimpleNamespace(
            ok=False,
            error="preflight failed",
            payload={
                "guidance": ["Initialize git repo", "Commit changes first"],
                "_meta": {"payload_type": "integration", "schema_version": "1.0.0"},
            },
        ),
    )

    args = SimpleNamespace(
        repo_path="/tmp/repo",
        spec="rmsnorm",
        dry_run=False,
        require_clean=True,
        output_dir=None,
        output_json=False,
    )

    with pytest.raises(SystemExit) as exc:
        cli.cmd_integrate(args)

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Integration failed." in captured.err
    assert "Guidance:" in captured.err
    assert "Initialize git repo" in captured.err


def test_cmd_integrate_success_json_output(monkeypatch, capsys):
    import scholardevclaw.application.pipeline as pipeline

    payload = {
        "dry_run": True,
        "spec": "rmsnorm",
        "preflight": {"is_clean": True},
        "validation": None,
        "_meta": {"payload_type": "integration", "schema_version": "1.0.0"},
    }
    monkeypatch.setattr(
        pipeline,
        "run_integrate",
        lambda *_a, **_k: SimpleNamespace(ok=True, error=None, payload=payload),
    )

    args = SimpleNamespace(
        repo_path="/tmp/repo",
        spec="rmsnorm",
        dry_run=True,
        require_clean=False,
        output_dir=None,
        output_json=True,
    )

    cli.cmd_integrate(args)
    captured = capsys.readouterr()
    assert "Dry-run complete" in captured.out
    assert json.dumps(payload, indent=2) in captured.out


def test_cmd_tui_importerror_prints_install_hint(monkeypatch, capsys):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "scholardevclaw.tui":
            raise ImportError("textual missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SystemExit) as exc:
        cli.cmd_tui(SimpleNamespace())

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "TUI dependencies are not installed" in captured.err
    assert 'pip install -e ".[tui]"' in captured.err


def test_workspace_parser_list(monkeypatch):
    called = {}

    def fake_cmd(args):
        called["action"] = args.workspace_action
        called["repo"] = args.repo_id_or_path
        called["output_json"] = args.output_json

    monkeypatch.setattr(cli, "cmd_workspace", fake_cmd)
    monkeypatch.setattr(cli.sys, "argv", ["scholardevclaw", "workspace", "list"])

    cli.main()

    assert called == {"action": "list", "repo": None, "output_json": False}


def test_workspace_parser_add_with_name(monkeypatch):
    called = {}

    def fake_cmd(args):
        called["action"] = args.workspace_action
        called["repo"] = args.repo_id_or_path
        called["name"] = args.name

    monkeypatch.setattr(cli, "cmd_workspace", fake_cmd)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["scholardevclaw", "workspace", "add", "/tmp/repo", "--name", "demo"],
    )

    cli.main()

    assert called == {"action": "add", "repo": "/tmp/repo", "name": "demo"}


def test_workspace_parser_analyze_all(monkeypatch):
    called = {}

    def fake_cmd(args):
        called["action"] = args.workspace_action
        called["all"] = args.all
        called["repo"] = args.repo_id_or_path

    monkeypatch.setattr(cli, "cmd_workspace", fake_cmd)
    monkeypatch.setattr(cli.sys, "argv", ["scholardevclaw", "workspace", "analyze", "--all"])

    cli.main()

    assert called == {"action": "analyze", "all": True, "repo": None}


def test_workspace_parser_analyze_single_repo(monkeypatch):
    called = {}

    def fake_cmd(args):
        called["action"] = args.workspace_action
        called["all"] = args.all
        called["repo"] = args.repo_id_or_path

    monkeypatch.setattr(cli, "cmd_workspace", fake_cmd)
    monkeypatch.setattr(cli.sys, "argv", ["scholardevclaw", "workspace", "analyze", "my-repo"])

    cli.main()

    assert called == {"action": "analyze", "all": False, "repo": "my-repo"}


def test_ingest_parser_with_output_dir(monkeypatch):
    called = {}

    def fake_cmd(args):
        called["source"] = args.source
        called["output_dir"] = args.output_dir
        called["no_cache"] = args.no_cache
        called["verbose"] = args.verbose

    monkeypatch.setattr(cli, "cmd_ingest", fake_cmd)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "scholardevclaw",
            "ingest",
            "10.1000/xyz123",
            "--output-dir",
            "/tmp/out",
            "--no-cache",
            "--verbose",
        ],
    )

    cli.main()

    assert called == {
        "source": "10.1000/xyz123",
        "output_dir": "/tmp/out",
        "no_cache": True,
        "verbose": True,
    }


def test_kb_parser_stats(monkeypatch):
    called = {}

    def fake_cmd(args):
        called["kb_action"] = args.kb_action

    monkeypatch.setattr(cli, "cmd_kb", fake_cmd)
    monkeypatch.setattr(cli.sys, "argv", ["scholardevclaw", "kb", "stats"])

    cli.main()

    assert called == {"kb_action": "stats"}


def test_kb_parser_search_with_limit_and_domain(monkeypatch):
    called = {}

    def fake_cmd(args):
        called["kb_action"] = args.kb_action
        called["query"] = args.query
        called["limit"] = args.limit
        called["domain"] = args.domain

    monkeypatch.setattr(cli, "cmd_kb", fake_cmd)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "scholardevclaw",
            "kb",
            "search",
            "attention mechanism",
            "--limit",
            "7",
            "--domain",
            "nlp",
        ],
    )

    cli.main()

    assert called == {
        "kb_action": "search",
        "query": "attention mechanism",
        "limit": 7,
        "domain": "nlp",
    }


def test_kb_parser_clear(monkeypatch):
    called = {}

    def fake_cmd(args):
        called["kb_action"] = args.kb_action

    monkeypatch.setattr(cli, "cmd_kb", fake_cmd)
    monkeypatch.setattr(cli.sys, "argv", ["scholardevclaw", "kb", "clear"])

    cli.main()

    assert called == {"kb_action": "clear"}


def test_cmd_kb_stats_with_mocked_knowledge_base(monkeypatch, capsys):
    class FakeKnowledgeBase:
        def stats(self):
            return {
                "persist_dir": "/tmp/fake-kb",
                "papers": 2,
                "implementations": 4,
                "patterns": 0,
            }

    monkeypatch.setattr(cli, "_initialize_knowledge_base", lambda *, strict: FakeKnowledgeBase())

    cli.cmd_kb(SimpleNamespace(kb_action="stats"))

    captured = capsys.readouterr()
    assert "Knowledge base statistics" in captured.out
    assert "Papers: 2" in captured.out
    assert "Implementations: 4" in captured.out


def test_cmd_kb_search_with_mocked_knowledge_base(monkeypatch, capsys):
    observed: dict[str, object] = {}

    class FakeKnowledgeBase:
        def retrieve_similar_papers(self, query: str, n: int, domain_filter: str | None):
            observed["query"] = query
            observed["n"] = n
            observed["domain_filter"] = domain_filter
            return [
                {
                    "text": "Attention and layer normalization details",
                    "metadata": {
                        "title": "Transformer Paper",
                        "domain": "nlp",
                        "complexity": "medium",
                    },
                }
            ]

    monkeypatch.setattr(cli, "_initialize_knowledge_base", lambda *, strict: FakeKnowledgeBase())

    cli.cmd_kb(
        SimpleNamespace(
            kb_action="search",
            query="attention",
            limit=3,
            domain="nlp",
        )
    )

    captured = capsys.readouterr()
    assert observed == {"query": "attention", "n": 3, "domain_filter": "nlp"}
    assert "Transformer Paper" in captured.out
    assert "domain=nlp" in captured.out


def test_cmd_kb_clear_with_mocked_knowledge_base(monkeypatch, capsys):
    observed = {"cleared": False}

    class FakeKnowledgeBase:
        def clear(self):
            observed["cleared"] = True

        def stats(self):
            return {"papers": 0, "implementations": 0, "patterns": 0}

    monkeypatch.setattr(cli, "_initialize_knowledge_base", lambda *, strict: FakeKnowledgeBase())

    cli.cmd_kb(SimpleNamespace(kb_action="clear"))

    captured = capsys.readouterr()
    assert observed["cleared"] is True
    assert "Knowledge base cleared." in captured.out


def test_understand_parser_with_model_and_output_dir(monkeypatch):
    called = {}

    def fake_cmd(args):
        called["paper_document_json"] = args.paper_document_json
        called["model"] = args.model
        called["provider"] = args.provider
        called["output_dir"] = args.output_dir

    monkeypatch.setattr(cli, "cmd_understand", fake_cmd)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "scholardevclaw",
            "understand",
            "/tmp/paper_document.json",
            "--model",
            "claude-opus-4-5",
            "--provider",
            "openrouter",
            "--output-dir",
            "/tmp/out",
        ],
    )

    cli.main()

    assert called == {
        "paper_document_json": "/tmp/paper_document.json",
        "model": "claude-opus-4-5",
        "provider": "openrouter",
        "output_dir": "/tmp/out",
    }


def test_plan_parser_with_stack_and_output_dir(monkeypatch):
    called = {}

    def fake_cmd(args):
        called["understanding_json"] = args.understanding_json
        called["stack"] = args.stack
        called["provider"] = args.provider
        called["output_dir"] = args.output_dir

    monkeypatch.setattr(cli, "cmd_plan", fake_cmd)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "scholardevclaw",
            "plan",
            "/tmp/understanding.json",
            "--stack",
            "numpy-only",
            "--provider",
            "openrouter",
            "--output-dir",
            "/tmp/plan-out",
        ],
    )

    cli.main()

    assert called == {
        "understanding_json": "/tmp/understanding.json",
        "stack": "numpy-only",
        "provider": "openrouter",
        "output_dir": "/tmp/plan-out",
    }


def test_plan_parser_accepts_numpy_alias(monkeypatch):
    called = {}

    def fake_cmd(args):
        called["stack"] = args.stack

    monkeypatch.setattr(cli, "cmd_plan", fake_cmd)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "scholardevclaw",
            "plan",
            "/tmp/understanding.json",
            "--stack",
            "numpy",
        ],
    )

    cli.main()

    assert called == {"stack": "numpy"}


def test_legacy_commands_accept_use_specs_flag(monkeypatch):
    observed: dict[str, bool] = {}

    def _record(command_name: str):
        def _handler(args):
            observed[command_name] = bool(getattr(args, "use_specs", False))

        return _handler

    monkeypatch.setattr(cli, "cmd_suggest", _record("suggest"))
    monkeypatch.setattr(cli, "cmd_map", _record("map"))
    monkeypatch.setattr(cli, "cmd_generate", _record("generate"))
    monkeypatch.setattr(cli, "cmd_integrate", _record("integrate"))

    command_argv = [
        ["scholardevclaw", "suggest", "/tmp/repo", "--use-specs"],
        ["scholardevclaw", "map", "/tmp/repo", "rmsnorm", "--use-specs"],
        ["scholardevclaw", "generate", "/tmp/repo", "rmsnorm", "--use-specs"],
        ["scholardevclaw", "integrate", "/tmp/repo", "rmsnorm", "--use-specs"],
    ]

    for argv in command_argv:
        monkeypatch.setattr(cli.sys, "argv", argv)
        cli.main()

    assert observed == {
        "suggest": True,
        "map": True,
        "generate": True,
        "integrate": True,
    }


def test_generate_parser_dynamic_mode_arguments(monkeypatch):
    called = {}

    def fake_cmd(args):
        called["arg1"] = args.arg1
        called["arg2"] = args.arg2
        called["max_parallel"] = args.max_parallel
        called["model"] = args.model
        called["provider"] = args.provider
        called["output_dir"] = args.output_dir

    monkeypatch.setattr(cli, "cmd_generate", fake_cmd)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "scholardevclaw",
            "generate",
            "/tmp/implementation_plan.json",
            "/tmp/understanding.json",
            "--max-parallel",
            "3",
            "--model",
            "claude-opus-4-5",
            "--provider",
            "anthropic",
            "--output-dir",
            "/tmp/generated",
        ],
    )

    cli.main()

    assert called == {
        "arg1": "/tmp/implementation_plan.json",
        "arg2": "/tmp/understanding.json",
        "max_parallel": 3,
        "model": "claude-opus-4-5",
        "provider": "anthropic",
        "output_dir": "/tmp/generated",
    }


def test_execute_parser_with_heal_timeout_and_output_dir(monkeypatch):
    called = {}

    def fake_cmd(args):
        called["project_dir"] = args.project_dir
        called["heal"] = args.heal
        called["timeout"] = args.timeout
        called["provider"] = args.provider
        called["output_dir"] = args.output_dir

    monkeypatch.setattr(cli, "cmd_execute", fake_cmd)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "scholardevclaw",
            "execute",
            "/tmp/project",
            "--heal",
            "--timeout",
            "120",
            "--provider",
            "openrouter",
            "--output-dir",
            "/tmp/out",
        ],
    )

    cli.main()

    assert called == {
        "project_dir": "/tmp/project",
        "heal": True,
        "timeout": 120,
        "provider": "openrouter",
        "output_dir": "/tmp/out",
    }


def test_scaffold_parser_with_output_dir(monkeypatch):
    called = {}

    def fake_cmd(args):
        called["project_dir"] = args.project_dir
        called["plan_json"] = args.plan_json
        called["understanding_json"] = args.understanding_json
        called["reproducibility_report_json"] = args.reproducibility_report_json
        called["output_dir"] = args.output_dir

    monkeypatch.setattr(cli, "cmd_scaffold", fake_cmd)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "scholardevclaw",
            "scaffold",
            "/tmp/project",
            "/tmp/plan.json",
            "/tmp/understanding.json",
            "/tmp/reproducibility_report.json",
            "--output-dir",
            "/tmp/scaffold-out",
        ],
    )

    cli.main()

    assert called == {
        "project_dir": "/tmp/project",
        "plan_json": "/tmp/plan.json",
        "understanding_json": "/tmp/understanding.json",
        "reproducibility_report_json": "/tmp/reproducibility_report.json",
        "output_dir": "/tmp/scaffold-out",
    }


def test_from_paper_parser_with_all_flags(monkeypatch):
    called = {}

    def fake_cmd(args):
        called["source"] = args.source
        called["output_dir"] = args.output_dir
        called["heal"] = args.heal
        called["scaffold"] = args.scaffold
        called["max_parallel"] = args.max_parallel
        called["model"] = args.model
        called["provider"] = args.provider
        called["no_kb"] = args.no_kb
        called["dry_run"] = args.dry_run

    monkeypatch.setattr(cli, "cmd_from_paper", fake_cmd)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "scholardevclaw",
            "from-paper",
            "arxiv:2005.14165",
            "--output-dir",
            "/tmp/from-paper",
            "--heal",
            "--scaffold",
            "--max-parallel",
            "8",
            "--model",
            "claude-opus-4-5",
            "--provider",
            "openrouter",
            "--no-kb",
            "--dry-run",
        ],
    )

    cli.main()

    assert called == {
        "source": "arxiv:2005.14165",
        "output_dir": "/tmp/from-paper",
        "heal": True,
        "scaffold": True,
        "max_parallel": 8,
        "model": "claude-opus-4-5",
        "provider": "openrouter",
        "no_kb": True,
        "dry_run": True,
    }


def test_cmd_from_paper_dry_run_writes_phase_artifacts(monkeypatch, tmp_path):
    import scholardevclaw.ingestion.paper_fetcher as paper_fetcher
    import scholardevclaw.planning as planning
    import scholardevclaw.understanding.agent as understanding_agent
    from scholardevclaw.ingestion.models import PaperDocument
    from scholardevclaw.planning.models import CodeModule, ImplementationPlan
    from scholardevclaw.understanding.models import PaperUnderstanding

    fake_graph_module = types.ModuleType("scholardevclaw.understanding.graph")
    fake_graph_module.build_concept_graph = lambda _u: object()  # type: ignore[attr-defined]
    fake_graph_module.export_graph_json = lambda _g: {"nodes": [], "links": []}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scholardevclaw.understanding.graph", fake_graph_module)

    class FakeFetcher:
        def fetch_auto(self, source: str, dest_dir: Path):
            del source
            dest_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = dest_dir / "paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            return PaperDocument(
                title="Dry Run Paper",
                authors=["Author"],
                arxiv_id="2003.00744",
                doi=None,
                year=2020,
                abstract="Abstract",
                sections=[],
                equations=[],
                algorithms=[],
                figures=[],
                full_text="text",
                pdf_path=pdf_path,
                references=[],
                keywords=[],
                domain="cv",
            )

    class FakeUnderstandingAgent:
        def __init__(self, api_key: str, model: str, *, provider: str | None = None) -> None:
            del api_key, model, provider

        def understand(self, document: PaperDocument) -> PaperUnderstanding:
            del document
            return PaperUnderstanding(
                paper_title="Dry Run Paper",
                one_line_summary="summary",
                problem_statement="problem",
                key_insight="insight",
                core_algorithm_description="algorithm",
                input_output_spec="input->output",
                evaluation_protocol="accuracy",
                complexity="medium",
                estimated_impl_hours=6,
                confidence=0.8,
            )

    class FakePlanner:
        def __init__(self, api_key: str, model: str, *, provider: str | None = None) -> None:
            del api_key, model, provider

        def plan(
            self, understanding: PaperUnderstanding, document: PaperDocument
        ) -> ImplementationPlan:
            del understanding, document
            return ImplementationPlan(
                project_name="dry_run_project",
                target_language="python",
                tech_stack="pytorch",
                modules=[
                    CodeModule(
                        id="module_a",
                        name="Module A",
                        description="demo",
                        file_path="src/module_a.py",
                        depends_on=[],
                        priority=1,
                        estimated_lines=20,
                        test_file_path="tests/test_module_a.py",
                        tech_stack="pytorch",
                    )
                ],
                directory_structure={"src": {}, "tests": {}},
                environment={"numpy": ">=1.26.0"},
                entry_points=["module_a"],
                estimated_total_lines=20,
            )

    monkeypatch.setattr(paper_fetcher, "PaperFetcher", FakeFetcher)
    monkeypatch.setattr(understanding_agent, "UnderstandingAgent", FakeUnderstandingAgent)
    monkeypatch.setattr(planning, "ImplementationPlanner", FakePlanner)

    kb_init_calls = {"count": 0}

    def fake_init_kb(*, strict: bool):
        del strict
        kb_init_calls["count"] += 1
        return object()

    monkeypatch.setattr(cli, "_initialize_knowledge_base", fake_init_kb)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")

    output_root = tmp_path / "from-paper-dry-run"
    args = SimpleNamespace(
        source="arxiv:2003.00744",
        output_dir=str(output_root),
        heal=True,
        scaffold=True,
        max_parallel=4,
        model="claude-opus-4-5",
        provider="openrouter",
        no_kb=True,
        dry_run=True,
    )

    cli.cmd_from_paper(args)

    work_dir = output_root / "work"
    assert (work_dir / "paper_document.json").exists()
    assert (work_dir / "understanding.json").exists()
    assert (work_dir / "concept_graph.json").exists()
    assert (work_dir / "implementation_plan.json").exists()

    project_dir = output_root / "dry_run_project"
    assert project_dir.exists()
    assert not (project_dir / "generation_report.json").exists()
    assert kb_init_calls == {"count": 0}


def test_cmd_from_paper_full_pipeline_with_heal_and_scaffold(monkeypatch, tmp_path):
    import scholardevclaw.execution as execution
    import scholardevclaw.generation as generation
    import scholardevclaw.ingestion.paper_fetcher as paper_fetcher
    import scholardevclaw.planning as planning
    import scholardevclaw.product as product
    import scholardevclaw.understanding.agent as understanding_agent
    from scholardevclaw.generation.models import GenerationResult, ModuleResult
    from scholardevclaw.ingestion.models import PaperDocument
    from scholardevclaw.planning.models import CodeModule, ImplementationPlan
    from scholardevclaw.understanding.models import PaperUnderstanding

    fake_graph_module = types.ModuleType("scholardevclaw.understanding.graph")
    fake_graph_module.build_concept_graph = lambda _u: object()  # type: ignore[attr-defined]
    fake_graph_module.export_graph_json = lambda _g: {"nodes": [], "links": []}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scholardevclaw.understanding.graph", fake_graph_module)

    class FakeKnowledgeBase:
        def __init__(self) -> None:
            self.papers_stored = 0
            self.implementations_stored = 0

        def store_paper(self, document: PaperDocument, understanding: PaperUnderstanding) -> None:
            del document, understanding
            self.papers_stored += 1

        def store_implementation(
            self,
            module: CodeModule,
            code: str,
            understanding: PaperUnderstanding,
        ) -> None:
            del module, code, understanding
            self.implementations_stored += 1

    class FakeFetcher:
        def fetch_auto(self, source: str, dest_dir: Path):
            del source
            dest_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = dest_dir / "paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            return PaperDocument(
                title="Full Flow Paper",
                authors=["Author"],
                arxiv_id="2005.14165",
                doi=None,
                year=2020,
                abstract="Abstract",
                sections=[],
                equations=[],
                algorithms=[],
                figures=[],
                full_text="text",
                pdf_path=pdf_path,
                references=[],
                keywords=[],
                domain="nlp",
            )

    class FakeUnderstandingAgent:
        def __init__(self, api_key: str, model: str, *, provider: str | None = None) -> None:
            del api_key, model, provider

        def understand(self, document: PaperDocument) -> PaperUnderstanding:
            del document
            return PaperUnderstanding(
                paper_title="Full Flow Paper",
                one_line_summary="summary",
                problem_statement="problem",
                key_insight="insight",
                core_algorithm_description="algorithm",
                input_output_spec="input->output",
                evaluation_protocol="accuracy: 0.90",
                complexity="medium",
                estimated_impl_hours=12,
                confidence=0.9,
            )

    class FakePlanner:
        def __init__(self, api_key: str, model: str, *, provider: str | None = None) -> None:
            del api_key, model, provider

        def plan(
            self, understanding: PaperUnderstanding, document: PaperDocument
        ) -> ImplementationPlan:
            del understanding, document
            return ImplementationPlan(
                project_name="phase8_project",
                target_language="python",
                tech_stack="pytorch",
                modules=[
                    CodeModule(
                        id="module_a",
                        name="Module A",
                        description="demo",
                        file_path="src/module_a.py",
                        depends_on=[],
                        priority=1,
                        estimated_lines=20,
                        test_file_path="tests/test_module_a.py",
                        tech_stack="pytorch",
                    )
                ],
                directory_structure={"src": {}, "tests": {}},
                environment={"numpy": ">=1.26.0"},
                entry_points=["module_a"],
                estimated_total_lines=20,
            )

    observed: dict[str, object] = {}

    class FakeOrchestrator:
        def __init__(
            self,
            api_key: str,
            model: str,
            *,
            client=None,
            knowledge_base=None,
        ) -> None:
            observed["orchestrator_api_key"] = api_key
            observed["orchestrator_model"] = model
            observed["orchestrator_client"] = client
            observed["orchestrator_kb"] = knowledge_base

        def generate_sync(self, *, plan, understanding, output_dir: Path, max_parallel: int):
            observed["generate_project_name"] = plan.project_name
            observed["generate_understanding"] = understanding.paper_title
            observed["generate_output_dir"] = str(output_dir)
            observed["generate_max_parallel"] = max_parallel
            return GenerationResult(
                plan=plan,
                module_results=[
                    ModuleResult(
                        module_id="module_a",
                        file_path="src/module_a.py",
                        test_file_path="tests/test_module_a.py",
                        code="def module_a() -> int:\n    return 1\n",
                        test_code="def test_module_a() -> None:\n    assert True\n",
                        generation_attempts=1,
                        final_errors=[],
                        tokens_used=5,
                    )
                ],
                output_dir=output_dir,
                success_rate=1.0,
                total_tokens_used=5,
                duration_seconds=0.25,
            )

    class FakeRunner:
        def __init__(self, timeout_seconds: int = 300, memory_limit_mb: int = 4096) -> None:
            del timeout_seconds, memory_limit_mb
            self._index = 0

        def run_tests(self, project_dir: Path):
            del project_dir
            self._index += 1
            success = self._index >= 2
            failed = 0 if success else 1
            return SimpleNamespace(
                tests_passed=3 if success else 2,
                tests_failed=failed,
                tests_errors=0,
                success=success,
                to_dict=lambda: {
                    "exit_code": 0 if success else 1,
                    "stdout": "ok" if success else "",
                    "stderr": "" if success else "FAILED tests/test_module_a.py::test_x",
                    "duration_seconds": 0.01,
                    "peak_memory_mb": 8.0,
                    "tests_passed": 3 if success else 2,
                    "tests_failed": failed,
                    "tests_errors": 0,
                    "success": success,
                },
            )

    class FakeHealer:
        def __init__(self, orchestrator, runner) -> None:
            del orchestrator, runner
            self.round_reports = [
                {
                    "round": 1,
                    "tests_passed": 2,
                    "tests_failed": 1,
                    "tests_errors": 0,
                    "success": False,
                    "failing_modules": ["module_a"],
                }
            ]

        def heal(
            self,
            generation_result: GenerationResult,
            plan: ImplementationPlan,
            understanding: PaperUnderstanding,
        ) -> GenerationResult:
            del plan, understanding
            return generation_result

    class FakeScorer:
        def __init__(
            self,
            api_key: str | None = None,
            model: str = "claude-sonnet-4-5",
            *,
            provider: str | None = None,
        ) -> None:
            observed["scorer_api_key"] = api_key
            observed["scorer_model"] = model
            observed["scorer_provider"] = provider

        def score(self, understanding: PaperUnderstanding, execution_report):
            del understanding, execution_report
            return SimpleNamespace(
                score=0.82,
                verdict="partial",
                to_dict=lambda: {
                    "paper_title": "Full Flow Paper",
                    "claimed_metrics": {"accuracy": 0.9},
                    "achieved_metrics": {"accuracy": 0.82},
                    "delta": {"accuracy": -0.08},
                    "score": 0.82,
                    "verdict": "partial",
                },
            )

    class FakeScaffolder:
        def scaffold(self, project_dir: Path, plan, understanding, reproducibility_report) -> None:
            del plan, understanding, reproducibility_report
            (project_dir / "api").mkdir(parents=True, exist_ok=True)
            (project_dir / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
            (project_dir / "api" / "main.py").write_text("# api\n", encoding="utf-8")
            (project_dir / "demo.py").write_text("# demo\n", encoding="utf-8")
            (project_dir / "README.md").write_text("# readme\n", encoding="utf-8")
            (project_dir / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
            (project_dir / "pyproject.toml").write_text(
                "[project]\nname='demo'\n", encoding="utf-8"
            )
            (project_dir / ".github" / "workflows" / "ci.yml").write_text(
                "name: CI\n",
                encoding="utf-8",
            )

    fake_kb = FakeKnowledgeBase()

    monkeypatch.setattr(paper_fetcher, "PaperFetcher", FakeFetcher)
    monkeypatch.setattr(understanding_agent, "UnderstandingAgent", FakeUnderstandingAgent)
    monkeypatch.setattr(planning, "ImplementationPlanner", FakePlanner)
    monkeypatch.setattr(generation, "CodeOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(execution, "SandboxRunner", FakeRunner)
    monkeypatch.setattr(execution, "SelfHealingLoop", FakeHealer)
    monkeypatch.setattr(execution, "ReproducibilityScorer", FakeScorer)
    monkeypatch.setattr(product, "ProductScaffolder", FakeScaffolder)
    monkeypatch.setattr(cli, "_initialize_knowledge_base", lambda *, strict: fake_kb)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-openrouter-key")

    output_root = tmp_path / "from-paper-full"
    args = SimpleNamespace(
        source="arxiv:2005.14165",
        output_dir=str(output_root),
        heal=True,
        scaffold=True,
        max_parallel=2,
        model="claude-opus-4-5",
        provider="openrouter",
        no_kb=False,
        dry_run=False,
    )

    cli.cmd_from_paper(args)

    project_dir = output_root / "phase8_project"
    work_dir = output_root / "work"

    assert (project_dir / "generation_report.json").exists()
    assert (work_dir / "execution_report.json").exists()
    assert (work_dir / "reproducibility_report.json").exists()
    assert (project_dir / "api" / "main.py").exists()
    assert (project_dir / "demo.py").exists()
    assert (project_dir / "README.md").exists()

    generation_payload = json.loads(
        (project_dir / "generation_report.json").read_text(encoding="utf-8")
    )
    assert generation_payload["success_rate"] == 1.0
    assert generation_payload["healing"]["round_count"] == 1
    assert fake_kb.papers_stored == 1
    assert fake_kb.implementations_stored == 1

    # When provider is non-Anthropic, orchestrator is created with empty api_key/model but with client
    assert observed["orchestrator_api_key"] == ""
    assert observed["orchestrator_model"] == ""
    assert observed["orchestrator_client"] is not None
    assert observed["orchestrator_kb"] == fake_kb
    assert observed["generate_project_name"] == "phase8_project"
    assert observed["generate_understanding"] == "Full Flow Paper"
    assert observed["generate_output_dir"] == str(project_dir.resolve())
    assert observed["generate_max_parallel"] == 2
    # Scorer uses the same provider as the main orchestrator (openrouter)
    # When provider is non-Anthropic, model defaults to provider's default
    assert observed["scorer_api_key"] == "fake-openrouter-key"
    assert observed["scorer_model"] == "openai/gpt-4.1-mini"
    assert observed["scorer_provider"] == "openrouter"


def test_cmd_generate_dynamic_mode_writes_generation_report(monkeypatch, tmp_path):
    import scholardevclaw.generation as generation

    class FakeResult:
        success_rate = 1.0
        module_results = [SimpleNamespace(module_id="module_0")]
        duration_seconds = 0.12

        @staticmethod
        def to_dict() -> dict[str, object]:
            return {
                "success_rate": 1.0,
                "duration_seconds": 0.12,
                "module_results": [
                    {
                        "module_id": "module_0",
                        "generation_attempts": 1,
                    }
                ],
                "total_tokens_used": 20,
            }

    observed: dict[str, object] = {}
    fake_kb = object()

    class FakeOrchestrator:
        def __init__(
            self,
            api_key: str,
            model: str,
            *,
            client=None,
            knowledge_base=None,
        ) -> None:
            observed["api_key"] = api_key
            observed["model"] = model
            observed["client"] = client
            observed["knowledge_base"] = knowledge_base

        def generate_sync(self, *, plan, understanding, output_dir: Path, max_parallel: int):
            observed["plan_project_name"] = plan.project_name
            observed["understanding_title"] = understanding.paper_title
            observed["output_dir"] = str(output_dir)
            observed["max_parallel"] = max_parallel
            return FakeResult()

    monkeypatch.setattr(generation, "CodeOrchestrator", FakeOrchestrator)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")
    monkeypatch.setattr(cli, "_initialize_knowledge_base", lambda *, strict: fake_kb)

    plan_path = tmp_path / "implementation_plan.json"
    understanding_path = tmp_path / "understanding.json"
    output_dir = tmp_path / "generated"

    plan_path.write_text(
        json.dumps(
            {
                "project_name": "demo-project",
                "target_language": "python",
                "tech_stack": "python",
                "modules": [],
            }
        ),
        encoding="utf-8",
    )
    understanding_path.write_text(
        json.dumps(
            {
                "paper_title": "Demo Paper",
                "core_algorithm_description": "Demo algorithm",
            }
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(
        arg1=str(plan_path),
        arg2=str(understanding_path),
        output_dir=str(output_dir),
        max_parallel=2,
        model="claude-sonnet-4-5",
        provider="openrouter",
        output_json=False,
        use_specs=False,
    )

    cli.cmd_generate(args)

    report_path = output_dir / "generation_report.json"
    assert report_path.exists()
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report_payload) >= {
        "success_rate",
        "duration_seconds",
        "module_results",
        "total_tokens_used",
    }
    # When provider is non-Anthropic, orchestrator is created with empty api_key/model but with client
    assert observed["api_key"] == ""
    assert observed["model"] == ""
    assert observed["client"] is not None
    assert observed["knowledge_base"] == fake_kb
    assert observed["plan_project_name"] == "demo-project"
    assert observed["understanding_title"] == "Demo Paper"
    assert observed["output_dir"] == str(output_dir.resolve())
    assert observed["max_parallel"] == 2


def test_cmd_scaffold_writes_artifacts_to_output_dir(tmp_path):
    project_dir = tmp_path / "project"
    output_dir = tmp_path / "release"
    project_dir.mkdir()

    plan_path = tmp_path / "implementation_plan.json"
    understanding_path = tmp_path / "understanding.json"
    reproducibility_path = tmp_path / "reproducibility_report.json"

    plan_path.write_text(
        json.dumps(
            {
                "project_name": "demo-product",
                "target_language": "python",
                "tech_stack": "pytorch",
                "modules": [],
                "environment": {"fastapi": ">=0.111.0"},
                "entry_points": ["src/main.py"],
            }
        ),
        encoding="utf-8",
    )
    understanding_path.write_text(
        json.dumps(
            {
                "paper_title": "Demo Paper",
                "one_line_summary": "Summary",
                "problem_statement": "Problem",
                "key_insight": "Insight",
                "input_output_spec": "Input to output",
                "evaluation_protocol": "Accuracy: 0.95",
                "requirements": [{"name": "Python", "type": "runtime", "is_optional": False}],
            }
        ),
        encoding="utf-8",
    )
    reproducibility_path.write_text(
        json.dumps(
            {
                "paper_title": "Demo Paper",
                "claimed_metrics": {"accuracy": 0.95},
                "achieved_metrics": {"accuracy": 0.94},
                "score": 0.98,
                "verdict": "reproduced",
            }
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(
        project_dir=str(project_dir),
        plan_json=str(plan_path),
        understanding_json=str(understanding_path),
        reproducibility_report_json=str(reproducibility_path),
        output_dir=str(output_dir),
    )

    cli.cmd_scaffold(args)

    expected_paths = [
        output_dir / "api" / "main.py",
        output_dir / "demo.py",
        output_dir / "pyproject.toml",
        output_dir / "Dockerfile",
        output_dir / "README.md",
        output_dir / ".github" / "workflows" / "ci.yml",
    ]
    for path in expected_paths:
        assert path.exists()


def test_cmd_execute_writes_execution_and_reproducibility_reports(monkeypatch, tmp_path):
    import scholardevclaw.execution as execution

    class FakeRunner:
        def __init__(self, *, timeout_seconds: int = 0, memory_limit_mb: int = 4096) -> None:
            del memory_limit_mb
            self.timeout_seconds = timeout_seconds

        def run_tests(self, _project_dir: Path):
            return SimpleNamespace(
                tests_passed=3,
                tests_failed=0,
                tests_errors=0,
                to_dict=lambda: {
                    "exit_code": 0,
                    "stdout": "ok",
                    "stderr": "",
                    "duration_seconds": 0.01,
                    "peak_memory_mb": 16.0,
                    "tests_passed": 3,
                    "tests_failed": 0,
                    "tests_errors": 0,
                    "success": True,
                },
                success=True,
            )

    class FakeScorer:
        def __init__(
            self, *, api_key=None, model: str = "claude-sonnet-4-5", provider: str | None = None
        ) -> None:
            del api_key, model, provider

        def score(self, understanding, execution_report):
            del understanding, execution_report
            return SimpleNamespace(
                score=1.0,
                verdict="reproduced",
                to_dict=lambda: {
                    "paper_title": "Demo",
                    "claimed_metrics": {},
                    "achieved_metrics": {},
                    "delta": {},
                    "score": 1.0,
                    "verdict": "reproduced",
                },
            )

    monkeypatch.setattr(execution, "SandboxRunner", FakeRunner)
    monkeypatch.setattr(execution, "ReproducibilityScorer", FakeScorer)

    project_dir = tmp_path / "project"
    output_dir = tmp_path / "out"
    project_dir.mkdir()

    args = SimpleNamespace(
        project_dir=str(project_dir),
        heal=False,
        timeout=120,
        output_dir=str(output_dir),
    )
    cli.cmd_execute(args)

    assert (output_dir / "execution_report.json").exists()
    assert (output_dir / "reproducibility_report.json").exists()


def test_cmd_execute_with_heal_updates_generation_report_metadata(monkeypatch, tmp_path):
    import scholardevclaw.execution as execution
    import scholardevclaw.generation as generation
    from scholardevclaw.generation.models import GenerationResult, ModuleResult
    from scholardevclaw.planning.models import CodeModule, ImplementationPlan

    class FakeRunner:
        def __init__(self, *, timeout_seconds: int = 0, memory_limit_mb: int = 4096) -> None:
            del timeout_seconds, memory_limit_mb
            self._reports = [
                SimpleNamespace(
                    tests_passed=2,
                    tests_failed=2,
                    tests_errors=0,
                    success=False,
                    to_dict=lambda: {
                        "exit_code": 1,
                        "stdout": "",
                        "stderr": "FAILED tests/test_module_a.py::test_a",
                        "duration_seconds": 0.01,
                        "peak_memory_mb": 32.0,
                        "tests_passed": 2,
                        "tests_failed": 2,
                        "tests_errors": 0,
                        "success": False,
                    },
                ),
                SimpleNamespace(
                    tests_passed=4,
                    tests_failed=0,
                    tests_errors=0,
                    success=True,
                    to_dict=lambda: {
                        "exit_code": 0,
                        "stdout": "all good",
                        "stderr": "",
                        "duration_seconds": 0.01,
                        "peak_memory_mb": 32.0,
                        "tests_passed": 4,
                        "tests_failed": 0,
                        "tests_errors": 0,
                        "success": True,
                    },
                ),
            ]
            self._idx = 0

        def run_tests(self, _project_dir: Path):
            report = self._reports[min(self._idx, len(self._reports) - 1)]
            self._idx += 1
            return report

    class FakeScorer:
        def __init__(
            self, *, api_key=None, model: str = "claude-sonnet-4-5", provider: str | None = None
        ) -> None:
            del api_key, model, provider

        def score(self, understanding, execution_report):
            del understanding, execution_report
            return SimpleNamespace(
                score=0.6,
                verdict="partial",
                to_dict=lambda: {
                    "paper_title": "Demo",
                    "claimed_metrics": {},
                    "achieved_metrics": {},
                    "delta": {},
                    "score": 0.6,
                    "verdict": "partial",
                },
            )

    class FakeOrchestrator:
        def __init__(
            self,
            api_key: str,
            model: str,
            *,
            client=None,
            knowledge_base=None,
        ) -> None:
            self.api_key = api_key
            self.model = model
            self.client = client
            self.knowledge_base = knowledge_base

    class FakeHealer:
        def __init__(self, orchestrator, runner):
            self.orchestrator = orchestrator
            self.runner = runner
            self.round_reports = [
                {
                    "round": 1,
                    "tests_passed": 2,
                    "tests_failed": 2,
                    "tests_errors": 0,
                    "success": False,
                    "failing_modules": ["module_a"],
                }
            ]

        def heal(self, generation_result, plan, understanding):
            del generation_result, understanding
            return GenerationResult(
                plan=plan,
                module_results=[
                    ModuleResult(
                        module_id="module_a",
                        file_path="src/module_a.py",
                        test_file_path="tests/test_module_a.py",
                        code="def module_a():\n    return 1\n",
                        test_code="def test_module_a():\n    assert True\n",
                        generation_attempts=2,
                        final_errors=[],
                        tokens_used=4,
                    )
                ],
                output_dir=tmp_path / "project",
                success_rate=1.0,
                total_tokens_used=9,
                duration_seconds=0.2,
            )

    monkeypatch.setattr(execution, "SandboxRunner", FakeRunner)
    monkeypatch.setattr(execution, "ReproducibilityScorer", FakeScorer)
    monkeypatch.setattr(execution, "SelfHealingLoop", FakeHealer)
    monkeypatch.setattr(generation, "CodeOrchestrator", FakeOrchestrator)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")

    project_dir = tmp_path / "project"
    project_dir.mkdir()

    plan_payload = ImplementationPlan(
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
            )
        ],
    ).to_dict()

    generation_report_path = project_dir / "generation_report.json"
    generation_report_path.write_text(
        json.dumps(
            {
                "plan": plan_payload,
                "module_results": [
                    {
                        "module_id": "module_a",
                        "file_path": "src/module_a.py",
                        "test_file_path": "tests/test_module_a.py",
                        "code": "def module_a():\n    return 0\n",
                        "test_code": "def test_module_a():\n    assert False\n",
                        "generation_attempts": 1,
                        "final_errors": ["initial"],
                        "tokens_used": 5,
                    }
                ],
                "output_dir": str(project_dir),
                "success_rate": 0.0,
                "total_tokens_used": 5,
                "duration_seconds": 0.1,
            }
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(project_dir=str(project_dir), heal=True, timeout=120, output_dir=None)
    cli.cmd_execute(args)

    updated_generation = json.loads(generation_report_path.read_text(encoding="utf-8"))
    healing = updated_generation.get("healing", {})

    assert healing.get("initial_failed_tests", 0) > healing.get("final_failed_tests", 0)
    assert isinstance(healing.get("round_count"), int)
    assert healing["round_count"] >= 1


def test_cmd_execute_exits_nonzero_when_final_execution_fails(monkeypatch, tmp_path):
    import scholardevclaw.execution as execution

    class FakeRunner:
        def __init__(self, *, timeout_seconds: int = 0, memory_limit_mb: int = 4096) -> None:
            del timeout_seconds, memory_limit_mb

        def run_tests(self, _project_dir: Path):
            return SimpleNamespace(
                tests_passed=1,
                tests_failed=1,
                tests_errors=0,
                success=False,
                to_dict=lambda: {
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": "FAILED tests/test_module.py::test_x",
                    "duration_seconds": 0.01,
                    "peak_memory_mb": 32.0,
                    "tests_passed": 1,
                    "tests_failed": 1,
                    "tests_errors": 0,
                    "success": False,
                },
            )

    class FakeScorer:
        def __init__(
            self, *, api_key=None, model: str = "claude-sonnet-4-5", provider: str | None = None
        ) -> None:
            del api_key, model, provider

        def score(self, understanding, execution_report):
            del understanding, execution_report
            return SimpleNamespace(
                score=0.3,
                verdict="failed",
                to_dict=lambda: {
                    "paper_title": "Demo",
                    "claimed_metrics": {},
                    "achieved_metrics": {},
                    "delta": {},
                    "score": 0.3,
                    "verdict": "failed",
                },
            )

    monkeypatch.setattr(execution, "SandboxRunner", FakeRunner)
    monkeypatch.setattr(execution, "ReproducibilityScorer", FakeScorer)

    project_dir = tmp_path / "project"
    project_dir.mkdir()

    args = SimpleNamespace(project_dir=str(project_dir), heal=False, timeout=120, output_dir=None)
    with pytest.raises(SystemExit) as exc:
        cli.cmd_execute(args)

    assert exc.value.code == 1
    assert (project_dir / "execution_report.json").exists()
    assert (project_dir / "reproducibility_report.json").exists()


def test_cmd_deploy_check_json_success(tmp_path, capsys):
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
SCHOLARDEVCLAW_API_AUTH_KEY=real-api-key
SCHOLARDEVCLAW_ALLOWED_REPO_DIRS=/repos
SCHOLARDEVCLAW_CORS_ORIGINS=https://scholardevclaw.ai
OPENCLAW_TOKEN=real-openclaw-token
OPENCLAW_API_URL=https://api.openclaw.example
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=super-secure-password
CORE_BRIDGE_MODE=http
CORE_API_URL=http://core-api:8000
""",
        encoding="utf-8",
    )

    args = SimpleNamespace(env_file=str(env_file), output_json=True)
    cli.cmd_deploy_check(args)

    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["errors"] == []


def test_cmd_deploy_check_missing_env_file_exits_1(tmp_path, capsys):
    args = SimpleNamespace(env_file=str(tmp_path / "missing.env"), output_json=False)

    with pytest.raises(SystemExit) as exc:
        cli.cmd_deploy_check(args)

    assert exc.value.code == 1
    assert "Environment file not found" in capsys.readouterr().err
