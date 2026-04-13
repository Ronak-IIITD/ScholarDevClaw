from __future__ import annotations

import builtins
import json
from types import SimpleNamespace

import pytest

import scholardevclaw.cli as cli


@pytest.mark.parametrize(
    ("command", "argv", "handler_name"),
    [
        ("analyze", ["scholardevclaw", "analyze", "/tmp/repo"], "cmd_analyze"),
        ("search", ["scholardevclaw", "search", "rmsnorm"], "cmd_search"),
        ("suggest", ["scholardevclaw", "suggest", "/tmp/repo"], "cmd_suggest"),
        ("integrate", ["scholardevclaw", "integrate", "/tmp/repo"], "cmd_integrate"),
        ("map", ["scholardevclaw", "map", "/tmp/repo", "rmsnorm"], "cmd_map"),
        ("generate", ["scholardevclaw", "generate", "/tmp/repo", "rmsnorm"], "cmd_generate"),
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
