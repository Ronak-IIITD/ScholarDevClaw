from __future__ import annotations

import builtins
import json
from types import SimpleNamespace

import pytest

import scholardevclaw.cli as cli


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
