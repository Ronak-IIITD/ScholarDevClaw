from __future__ import annotations

from types import ModuleType


def test_run_tui_delegates_to_app_run_tui(monkeypatch):
    import scholardevclaw.tui as tui

    app_module = ModuleType("scholardevclaw.tui.app")
    calls = {"count": 0}

    def fake_run_tui():
        calls["count"] += 1
        return "ok"

    app_module.run_tui = fake_run_tui
    monkeypatch.setitem(__import__("sys").modules, "scholardevclaw.tui.app", app_module)

    result = tui.run_tui()

    assert result == "ok"
    assert calls["count"] == 1


def test_public_symbols_in_all_are_resolvable():
    import scholardevclaw.tui as tui

    for symbol in tui.__all__:
        resolved = getattr(tui, symbol)
        assert resolved is not None
