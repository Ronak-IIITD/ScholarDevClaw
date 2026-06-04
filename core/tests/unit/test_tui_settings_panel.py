"""Tests for the settings panel modal."""

from __future__ import annotations

import pytest

from scholardevclaw.tui.settings_panel import (
    Setting,
    SettingsPanel,
    _SectionHeader,
    _SettingRow,
)

# -----------------------------------------------------------------------------
# Setting
# -----------------------------------------------------------------------------


class TestSetting:
    """The Setting data class and its value-rotation helpers."""

    def test_construction_choice(self) -> None:
        s = Setting(
            key="llm.provider",
            label="Provider",
            kind="choice",
            current="anthropic",
            options=("anthropic", "openai", "ollama"),
        )
        assert s.current == "anthropic"
        assert s.options == ("anthropic", "openai", "ollama")

    def test_construction_text(self) -> None:
        s = Setting(
            key="llm.model",
            label="Model",
            kind="text",
            current="claude-sonnet-4-20250514",
        )
        assert s.current == "claude-sonnet-4-20250514"
        assert s.options == ()

    def test_construction_toggle(self) -> None:
        s = Setting(
            key="behavior.yolo",
            label="YOLO mode",
            kind="toggle",
            current="off",
        )
        assert s.current == "off"

    def test_frozen(self) -> None:
        s = Setting(key="x", label="X", kind="text", current="y")
        with pytest.raises((AttributeError, Exception)):
            s.label = "Z"  # type: ignore[misc]

    # ----- cycle -----

    def test_cycle_choice_advances(self) -> None:
        s = Setting(key="k", label="L", kind="choice", current="a", options=("a", "b", "c"))
        assert s.cycle() == "b"

    def test_cycle_choice_wraps(self) -> None:
        s = Setting(key="k", label="L", kind="choice", current="c", options=("a", "b", "c"))
        assert s.cycle() == "a"

    def test_cycle_choice_unknown_current(self) -> None:
        s = Setting(key="k", label="L", kind="choice", current="zzz", options=("a", "b"))
        # Falls back to first option, then advances: (0 + 1) % 2 = 1
        assert s.cycle() == "b"

    def test_cycle_choice_empty_options(self) -> None:
        s = Setting(key="k", label="L", kind="choice", current="a", options=())
        assert s.cycle() == "a"

    def test_cycle_toggle_off_to_on(self) -> None:
        s = Setting(key="k", label="L", kind="toggle", current="off")
        assert s.cycle() == "on"

    def test_cycle_toggle_on_to_off(self) -> None:
        s = Setting(key="k", label="L", kind="toggle", current="on")
        assert s.cycle() == "off"

    def test_cycle_text_returns_current(self) -> None:
        s = Setting(key="k", label="L", kind="text", current="hello")
        assert s.cycle() == "hello"

    # ----- display -----

    def test_display_toggle_on(self) -> None:
        s = Setting(key="k", label="L", kind="toggle", current="on")
        d = s.display()
        assert "on" in d
        assert "●" in d

    def test_display_toggle_off(self) -> None:
        s = Setting(key="k", label="L", kind="toggle", current="off")
        d = s.display()
        assert "off" in d
        assert "○" in d

    def test_display_choice(self) -> None:
        s = Setting(key="k", label="L", kind="choice", current="a", options=("a", "b"))
        d = s.display()
        assert "a" in d
        # Has arrow indicators
        assert "‹" in d
        assert "›" in d

    def test_display_text(self) -> None:
        s = Setting(key="k", label="L", kind="text", current="hello")
        assert s.display() == "hello"


# -----------------------------------------------------------------------------
# _SettingRow
# -----------------------------------------------------------------------------


class TestSettingRow:
    """Row state management."""

    def test_initial_current_from_setting(self) -> None:
        s = Setting(key="k", label="L", kind="text", current="hello")
        row = _SettingRow(s, 0)
        assert row.current_value == "hello"
        assert row.initial_value == "hello"

    def test_set_value_updates_current(self) -> None:
        s = Setting(key="k", label="L", kind="text", current="a")
        row = _SettingRow(s, 0)
        row.set_value("b")
        assert row.current_value == "b"
        assert row.initial_value == "a"  # unchanged

    def test_focus_toggle(self) -> None:
        s = Setting(key="k", label="L", kind="text", current="x")
        row = _SettingRow(s, 0)
        assert row.has_class("-focused") is False
        row.set_focused(True)
        assert row.has_class("-focused") is True
        row.set_focused(False)
        assert row.has_class("-focused") is False

    def test_stores_setting_and_index(self) -> None:
        s = Setting(key="k", label="L", kind="text", current="x")
        row = _SettingRow(s, 7)
        assert row.setting is s
        assert row.index == 7


# -----------------------------------------------------------------------------
# _SectionHeader
# -----------------------------------------------------------------------------


class TestSectionHeader:
    """Section header widget is constructible."""

    def test_construction(self) -> None:
        h = _SectionHeader("── LLM ──")
        assert h is not None


# -----------------------------------------------------------------------------
# SettingsPanel (construction + section ordering)
# -----------------------------------------------------------------------------


class TestSettingsPanel:
    """Modal screen construction and section detection."""

    def test_constructible_empty(self) -> None:
        panel = SettingsPanel(settings=[])
        assert panel._rows == []
        assert panel._focus_index == 0

    def test_constructible_single_section(self) -> None:
        settings = [
            Setting(
                key="llm.provider", label="Provider", kind="choice", current="a", options=("a", "b")
            ),
            Setting(key="llm.model", label="Model", kind="text", current="m"),
        ]
        panel = SettingsPanel(settings=settings)
        assert len(panel._rows) == 2
        assert panel._rows[0].setting.key == "llm.provider"
        assert panel._rows[1].setting.key == "llm.model"

    def test_constructible_multi_section(self) -> None:
        settings = [
            Setting(
                key="llm.provider", label="Provider", kind="choice", current="a", options=("a", "b")
            ),
            Setting(key="behavior.yolo", label="YOLO", kind="toggle", current="off"),
            Setting(
                key="appearance.theme",
                label="Theme",
                kind="choice",
                current="dark",
                options=("dark", "light"),
            ),
        ]
        panel = SettingsPanel(settings=settings)
        assert len(panel._rows) == 3

    def test_constructible_no_section_prefix(self) -> None:
        """Settings without a '.' in the key fall under 'general'."""
        settings = [
            Setting(key="no_section", label="X", kind="text", current="x"),
        ]
        panel = SettingsPanel(settings=settings)
        assert len(panel._rows) == 1

    def test_modal_screen_subclass(self) -> None:
        from textual.screen import ModalScreen

        assert issubclass(SettingsPanel, ModalScreen)

    def test_focus_index_in_bounds(self) -> None:
        settings = [
            Setting(key="a.b", label="A", kind="text", current="x"),
        ]
        panel = SettingsPanel(settings=settings)
        assert 0 <= panel._focus_index < len(panel._rows)

    def test_initial_settings_stored(self) -> None:
        settings = [
            Setting(key="a.b", label="A", kind="text", current="x"),
        ]
        panel = SettingsPanel(settings=settings)
        assert panel._initial_settings == settings

    def test_iter_widgets_yields_headers_and_rows(self) -> None:
        settings = [
            Setting(key="llm.x", label="X", kind="text", current="x"),
            Setting(key="behavior.y", label="Y", kind="toggle", current="off"),
        ]
        panel = SettingsPanel(settings=settings)
        widgets = list(panel._iter_widgets())
        # One section header per section + one row per setting
        assert len(widgets) == 2 + 2  # 2 sections * 1 header + 2 rows
        # First widget per section is a _SectionHeader
        assert isinstance(widgets[0], _SectionHeader)
        assert isinstance(widgets[2], _SectionHeader)
        # Settings rows come after their section header
        assert isinstance(widgets[1], _SettingRow)
        assert isinstance(widgets[3], _SettingRow)
