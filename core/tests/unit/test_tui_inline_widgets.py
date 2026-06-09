"""Tests for inline workflow widgets (Phase 2: inline conversation flow)."""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from scholardevclaw.tui.widgets_new import (
    ConversationView,
    InlineConfirmBar,
    InlineInput,
    InlineProgressCard,
    WorkflowCard,
    make_assistant_message,
    make_system_message,
)


# ---------------------------------------------------------------------------
# WorkflowCard
# ---------------------------------------------------------------------------


def test_workflowcard_add_row():
    card = WorkflowCard(title="Test Card")
    card.add_row("Hello", "value")
    card.add_row("Warning", "warning")
    assert len(card._body_rows) == 2
    assert card._body_rows[0] == ("Hello", "-value")
    assert card._body_rows[1] == ("Warning", "-warning")


def test_workflowcard_clear_rows():
    card = WorkflowCard(title="Test")
    card.add_row("A", "value")
    card.add_row("B", "label")
    card.clear_rows()
    assert len(card._body_rows) == 0


def test_workflowcard_set_rows():
    card = WorkflowCard(title="Test")
    card.add_row("old", "value")
    card.set_rows([("new1", "accent"), ("new2", "success")])
    assert len(card._body_rows) == 2
    assert card._body_rows[0] == ("new1", "-accent")


def test_workflowcard_set_progress():
    card = WorkflowCard(title="Test")
    card.set_progress(0.5, "50%")
    assert card._progress == 0.5
    assert card._progress_text == "50%"


def test_workflowcard_set_progress_clamps():
    card = WorkflowCard(title="Test")
    card.set_progress(1.5)
    assert card._progress == 1.0
    card.set_progress(-0.5)
    assert card._progress == 0.0


def test_workflowcard_clear_progress():
    card = WorkflowCard(title="Test")
    card.set_progress(0.5)
    card.clear_progress()
    assert card._progress is None
    assert card._progress_text == ""


def test_workflowcard_set_title():
    card = WorkflowCard(title="Old")
    card.set_title("New")
    assert card._title == "New"


# ---------------------------------------------------------------------------
# InlineInput
# ---------------------------------------------------------------------------


def test_inline_input_init():
    inp = InlineInput(label="Source", placeholder="Enter text", hint="Press Enter")
    assert inp._label == "Source"
    assert inp._placeholder == "Enter text"
    assert inp._hint == "Press Enter"
    assert inp._initial_value == ""


def test_inline_input_with_value():
    inp = InlineInput(value="initial")
    assert inp._initial_value == "initial"


def test_inline_input_messages():
    """Test that Submitted and Cancelled messages can be created."""
    msg = InlineInput.Submitted("hello")
    assert msg.value == "hello"

    cancel = InlineInput.Cancelled()
    assert cancel is not None


# ---------------------------------------------------------------------------
# InlineConfirmBar
# ---------------------------------------------------------------------------


def test_inline_confirm_bar_init():
    bar = InlineConfirmBar(label="Approve plan?")
    assert bar._label == "Approve plan?"


def test_inline_confirm_bar_default_label():
    bar = InlineConfirmBar()
    assert bar._label == "Approval gate:"


def test_inline_confirm_bar_message():
    msg = InlineConfirmBar.Confirmed("approve")
    assert msg.decision == "approve"

    msg2 = InlineConfirmBar.Confirmed("reject")
    assert msg2.decision == "reject"


# ---------------------------------------------------------------------------
# InlineProgressCard
# ---------------------------------------------------------------------------


def test_inline_progress_card_init():
    card = InlineProgressCard(title="Generation")
    assert card._title == "Generation"
    assert card._phases == []
    assert card._tokens == 0
    assert card._elapsed == 0.0


def test_inline_progress_card_set_phases():
    card = InlineProgressCard(title="Test")
    phases = [
        {"name": "module1", "status": "pending", "duration": None},
        {"name": "module2", "status": "running", "duration": 1.5},
    ]
    card.set_phases(phases)
    assert len(card._phases) == 2
    assert card._phases[0]["name"] == "module1"


def test_inline_progress_card_update_phase():
    card = InlineProgressCard(title="Test")
    card.set_phases([{"name": "m1", "status": "pending", "duration": None}])
    card.update_phase("m1", "complete", 2.5)
    assert card._phases[0]["status"] == "complete"
    assert card._phases[0]["duration"] == 2.5


def test_inline_progress_card_update_phase_not_found():
    card = InlineProgressCard(title="Test")
    card.set_phases([{"name": "m1", "status": "pending", "duration": None}])
    # Should not raise
    card.update_phase("nonexistent", "complete")


def test_inline_progress_card_set_stats():
    card = InlineProgressCard(title="Test")
    card.set_stats(tokens=1500, elapsed=3.5)
    assert card._tokens == 1500
    assert card._elapsed == 3.5


# ---------------------------------------------------------------------------
# ConversationView with inline widgets
# ---------------------------------------------------------------------------


def test_conversationview_add_system_message():
    """Test that system messages can be added to conversation view."""
    view = ConversationView()
    msg = make_system_message("Hello world")
    assert msg.role.value == "system"
    assert msg.content == "Hello world"


def test_conversationview_add_assistant_message():
    """Test that assistant messages can be added to conversation view."""
    view = ConversationView()
    msg = make_assistant_message("I can help with that")
    assert msg.role.value == "assistant"
    assert msg.content == "I can help with that"
