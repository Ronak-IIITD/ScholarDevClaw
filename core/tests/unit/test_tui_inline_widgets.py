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


# ---------------------------------------------------------------------------
# InlineDiffCard
# ---------------------------------------------------------------------------


def test_inlinediffcard_set_file_info():
    from scholardevclaw.tui.widgets_new import InlineDiffCard

    card = InlineDiffCard()
    card.set_file_info("test.py", "modified", additions=10, deletions=5)
    assert card._file_path == "test.py"
    assert card._status == "modified"
    assert card._additions == 10
    assert card._deletions == 5


def test_inlinediffcard_set_diff_lines():
    from scholardevclaw.tui.widgets_new import InlineDiffCard

    card = InlineDiffCard()
    lines = [
        ("+ new line", "addition"),
        ("- old line", "deletion"),
        ("  context line", "context"),
    ]
    card.set_diff_lines(lines)
    assert len(card._diff_lines) == 3
    assert card._diff_lines[0] == ("+ new line", "addition")


def test_inlinediffcard_add_diff_line():
    from scholardevclaw.tui.widgets_new import InlineDiffCard

    card = InlineDiffCard()
    card.add_diff_line("+ added", "addition")
    card.add_diff_line("- removed", "deletion")
    assert len(card._diff_lines) == 2


def test_inlinediffcard_clear_diff_lines():
    from scholardevclaw.tui.widgets_new import InlineDiffCard

    card = InlineDiffCard()
    card.add_diff_line("+ line", "addition")
    card.clear_diff_lines()
    assert len(card._diff_lines) == 0


# ---------------------------------------------------------------------------
# InlinePatchReview
# ---------------------------------------------------------------------------


def test_inlinepatchreview_set_files():
    from scholardevclaw.tui.widgets_new import InlinePatchReview

    review = InlinePatchReview(title="Test Patch")
    files = [
        {"path": "a.py", "status": "modified", "additions": 5, "deletions": 2, "diff_lines": []},
        {"path": "b.py", "status": "added", "additions": 10, "deletions": 0, "diff_lines": []},
    ]
    review.set_files(files)
    assert len(review._files) == 2
    assert review._current_file_index == 0


def test_inlinepatchreview_add_file():
    from scholardevclaw.tui.widgets_new import InlinePatchReview

    review = InlinePatchReview()
    review.add_file("test.py", "modified", additions=3, deletions=1)
    assert len(review._files) == 1
    assert review._files[0]["path"] == "test.py"


def test_inlinepatchreview_next_prev_file():
    from scholardevclaw.tui.widgets_new import InlinePatchReview

    review = InlinePatchReview()
    review.add_file("a.py", "modified")
    review.add_file("b.py", "added")
    review.add_file("c.py", "deleted")

    assert review._current_file_index == 0
    review.next_file()
    assert review._current_file_index == 1
    review.next_file()
    assert review._current_file_index == 2
    review.next_file()  # Wraps around
    assert review._current_file_index == 0
    review.prev_file()  # Wraps around
    assert review._current_file_index == 2


def test_inlinepatchreview_set_file_decision():
    from scholardevclaw.tui.widgets_new import InlinePatchReview

    review = InlinePatchReview()
    review.add_file("a.py", "modified")
    review.set_file_decision("a.py", "accept")
    assert review._file_decisions["a.py"] == "accept"


def test_inlinepatchreview_get_total_stats():
    from scholardevclaw.tui.widgets_new import InlinePatchReview

    review = InlinePatchReview()
    review.add_file("a.py", "modified", additions=5, deletions=2)
    review.add_file("b.py", "added", additions=10, deletions=0)
    total_adds, total_dels = review._get_total_stats()
    assert total_adds == 15
    assert total_dels == 2


def test_inlinepatchreview_messages():
    from scholardevclaw.tui.widgets_new import InlinePatchReview

    msg1 = InlinePatchReview.FileAction("test.py", "accept")
    assert msg1.file_path == "test.py"
    assert msg1.action == "accept"

    msg2 = InlinePatchReview.AllFilesAction("accept_all")
    assert msg2.action == "accept_all"
