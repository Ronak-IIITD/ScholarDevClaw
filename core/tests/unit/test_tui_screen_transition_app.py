"""Integration tests for the screen transition system.

These tests use :func:`App.run_test` to drive a real Textual app
end-to-end and verify that :class:`Widget.animate` actually mutates
the underlying widget state over time.

The unit tests in :mod:`tests.unit.test_tui_screen_transitions`
cover the pure logic of transitions; the tests in this module verify
that the integration with Textual's animation system works as
expected (e.g. opacity reaches the target after ``pilot.pause``).

We deliberately do not exercise offset animation through the
Textual animator — Textual 8.2.x has known issues animating
``(x, y)`` offsets through :meth:`Widget.animate`. The transition
helpers fall back to snapping the offset on completion, which is
covered here.
"""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from textual.app import App  # noqa: E402
from textual.containers import Vertical  # noqa: E402
from textual.screen import ModalScreen  # noqa: E402
from textual.widgets import Static  # noqa: E402

from scholardevclaw.tui.screen_transitions import (  # noqa: E402
    FADE,
    NONE,
    SLIDE_LEFT,
    animate_enter,
    animate_exit,
    apply_starting_state,
    ending_state,
    starting_state,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _TransitionScreen(ModalScreen[None]):
    """A minimal modal screen used as a target for transitions."""

    def compose(self):  # type: ignore[override]
        with Vertical():
            yield Static("transition test", id="content")


class _HarnessApp(App[None]):
    """A bare-bones :class:`App` for transition tests.

    The class is intentionally minimal — no compose, no CSS, no
    bindings — so that the test can control exactly when a screen is
    pushed and observe the resulting animation.
    """


# ---------------------------------------------------------------------------
# apply_starting_state with a real Textual widget
# ---------------------------------------------------------------------------


class TestApplyStartingStateReal:
    @pytest.mark.asyncio
    async def test_fade_enter_sets_opacity_to_zero(self) -> None:
        app = _HarnessApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _TransitionScreen()
            await app.push_screen(screen)
            await pilot.pause()
            apply_starting_state(screen, FADE, entering=True)
            # Opacity should be 0.0 after apply_starting_state.
            assert screen.opacity == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_slide_left_enter_sets_offset(self) -> None:
        app = _HarnessApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _TransitionScreen()
            await app.push_screen(screen)
            await pilot.pause()
            apply_starting_state(screen, SLIDE_LEFT, entering=True)
            # Offset should be (8, 0) — the starting position for a
            # left-slide enter.
            assert screen.offset == (8, 0)
            assert screen.opacity == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_apply_starting_state_is_idempotent_for_none(self) -> None:
        app = _HarnessApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _TransitionScreen()
            await app.push_screen(screen)
            await pilot.pause()
            apply_starting_state(screen, NONE, entering=True)
            # None transition ends at the identity state.
            assert screen.opacity == pytest.approx(1.0)
            assert screen.offset == (0, 0)


# ---------------------------------------------------------------------------
# animate_enter drives opacity to the target value
# ---------------------------------------------------------------------------


class TestAnimateEnterReal:
    @pytest.mark.asyncio
    async def test_fade_enter_completes_to_full_opacity(self) -> None:
        app = _HarnessApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _TransitionScreen()
            await app.push_screen(screen)
            await pilot.pause()

            apply_starting_state(screen, FADE, entering=True)
            assert screen.opacity == pytest.approx(0.0)

            animate_enter(screen, FADE)
            # Wait for the animation to finish (FADE is 0.18s).
            await pilot.pause(0.3)
            # Opacity should be at the target value.
            assert screen.opacity == pytest.approx(
                ending_state(FADE, entering=True)["opacity"],
                abs=0.01,
            )

    @pytest.mark.asyncio
    async def test_fade_enter_snaps_offset_at_completion(self) -> None:
        app = _HarnessApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _TransitionScreen()
            await app.push_screen(screen)
            await pilot.pause()

            apply_starting_state(screen, SLIDE_LEFT, entering=True)
            # Offset starts at the slide-left starting position.
            assert screen.offset == (8, 0)

            animate_enter(screen, SLIDE_LEFT)
            await pilot.pause(0.3)
            # Offset is snapped to the target at completion.
            assert screen.offset == ending_state(SLIDE_LEFT, entering=True)["offset"]

    @pytest.mark.asyncio
    async def test_animate_enter_for_none_is_synchronous(self) -> None:
        app = _HarnessApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _TransitionScreen()
            await app.push_screen(screen)
            await pilot.pause()

            apply_starting_state(screen, NONE, entering=True)
            animate_enter(screen, NONE)
            # None has duration 0, so no pause is required. The screen
            # is already at its target state.
            assert screen.opacity == pytest.approx(1.0)
            assert screen.offset == (0, 0)


# ---------------------------------------------------------------------------
# animate_exit drives opacity to 0 and fires on_complete
# ---------------------------------------------------------------------------


class TestAnimateExitReal:
    @pytest.mark.asyncio
    async def test_fade_exit_completes_to_zero_opacity(self) -> None:
        app = _HarnessApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _TransitionScreen()
            await app.push_screen(screen)
            await pilot.pause()

            apply_starting_state(screen, FADE, entering=False)
            animate_exit(screen, FADE)
            await pilot.pause(0.3)
            assert screen.opacity == pytest.approx(
                ending_state(FADE, entering=False)["opacity"],
                abs=0.01,
            )

    @pytest.mark.asyncio
    async def test_animate_exit_fires_on_complete(self) -> None:
        app = _HarnessApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _TransitionScreen()
            await app.push_screen(screen)
            await pilot.pause()

            fired: list[bool] = []
            apply_starting_state(screen, FADE, entering=False)
            animate_exit(screen, FADE, on_complete=lambda: fired.append(True))
            # Before the animation finishes the callback hasn't fired.
            assert fired == []
            await pilot.pause(0.3)
            # After the animation the callback has fired.
            assert fired == [True]

    @pytest.mark.asyncio
    async def test_animate_exit_for_none_fires_on_complete_synchronously(
        self,
    ) -> None:
        app = _HarnessApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _TransitionScreen()
            await app.push_screen(screen)
            await pilot.pause()

            fired: list[bool] = []
            animate_exit(screen, NONE, on_complete=lambda: fired.append(True))
            # The callback should have fired without a pause.
            assert fired == [True]


# ---------------------------------------------------------------------------
# starting_state / ending_state with a real widget
# ---------------------------------------------------------------------------


class TestStateReal:
    @pytest.mark.asyncio
    async def test_starting_state_for_fade_enter(self) -> None:
        app = _HarnessApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _TransitionScreen()
            await app.push_screen(screen)
            await pilot.pause()
            state = starting_state(FADE, entering=True)
            assert state["offset"] == (0, 0)
            assert state["opacity"] == pytest.approx(0.0)
            # Sanity check: applying this state actually changes the
            # screen.
            apply_starting_state(screen, FADE, entering=True)
            assert screen.opacity == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_starting_state_for_slide_left_enter(self) -> None:
        app = _HarnessApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _TransitionScreen()
            await app.push_screen(screen)
            await pilot.pause()
            state = starting_state(SLIDE_LEFT, entering=True)
            assert state["offset"] == (8, 0)
            assert state["opacity"] == pytest.approx(0.0)
            apply_starting_state(screen, SLIDE_LEFT, entering=True)
            assert screen.offset == (8, 0)
            assert screen.opacity == pytest.approx(0.0)
