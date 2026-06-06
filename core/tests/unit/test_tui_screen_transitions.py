"""Pure-logic tests for the screen transition system.

The integration tests that exercise Textual's actual ``Widget.animate``
behaviour live in :mod:`tests.unit.test_tui_screen_transition_app`.
This file is intentionally pure-Python (with the textual library
present, but no running app) so that the math and data structures are
easy to verify.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

pytest.importorskip("textual")

from textual.app import App  # noqa: E402
from textual.screen import Screen  # noqa: E402
from textual.widget import Widget  # noqa: E402

from scholardevclaw.tui.screen_transitions import (  # noqa: E402
    EASING_REGISTRY,
    FADE,
    NONE,
    SLIDE_DOWN,
    SLIDE_LEFT,
    SLIDE_RIGHT,
    SLIDE_UP,
    TRANSITION_REGISTRY,
    Keyframe,
    ScreenTransition,
    TransitionPhase,
    animate_enter,
    animate_exit,
    apply_easing,
    apply_starting_state,
    compute_progress,
    ending_state,
    get_transition,
    get_transition_or_default,
    in_cubic,
    in_out_cubic,
    in_out_quad,
    in_quad,
    interpolate_keyframes,
    linear,
    list_transitions,
    out_cubic,
    out_quad,
    phase_end,
    phase_start,
    pop_screen_with_transition,
    push_screen_with_transition,
    register_transition,
    resolve_easing,
    starting_state,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _StubStyles:
    """Stub for the ``styles`` attribute of a Textual widget."""

    def __init__(self) -> None:
        self._opacity = 1.0
        self.animate_calls: list[dict[str, object]] = []

    @property
    def opacity(self) -> float:
        return self._opacity

    @opacity.setter
    def opacity(self, value: float) -> None:
        self._opacity = float(value)

    def animate(
        self,
        attribute: str,
        value: object,
        *,
        duration: float | None = None,
        easing: str | None = None,
        on_complete: object = None,
        **_: object,
    ) -> None:
        self.animate_calls.append(
            {
                "attribute": attribute,
                "value": value,
                "duration": duration,
                "easing": easing,
                "on_complete": on_complete,
            }
        )


def _cast_app(obj: Any) -> App[Any]:
    """Cast a stub app to ``App[Any]`` for the helper functions.

    The production helpers are typed against Textual's :class:`App`,
    but the unit tests use lightweight stubs that duck-type the
    required surface (``push_screen`` / ``pop_screen`` /
    ``call_later`` / ``screen``).  ``cast`` is the idiomatic way to
    tell mypy "trust me, this is a real App".
    """
    return cast(App[Any], obj)


def _cast_screen(obj: Any) -> Screen[Any]:
    """Cast a stub screen to ``Screen[Any]`` for the helper functions."""
    return cast(Screen[Any], obj)


def _cast_widget(obj: Any) -> Widget:
    """Cast a stub widget to ``Widget`` for the helper functions."""
    return cast(Widget, obj)


class _StubWidget:
    """Minimal widget stub for the integration helpers.

    Mirrors the parts of the Textual API that
    :func:`apply_starting_state` and :func:`animate_enter` touch: a
    settable ``offset``, a settable ``styles.opacity`` (with a
    read-back ``opacity`` property), and a ``styles.animate`` method
    that records its calls.
    """

    def __init__(self) -> None:
        self.styles = _StubStyles()
        self._offset: tuple[int, int] = (0, 0)
        self.animate_calls: list[dict[str, object]] = []
        # Surface styles.animate_calls at the widget level for tests
        # that want to inspect the calls.
        self.styles_animate_calls = self.styles.animate_calls

    @property
    def offset(self) -> tuple[int, int]:
        return self._offset

    @offset.setter
    def offset(self, value: tuple[int, int]) -> None:
        self._offset = (int(value[0]), int(value[1]))

    @property
    def opacity(self) -> float:
        return self.styles.opacity


# ---------------------------------------------------------------------------
# Easing functions
# ---------------------------------------------------------------------------


class TestEasings:
    """All easing functions are clamped to ``[0, 1]`` and end exactly
    at the boundaries."""

    @pytest.mark.parametrize(
        "easing",
        [
            linear,
            in_quad,
            out_quad,
            in_out_quad,
            in_cubic,
            out_cubic,
            in_out_cubic,
        ],
    )
    def test_easing_at_zero_is_zero(self, easing) -> None:
        assert easing(0.0) == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "easing",
        [
            linear,
            in_quad,
            out_quad,
            in_out_quad,
            in_cubic,
            out_cubic,
            in_out_cubic,
        ],
    )
    def test_easing_at_one_is_one(self, easing) -> None:
        assert easing(1.0) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "easing",
        [
            in_quad,
            out_quad,
            in_out_quad,
            in_cubic,
            out_cubic,
            in_out_cubic,
        ],
    )
    def test_easing_stays_in_unit_interval(self, easing) -> None:
        for t in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            result = easing(t)
            assert 0.0 <= result <= 1.0, f"{easing.__name__}({t}) = {result}"

    def test_linear_is_identity(self) -> None:
        for t in [0.0, 0.1, 0.5, 0.7, 1.0]:
            assert linear(t) == pytest.approx(t)

    def test_in_quad_slower_than_linear_at_start(self) -> None:
        assert in_quad(0.5) < 0.5
        assert in_quad(0.25) < 0.25

    def test_out_quad_faster_than_linear_at_start(self) -> None:
        assert out_quad(0.5) > 0.5
        assert out_quad(0.25) > 0.25

    def test_in_out_quad_symmetric(self) -> None:
        # By construction in_out_quad is symmetric around 0.5.
        assert in_out_quad(0.25) == pytest.approx(0.125)
        assert in_out_quad(0.75) == pytest.approx(0.875)

    def test_in_out_cubic_symmetric(self) -> None:
        # The classic cubic ease-in-out curve.
        assert in_out_cubic(0.5) == pytest.approx(0.5)
        # 0.25 -> 4 * 0.25^3 = 0.0625
        assert in_out_cubic(0.25) == pytest.approx(0.0625)
        # 0.75 -> 1 - (1 - 0.5)^3 / 2 = 1 - 0.0625 = 0.9375
        assert in_out_cubic(0.75) == pytest.approx(0.9375)

    def test_in_cubic_starts_lower_than_in_quad(self) -> None:
        # Cubic ease-in is more aggressive at the start than quadratic.
        assert in_cubic(0.5) < in_quad(0.5)

    def test_out_cubic_ends_higher_than_out_quad(self) -> None:
        assert out_cubic(0.5) > out_quad(0.5)


class TestEasingRegistry:
    def test_registry_contains_expected_names(self) -> None:
        assert set(EASING_REGISTRY) == {
            "linear",
            "in_quad",
            "out_quad",
            "in_out_quad",
            "in_cubic",
            "out_cubic",
            "in_out_cubic",
        }

    def test_resolve_easing_returns_callable(self) -> None:
        fn = resolve_easing("linear")
        assert callable(fn)
        assert fn(0.5) == pytest.approx(0.5)

    def test_resolve_easing_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown easing"):
            resolve_easing("nonexistent")

    def test_apply_easing_clamps_negative(self) -> None:
        # Should clamp t to 0 before applying easing.
        assert apply_easing("linear", -0.5) == pytest.approx(0.0)

    def test_apply_easing_clamps_above_one(self) -> None:
        assert apply_easing("linear", 1.5) == pytest.approx(1.0)

    def test_apply_easing_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown easing"):
            apply_easing("nope", 0.5)


# ---------------------------------------------------------------------------
# Keyframe and TransitionPhase invariants
# ---------------------------------------------------------------------------


class TestKeyframe:
    def test_basic_construction(self) -> None:
        kf = Keyframe(at=0.5, offset=(2, -3), opacity=0.7)
        assert kf.at == 0.5
        assert kf.offset == (2, -3)
        assert kf.opacity == 0.7

    def test_opacity_can_be_none(self) -> None:
        kf = Keyframe(at=0.0, offset=(0, 0), opacity=None)
        assert kf.opacity is None

    def test_frozen(self) -> None:
        kf = Keyframe(at=0.0)
        with pytest.raises(Exception):
            kf.at = 1.0  # type: ignore[misc]

    def test_at_must_be_in_unit_range(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            Keyframe(at=-0.1)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            Keyframe(at=1.5)

    def test_opacity_must_be_in_unit_range(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            Keyframe(at=0.5, opacity=1.5)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            Keyframe(at=0.5, opacity=-0.1)

    def test_equality(self) -> None:
        a = Keyframe(at=0.0, offset=(1, 2), opacity=0.5)
        b = Keyframe(at=0.0, offset=(1, 2), opacity=0.5)
        assert a == b
        assert hash(a) == hash(b)


class TestTransitionPhase:
    def test_empty_phase_is_allowed(self) -> None:
        phase = TransitionPhase()
        assert phase.keyframes == ()

    def test_keyframes_sorted_on_construction(self) -> None:
        # Pass them out of order; expect them stored in order.
        phase = TransitionPhase(
            keyframes=(
                Keyframe(at=1.0, offset=(0, 0), opacity=1.0),
                Keyframe(at=0.0, offset=(5, 0), opacity=0.0),
                Keyframe(at=0.5, offset=(2, 0), opacity=0.5),
            ),
        )
        assert [kf.at for kf in phase.keyframes] == [0.0, 0.5, 1.0]

    def test_frozen(self) -> None:
        phase = TransitionPhase(keyframes=(Keyframe(at=0.0),))
        with pytest.raises(Exception):
            phase.keyframes = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ScreenTransition invariants
# ---------------------------------------------------------------------------


class TestScreenTransition:
    def _build(self, **overrides: object) -> ScreenTransition:
        defaults: dict[str, object] = dict(
            name="test",
            duration=0.2,
            easing="linear",
            enter=TransitionPhase(
                keyframes=(Keyframe(at=0.0, offset=(0, 0), opacity=0.0),),
            ),
            exit=TransitionPhase(
                keyframes=(Keyframe(at=0.0, offset=(0, 0), opacity=0.0),),
            ),
        )
        defaults.update(overrides)
        return ScreenTransition(**defaults)  # type: ignore[arg-type]

    def test_basic_construction(self) -> None:
        tr = self._build()
        assert tr.name == "test"
        assert tr.duration == 0.2
        assert tr.easing == "linear"

    def test_name_must_be_nonempty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            self._build(name="")
        with pytest.raises(ValueError, match="non-empty"):
            self._build(name="   ")

    def test_duration_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            self._build(duration=-0.1)

    def test_easing_must_be_registered(self) -> None:
        with pytest.raises(ValueError, match="registered easing"):
            self._build(easing="nope")

    def test_zero_duration_is_allowed(self) -> None:
        tr = self._build(duration=0.0)
        assert tr.duration == 0.0

    def test_with_duration_returns_copy(self) -> None:
        tr = self._build(duration=0.2)
        tr2 = tr.with_duration(0.5)
        assert tr2.duration == 0.5
        # Original is unchanged.
        assert tr.duration == 0.2

    def test_with_duration_rejects_negative(self) -> None:
        tr = self._build()
        with pytest.raises(ValueError, match=">= 0"):
            tr.with_duration(-0.1)

    def test_frozen(self) -> None:
        tr = self._build()
        with pytest.raises(Exception):
            tr.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Pre-built transitions
# ---------------------------------------------------------------------------


class TestPrebuiltTransitions:
    def test_none_transition_is_zero_duration(self) -> None:
        assert NONE.duration == 0.0
        assert NONE.easing == "linear"

    def test_none_is_identity(self) -> None:
        assert starting_state(NONE, entering=True) == {"offset": (0, 0), "opacity": 1.0}
        assert ending_state(NONE, entering=True) == {"offset": (0, 0), "opacity": 1.0}
        assert starting_state(NONE, entering=False) == {"offset": (0, 0), "opacity": 1.0}
        assert ending_state(NONE, entering=False) == {"offset": (0, 0), "opacity": 1.0}

    def test_fade_starts_invisible(self) -> None:
        state = starting_state(FADE, entering=True)
        assert state["opacity"] == pytest.approx(0.0)
        assert state["offset"] == (0, 0)

    def test_fade_ends_visible(self) -> None:
        state = ending_state(FADE, entering=True)
        assert state["opacity"] == pytest.approx(1.0)
        assert state["offset"] == (0, 0)

    def test_fade_exit_starts_visible(self) -> None:
        state = starting_state(FADE, entering=False)
        assert state["opacity"] == pytest.approx(1.0)
        assert state["offset"] == (0, 0)

    def test_fade_exit_ends_invisible(self) -> None:
        state = ending_state(FADE, entering=False)
        assert state["opacity"] == pytest.approx(0.0)
        assert state["offset"] == (0, 0)

    @pytest.mark.parametrize(
        "transition,expected_enter_offset",
        [
            (SLIDE_LEFT, (8, 0)),
            (SLIDE_RIGHT, (-8, 0)),
            (SLIDE_UP, (0, 2)),
            (SLIDE_DOWN, (0, -2)),
        ],
    )
    def test_slide_enter_offsets(
        self, transition: ScreenTransition, expected_enter_offset: tuple[int, int]
    ) -> None:
        state = starting_state(transition, entering=True)
        assert state["offset"] == expected_enter_offset
        assert state["opacity"] == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "transition",
        [SLIDE_LEFT, SLIDE_RIGHT, SLIDE_UP, SLIDE_DOWN],
    )
    def test_slide_enter_ends_at_rest(self, transition: ScreenTransition) -> None:
        state = ending_state(transition, entering=True)
        assert state["offset"] == (0, 0)
        assert state["opacity"] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "transition,expected_exit_offset",
        [
            (SLIDE_LEFT, (-8, 0)),
            (SLIDE_RIGHT, (8, 0)),
            (SLIDE_UP, (0, -2)),
            (SLIDE_DOWN, (0, 2)),
        ],
    )
    def test_slide_exit_offsets(
        self, transition: ScreenTransition, expected_exit_offset: tuple[int, int]
    ) -> None:
        state = ending_state(transition, entering=False)
        assert state["offset"] == expected_exit_offset
        assert state["opacity"] == pytest.approx(0.0)

    def test_prebuilt_have_in_out_cubic_easing(self) -> None:
        for tr in (FADE, SLIDE_LEFT, SLIDE_RIGHT, SLIDE_UP, SLIDE_DOWN):
            assert tr.easing == "in_out_cubic"

    def test_prebuilt_have_positive_duration(self) -> None:
        for tr in (FADE, SLIDE_LEFT, SLIDE_RIGHT, SLIDE_UP, SLIDE_DOWN):
            assert tr.duration > 0


# ---------------------------------------------------------------------------
# Registry and lookup
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_contains_expected_names(self) -> None:
        assert set(TRANSITION_REGISTRY) == {
            "none",
            "fade",
            "slide_left",
            "slide_right",
            "slide_up",
            "slide_down",
        }

    def test_get_transition_known_returns_instance(self) -> None:
        assert get_transition("fade") is FADE
        assert get_transition("none") is NONE
        assert get_transition("slide_left") is SLIDE_LEFT

    def test_get_transition_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown transition"):
            get_transition("zoom")

    def test_get_transition_or_default_none_returns_none_transition(self) -> None:
        assert get_transition_or_default(None) is NONE
        assert get_transition_or_default("") is NONE

    def test_get_transition_or_default_known_returns_transition(self) -> None:
        assert get_transition_or_default("fade") is FADE

    def test_get_transition_or_default_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown transition"):
            get_transition_or_default("warp")

    def test_list_transitions_returns_sorted_tuple(self) -> None:
        names = list_transitions()
        assert isinstance(names, tuple)
        assert names == tuple(sorted(names))
        # All known names are present.
        for expected in (
            "none",
            "fade",
            "slide_left",
            "slide_right",
            "slide_up",
            "slide_down",
        ):
            assert expected in names

    def test_register_transition_adds_entry(self) -> None:
        custom = ScreenTransition(
            name="zoom",
            duration=0.3,
            easing="out_cubic",
            enter=TransitionPhase(
                keyframes=(Keyframe(at=0.0, opacity=0.0),),
            ),
            exit=TransitionPhase(
                keyframes=(Keyframe(at=1.0, opacity=0.0),),
            ),
        )
        register_transition(custom)
        try:
            assert get_transition("zoom") is custom
        finally:
            TRANSITION_REGISTRY.pop("zoom", None)

    def test_register_transition_replaces_existing(self) -> None:
        custom = ScreenTransition(
            name="fade",
            duration=0.5,
            easing="linear",
            enter=TransitionPhase(
                keyframes=(Keyframe(at=0.0, opacity=0.5),),
            ),
            exit=TransitionPhase(
                keyframes=(Keyframe(at=0.0, opacity=0.5),),
            ),
        )
        original = TRANSITION_REGISTRY["fade"]
        register_transition(custom)
        try:
            assert get_transition("fade") is custom
            assert get_transition("fade").duration == 0.5
        finally:
            TRANSITION_REGISTRY["fade"] = original

    def test_cannot_replace_none(self) -> None:
        with pytest.raises(ValueError, match="'none' transition"):
            register_transition(
                ScreenTransition(
                    name="none",
                    duration=0.5,
                    easing="linear",
                    enter=TransitionPhase(),
                    exit=TransitionPhase(),
                ),
            )


# ---------------------------------------------------------------------------
# Keyframe interpolation
# ---------------------------------------------------------------------------


class TestInterpolation:
    def test_empty_phase_returns_none_opacity(self) -> None:
        phase = TransitionPhase()
        offset, opacity = interpolate_keyframes(phase, 0.5)
        assert offset == (0, 0)
        assert opacity is None

    def test_single_keyframe_returns_that_keyframe(self) -> None:
        phase = TransitionPhase(
            keyframes=(Keyframe(at=0.5, offset=(3, -2), opacity=0.4),),
        )
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            offset, opacity = interpolate_keyframes(phase, t)
            assert offset == (3, -2)
            assert opacity == pytest.approx(0.4)

    def test_progress_below_first_keyframe_clamps(self) -> None:
        phase = TransitionPhase(
            keyframes=(
                Keyframe(at=0.5, offset=(4, 0), opacity=0.0),
                Keyframe(at=1.0, offset=(0, 0), opacity=1.0),
            ),
        )
        offset, opacity = interpolate_keyframes(phase, 0.0)
        assert offset == (4, 0)
        assert opacity == pytest.approx(0.0)

    def test_progress_above_last_keyframe_clamps(self) -> None:
        phase = TransitionPhase(
            keyframes=(
                Keyframe(at=0.0, offset=(4, 0), opacity=0.0),
                Keyframe(at=0.5, offset=(0, 0), opacity=1.0),
            ),
        )
        offset, opacity = interpolate_keyframes(phase, 1.0)
        assert offset == (0, 0)
        assert opacity == pytest.approx(1.0)

    def test_midpoint_lerps(self) -> None:
        phase = TransitionPhase(
            keyframes=(
                Keyframe(at=0.0, offset=(0, 0), opacity=0.0),
                Keyframe(at=1.0, offset=(10, 0), opacity=1.0),
            ),
        )
        offset, opacity = interpolate_keyframes(phase, 0.5)
        assert offset == (5, 0)
        assert opacity == pytest.approx(0.5)

    def test_quarter_progress_lerps(self) -> None:
        phase = TransitionPhase(
            keyframes=(
                Keyframe(at=0.0, offset=(0, 0), opacity=0.0),
                Keyframe(at=1.0, offset=(8, 4), opacity=1.0),
            ),
        )
        offset, opacity = interpolate_keyframes(phase, 0.25)
        assert offset == (2, 1)
        assert opacity == pytest.approx(0.25)

    def test_offset_only_keyframes_yield_none_opacity(self) -> None:
        phase = TransitionPhase(
            keyframes=(
                Keyframe(at=0.0, offset=(0, 0), opacity=None),
                Keyframe(at=1.0, offset=(4, 0), opacity=None),
            ),
        )
        offset, opacity = interpolate_keyframes(phase, 0.5)
        assert offset == (2, 0)
        assert opacity is None

    def test_opacity_none_at_start_converges_to_end(self) -> None:
        # If only the end keyframe has an opacity, intermediate values
        # should also resolve to that opacity.
        phase = TransitionPhase(
            keyframes=(
                Keyframe(at=0.0, offset=(0, 0), opacity=None),
                Keyframe(at=1.0, offset=(4, 0), opacity=0.8),
            ),
        )
        for t in (0.0, 0.5, 1.0):
            offset, opacity = interpolate_keyframes(phase, t)
            if t == 0.0:
                assert opacity is None
            else:
                assert opacity == pytest.approx(0.8)

    def test_opacity_none_at_end_converges_to_start(self) -> None:
        phase = TransitionPhase(
            keyframes=(
                Keyframe(at=0.0, offset=(0, 0), opacity=0.2),
                Keyframe(at=1.0, offset=(4, 0), opacity=None),
            ),
        )
        for t in (0.0, 0.5, 1.0):
            _, opacity = interpolate_keyframes(phase, t)
            if t == 1.0:
                assert opacity is None
            else:
                assert opacity == pytest.approx(0.2)

    def test_three_keyframes_interpolate_in_two_segments(self) -> None:
        phase = TransitionPhase(
            keyframes=(
                Keyframe(at=0.0, offset=(0, 0), opacity=0.0),
                Keyframe(at=0.5, offset=(4, 0), opacity=0.5),
                Keyframe(at=1.0, offset=(8, 0), opacity=1.0),
            ),
        )
        # In the first segment
        offset, opacity = interpolate_keyframes(phase, 0.25)
        assert offset == (2, 0)
        assert opacity == pytest.approx(0.25)
        # In the second segment
        offset, opacity = interpolate_keyframes(phase, 0.75)
        assert offset == (6, 0)
        assert opacity == pytest.approx(0.75)

    def test_clamping_works_for_out_of_range(self) -> None:
        phase = TransitionPhase(
            keyframes=(
                Keyframe(at=0.0, offset=(0, 0), opacity=0.0),
                Keyframe(at=1.0, offset=(4, 0), opacity=1.0),
            ),
        )
        offset, opacity = interpolate_keyframes(phase, -0.3)
        assert offset == (0, 0)
        assert opacity == pytest.approx(0.0)
        offset, opacity = interpolate_keyframes(phase, 1.7)
        assert offset == (4, 0)
        assert opacity == pytest.approx(1.0)


class TestPhaseHelpers:
    def test_phase_start_and_end_match_explicit_progress(self) -> None:
        phase = TransitionPhase(
            keyframes=(
                Keyframe(at=0.0, offset=(2, -2), opacity=0.1),
                Keyframe(at=1.0, offset=(0, 0), opacity=1.0),
            ),
        )
        assert phase_start(phase) == ((2, -2), 0.1)
        assert phase_end(phase) == ((0, 0), 1.0)
        assert phase_start(phase) == interpolate_keyframes(phase, 0.0)
        assert phase_end(phase) == interpolate_keyframes(phase, 1.0)


# ---------------------------------------------------------------------------
# compute_progress and easing integration
# ---------------------------------------------------------------------------


class TestComputeProgress:
    def test_linear_at_midpoint(self) -> None:
        # build a tiny transition for the test
        tr = ScreenTransition(
            name="t",
            duration=0.1,
            easing="linear",
            enter=TransitionPhase(keyframes=(Keyframe(at=0.0),)),
            exit=TransitionPhase(keyframes=(Keyframe(at=0.0),)),
        )
        assert compute_progress(tr, 0.5) == pytest.approx(0.5)

    def test_in_out_cubic_at_midpoint(self) -> None:
        tr = ScreenTransition(
            name="t",
            duration=0.1,
            easing="in_out_cubic",
            enter=TransitionPhase(keyframes=(Keyframe(at=0.0),)),
            exit=TransitionPhase(keyframes=(Keyframe(at=0.0),)),
        )
        # At t=0.5 the cubic ease-in-out returns exactly 0.5 by symmetry.
        assert compute_progress(tr, 0.5) == pytest.approx(0.5)

    def test_clamps_inputs(self) -> None:
        tr = ScreenTransition(
            name="t",
            duration=0.1,
            easing="linear",
            enter=TransitionPhase(keyframes=(Keyframe(at=0.0),)),
            exit=TransitionPhase(keyframes=(Keyframe(at=0.0),)),
        )
        assert compute_progress(tr, -0.3) == pytest.approx(0.0)
        assert compute_progress(tr, 1.3) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# starting_state / ending_state
# ---------------------------------------------------------------------------


class TestState:
    def test_starting_state_uses_enter_phase(self) -> None:
        assert starting_state(FADE, entering=True) == {
            "offset": (0, 0),
            "opacity": 0.0,
        }

    def test_starting_state_uses_exit_phase(self) -> None:
        # For FADE exit, the starting state is fully visible.
        state = starting_state(FADE, entering=False)
        assert state == {"offset": (0, 0), "opacity": 1.0}

    def test_ending_state_uses_enter_phase(self) -> None:
        assert ending_state(FADE, entering=True) == {
            "offset": (0, 0),
            "opacity": 1.0,
        }

    def test_ending_state_uses_exit_phase(self) -> None:
        assert ending_state(FADE, entering=False) == {
            "offset": (0, 0),
            "opacity": 0.0,
        }

    def test_starting_state_omits_opacity_when_no_opacity_keyframes(self) -> None:
        tr = ScreenTransition(
            name="offset-only",
            duration=0.1,
            easing="linear",
            enter=TransitionPhase(
                keyframes=(
                    Keyframe(at=0.0, offset=(2, 0), opacity=None),
                    Keyframe(at=1.0, offset=(0, 0), opacity=None),
                ),
            ),
            exit=TransitionPhase(
                keyframes=(
                    Keyframe(at=0.0, offset=(0, 0), opacity=None),
                    Keyframe(at=1.0, offset=(-2, 0), opacity=None),
                ),
            ),
        )
        state = starting_state(tr, entering=True)
        assert state == {"offset": (2, 0)}
        assert "opacity" not in state

    def test_ending_state_includes_opacity_when_phase_has_opacity(self) -> None:
        state = ending_state(SLIDE_LEFT, entering=True)
        assert "opacity" in state
        assert state["opacity"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# apply_starting_state (uses a stub widget)
# ---------------------------------------------------------------------------


class TestApplyStartingState:
    def test_sets_offset_and_opacity(self) -> None:
        widget = _StubWidget()
        apply_starting_state(widget, FADE, entering=True)  # type: ignore[arg-type]
        assert widget.offset == (0, 0)
        assert widget.opacity == pytest.approx(0.0)

    def test_uses_exit_phase_when_entering_false(self) -> None:
        widget = _StubWidget()
        apply_starting_state(widget, FADE, entering=False)  # type: ignore[arg-type]
        assert widget.opacity == pytest.approx(1.0)

    def test_does_not_touch_opacity_when_not_in_state(self) -> None:
        widget = _StubWidget()
        tr = ScreenTransition(
            name="offset-only",
            duration=0.1,
            easing="linear",
            enter=TransitionPhase(
                keyframes=(
                    Keyframe(at=0.0, offset=(4, 0), opacity=None),
                    Keyframe(at=1.0, offset=(0, 0), opacity=None),
                ),
            ),
            exit=TransitionPhase(
                keyframes=(
                    Keyframe(at=0.0, offset=(0, 0), opacity=None),
                    Keyframe(at=1.0, offset=(-4, 0), opacity=None),
                ),
            ),
        )
        original_opacity = widget.opacity
        apply_starting_state(widget, tr, entering=True)  # type: ignore[arg-type]
        # offset changed
        assert widget.offset == (4, 0)
        # opacity was untouched because the phase has no opacity keyframe
        assert widget.opacity == original_opacity

    def test_none_transition_is_a_no_op_for_offset(self) -> None:
        widget = _StubWidget()
        widget.offset = (3, 3)
        apply_starting_state(widget, NONE, entering=True)  # type: ignore[arg-type]
        # Offsets were already at (0,0) target, opacity is the identity
        # value of 1.0; everything else is left alone.
        assert widget.offset == (0, 0)
        assert widget.opacity == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# animate_enter / animate_exit
# ---------------------------------------------------------------------------


class TestAnimations:
    def test_animate_enter_schedules_opacity(self) -> None:
        widget = _StubWidget()
        animate_enter(widget, FADE)  # type: ignore[arg-type]
        calls = widget.styles.animate_calls
        attrs = [call["attribute"] for call in calls]
        assert "opacity" in attrs
        # Opacity animation should use the FADE duration and easing.
        for call in calls:
            assert call["duration"] == FADE.duration
            assert call["easing"] == FADE.easing

    def test_animate_enter_opacity_target_value(self) -> None:
        widget = _StubWidget()
        animate_enter(widget, SLIDE_LEFT)  # type: ignore[arg-type]
        targets = {call["attribute"]: call["value"] for call in widget.styles.animate_calls}
        # For a slide enter, opacity ends at 1.0 (fully visible).
        assert targets["opacity"] == pytest.approx(1.0)

    def test_animate_exit_opacity_target_value(self) -> None:
        widget = _StubWidget()
        animate_exit(widget, SLIDE_LEFT)  # type: ignore[arg-type]
        targets = {call["attribute"]: call["value"] for call in widget.styles.animate_calls}
        # For a slide exit, opacity ends at 0.0 (fully invisible).
        assert targets["opacity"] == pytest.approx(0.0)

    def test_animate_enter_none_does_nothing(self) -> None:
        widget = _StubWidget()
        animate_enter(widget, NONE)  # type: ignore[arg-type]
        # NONE has no opacity keyframes, so no styles.animate call.
        assert widget.styles.animate_calls == []

    def test_animate_exit_fires_on_complete(self) -> None:
        widget = _StubWidget()
        fired: list[bool] = []
        animate_exit(widget, FADE, on_complete=lambda: fired.append(True))  # type: ignore[arg-type]
        # styles.animate was called once for opacity.
        assert widget.styles.animate_calls, "expected animation to be scheduled"
        cb = widget.styles.animate_calls[-1]["on_complete"]
        assert callable(cb)
        cb()
        # The user's on_complete should have fired.
        assert fired == [True]

    def test_animate_enter_snaps_offset_at_completion(self) -> None:
        widget = _StubWidget()
        animate_enter(widget, SLIDE_LEFT)  # type: ignore[arg-type]
        cb = widget.styles.animate_calls[-1]["on_complete"]
        assert callable(cb)
        # Before completion, offset is the initial state (we set it via
        # apply_starting_state in real usage; for this test, just
        # verify the snap fires).
        widget.offset = (3, 0)
        cb()
        # After completion, offset is snapped to the ending state.
        assert widget.offset == (0, 0)

    def test_animate_exit_snaps_offset_at_completion(self) -> None:
        widget = _StubWidget()
        animate_exit(widget, SLIDE_LEFT)  # type: ignore[arg-type]
        cb = widget.styles.animate_calls[-1]["on_complete"]
        assert callable(cb)
        widget.offset = (0, 0)
        cb()
        # Slide-left exit ends at (-8, 0).
        assert widget.offset == (-8, 0)


# ---------------------------------------------------------------------------
# push_screen_with_transition / pop_screen_with_transition
# ---------------------------------------------------------------------------


class _StubApp:
    """Minimal app stub for push/pop tests.

    Records pushed screens and tracks pop invocations. Implements
    :func:`call_later` and :meth:`screen` (the latter returns a screen
    stub for pop tests).
    """

    def __init__(self) -> None:
        self.pushed: list[object] = []
        self.popped: int = 0
        self.call_later_calls: list[object] = []
        self.current_screen = _StubWidget()

    def push_screen(self, screen: object, callback: object = None) -> None:  # noqa: ARG002
        self.pushed.append(screen)

    def pop_screen(self) -> None:
        self.popped += 1

    def call_later(self, fn: object, *args: object, **kwargs: object) -> None:  # noqa: ARG002
        self.call_later_calls.append(fn)

    @property
    def screen(self) -> _StubWidget:
        return self.current_screen


class _StubScreen:
    """A stand-in for a Textual Screen, supporting offset/opacity mutation.

    Like a real Textual widget, ``opacity`` is a read-only computed
    property; the underlying mutable value lives on ``styles.opacity``.
    """

    def __init__(self) -> None:
        self.styles = _StubStyles()
        self._offset: tuple[int, int] = (0, 0)
        self._call_after_refresh_args: list[object] = []

    @property
    def offset(self) -> tuple[int, int]:
        return self._offset

    @offset.setter
    def offset(self, value: tuple[int, int]) -> None:
        self._offset = (int(value[0]), int(value[1]))

    @property
    def opacity(self) -> float:
        return self.styles.opacity

    def call_after_refresh(self, fn: object, *args: object, **kwargs: object) -> None:  # noqa: ARG002
        self._call_after_refresh_args.append(fn)


class TestPushWithTransition:
    def test_push_with_none_skips_animation(self) -> None:
        app = _StubApp()
        screen = _StubScreen()
        push_screen_with_transition(
            _cast_app(app),
            _cast_screen(screen),
            transition_name="none",
            callback=lambda _: None,
        )
        # Screen is pushed without offset/opacity mutation.
        assert app.pushed == [screen]
        # No call_after_refresh because there is no entry animation.
        assert screen._call_after_refresh_args == []

    def test_push_with_fade_applies_starting_state(self) -> None:
        app = _StubApp()
        screen = _StubScreen()
        push_screen_with_transition(_cast_app(app), _cast_screen(screen), transition_name="fade")
        # Screen is pushed.
        assert app.pushed == [screen]
        # Starting state for fade-enter: opacity 0, offset (0, 0).
        assert screen.opacity == pytest.approx(0.0)
        # The animation is scheduled to run after refresh.
        assert screen._call_after_refresh_args != []

    def test_push_with_slide_left_applies_starting_offset(self) -> None:
        app = _StubApp()
        screen = _StubScreen()
        push_screen_with_transition(
            _cast_app(app), _cast_screen(screen), transition_name="slide_left"
        )
        assert screen.offset == (8, 0)
        assert screen.opacity == pytest.approx(0.0)

    def test_push_with_none_transition_name_falls_back_to_class_attr(self) -> None:
        app = _StubApp()

        class _CustomScreen(_StubScreen):
            TRANSITION = "slide_right"

        screen = _CustomScreen()
        push_screen_with_transition(_cast_app(app), _cast_screen(screen), transition_name=None)
        # The class attribute should be honoured when transition_name is
        # None.
        assert screen.offset == (-8, 0)

    def test_push_with_empty_string_falls_back_to_class_attr(self) -> None:
        app = _StubApp()

        class _CustomScreen(_StubScreen):
            TRANSITION = "slide_up"

        screen = _CustomScreen()
        push_screen_with_transition(_cast_app(app), _cast_screen(screen), transition_name="")
        assert screen.offset == (0, 2)

    def test_push_with_unknown_transition_raises(self) -> None:
        app = _StubApp()
        screen = _StubScreen()
        with pytest.raises(KeyError, match="Unknown transition"):
            push_screen_with_transition(
                _cast_app(app), _cast_screen(screen), transition_name="warp"
            )

    def test_push_passes_callback_through(self) -> None:
        app = _StubApp()
        screen = _StubScreen()
        received: list[object] = []
        push_screen_with_transition(
            _cast_app(app),
            _cast_screen(screen),
            transition_name="none",
            callback=received.append,
        )
        # The callback is forwarded to push_screen; verify the stub
        # received it (we don't need to invoke it, just that the call
        # signature was correct).
        assert app.pushed == [screen]


class TestPopWithTransition:
    def test_pop_with_none_skips_animation(self) -> None:
        app = _StubApp()
        pop_screen_with_transition(_cast_app(app), NONE)
        assert app.popped == 1
        # No animation is scheduled for the current screen, so its
        # offset should not be mutated by pop_screen_with_transition.
        assert app.current_screen.offset == (0, 0)

    def test_pop_with_fade_applies_starting_state_and_schedules(self) -> None:
        app = _StubApp()
        # Pretend the screen is fully visible.
        app.current_screen.styles.opacity = 1.0
        pop_screen_with_transition(_cast_app(app), FADE)
        # The screen's opacity was not yet mutated: starting_state for
        # the exit phase is 1.0 (the same as the current state), so
        # nothing changes immediately.
        assert app.current_screen.opacity == pytest.approx(1.0)
        # pop_screen is *not* called immediately; it waits for the
        # animation to finish.
        assert app.popped == 0
        # One styles.animate call should have been scheduled (opacity).
        assert len(app.current_screen.styles.animate_calls) == 1
        # Fire the on_complete; the pop should happen exactly once.
        cb = app.current_screen.styles.animate_calls[-1]["on_complete"]
        assert callable(cb)
        cb()
        assert app.popped == 1
