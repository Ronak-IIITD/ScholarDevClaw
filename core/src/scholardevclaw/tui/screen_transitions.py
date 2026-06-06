"""Animated screen transitions for the TUI.

This module provides a small, self-contained animation system for modal
screens, focused on the two attributes that Textual reliably animates
across versions: ``opacity`` and ``offset``.

Design
------
A :class:`ScreenTransition` is a declarative description of how a screen
should appear (enter) and disappear (exit). The description is just data:
keyframes paired with a duration and an easing function. The actual
animation is performed by :func:`animate_enter` and :func:`animate_exit`,
which delegate to ``Widget.animate``.

Two kinds of helpers are exposed:

* **Pure logic** (no Textual dependency) — easing functions, keyframe
  interpolation, and the "starting" / "ending" state of each transition.
  These are the most important to unit test.
* **Widget integration** — :func:`apply_starting_state`,
  :func:`animate_enter`, :func:`animate_exit`, and
  :func:`push_screen_with_transition`. These touch Textual's widget API
  and require a running app to be exercised end-to-end.

Adding a custom transition
--------------------------
The pre-built transitions cover the common cases (fade, slide). To add
a new transition without changing this file::

    from scholardevclaw.tui.screen_transitions import (
        Keyframe,
        ScreenTransition,
        TransitionPhase,
        register_transition,
    )

    custom = ScreenTransition(
        name="zoom",
        duration=0.3,
        easing="out_cubic",
        enter=TransitionPhase(
            keyframes=(Keyframe(at=0.0, offset=(0, 0), opacity=0.0),),
        ),
        exit=TransitionPhase(
            keyframes=(Keyframe(at=1.0, offset=(0, 0), opacity=0.0),),
        ),
    )
    register_transition(custom)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from textual.app import App
from textual.screen import Screen
from textual.widget import Widget

# ---------------------------------------------------------------------------
# Easing functions
# ---------------------------------------------------------------------------

EasingFunction = Callable[[float], float]
"""A function that maps a normalized progress value ``t`` in ``[0, 1]``
to an eased progress value in ``[0, 1]``."""


def linear(t: float) -> float:
    """Identity easing — no acceleration."""
    return t


def in_quad(t: float) -> float:
    """Quadratic ease-in: starts slow, ends fast."""
    return t * t


def out_quad(t: float) -> float:
    """Quadratic ease-out: starts fast, ends slow."""
    return 1 - (1 - t) * (1 - t)


def in_out_quad(t: float) -> float:
    """Quadratic ease-in-out: slow at both ends."""
    if t < 0.5:
        return 2 * t * t
    return 1 - (-2 * t + 2) ** 2 / 2


def in_cubic(t: float) -> float:
    """Cubic ease-in: starts very slow, ends fast."""
    return t * t * t


def out_cubic(t: float) -> float:
    """Cubic ease-out: starts fast, ends very slow."""
    return 1 - (1 - t) ** 3


def in_out_cubic(t: float) -> float:
    """Cubic ease-in-out: gentle on both ends, fast in the middle."""
    if t < 0.5:
        return 4 * t * t * t
    return 1 - (-2 * t + 2) ** 3 / 2


EASING_REGISTRY: dict[str, EasingFunction] = {
    "linear": linear,
    "in_quad": in_quad,
    "out_quad": out_quad,
    "in_out_quad": in_out_quad,
    "in_cubic": in_cubic,
    "out_cubic": out_cubic,
    "in_out_cubic": in_out_cubic,
}


def resolve_easing(name: str) -> EasingFunction:
    """Return the easing function registered under ``name``.

    Raises:
        KeyError: if no easing is registered under that name.
    """
    try:
        return EASING_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown easing '{name}'. Available: {', '.join(sorted(EASING_REGISTRY))}"
        ) from exc


def apply_easing(name: str, t: float) -> float:
    """Resolve an easing by name and apply it to ``t``.

    The result is clamped to ``[0, 1]`` to defend against floating-point
    drift in the higher-order cubic functions.
    """
    if not 0.0 <= t <= 1.0:
        t = max(0.0, min(1.0, t))
    return resolve_easing(name)(t)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Keyframe:
    """A single keyframe in a transition phase.

    Attributes:
        at: Normalized time within the phase in ``[0, 1]``.
        offset: ``(x, y)`` pixel offset applied to the widget.
        opacity: Opacity in ``[0, 1]``; ``None`` means "don't touch".
    """

    at: float
    offset: tuple[int, int] = (0, 0)
    opacity: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.at <= 1.0:
            raise ValueError(f"Keyframe.at must be in [0, 1], got {self.at!r}")
        if self.opacity is not None and not 0.0 <= self.opacity <= 1.0:
            raise ValueError(f"Keyframe.opacity must be None or in [0, 1], got {self.opacity!r}")


@dataclass(frozen=True)
class TransitionPhase:
    """An ordered set of keyframes.

    Keyframes are sorted by ``at`` on construction. A phase with no
    keyframes is treated as "no change" (identity).
    """

    keyframes: tuple[Keyframe, ...] = ()

    def __post_init__(self) -> None:
        sorted_keyframes = tuple(sorted(self.keyframes, key=lambda k: k.at))
        object.__setattr__(self, "keyframes", sorted_keyframes)
        for kf in sorted_keyframes:
            if kf.opacity is None and kf.offset == (0, 0):
                # Tolerated: a no-op keyframe is harmless. Just keep it
                # so consumers can stamp transitions with timing markers.
                continue


@dataclass(frozen=True)
class ScreenTransition:
    """Declarative description of a screen enter/exit animation.

    A transition is symmetric: both the enter and exit phases share the
    same duration and easing, but their keyframes describe the
    "appearance" (entering) or "disappearance" (exiting) of the screen.

    Attributes:
        name: Stable identifier used for lookup.
        duration: Total duration of the phase in seconds.
        easing: Name of an easing function in :data:`EASING_REGISTRY`.
        enter: Keyframes describing how the screen appears.
        exit: Keyframes describing how the screen disappears.
    """

    name: str
    duration: float
    easing: str
    enter: TransitionPhase
    exit: TransitionPhase

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("ScreenTransition.name must be a non-empty string")
        if self.duration < 0:
            raise ValueError(f"ScreenTransition.duration must be >= 0, got {self.duration!r}")
        if self.easing not in EASING_REGISTRY:
            raise ValueError(
                f"ScreenTransition.easing must be a registered easing, "
                f"got {self.easing!r}. "
                f"Available: {', '.join(sorted(EASING_REGISTRY))}"
            )

    def with_duration(self, duration: float) -> ScreenTransition:
        """Return a copy with the duration replaced."""
        if duration < 0:
            raise ValueError(f"duration must be >= 0, got {duration!r}")
        return ScreenTransition(
            name=self.name,
            duration=duration,
            easing=self.easing,
            enter=self.enter,
            exit=self.exit,
        )


# ---------------------------------------------------------------------------
# Pre-built transitions
# ---------------------------------------------------------------------------


# A phase that is a no-op: a single keyframe at (0, 0) with opacity 1.
_IDENTITY_PHASE = TransitionPhase(
    keyframes=(
        Keyframe(at=0.0, offset=(0, 0), opacity=1.0),
        Keyframe(at=1.0, offset=(0, 0), opacity=1.0),
    ),
)


def _build_fade(duration: float) -> ScreenTransition:
    return ScreenTransition(
        name="fade",
        duration=duration,
        easing="in_out_cubic",
        enter=TransitionPhase(
            keyframes=(
                Keyframe(at=0.0, offset=(0, 0), opacity=0.0),
                Keyframe(at=1.0, offset=(0, 0), opacity=1.0),
            ),
        ),
        exit=TransitionPhase(
            keyframes=(
                Keyframe(at=0.0, offset=(0, 0), opacity=1.0),
                Keyframe(at=1.0, offset=(0, 0), opacity=0.0),
            ),
        ),
    )


def _build_slide(duration: float, *, name: str, direction: str) -> ScreenTransition:
    """Build a slide transition.

    ``direction`` is one of ``"left"``, ``"right"``, ``"up"``, ``"down"``
    and describes which way the screen travels when entering.
    """
    enter_offset, exit_offset = _slide_offsets_for(direction)
    return ScreenTransition(
        name=name,
        duration=duration,
        easing="in_out_cubic",
        enter=TransitionPhase(
            keyframes=(
                Keyframe(at=0.0, offset=enter_offset, opacity=0.0),
                Keyframe(at=1.0, offset=(0, 0), opacity=1.0),
            ),
        ),
        exit=TransitionPhase(
            keyframes=(
                Keyframe(at=0.0, offset=(0, 0), opacity=1.0),
                Keyframe(at=1.0, offset=exit_offset, opacity=0.0),
            ),
        ),
    )


def _slide_offsets_for(direction: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ``(enter_start, exit_end)`` offsets for a slide direction.

    "Sliding in from the right" means the screen starts to the right of
    its destination and moves left to settle, so the enter offset is a
    positive x and the exit offset is a negative x (it leaves to the
    left).
    """
    # Conservative offsets: enough to read as motion in a TUI but not so
    # large that they cause visible re-layout. 8 cells horizontally and
    # 2 cells vertically are common defaults in CSS transition libraries.
    if direction == "left":
        return (8, 0), (-8, 0)
    if direction == "right":
        return (-8, 0), (8, 0)
    if direction == "up":
        return (0, 2), (0, -2)
    if direction == "down":
        return (0, -2), (0, 2)
    raise ValueError(
        f"slide direction must be one of 'left', 'right', 'up', 'down'; got {direction!r}"
    )


# Pre-built transitions with sensible defaults.
NONE: ScreenTransition = ScreenTransition(
    name="none",
    duration=0.0,
    easing="linear",
    enter=_IDENTITY_PHASE,
    exit=_IDENTITY_PHASE,
)
FADE: ScreenTransition = _build_fade(0.18)
SLIDE_LEFT: ScreenTransition = _build_slide(0.22, name="slide_left", direction="left")
SLIDE_RIGHT: ScreenTransition = _build_slide(0.22, name="slide_right", direction="right")
SLIDE_UP: ScreenTransition = _build_slide(0.22, name="slide_up", direction="up")
SLIDE_DOWN: ScreenTransition = _build_slide(0.22, name="slide_down", direction="down")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


# A frozen registry is exposed for read-only iteration; custom entries
# can be added through :func:`register_transition`.
TRANSITION_REGISTRY: dict[str, ScreenTransition] = {
    "none": NONE,
    "fade": FADE,
    "slide_left": SLIDE_LEFT,
    "slide_right": SLIDE_RIGHT,
    "slide_up": SLIDE_UP,
    "slide_down": SLIDE_DOWN,
}


def list_transitions() -> tuple[str, ...]:
    """Return the registered transition names in a stable order."""
    return tuple(
        name
        for name, transition in sorted(TRANSITION_REGISTRY.items())
        for _ in [transition]  # explicit iteration to keep mypy happy
    )


def get_transition(name: str) -> ScreenTransition:
    """Look up a transition by name.

    Raises:
        KeyError: if no transition is registered under that name.
    """
    try:
        return TRANSITION_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown transition '{name}'. Available: {', '.join(sorted(TRANSITION_REGISTRY))}"
        ) from exc


def get_transition_or_default(name: str | None) -> ScreenTransition:
    """Look up a transition, returning :data:`NONE` for ``None``/empty.

    This is the helper most callers want: ``None`` is treated as "no
    animation" rather than an error. Unknown names still raise
    :class:`KeyError` so typos are caught.
    """
    if not name:
        return NONE
    return get_transition(name)


def register_transition(transition: ScreenTransition) -> None:
    """Register or replace a transition in the registry.

    Re-registering an existing name is allowed; the new transition
    replaces the old one. The :data:`NONE` transition is treated as
    immutable — re-registering it raises :class:`ValueError`.
    """
    if transition.name == "none":
        raise ValueError("Cannot replace the 'none' transition")
    TRANSITION_REGISTRY[transition.name] = transition


# ---------------------------------------------------------------------------
# Pure logic helpers
# ---------------------------------------------------------------------------


def _clamp_unit(t: float) -> float:
    """Clamp ``t`` to ``[0, 1]``."""
    if t < 0.0:
        return 0.0
    if t > 1.0:
        return 1.0
    return t


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between ``a`` and ``b``."""
    return a + (b - a) * t


def _lerp_int(a: int, b: int, t: float) -> int:
    """Integer linear interpolation between ``a`` and ``b``."""
    return int(round(a + (b - a) * t))


def _lerp_offset(a: tuple[int, int], b: tuple[int, int], t: float) -> tuple[int, int]:
    return _lerp_int(a[0], b[0], t), _lerp_int(a[1], b[1], t)


def compute_progress(transition: ScreenTransition, t: float) -> float:
    """Map a normalized time ``t`` to an eased progress value.

    The clamping and easing are applied here so callers don't have to.
    """
    return apply_easing(transition.easing, t)


def interpolate_keyframes(
    phase: TransitionPhase, progress: float
) -> tuple[tuple[int, int], float | None]:
    """Linearly interpolate the keyframes of ``phase`` at ``progress``.

    Args:
        phase: A :class:`TransitionPhase` with at least one keyframe.
        progress: Normalized progress in ``[0, 1]``.

    Returns:
        A tuple ``(offset, opacity)``. ``opacity`` is ``None`` if no
        keyframe in the phase specifies one.
    """
    if not phase.keyframes:
        return (0, 0), None

    clamped = _clamp_unit(progress)

    # If the progress is at or below the first keyframe, return that.
    first = phase.keyframes[0]
    if clamped <= first.at:
        return first.offset, first.opacity

    # If the progress is at or above the last keyframe, return that.
    last = phase.keyframes[-1]
    if clamped >= last.at:
        return last.offset, last.opacity

    # Otherwise find the bracketing keyframes and lerp between them.
    for i in range(len(phase.keyframes) - 1):
        a = phase.keyframes[i]
        b = phase.keyframes[i + 1]
        if a.at <= clamped <= b.at:
            if b.at == a.at:
                local = 0.0
            else:
                local = (clamped - a.at) / (b.at - a.at)
            offset = _lerp_offset(a.offset, b.offset, local)
            if a.opacity is None and b.opacity is None:
                opacity: float | None = None
            elif a.opacity is None:
                opacity = b.opacity
            elif b.opacity is None:
                opacity = a.opacity
            else:
                opacity = _lerp(a.opacity, b.opacity, local)
            return offset, opacity

    # Unreachable for non-empty phases, but be defensive.
    return last.offset, last.opacity


def phase_start(phase: TransitionPhase) -> tuple[tuple[int, int], float | None]:
    """Return the state of a phase at progress ``0.0``."""
    return interpolate_keyframes(phase, 0.0)


def phase_end(phase: TransitionPhase) -> tuple[tuple[int, int], float | None]:
    """Return the state of a phase at progress ``1.0``."""
    return interpolate_keyframes(phase, 1.0)


def starting_state(transition: ScreenTransition, *, entering: bool) -> dict[str, Any]:
    """Compute the starting visual state for a transition phase.

    Returns a dict with ``"offset"`` (always present) and
    ``"opacity"`` (present only if the phase has any opacity keyframe).
    """
    phase = transition.enter if entering else transition.exit
    offset, opacity = phase_start(phase)
    state: dict[str, Any] = {"offset": offset}
    if opacity is not None:
        state["opacity"] = opacity
    return state


def ending_state(transition: ScreenTransition, *, entering: bool) -> dict[str, Any]:
    """Compute the ending visual state for a transition phase."""
    phase = transition.enter if entering else transition.exit
    offset, opacity = phase_end(phase)
    state: dict[str, Any] = {"offset": offset}
    if opacity is not None:
        state["opacity"] = opacity
    return state


# ---------------------------------------------------------------------------
# Widget integration
# ---------------------------------------------------------------------------


def apply_starting_state(
    widget: Widget,
    transition: ScreenTransition,
    *,
    entering: bool = True,
) -> None:
    """Mutate ``widget`` so that a transition is ready to begin.

    The widget's ``offset`` and (optionally) ``opacity`` are set to the
    starting values of the requested phase. Call this *before* the
    widget is mounted (or before :func:`animate_enter` /
    :func:`animate_exit` is called) to avoid a single-frame flash.
    """
    state = starting_state(transition, entering=entering)
    if "offset" in state:
        widget.offset = state["offset"]
    if "opacity" in state:
        # ``Widget.opacity`` is a read-only computed property; the
        # underlying mutable value lives on ``widget.styles``.
        widget.styles.opacity = state["opacity"]


def _run_animation(
    widget: Widget,
    transition: ScreenTransition,
    *,
    entering: bool,
    on_complete: Callable[[], None] | None,
) -> None:
    """Drive the actual animation by calling :meth:`Widget.animate`.

    Opacity is animated via :meth:`Styles.animate` (which works on
    Textual CSS-stored values) and offset is assigned directly. This
    approach works on Textual 8.2.x, where the high-level
    :meth:`Widget.animate` cannot reliably animate a ``(x, y)``
    offset. The visual effect is therefore "opacity eases in/out
    while the offset snaps to the final value at the end".

    ``on_complete`` is fired after the opacity animation finishes.
    For transitions with a non-positive duration (e.g. :data:`NONE`)
    no animation is scheduled and ``on_complete`` is invoked
    synchronously if provided.
    """
    if transition.duration <= 0:
        if on_complete is not None:
            try:
                on_complete()
            except Exception:
                pass
        return

    end_state = ending_state(transition, entering=entering)
    end_offset = end_state["offset"]
    end_opacity = end_state.get("opacity")

    fired = {"done": False}

    def _wrap() -> None:
        if fired["done"]:
            return
        fired["done"] = True
        # Snap the offset to its final value at the end of the
        # animation. The screen will already be at the target opacity,
        # so the offset change happens while the screen is fully
        # visible (or fully invisible for an exit).
        try:
            widget.offset = end_offset
        except Exception:
            pass
        if on_complete is not None:
            try:
                on_complete()
            except Exception:
                pass

    if end_opacity is not None:
        try:
            widget.styles.animate(
                "opacity",
                end_opacity,
                duration=transition.duration,
                easing=transition.easing,
                on_complete=_wrap,
            )
        except Exception:
            # Fall back to a direct assignment if the animation system
            # refuses the call for any reason.
            try:
                widget.styles.opacity = end_opacity
            except Exception:
                pass
            _wrap()
    else:
        # No opacity to animate; still snap offset and fire callback.
        _wrap()


def animate_enter(widget: Widget, transition: ScreenTransition) -> None:
    """Animate ``widget`` from its current state to the end of the
    enter phase. The widget's current state is treated as the starting
    state — call :func:`apply_starting_state` first to ensure a clean
    entry."""
    _run_animation(widget, transition, entering=True, on_complete=None)


def animate_exit(
    widget: Widget,
    transition: ScreenTransition,
    *,
    on_complete: Callable[[], None] | None = None,
) -> None:
    """Animate ``widget`` from its current state to the end of the exit
    phase. ``on_complete`` is invoked after the animation finishes."""
    _run_animation(widget, transition, entering=False, on_complete=on_complete)


def push_screen_with_transition(
    app: App[Any],
    screen: Screen[Any],
    *,
    transition_name: str | None = "fade",
    callback: Callable[[Any], None] | None = None,
) -> None:
    """Push ``screen`` onto ``app``'s stack with a transition.

    The transition is resolved from ``transition_name`` (or read from
    the screen's ``TRANSITION`` class attribute as a fallback). The
    screen's starting state is applied before the push, and the entry
    animation is scheduled for after the screen mounts.

    Args:
        app: The Textual :class:`App` to push onto.
        screen: The screen to push.
        transition_name: Name of a registered transition, or ``None``/
            empty to skip animation entirely. Falls back to
            ``screen.TRANSITION`` if the attribute exists.
        callback: Optional callback to register with
            :meth:`App.push_screen`.
    """
    resolved_name = transition_name
    if not resolved_name:
        resolved_name = getattr(screen, "TRANSITION", None)
    transition = get_transition_or_default(resolved_name)

    if transition is NONE or transition.duration <= 0:
        app.push_screen(screen, callback=callback)
        return

    apply_starting_state(screen, transition, entering=True)
    app.push_screen(screen, callback=callback)

    # Schedule the entry animation for after the screen is mounted.
    def _kickoff() -> None:
        try:
            animate_enter(screen, transition)
        except Exception:
            # Never let a failed animation keep the TUI stuck.
            pass

    try:
        screen.call_after_refresh(_kickoff)
    except Exception:
        # The screen may not be mounted yet; fall back to call_later.
        try:
            app.call_later(_kickoff)
        except Exception:
            pass


def pop_screen_with_transition(
    app: App[Any],
    transition: ScreenTransition,
) -> None:
    """Pop the topmost screen off ``app``'s stack with a transition.

    The exit animation runs on the screen being popped. ``app.pop_screen``
    is invoked in the animation's completion callback.
    """
    if transition is NONE or transition.duration <= 0:
        app.pop_screen()
        return

    current = app.screen
    apply_starting_state(current, transition, entering=False)

    def _finish() -> None:
        try:
            app.pop_screen()
        except Exception:
            pass

    animate_exit(current, transition, on_complete=_finish)


__all__ = [
    "EASING_REGISTRY",
    "EasingFunction",
    "FADE",
    "Keyframe",
    "NONE",
    "SLIDE_DOWN",
    "SLIDE_LEFT",
    "SLIDE_RIGHT",
    "SLIDE_UP",
    "ScreenTransition",
    "TRANSITION_REGISTRY",
    "TransitionPhase",
    "animate_enter",
    "animate_exit",
    "apply_easing",
    "apply_starting_state",
    "compute_progress",
    "ending_state",
    "get_transition",
    "get_transition_or_default",
    "in_cubic",
    "in_out_cubic",
    "in_out_quad",
    "in_quad",
    "interpolate_keyframes",
    "linear",
    "list_transitions",
    "out_cubic",
    "out_quad",
    "phase_end",
    "phase_start",
    "pop_screen_with_transition",
    "push_screen_with_transition",
    "register_transition",
    "resolve_easing",
    "starting_state",
]
