"""
Theme constants and color scheme for ScholarDevClaw TUI.

Provides a cohesive dark theme with accent colors for different message types.
"""

from __future__ import annotations

from typing import Any

# -----------------------------------------------------------------------------
# Core Color Palette (Dark Theme)
# -----------------------------------------------------------------------------

COLORS = {
    # Backgrounds
    "background": "#0b0f12",
    "surface": "#151a21",
    "surface-elevated": "#1c222b",
    # Text
    "text": "#d7dee7",
    "text-muted": "#64748b",
    "text-bright": "#f1f5f9",
    # Accents
    "accent": "#38bdf8",  # Sky blue - primary actions
    "accent-dim": "#0ea5e9",  # Darker accent
    # Semantic
    "success": "#22c55e",  # Green
    "warning": "#f59e0b",  # Amber
    "error": "#ef4444",  # Red
    "info": "#38bdf8",  # Blue
    # Borders/Separators
    "border": "#334155",
    "border-focus": "#38bdf8",
    # Provider-specific
    "provider-ollama": "#10b981",  # Emerald
    "provider-openrouter": "#8b5cf6",  # Violet
    "provider-anthropic": "#f59e0b",  # Amber
    "provider-openai": "#10b981",  # Emerald
}

# -----------------------------------------------------------------------------
# Typography
# -----------------------------------------------------------------------------

FONTS = {
    "mono": "JetBrains Mono, Fira Code, Consolas, monospace",
    "display": "Inter, system-ui, sans-serif",
}

FONT_SIZES = {
    "tiny": 10,
    "small": 12,
    "normal": 14,
    "large": 16,
    "xlarge": 20,
    "title": 24,
}

# -----------------------------------------------------------------------------
# Spacing
# -----------------------------------------------------------------------------

SPACING = {
    "xs": 2,
    "sm": 4,
    "md": 8,
    "lg": 16,
    "xl": 24,
    "xxl": 32,
}

# -----------------------------------------------------------------------------
# Effects
# -----------------------------------------------------------------------------

EFFECTS = {
    "shadow": "0 4px 6px -1px rgba(0, 0, 0, 0.3)",
    "glow-accent": "0 0 10px rgba(56, 189, 248, 0.3)",
    "glow-success": "0 0 10px rgba(34, 197, 94, 0.3)",
    "glow-error": "0 0 10px rgba(239, 68, 68, 0.3)",
}

# -----------------------------------------------------------------------------
# Animation Durations (seconds)
# -----------------------------------------------------------------------------

ANIMATION = {
    "fast": 0.1,
    "normal": 0.2,
    "slow": 0.4,
    "very_slow": 0.8,
}

# -----------------------------------------------------------------------------
# CSS Template Helpers
# -----------------------------------------------------------------------------


def bg_color(name: str) -> str:
    """Get background color by name."""
    return COLORS.get(name, COLORS["background"])


def text_color(name: str) -> str:
    """Get text color by name."""
    return COLORS.get(name, COLORS["text"])


def css_var(name: str, fallback: str | None = None) -> str:
    """Generate CSS variable reference."""
    if fallback:
        return f"${name}"
    return f"${name}"


def make_gradient(colors: list[str]) -> str:
    """Create a linear gradient from colors."""
    return f"linear-gradient(to right, {', '.join(colors)})"


# -----------------------------------------------------------------------------
# Preset Themes
# -----------------------------------------------------------------------------


def get_theme(theme_name: str = "default") -> dict[str, Any]:
    """Get theme configuration by name."""
    themes = {
        "default": {
            "colors": COLORS,
            "fonts": FONTS,
            "font_sizes": FONT_SIZES,
            "spacing": SPACING,
            "effects": EFFECTS,
            "animation": ANIMATION,
        },
        "minimal": {
            "colors": {
                **COLORS,
                "surface": COLORS["background"],
                "text-muted": "#475569",
            },
            "fonts": FONTS,
            "font_sizes": FONT_SIZES,
            "spacing": SPACING,
            "effects": {},
            "animation": ANIMATION,
        },
        "high_contrast": {
            "colors": {
                **COLORS,
                "background": "#000000",
                "text": "#ffffff",
                "accent": "#00d4ff",
            },
            "fonts": FONTS,
            "font_sizes": {k: v + 2 for k, v in FONT_SIZES.items()},
            "spacing": SPACING,
            "effects": EFFECTS,
            "animation": ANIMATION,
        },
    }
    return themes.get(theme_name, themes["default"])
