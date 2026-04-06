"""
Theme constants and color scheme for ScholarDevClaw TUI.

Custom color scheme provided by user.
"""

from __future__ import annotations

from typing import Any

# -----------------------------------------------------------------------------
# Core Color Palette (User-Provided)
# -----------------------------------------------------------------------------

COLORS = {
    # Backgrounds (darkest to lightest)
    "background": "#06141B",  # Darkest - main background
    "surface": "#11212D",  # Dark - panels/cards
    "surface-elevated": "#253754",  # Medium dark - elevated elements
    # Text
    "text": "#CCD0CF",  # Lightest - primary text
    "text-muted": "#9BA8AB",  # Light muted - secondary text
    "text-bright": "#CCD0CF",  # Primary text (same as text)
    # Accents
    "accent": "#253754",  # Medium dark - primary actions
    "accent-dim": "#4A5C6A",  # Medium - darker accent
    "accent-bright": "#4A5C6A",  # Brighter accent for highlights
    # Semantic (using palette)
    "success": "#9BA8AB",  # Light muted - readable success on dark bg
    "warning": "#CCD0CF",  # High contrast warning text
    "error": "#CCD0CF",  # High contrast error text
    "info": "#9BA8AB",  # Light muted info
    # Borders/Separators
    "border": "#4A5C6A",  # Medium
    "border-focus": "#9BA8AB",  # Light muted
    # Provider-specific (using palette)
    "provider-ollama": "#4A5C6A",  # Medium
    "provider-openrouter": "#253754",  # Medium dark
    "provider-anthropic": "#9BA8AB",  # Light muted
    "provider-openai": "#4A5C6A",  # Medium
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
