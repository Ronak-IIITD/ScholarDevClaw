"""
Theme system for ScholarDevClaw TUI.

Provides four polished themes with OpenCode-style (cyan accent) as default.
All themes use CSS variables that map to Textual's theme system.
"""

from __future__ import annotations

from typing import Any

# -----------------------------------------------------------------------------
# Theme Definitions
# -----------------------------------------------------------------------------


def _opencode_colors() -> dict[str, str]:
    """OpenCode-style: cool dark with cyan accent."""
    return {
        # Backgrounds
        "background": "#0a0e14",
        "surface": "#111827",
        "surface-elevated": "#1e293b",
        "surface-overlay": "#1e293b",
        # Text
        "text": "#f1f5f9",
        "text-muted": "#94a3b8",
        "text-bright": "#f8fafc",
        # Accents
        "accent": "#06b6d4",
        "accent-dim": "#0891b2",
        "accent-bright": "#22d3ee",
        "accent-subtle": "#164e63",
        # Semantic
        "success": "#22c55e",
        "success-dim": "#16a34a",
        "warning": "#fbbf24",
        "warning-dim": "#d97706",
        "error": "#ef4444",
        "error-dim": "#dc2626",
        "info": "#3b82f6",
        "info-dim": "#2563eb",
        # Borders
        "border": "#1f2937",
        "border-focus": "#06b6d4",
        "border-subtle": "#374151",
        # Status
        "status-idle": "#64748b",
        "status-running": "#06b6d4",
        "status-success": "#22c55e",
        "status-error": "#ef4444",
        # Code
        "code-bg": "#0f172a",
        "code-border": "#1e293b",
        "code-text": "#e2e8f0",
        # Diff
        "diff-add-bg": "#052e16",
        "diff-add-text": "#4ade80",
        "diff-del-bg": "#450a0a",
        "diff-del-text": "#f87171",
        "diff-hunk": "#94a3b8",
        "diff-file": "#22d3ee",
        # User/Assistant message
        "user-bg": "#1e293b",
        "assistant-bg": "#0f172a",
        # Provider-specific
        "provider-ollama": "#06b6d4",
        "provider-openrouter": "#3b82f6",
        "provider-anthropic": "#a78bfa",
        "provider-openai": "#22c55e",
    }


def _claude_colors() -> dict[str, str]:
    """Claude-style: warm dark with blue accent."""
    return {
        # Backgrounds
        "background": "#0d1117",
        "surface": "#161b22",
        "surface-elevated": "#21262d",
        "surface-overlay": "#21262d",
        # Text
        "text": "#e6edf3",
        "text-muted": "#8b949e",
        "text-bright": "#f0f6fc",
        # Accents
        "accent": "#58a6ff",
        "accent-dim": "#388bfd",
        "accent-bright": "#79c0ff",
        "accent-subtle": "#1f3a5f",
        # Semantic
        "success": "#3fb950",
        "success-dim": "#238636",
        "warning": "#d29922",
        "warning-dim": "#bb8009",
        "error": "#f85149",
        "error-dim": "#da3633",
        "info": "#58a6ff",
        "info-dim": "#388bfd",
        # Borders
        "border": "#30363d",
        "border-focus": "#58a6ff",
        "border-subtle": "#21262d",
        # Status
        "status-idle": "#8b949e",
        "status-running": "#58a6ff",
        "status-success": "#3fb950",
        "status-error": "#f85149",
        # Code
        "code-bg": "#0d1117",
        "code-border": "#30363d",
        "code-text": "#e6edf3",
        # Diff
        "diff-add-bg": "#12261e",
        "diff-add-text": "#3fb950",
        "diff-del-bg": "#3d1418",
        "diff-del-text": "#f85149",
        "diff-hunk": "#8b949e",
        "diff-file": "#79c0ff",
        # User/Assistant message
        "user-bg": "#161b22",
        "assistant-bg": "#0d1117",
        # Provider-specific
        "provider-ollama": "#58a6ff",
        "provider-openrouter": "#3fb950",
        "provider-anthropic": "#a78bfa",
        "provider-openai": "#3fb950",
    }


def _minimal_colors() -> dict[str, str]:
    """Minimal: clean monochrome with subtle accents."""
    return {
        # Backgrounds
        "background": "#000000",
        "surface": "#0a0a0a",
        "surface-elevated": "#141414",
        "surface-overlay": "#141414",
        # Text
        "text": "#d4d4d4",
        "text-muted": "#737373",
        "text-bright": "#fafafa",
        # Accents
        "accent": "#a3a3a3",
        "accent-dim": "#737373",
        "accent-bright": "#d4d4d4",
        "accent-subtle": "#262626",
        # Semantic
        "success": "#a3a3a3",
        "success-dim": "#737373",
        "warning": "#d4d4d4",
        "warning-dim": "#a3a3a3",
        "error": "#d4d4d4",
        "error-dim": "#a3a3a3",
        "info": "#d4d4d4",
        "info-dim": "#a3a3a3",
        # Borders
        "border": "#262626",
        "border-focus": "#a3a3a3",
        "border-subtle": "#1a1a1a",
        # Status
        "status-idle": "#525252",
        "status-running": "#a3a3a3",
        "status-success": "#a3a3a3",
        "status-error": "#d4d4d4",
        # Code
        "code-bg": "#0a0a0a",
        "code-border": "#262626",
        "code-text": "#d4d4d4",
        # Diff
        "diff-add-bg": "#0a1a0a",
        "diff-add-text": "#a3a3a3",
        "diff-del-bg": "#1a0a0a",
        "diff-del-text": "#737373",
        "diff-hunk": "#525252",
        "diff-file": "#d4d4d4",
        # User/Assistant message
        "user-bg": "#0a0a0a",
        "assistant-bg": "#000000",
        # Provider-specific
        "provider-ollama": "#a3a3a3",
        "provider-openrouter": "#a3a3a3",
        "provider-anthropic": "#a3a3a3",
        "provider-openai": "#a3a3a3",
    }


def _high_contrast_colors() -> dict[str, str]:
    """High contrast: maximum readability for accessibility."""
    return {
        # Backgrounds
        "background": "#000000",
        "surface": "#0a0a0a",
        "surface-elevated": "#1a1a1a",
        "surface-overlay": "#1a1a1a",
        # Text
        "text": "#ffffff",
        "text-muted": "#b0b0b0",
        "text-bright": "#ffffff",
        # Accents
        "accent": "#00ffff",
        "accent-dim": "#00cccc",
        "accent-bright": "#66ffff",
        "accent-subtle": "#003333",
        # Semantic
        "success": "#00ff00",
        "success-dim": "#00cc00",
        "warning": "#ffff00",
        "warning-dim": "#cccc00",
        "error": "#ff3333",
        "error-dim": "#cc0000",
        "info": "#3399ff",
        "info-dim": "#0066cc",
        # Borders
        "border": "#404040",
        "border-focus": "#00ffff",
        "border-subtle": "#333333",
        # Status
        "status-idle": "#808080",
        "status-running": "#00ffff",
        "status-success": "#00ff00",
        "status-error": "#ff3333",
        # Code
        "code-bg": "#0a0a0a",
        "code-border": "#404040",
        "code-text": "#ffffff",
        # Diff
        "diff-add-bg": "#002200",
        "diff-add-text": "#00ff00",
        "diff-del-bg": "#220000",
        "diff-del-text": "#ff3333",
        "diff-hunk": "#808080",
        "diff-file": "#00ffff",
        # User/Assistant message
        "user-bg": "#0a0a0a",
        "assistant-bg": "#000000",
        # Provider-specific
        "provider-ollama": "#00ffff",
        "provider-openrouter": "#3399ff",
        "provider-anthropic": "#cc66ff",
        "provider-openai": "#00ff00",
    }


# Theme registry
THEMES: dict[str, dict[str, Any]] = {
    "opencode": {
        "name": "OpenCode",
        "description": "Cool dark with cyan accent",
        "colors": _opencode_colors(),
    },
    "claude": {
        "name": "Claude",
        "description": "Warm dark with blue accent",
        "colors": _claude_colors(),
    },
    "minimal": {
        "name": "Minimal",
        "description": "Clean monochrome",
        "colors": _minimal_colors(),
    },
    "high_contrast": {
        "name": "High Contrast",
        "description": "Maximum readability",
        "colors": _high_contrast_colors(),
    },
}

# Default theme
DEFAULT_THEME = "opencode"

# Backward compatibility: default COLORS dict points to opencode
COLORS: dict[str, str] = _opencode_colors()

# All available theme names in display order
THEME_NAMES: tuple[str, ...] = ("opencode", "claude", "minimal", "high_contrast")


# -----------------------------------------------------------------------------
# Theme accessor functions
# -----------------------------------------------------------------------------


def get_theme(theme_name: str | None = None) -> dict[str, Any]:
    """Get theme configuration by name.

    Returns the full theme dict with 'name', 'description', 'colors' keys.
    Falls back to the default theme for unknown names.
    """
    name = theme_name or DEFAULT_THEME
    return THEMES.get(name, THEMES[DEFAULT_THEME])


def get_theme_colors(theme_name: str | None = None) -> dict[str, str]:
    """Get just the color palette for a theme."""
    return get_theme(theme_name)["colors"]


def get_color(color_key: str, theme_name: str | None = None) -> str:
    """Get a specific color value by key, falling back to the default theme."""
    colors = get_theme_colors(theme_name)
    return colors.get(color_key, COLORS.get(color_key, "#ffffff"))


def list_themes() -> list[dict[str, str]]:
    """Return a list of theme summaries for the theme switcher."""
    return [
        {
            "key": key,
            "name": theme["name"],
            "description": theme["description"],
            "accent": theme["colors"]["accent"],
        }
        for key, theme in THEMES.items()
    ]


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
    return f"${name}"


def make_gradient(colors: list[str]) -> str:
    """Create a linear gradient from colors."""
    return f"linear-gradient(to right, {', '.join(colors)})"


# -----------------------------------------------------------------------------
# Typography (unchanged)
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
# Spacing (unchanged)
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
# Effects (unchanged)
# -----------------------------------------------------------------------------

EFFECTS = {
    "shadow": "0 4px 6px -1px rgba(0, 0, 0, 0.3)",
    "glow-accent": "0 0 10px rgba(6, 182, 212, 0.3)",
    "glow-success": "0 0 10px rgba(34, 197, 94, 0.3)",
    "glow-error": "0 0 10px rgba(239, 68, 68, 0.3)",
}

# -----------------------------------------------------------------------------
# Animation Durations (seconds)
# -----------------------------------------------------------------------------

ANIMATION = {
    "fast": 0.1,
    "normal": 0.18,
    "slow": 0.3,
    "very_slow": 0.5,
}
