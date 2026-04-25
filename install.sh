#!/usr/bin/env bash
set -e

# ScholarDevClaw Installer
# Quick install for the ScholarDevClaw TUI

SCHOLARDEVCLAW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$SCHOLARDEVCLAW_ROOT/core"
VENV_DIR="$CORE_DIR/.venv"

echo "==> ScholarDevClaw Installer"
echo ""

# Detect platform
PLATFORM="$(uname -s)"
echo "Platform: $PLATFORM"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python: $PYTHON_VERSION"

# Create venv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "==> Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo ""
echo "==> Upgrading pip..."
pip install --upgrade pip --quiet

# Install core with all extras
echo ""
echo "==> Installing ScholarDevClaw core..."
pip install -e "$CORE_DIR[arxiv,ml,dev,tui]" --quiet

# Verify TUI is available
echo ""
echo "==> Verifying TUI..."
if ! command -v scholardevclaw &> /dev/null; then
    echo "Warning: scholardevclaw command not found in PATH"
    echo "Try: source $VENV_DIR/bin/activate"
else
    echo "TUI: OK ($(scholardevclaw --version 2>/dev/null || echo 'installed'))"
fi

# Check for optional tools
echo ""
echo "==> Optional tools:"

if command -v git &> /dev/null; then
    echo "  git:   OK"
else
    echo "  git:   missing (optional but recommended)"
fi

if command -v node &> /dev/null; then
    echo "  node:  OK"
else
    echo "  node:  missing (needed for agent/)"
fi

if command -v bun &> /dev/null; then
    echo "  bun:   OK"
else
    echo "  bun:   missing (optional for agent)"
fi

echo ""
echo "==> Installation complete!"
echo ""
echo "Quick start:"
echo "  source $VENV_DIR/bin/activate"
echo "  scholardevclaw tui"
echo ""
echo "First time setup:"
echo "  scholardevclaw tui"
echo "  # type: setup"
echo "  # paste your API key"
echo "  # then: paper arxiv:1706.03762"
echo ""
