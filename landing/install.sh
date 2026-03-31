#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ScholarDevClaw Installer
# One-line: curl -fsSL https://Ronak-IIITD.github.io/ScholarDevClaw/install.sh | bash
#
# To use a custom domain, replace the URL above with:
#   curl -fsSL https://scholardevclaw.ai/install.sh | bash
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ─── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
RESET='\033[0m'

# ─── State ────────────────────────────────────────────────────────────────────
PREFIX="${SCHOLARDEVCLAW_PREFIX:-${HOME}/.local}"
PYTHON_CMD=""
VENV_DIR=""
INSTALL_METHOD="pip"
NO_COLOR="${NO_COLOR:-}"

log_info()  { printf "${CYAN}[i]${RESET} %s\n" "$*"; }
log_ok()    { printf "${GREEN}[✓]${RESET} %s\n" "$*"; }
log_warn()  { printf "${YELLOW}[!]${RESET} %s\n" "$*"; }
log_error() { printf "${RED}[✗]${RESET} %s\n" "$*" >&2; }
log_step()  { printf "\n${BOLD}${CYAN}▸${RESET} %s\n" "$*"; }
log_cmd()   { printf "  ${CYAN}$${RESET} %s\n" "$*"; }

# ─── Helpers ──────────────────────────────────────────────────────────────────
requires() {
  if ! command -v "$1" &>/dev/null; then
    log_error "Required command not found: $1"
    log_error "Please install $1 and try again."
    exit 1
  fi
}

parse_args() {
  for arg in "$@"; do
    case $arg in
      --pip)       INSTALL_METHOD="pip" ;;
      --no-venv)   NO_VENV=1 ;;
      --prefix=*)  PREFIX="${arg#*=}";;
      --help|-h)   show_help ;;
    esac
  done
}

show_help() {
  cat <<EOF
ScholarDevClaw Installer

Usage: curl -fsSL https://Ronak-IIITD.github.io/ScholarDevClaw/install.sh | bash [flags]
       (or replace URL with your custom domain when configured)

Flags:
  --pip         Force install via pip (default: auto-detect)
  --no-venv     Skip virtual environment creation
  --prefix=PATH Custom install prefix (default: ~/.local)
  --help, -h    Show this message

Examples:
  curl -fsSL https://Ronak-IIITD.github.io/ScholarDevClaw/install.sh | bash
  curl -fsSL https://Ronak-IIITD.github.io/ScholarDevClaw/install.sh | bash -s -- --pip
EOF
  exit 0
}

# ─── System Detection ─────────────────────────────────────────────────────────
detect_python() {
  log_step "Detecting Python..."

  for cmd in python3 python python3.12 python3.11 python3.10; do
    if command -v "$cmd" &>/dev/null; then
      local ver
      ver=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
      if [[ -n "$ver" ]]; then
        major="${ver%.*}"
        minor="${ver#*.}"
        if [[ "$major" -eq 3 ]] && [[ "$minor" -ge 10 ]]; then
          PYTHON_CMD="$cmd"
          log_ok "Found $cmd $ver"
          return 0
        fi
      fi
    fi
  done

  log_error "Python 3.10+ not found."
  log_error "Please install Python 3.10 or later: https://www.python.org/downloads/"
  exit 1
}

check_pip() {
  log_step "Checking pip..."

  requires pip

  if ! "$PYTHON_CMD" -m pip --version &>/dev/null; then
    log_warn "pip module not available. Trying pip3..."
    requires pip3
  fi

  log_ok "pip is available"
}

# ─── Install ──────────────────────────────────────────────────────────────────
install_package() {
  log_step "Installing ScholarDevClaw..."

  local install_cmd
  local pip_flags=(
    --quiet
    --disable-pip-version-check
  )

  if [[ -n "${SCHOLARDEVCLAW_UPGRADE:-}" ]]; then
    pip_flags+=(--upgrade)
  fi

  # Check if already installed
  if "$PYTHON_CMD" -m scholardevclaw --version &>/dev/null 2>&1; then
    local installed_ver
    installed_ver=$("$PYTHON_CMD" -m scholardevclaw --version 2>&1 | head -1)
    log_warn "ScholarDevClaw is already installed: $installed_ver"
    if [[ -z "${SCHOLARDEVCLAW_UPGRADE:-}" ]]; then
      log_info "Run with SCHOLARDEVCLAW_UPGRADE=1 to upgrade, or:"
      log_cmd "pip install --upgrade scholardevclaw"
      log_step "Installation complete!"
      return 0
    fi
    pip_flags+=(--upgrade)
  fi

  # Build install command
  install_cmd="$PYTHON_CMD -m pip install ${pip_flags[*]} ."

  log_info "Installing from local source..."
  if ! eval "$install_cmd"; then
    log_warn "Local install failed. Trying from GitHub source..."

    # Fallback: install from GitHub
    if ! $PYTHON_CMD -m pip install ${pip_flags[*]} "git+https://github.com/Ronak-IIITD/ScholarDevClaw.git@main"; then
      log_error "Installation failed."
      log_error "Please report this at: https://github.com/Ronak-IIITD/ScholarDevClaw/issues"
      exit 1
    fi
  fi

  log_ok "ScholarDevClaw installed successfully"
}

# ─── Verify ───────────────────────────────────────────────────────────────────
verify_install() {
  log_step "Verifying installation..."

  if ! "$PYTHON_CMD" -m scholardevclaw --version &>/dev/null; then
    log_error "Verification failed. 'scholardevclaw' command not found."
    log_error ""
    log_error "Try adding the pip bin directory to your PATH:"
    local bin_dir
    bin_dir=$("$PYTHON_CMD" -c 'import site; print(site.getusersitepackages() + "/bin" if site.getusersitepackages() else "")' 2>/dev/null || echo "${HOME}/.local/bin")
    log_cmd "export PATH=\"$bin_dir:\$PATH\""
    log_error ""
    log_error "Or run:"
    log_cmd "$PYTHON_CMD -m scholardevclaw --version"
    exit 1
  fi

  local ver
  ver=$("$PYTHON_CMD" -m scholardevclaw --version 2>&1 | head -1)
  log_ok "ScholarDevClaw $ver"
}

# ─── Post-install ─────────────────────────────────────────────────────────────
show_next_steps() {
  local bin_dir="${HOME}/.local/bin"
  if [[ ":$PATH:" != *":${bin_dir}:"* ]]; then
    log_warn "The ScholarDevClaw binary may not be in your PATH."
    log_info "Add this to your shell config (~/.bashrc, ~/.zshrc, etc.):"
    printf "\n  ${CYAN}export PATH=\"%s:\$PATH\"${RESET}\n\n" "$bin_dir"
    log_info "Then reload your shell:"
    log_cmd "source ~/.bashrc   # or ~/.zshrc"
    printf "\n"
  fi

  log_step "Next steps"
  log_info "Analyze a repository:"
  log_cmd "scholardevclaw analyze ./my-ml-project"
  log_info "Get research suggestions:"
  log_cmd "scholardevclaw suggest ./my-ml-project"
  log_info "Run the interactive TUI:"
  log_cmd "scholardevclaw tui"
  log_info "View all commands:"
  log_cmd "scholardevclaw --help"
  printf "\n"
  log_ok "ScholarDevClaw is ready!"
  log_info "Documentation: https://github.com/Ronak-IIITD/ScholarDevClaw"
  log_info "Issues: https://github.com/Ronak-IIITD/ScholarDevClaw/issues"
}

# ─── Banner ──────────────────────────────────────────────────────────────────
show_banner() {
  cat <<'EOF'

  ██████╗  ██████╗ ██████╗ ██████╗  █████╗ ███████╗██╗██╗  ██╗
  ██╔══██╗██╔════╝ ██╔═══██╗██╔══██╗██╔══██╗██╔════╝██║╚██╗██╔╝
  ██████╔╝██║  ███╗██║   ██║██████╔╝███████║███████╗██║ ╚███╔╝
  ██╔═══╝ ██║   ██║██║   ██║██╔══██╗██╔══██║╚════██║██║ ██╔██╗
  ██║     ╚██████╔╝╚██████╔╝██████╔╝██║  ██║███████║██║██╔╝ ██╗
  ╚═╝      ╚═════╝  ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝╚═╝  ╚═╝

  Autonomous ML Research Integration Engine
  https://github.com/Ronak-IIITD/ScholarDevClaw

EOF
}

# ─── Main ─────────────────────────────────────────────────────────────────────
main() {
  # Check for curl
  requires curl

  # Parse CLI args
  parse_args "$@"

  # Detect OS
  local os
  os=$(uname -s 2>/dev/null || echo "Linux")
  case "$os" in
    Linux*)     log_info "Detected: Linux" ;;
    Darwin*)     log_info "Detected: macOS" ;;
    *MINGW*|*CYGWIN*|MSYS*) log_info "Detected: Windows (WSL/MSYS)" ;;
    *)           log_info "Detected: $os" ;;
  esac

  # Detect Python
  detect_python

  # Check pip
  check_pip

  # Install
  install_package

  # Verify
  verify_install

  # Next steps
  show_next_steps
}

# Only run main if not being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  show_banner
  main "$@"
fi
