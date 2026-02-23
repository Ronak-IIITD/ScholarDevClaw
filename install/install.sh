#!/bin/bash
#
# ScholarDevClaw Installer
# Usage: curl -L https://scholardevclaw.com/install | bash
#

set -e

VERSION="1.0.0"
INSTALL_DIR="${HOME}/.scholardevclaw"
BIN_DIR="${HOME}/.local/bin"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║   ███████╗ █████╗ ██████╗ ██╗     ███████╗               ║"
echo "║   ██╔════╝██╔══██╗██╔══██╗██║     ██╔════╝               ║"
echo "║   █████╗  ███████║██████╔╝██║     █████╗                 ║"
echo "║   ██╔══╝  ██╔══██║██╔══██╗██║     ██╔══╝                 ║"
echo "║   ██║     ██║  ██║██║  ██║███████╗███████╗               ║"
echo "║   ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝               ║"
echo "║                    v${VERSION}                              ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${GREEN}Installing ScholarDevClaw...${NC}\n"

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     PLATFORM="linux";;
    Darwin*)    PLATFORM="macos";;
    *)          PLATFORM="unknown";;
esac

echo "Detected platform: ${PLATFORM}"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
    echo "Please install Python 3.10+ and try again."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: ${PYTHON_VERSION}"

# Create installation directory
echo -e "\n${YELLOW}Creating installation directory...${NC}"
mkdir -p "${INSTALL_DIR}"
mkdir -p "${BIN_DIR}"

# Check if pip is available
if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
    echo -e "${RED}Error: pip is required but not installed.${NC}"
    exit 1
fi

# Install using pip
echo -e "\n${YELLOW}Installing ScholarDevClaw...${NC}"
if command -v pip3 &> /dev/null; then
    pip3 install -e ".[tui,security]" --quiet
else
    python3 -m pip install -e ".[tui,security]" --quiet
fi

# Create launcher script
echo -e "\n${YELLOW}Creating launcher...${NC}"
cat > "${BIN_DIR}/scholardevclaw" << 'LAUNCHER'
#!/bin/bash
# ScholarDevClaw Launcher

# Check for updates
SCHOLARDEVCLAW_DIR="${HOME}/.scholardevclaw"
CONFIG_FILE="${SCHOLARDEVCLAW_DIR}/config.env"

# Load config if exists
if [ -f "${CONFIG_FILE}" ]; then
    source "${CONFIG_FILE}"
fi

# Check if in a git repository or has .scholardevclaw
if [ -d ".git" ] || [ -d ".scholardevclaw" ]; then
    # Run agent mode
    exec python3 -m scholardevclaw agent "$@"
else
    # Run normal CLI
    exec python3 -m scholardevclaw "$@"
fi
LAUNCHER

chmod +x "${BIN_DIR}/scholardevclaw"

# Add to PATH if not already there
SHELL_RC="${HOME}/.bashrc"
if [ -f "${HOME}/.zshrc" ]; then
    SHELL_RC="${HOME}/.zshrc"
fi

if ! grep -q "${BIN_DIR}" "${SHELL_RC}" 2>/dev/null; then
    echo "" >> "${SHELL_RC}"
    echo "# ScholarDevClaw" >> "${SHELL_RC}"
    echo "export PATH=\"\${HOME}/.local/bin:\${PATH}\"" >> "${SHELL_RC}"
    echo -e "${YELLOW}Added ${BIN_DIR} to PATH in ${SHELL_RC}${NC}"
    echo -e "${YELLOW}Please run 'source ${SHELL_RC}' or restart your terminal.${NC}"
fi

# Create config directory
mkdir -p "${INSTALL_DIR}"

# Print success message
echo -e "\n${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  Installation Complete!                    ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Run ScholarDevClaw with:"
echo -e "  ${BLUE}scholardevclaw agent${NC}          # Start interactive agent"
echo -e "  ${BLUE}scholardevclaw analyze <path>${NC} # Analyze a repository"
echo -e "  ${BLAD}scholardevclaw --help${NC}         # See all commands"
echo ""
echo -e "Quick start:"
echo -e "  1. cd to your project directory"
echo -e "  2. Run ${BLUE}scholardevclaw agent${NC}"
echo -e "  3. Type 'analyze .' to analyze your project"
echo ""
echo -e "For more info: ${BLUE}https://scholardevclaw.com${NC}"
echo ""
