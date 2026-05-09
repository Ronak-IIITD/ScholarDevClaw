# Contributing to ScholarDevClaw

Thank you for your interest in contributing to ScholarDevClaw! This document provides guidelines and instructions for contributing.

## 🛠️ Development Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ (for agent)
- Git

### Core (Python)

```bash
# Clone the repository
git clone https://github.com/Ronak-IIITD/ScholarDevClaw.git
cd ScholarDevClaw/core

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[arxiv,ml,tui,dev]"

# Run tests
pytest tests/unit/ -v
```

### Agent (TypeScript)

```bash
cd ScholarDevClaw/agent
bun install
bun run build
bun run test
```

### Web Dashboard (React)

```bash
cd ScholarDevClaw/web
npm install
npm run build
```

## 📋 Coding Standards

### Python

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Run `ruff check src/ tests/` before committing
- Ensure all tests pass: `pytest tests/unit/ -q`

### TypeScript

- Use strict TypeScript
- Run `bun run build` to verify compilation
- Run `bun run test` for unit tests

## 🔄 Branching Strategy

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Commit with a descriptive message
6. Push and create a Pull Request

## 📝 Commit Messages

Format: `type: short description`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Maintenance tasks

Example: `feat: add RoPE attention transformer`

## 🐛 Reporting Issues

When reporting issues, please include:

- ScholarDevClaw version
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error logs if applicable

Use the [issue templates](./.github/ISSUE_TEMPLATE/) for bug reports and feature requests.

## 🔍 Finding Issues to Work On

- Look for issues tagged with `good first issue`
- Check the [roadmap](../UPDATES.md) for planned features
- Issues tagged with `help wanted` are open for contribution

## 📤 Pull Request Process

1. Fork the repository
2. Create a feature branch from `main`
3. Write clear, descriptive commit messages
4. Include tests for new functionality
5. Ensure CI passes
6. Update documentation if needed
7. Request review from a maintainer

## 🧪 Testing

All new features should include tests:

```bash
# Run core tests
cd core && pytest tests/unit/ -v

# Run agent tests
cd agent && bun run test

# Run all checks
make test  # if available
```

## 📚 Resources

- [Documentation](../README.md)
- [Architecture](../ARCHITECTURE.md)
- [API Reference](../core/src/scholardevclaw/api/server.py)
- [Security Policy](./SECURITY.md)

## 💬 Getting Help

- Open a discussion for questions
- Check existing issues before creating new ones
- Join community channels (if available)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.