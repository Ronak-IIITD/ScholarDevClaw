# ScholarDevClaw v2 🚀

**Autonomous Research-to-Code AI Agent for All Developers**

ScholarDevClaw analyzes your codebase, researches relevant papers and implementations, and automatically generates improvements - supporting any programming language.

## 🌟 What's New in v2

- **Multi-Language Support**: Python, JavaScript/TypeScript, Go, Rust, Java, and more
- **Real-Time Research**: arXiv API integration for live paper search
- **Web Research**: GitHub, Papers with Code, Stack Overflow search
- **Smart Matching**: Automatically matches research to your code patterns
- **OpenClaw Integration**: Full orchestration with heartbeat and state management
- **Flexible Deployment**: Self-hosted or cloud

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/Ronak-IIITD/ScholarDevClaw.git
cd ScholarDevClaw

# Setup Python core
cd core
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[arxiv,ml]"

# Test installation
scholardevclaw --help
scholardevclaw demo
```

## 📖 Usage

All commands work through the unified `scholardevclaw` CLI:

```bash
# See all commands
scholardevclaw --help

# Analyze any codebase (auto-detects language)
scholardevclaw analyze /path/to/your/repo

# Search for research papers
scholardevclaw search "layer normalization" --arxiv --web

# Get AI-powered improvement suggestions
scholardevclaw suggest /path/to/your/repo

# Full integration workflow
scholardevclaw integrate /path/to/your/repo rmsnorm

# List available paper specs
scholardevclaw specs --list

# Run demo
scholardevclaw demo

# Optional: install and launch TUI
pip install -e ".[tui]"
scholardevclaw tui
```

## 🛠️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ScholarDevClaw v2                        │
├─────────────────────────────────────────────────────────────┤
│  Unified CLI Interface                                      │
│  • One command for all operations                          │
│  • Multi-language support                                   │
│  • Web + arXiv research                                    │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Code        │   │ Research     │   │  Code        │
│  Analyzer    │   │ Engine       │   │  Generator   │
│ (tree-sitter)│   │(arXiv + Web)│   │ (multi-lang) │
└───────────────┘   └───────────────┘   └───────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    OpenClaw Integration                     │
│  • Heartbeat Scheduling • State Management • GitHub PRs    │
└─────────────────────────────────────────────────────────────┘
```

## 🌍 Supported Languages

| Language | Status | Frameworks |
|----------|--------|------------|
| Python | ✅ Full | PyTorch, TensorFlow, Django, Flask, FastAPI |
| JavaScript | ✅ Full | Express, React, Vue, Angular |
| TypeScript | ✅ Full | Next.js, NestJS |
| Go | ✅ Basic | Gin, Echo |
| Rust | ✅ Basic | Actix, Rocket |
| Java | ✅ Basic | Spring, Maven |
| C/C++ | 🚧 Planned | - |
| Ruby | 🚧 Planned | Rails |

## 📚 Research Sources

- **arXiv**: 2.4M+ papers in CS, ML, Physics
- **GitHub**: Search for implementations
- **Papers with Code**: ML implementations
- **Stack Overflow**: Technical discussions
- **Technical Blogs**: Coming soon

## 🏗️ Deployment Options

### Self-Hosted (Local)

```bash
# Docker Compose
docker-compose up -d

# Systemd service
sudo systemctl start scholardevclaw

# PM2
pm2 start ecosystem.config.js
```

### Cloud

- **Fly.io**: `fly deploy`
- **Railway**: `railway up`
- **AWS EC2**: See docs/DEPLOYMENT.md

## 📖 Documentation

- [Quick Start Guide](demo.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Architecture Overview](AGENTS.md)
- [API Reference](docs/API.md)

## 🎯 Example Workflows

### Improve a Python ML Project

```bash
# 1. Analyze
cd my-ml-project
scholardevclaw analyze .

# 2. Get suggestions
scholardevclaw suggest .

# 3. Integrate RMSNorm
scholardevclaw integrate . rmsnorm
```

### Improve a JavaScript Backend

```bash
# Analyze Express.js app
scholardevclaw analyze ./my-api

# Search for caching papers
scholardevclaw search "api caching" --web

# Get suggestions
scholardevclaw suggest ./my-api
```

## 🔧 Configuration

```bash
# Environment variables
cat > .env << 'EOF'
SCHOLARDEVCLAW_WORKSPACE=~/.scholardevclaw/workspace
SCHOLARDEVCLAW_LOG_PATH=~/.scholardevclaw/logs
GITHUB_TOKEN=your_github_token
ANTHROPIC_API_KEY=your_anthropic_key
EOF
```

## 🧪 Testing

```bash
# Run tests
cd core
pytest

# Run demo
scholardevclaw demo

# Test with your repo
scholardevclaw analyze /path/to/repo
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a PR

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Acknowledgments

- OpenClaw framework for agent orchestration
- tree-sitter for multi-language parsing
- arXiv for paper access
- Papers with Code for implementations

## 📞 Support

- GitHub Issues: https://github.com/Ronak-IIITD/ScholarDevClaw/issues
- Discussions: https://github.com/Ronak-IIITD/ScholarDevClaw/discussions

---

**Built with ❤️ for researchers and developers**

Transform your codebase with cutting-edge research automatically. 🚀
