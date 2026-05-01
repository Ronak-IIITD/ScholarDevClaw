# ScholarDevClaw - Production Credentials Guide

This document lists all environment variables needed to run ScholarDevClaw in production.

---

## Required Credentials (Must Set)

### Core System
| Variable | Description | How to Get |
|----------|-------------|------------|
| `SCHOLARDEVCLAW_API_AUTH_KEY` | Core API authentication key | `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `SCHOLARDEVCLAW_ALLOWED_REPO_DIRS` | Colon-separated allowed repo paths | Set to your project directories, e.g., `/home/user/projects:/repos` |
| `SCHOLARDEVCLAW_CORS_ORIGINS` | Allowed CORS origins | Your frontend domain(s), e.g., `https://yourdomain.com` |
| `SCHOLARDEVCLAW_ENABLE_HSTS` | Enable HSTS headers | `true` for production |

### Convex (State Management)
| Variable | Description | How to Get |
|----------|-------------|------------|
| `CONVEX_URL` | Convex deployment URL | From `npx convex deploy` output |
| `SCHOLARDEVCLAW_CONVEX_AUTH_KEY` | Convex auth key | `python -c "import secrets; print(secrets.token_urlsafe(32))"` |

### LLM Provider (At Least One Required)
| Variable | Provider | How to Get |
|----------|----------|------------|
| `ANTHROPIC_API_KEY` | Anthropic (Claude) | https://console.anthropic.com/ |
| `OPENAI_API_KEY` | OpenAI | https://platform.openai.com/api-keys |
| `GEMINI_API_KEY` | Google Gemini | https://aistudio.google.com/app/apikey |
| `OPENROUTER_API_KEY` | OpenRouter | https://openrouter.ai/keys |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI | Azure Portal |
| `GROQ_API_KEY` | Groq | https://console.groq.com/ |
| `MISTRAL_API_KEY` | Mistral | https://console.mistral.ai/ |
| `DEEPSEEK_API_KEY` | DeepSeek | https://platform.deepseek.com/ |
| `COHERE_API_KEY` | Cohere | https://dashboard.cohere.com/ |
| `XAI_API_KEY` | xAI (Grok) | https://console.x.ai/ |
| `MOONSHOT_API_KEY` | Moonshot | https://platform.moonshot.cn/ |
| `GLM_API_KEY` | GLM (Zhipu) | https://open.bigmodel.cn/ |
| `MINIMAX_API_KEY` | MiniMax | https://platform.minimax.io/ |
| `TOGETHER_API_KEY` | Together AI | https://api.together.xyz/ |
| `FIREWORKS_API_KEY` | Fireworks AI | https://fireworks.ai/ |

### LLM Provider Selection
| Variable | Description | Options |
|----------|-------------|---------|
| `SCHOLARDEVCLAW_API_PROVIDER` | Which LLM to use | `anthropic`, `openai`, `gemini`, `openrouter`, etc. |
| `SCHOLARDEVCLAW_API_MODEL` | Specific model name | e.g., `claude-3-opus-20240229`, `gpt-4-turbo-preview` |

### GitHub
| Variable | Description | How to Get |
|----------|-------------|------------|
| `GITHUB_TOKEN` | GitHub personal access token | https://github.com/settings/tokens (repo scope) |

---

## Optional Credentials (Feature-Specific)

### OpenClaw Agent
| Variable | Description | How to Get |
|----------|-------------|------------|
| `OPENCLAW_TOKEN` | OpenClaw agent token | From OpenClaw service |
| `OPENCLAW_API_URL` | OpenClaw API URL | Default: `http://localhost:3000` |

### GitHub App (for PR automation)
| Variable | Description | How to Get |
|----------|-------------|------------|
| `GITHUB_APP_ID` | GitHub App ID | From GitHub App settings |
| `GITHUB_APP_PRIVATE_KEY` | GitHub App private key | From GitHub App settings |
| `GITHUB_APP_INSTALLATION_ID` | Installation ID | From GitHub App settings |
| `GITHUB_APP_WEBHOOK_SECRET` | Webhook secret | From GitHub App settings |
| `GITHUB_APP_REPOS` | Allowed repos | Comma-separated `owner/repo` |
| `GITHUB_APP_AUTO_APPLY` | Auto-apply PRs | `true` or `false` |
| `GITHUB_APP_REQUIRE_APPROVAL` | Require approval before PR | `true` or `false` |

### Execution & Sandbox
| Variable | Description | Default |
|----------|-------------|---------|
| `CORE_API_URL` | Python core API URL | `http://localhost:8000` |
| `PYTHON_COMMAND` | Python interpreter | `python3` |
| `CORE_BRIDGE_MODE` | Bridge mode | `http` (or `subprocess`) |
| `DEFAULT_MODE` | Execution mode | `step_approval` or `autonomous` |
| `SCHOLARDEVCLAW_VALIDATION_SANDBOX` | Sandbox type | `docker` or `none` |
| `SCHOLARDEVCLAW_VALIDATION_DOCKER_IMAGE` | Docker image | `python:3.10-slim` |
| `SCHOLARDEVCLAW_ROLLBACK_DIR` | Rollback directory | `/tmp/scholardevclaw_rollbacks` |

### Monitoring
| Variable | Description | How to Get |
|----------|-------------|------------|
| `GRAFANA_ADMIN_PASSWORD` | Grafana password | Set in Grafana |

### OAuth (Optional)
| Variable | Description |
|----------|-------------|
| `OAUTH_CLIENT_ID` | OAuth client ID |
| `OAUTH_CLIENT_SECRET` | OAuth client secret |
| `OAUTH_SCOPES` | OAuth scopes |

### TUI Settings
| Variable | Description | Default |
|----------|-------------|---------|
| `SCHOLARDEVCLAW_TUI_APPROVAL_GATES` | Enable approval gates | `true` |
| `SCHOLARDEVCLAW_TUI_PHASE9_AUTO_APPROVE` | Auto-approve all phases | `false` |

### Development
| Variable | Description | Default |
|----------|-------------|---------|
| `SCHOLARDEVCLAW_DEV_MODE` | Development mode | `false` |

---

## Minimum Viable Production Setup

For a basic production setup, you need these **minimum** credentials:

```bash
# 1. Core System
SCHOLARDEVCLAW_API_AUTH_KEY=<generate-with-python>
SCHOLARDEVCLAW_ALLOWED_REPO_DIRS=/home/user/projects
SCHOLARDEVCLAW_CORS_ORIGINS=https://yourdomain.com
SCHOLARDEVCLAW_ENABLE_HSTS=true

# 2. Convex
CONVEX_URL=https://your-deployment.convex.cloud
SCHOLARDEVCLAW_CONVEX_AUTH_KEY=<generate-with-python>

# 3. LLM Provider (choose one)
ANTHROPIC_API_KEY=<from-anthropic-console>
# OR
OPENAI_API_KEY=<from-openai-platform>
# OR
OPENROUTER_API_KEY=<from-openrouter>

# 4. LLM Selection
SCHOLARDEVCLAW_API_PROVIDER=anthropic  # or openai, openrouter, etc.
SCHOLARDEVCLAW_API_MODEL=claude-3-opus-20240229  # or your preferred model

# 5. GitHub
GITHUB_TOKEN=<from-github-settings-tokens>

# 6. Execution
CORE_API_URL=http://localhost:8000
DEFAULT_MODE=step_approval
SCHOLARDEVCLAW_VALIDATION_SANDBOX=docker
```

---

## Quick Setup Commands

### Generate Secure Keys
```bash
# Generate API auth key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate Convex auth key
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Verify Setup
```bash
# Test Python core
cd core
python -m scholardevclaw.cli --help

# Test agent
cd agent
bun run build

# Test API server
cd core
uvicorn scholardevclaw.api.server:app --reload
```

---

## Environment File Template

Copy this to your `.env` file:

```bash
# ============================================
# ScholarDevClaw Production Environment
# ============================================

# --- CORE SYSTEM (Required) ---
SCHOLARDEVCLAW_API_AUTH_KEY=<generate>
SCHOLARDEVCLAW_ALLOWED_REPO_DIRS=/home/user/projects
SCHOLARDEVCLAW_CORS_ORIGINS=https://yourdomain.com
SCHOLARDEVCLAW_ENABLE_HSTS=true

# --- CONVEX (Required) ---
CONVEX_URL=https://your-deployment.convex.cloud
SCHOLARDEVCLAW_CONVEX_AUTH_KEY=<generate>

# --- LLM PROVIDER (At least one required) ---
ANTHROPIC_API_KEY=<your-key>
SCHOLARDEVCLAW_API_PROVIDER=anthropic
SCHOLARDEVCLAW_API_MODEL=claude-3-opus-20240229

# --- GITHUB (Required for repo access) ---
GITHUB_TOKEN=<your-token>

# --- EXECUTION ---
CORE_API_URL=http://localhost:8000
DEFAULT_MODE=step_approval
SCHOLARDEVCLAW_VALIDATION_SANDBOX=docker

# --- OPTIONAL ---
# OPENCLAW_TOKEN=<your-token>
# GITHUB_APP_ID=<your-app-id>
# GITHUB_APP_PRIVATE_KEY=<your-private-key>
```

---

## Security Notes

1. **Never commit `.env`** - It's gitignored by default
2. **Use strong keys** - Generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`
3. **Restrict CORS** - Don't use `*` in production
4. **Rotate keys periodically** - Especially LLM API keys
5. **Use HTTPS** - Set `SCHOLARDEVCLAW_ENABLE_HSTS=true` in production