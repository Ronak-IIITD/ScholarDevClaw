# Security Policy

## 🔒 Reporting Security Vulnerabilities

If you discover a security vulnerability within ScholarDevClaw, please report it responsibly.

### How to Report

1. **Do NOT** create a public GitHub issue for security vulnerabilities
2. Email the maintainers directly or use GitHub's private vulnerability reporting
3. Include as much detail as possible:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (optional)

### What to Expect

- Acknowledgment within 48 hours
- Regular updates on progress
- Credit in the security advisory (if desired)

## 🔐 Security Best Practices

### API Keys and Tokens

- Never commit API keys or tokens to the repository
- Use environment variables for all sensitive configuration
- Rotate keys regularly

### Input Validation

ScholarDevClaw validates:
- Repository paths before processing
- arXiv IDs and paper sources
- User input in web forms

### Subprocess Execution

Validation runs in sandboxed environments:
- Docker isolation for untrusted code
- Host execution with strict mode disabled by default
- Timeout limits on all subprocess calls

### Path Security

- Repository paths are validated against allowed directories
- Path traversal attacks are blocked
- Temporary files are created in secure locations

## 🛡️ Dependencies

- Regular dependency updates via security advisories
- Python packages are pinned for reproducibility
- TypeScript dependencies are audited regularly

## 📋 Security Audit

A comprehensive security audit was conducted on 2026-05-06. See [SECURITY_UPDATES.md](./SECURITY_UPDATES.md) for details and fixes.

## 🔑 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SCHOLARDEVCLAW_API_AUTH_KEY` | API authentication key | Production |
| `ANTHROPIC_API_KEY` | Claude API key for LLM features | Optional |
| `GITHUB_TOKEN` | GitHub API token | Optional |
| `OPENCLAW_TOKEN` | OpenClaw integration token | Optional |
| `SCHOLARDEVCLAW_CORS_ORIGINS` | Allowed CORS origins | Production |

## 🚨 Known Limitations

- Multi-server deployments require external state management (Redis/Convex)
- OpenClaw integration is planned for future phases
- Dashboard state is single-process (documented limitation)

For questions about security, please open a discussion or contact maintainers.