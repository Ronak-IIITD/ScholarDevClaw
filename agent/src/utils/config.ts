import 'dotenv/config';

export const config = {
  openclaw: {
    token: process.env.OPENCLAW_TOKEN || 'dev-token',
    apiUrl: process.env.OPENCLAW_API_URL || 'http://localhost:3000',
  },
  convex: {
    deploymentUrl: process.env.CONVEX_URL || '',
  },
  github: {
    token: process.env.GITHUB_TOKEN || '',
  },
  anthropic: {
    apiKey: process.env.ANTHROPIC_API_KEY || '',
  },
  python: {
    coreApiUrl: process.env.CORE_API_URL || 'http://localhost:8000',
    subprocessCommand: process.env.PYTHON_COMMAND || 'python3',
  },
  execution: {
    defaultMode: process.env.DEFAULT_MODE || 'step_approval', // 'step_approval' | 'autonomous'
    maxRetries: 2,
    benchmarkTimeout: 300, // 5 minutes
  },
};
