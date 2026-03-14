import { logger } from './utils/logger.js';
import { ScholarDevClawOrchestrator } from './orchestrator.js';
import * as readline from 'readline';

function getArg(flag: string): string | undefined {
  const index = process.argv.indexOf(flag);
  if (index < 0 || index + 1 >= process.argv.length) {
    return undefined;
  }
  return process.argv[index + 1];
}

async function runRepl(): Promise<void> {
  console.log('🤖 ScholarDevClaw Agent - Interactive Mode');
  console.log('═'.repeat(50));
  console.log('Commands:');
  console.log('  analyze <path>     - Analyze a repository');
  console.log('  suggest <path>    - Get improvement suggestions');
  console.log('  integrate <path> <spec> - Run full integration');
  console.log('  help              - Show this help');
  console.log('  exit              - Exit interactive mode');
  console.log('═'.repeat(50));
  console.log('');

  const orchestrator = new ScholarDevClawOrchestrator(true);
  await orchestrator.initialize();

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: '🔮 ',
  });

  const currentContext: {
    repoPath?: string;
    paperSpec?: string;
  } = {};

  rl.prompt();

  rl.on('line', async (line) => {
    const input = line.trim();
    
    if (!input) {
      rl.prompt();
      return;
    }

    if (input.toLowerCase() === 'exit' || input.toLowerCase() === 'quit') {
      console.log('👋 Goodbye!');
      rl.close();
      return;
    }

    if (input.toLowerCase() === 'help') {
      console.log('📖 Available commands:');
      console.log('  analyze <path>              - Analyze a repository');
      console.log('  suggest <path>              - Get improvement suggestions');
      console.log('  search <query>              - Search research papers');
      console.log('  integrate <path> [spec]     - Run full integration workflow');
      console.log('  set repo <path>             - Set default repository path');
      console.log('  set spec <spec>             - Set default spec name');
      console.log('  context                     - Show current context');
      console.log('  help                        - Show this help');
      console.log('  exit                        - Exit interactive mode');
      console.log('');
      rl.prompt();
      return;
    }

    if (input.toLowerCase() === 'context') {
      console.log('📋 Current context:');
      console.log(`  Repository: ${currentContext.repoPath || '(not set)'}`);
      console.log(`  Spec: ${currentContext.paperSpec || '(not set)'}`);
      console.log('');
      rl.prompt();
      return;
    }

    const parts = input.split(/\s+/);
    const command = parts[0].toLowerCase();
    const args = parts.slice(1);

    try {
      switch (command) {
        case 'set': {
          if (args[0]?.toLowerCase() === 'repo' && args[1]) {
            currentContext.repoPath = args.slice(1).join(' ');
            console.log(`✅ Set repository to: ${currentContext.repoPath}`);
          } else if (args[0]?.toLowerCase() === 'spec' && args[1]) {
            currentContext.paperSpec = args[1];
            console.log(`✅ Set spec to: ${currentContext.paperSpec}`);
          } else {
            console.log('Usage: set repo <path> | set spec <name>');
          }
          break;
        }

        case 'analyze': {
          const path = args[0] || currentContext.repoPath;
          if (!path) {
            console.log('❌ Please provide a path or set default repo: set repo <path>');
          } else {
            console.log(`🔍 Analyzing: ${path}`);
            await orchestrator.runIntegration({
              repoUrl: path,
              mode: 'autonomous',
            });
            console.log('✅ Analysis complete');
          }
          break;
        }

        case 'suggest': {
          const path = args[0] || currentContext.repoPath;
          if (!path) {
            console.log('❌ Please provide a path or set default repo');
          } else {
            console.log(`💡 Getting suggestions for: ${path}`);
            await orchestrator.runIntegration({
              repoUrl: path,
              mode: 'autonomous',
            });
            console.log('✅ Suggestions complete');
          }
          break;
        }

        case 'search': {
          const query = args.join(' ') || 'machine learning';
          console.log(`🔎 Searching for: ${query}`);
          console.log('(Search functionality coming soon in REPL mode)');
          break;
        }

        case 'integrate': {
          const path = args[0] || currentContext.repoPath;
          const spec = args[1] || currentContext.paperSpec || 'rmsnorm';
          if (!path) {
            console.log('❌ Please provide a path or set default repo');
          } else {
            console.log(`🚀 Running integration: ${path} with ${spec}`);
            await orchestrator.runIntegration({
              repoUrl: path,
              paperUrl: `spec:${spec}`,
              mode: 'autonomous',
            });
            console.log('✅ Integration complete');
          }
          break;
        }

        default: {
          console.log(`❓ Unknown command: ${command}`);
          console.log('   Type "help" for available commands');
        }
      }
    } catch (error) {
      console.log(`❌ Error: ${error instanceof Error ? error.message : String(error)}`);
    }

    console.log('');
    rl.prompt();
  });

  rl.on('close', () => {
    process.exit(0);
  });
}

async function run(): Promise<void> {
  const command = process.argv[2] || 'heartbeat';
  
  // Check for REPL mode
  if (command === '--repl' || command === 'repl') {
    await runRepl();
    return;
  }

  const useHttp = process.env.CORE_BRIDGE_MODE !== 'subprocess';
  const orchestrator = new ScholarDevClawOrchestrator(useHttp);
  await orchestrator.initialize();

  if (command === 'run') {
    const repoUrl = getArg('--repo') || getArg('--repo-url');
    const paperUrl = getArg('--paper-url');
    const paperPdfPath = getArg('--paper-pdf');
    const mode = getArg('--mode') as 'step_approval' | 'autonomous' | undefined;

    if (!repoUrl) {
      throw new Error('Missing required argument: --repo <path-or-url>');
    }
    if (!paperUrl && !paperPdfPath) {
      throw new Error('Provide one research source: --paper-url <arxiv-id/url> or --paper-pdf <path>');
    }

    await orchestrator.runIntegration({
      repoUrl,
      paperUrl,
      paperPdfPath,
      mode,
    });

    logger.info('Run command completed');
    return;
  }

  if (command === 'resume') {
    const runId = getArg('--run-id');
    if (!runId) {
      throw new Error('Missing required argument: --run-id <run-id>');
    }

    const resumed = await orchestrator.resumeRun(runId);
    if (!resumed) {
      throw new Error(`Resume failed for run id: ${runId}`);
    }

    logger.info('Resume command completed', { runId });
    return;
  }

  await orchestrator.processPendingWork();
  logger.info('Heartbeat check completed');
}

run().catch((err: unknown) => {
  const message = err instanceof Error ? err.message : 'Unknown error';
  logger.error('Agent execution failed', { error: message });
  process.exitCode = 1;
});
