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

interface ParsedCommand {
  command: string;
  repoPath?: string;
  spec?: string;
  query?: string;
}

function parseNaturalInput(input: string): ParsedCommand {
  const lower = input.toLowerCase().trim();
  const result: ParsedCommand = { command: 'help' };
  
  const pathMatch = input.match(/(?:\s|^)([\/~.][^\s]+)/);
  if (pathMatch) {
    result.repoPath = pathMatch[1];
  }
  
  const specs = ['rmsnorm', 'flashattention', 'swiglu', 'geglu', 'gqa', 'rope', 'preln', 'alibi', 'qknorm'];
  for (const spec of specs) {
    if (lower.includes(spec)) {
      result.spec = spec;
      break;
    }
  }
  
  if (lower.includes('analyze') || lower.includes('scan') || lower.includes('inspect')) {
    result.command = 'analyze';
  } else if (lower.includes('suggest') || lower.includes('recommend') || lower.includes('ideas')) {
    result.command = 'suggest';
  } else if (lower.includes('integrate') || lower.includes('apply') || lower.includes('implement')) {
    result.command = 'integrate';
  } else if (lower.includes('search') || lower.includes('find')) {
    result.command = 'search';
    const queryMatch = lower.match(/(?:search|find)\s+(?:for\s+)?["']?(.+)/);
    if (queryMatch) result.query = queryMatch[1].trim();
  } else if (lower.includes('map')) {
    result.command = 'map';
  } else if (lower.includes('generate')) {
    result.command = 'generate';
  } else if (lower.includes('validate') || lower.includes('test')) {
    result.command = 'validate';
  } else if (lower.includes('specs') || lower.includes('list')) {
    result.command = 'specs';
  } else if (lower.startsWith('set ')) {
    result.command = 'set';
  } else if (result.repoPath) {
    result.command = 'analyze';
  }
  
  return result;
}

async function runRepl(): Promise<void> {
  console.log('ScholarDevClaw Agent - Interactive Mode');
  console.log('='.repeat(40));
  console.log('Commands: analyze, suggest, integrate, search, map, generate, validate, specs');
  console.log('Usage: <command> <path> [spec]');
  console.log('Or: set repo <path>, set spec <name>');
  console.log('='.repeat(40));
  console.log('');

  const orchestrator = new ScholarDevClawOrchestrator(true);
  await orchestrator.initialize();

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: '> ',
  });

  const currentContext: { repoPath?: string; paperSpec?: string } = {};

  console.log('Agent ready. Type "help" for commands.\n');
  rl.prompt();

  rl.on('line', async (line) => {
    const input = line.trim();
    
    if (!input) {
      rl.prompt();
      return;
    }

    if (input.toLowerCase() === 'exit' || input.toLowerCase() === 'quit') {
      console.log('Goodbye!');
      rl.close();
      return;
    }

    if (input.toLowerCase() === 'help') {
      console.log('Commands:');
      console.log('  analyze <path>       - Analyze a repository');
      console.log('  suggest <path>      - Get improvement suggestions');
      console.log('  search <query>     - Search research papers');
      console.log('  integrate <path> [spec] - Run full integration');
      console.log('  map <path> <spec>  - Map spec to repository');
      console.log('  generate <path> <spec> - Generate patches');
      console.log('  validate <path>    - Validate repository');
      console.log('  specs              - List available specs');
      console.log('  set repo <path>   - Set default repository');
      console.log('  set spec <name>   - Set default spec');
      console.log('');
      console.log('Natural language also works!');
      console.log('  apply rmsnorm to /path/to/repo');
      console.log('  suggest improvements for /repo\n');
      rl.prompt();
      return;
    }

    if (input.toLowerCase() === 'context') {
      console.log('Context:');
      console.log(`  Repository: ${currentContext.repoPath || '(not set)'}`);
      console.log(`  Spec: ${currentContext.paperSpec || '(not set)'}`);
      console.log('');
      rl.prompt();
      return;
    }

    const parts = input.split(/\s+/);
    const parsed = parseNaturalInput(input);
    
    if (parts[0] === 'set' && parts[1] === 'repo' && parts[2]) {
      parsed.command = 'set';
      parsed.repoPath = parts.slice(2).join(' ');
    }
    if (parts[0] === 'set' && parts[1] === 'spec' && parts[2]) {
      parsed.command = 'set';
      parsed.spec = parts[2];
    }
    if (['analyze', 'suggest', 'integrate', 'map', 'generate', 'validate'].includes(parts[0]) && parts[1]) {
      parsed.command = parts[0];
      if (!parsed.repoPath) parsed.repoPath = parts[1];
    }
    if (parts[0] === 'search' && parts[1]) {
      parsed.command = 'search';
      parsed.query = parts.slice(1).join(' ');
    }

    try {
      switch (parsed.command) {
        case 'set':
          if (parsed.repoPath) {
            currentContext.repoPath = parsed.repoPath;
            console.log(`Repository set to: ${parsed.repoPath}`);
          } else if (parsed.spec) {
            currentContext.paperSpec = parsed.spec;
            console.log(`Spec set to: ${parsed.spec}`);
          }
          break;

        case 'analyze':
          const analyzePath = parsed.repoPath || currentContext.repoPath;
          if (!analyzePath) {
            console.log('Error: Provide path or set default: set repo /path');
          } else {
            console.log(`Analyzing: ${analyzePath}`);
            await orchestrator.runIntegration({ repoUrl: analyzePath, mode: 'autonomous' });
            console.log('Analysis complete');
          }
          break;

        case 'suggest':
          const suggestPath = parsed.repoPath || currentContext.repoPath;
          if (!suggestPath) {
            console.log('Error: Provide path');
          } else {
            console.log(`Getting suggestions for: ${suggestPath}`);
            await orchestrator.runIntegration({ repoUrl: suggestPath, mode: 'autonomous' });
            console.log('Suggestions ready');
          }
          break;

        case 'search':
          const query = parsed.query || input.replace(/^(search|find)\s+/i, '').trim() || 'machine learning';
          console.log(`Searching for: ${query}`);
          console.log('(Search via TUI for full results)');
          break;

        case 'integrate':
          const intPath = parsed.repoPath || currentContext.repoPath;
          const spec = parsed.spec || currentContext.paperSpec || 'rmsnorm';
          if (!intPath) {
            console.log('Error: Provide path');
          } else {
            console.log(`Integrating ${spec} into: ${intPath}`);
            await orchestrator.runIntegration({ repoUrl: intPath, paperUrl: `spec:${spec}`, mode: 'autonomous' });
            console.log('Integration complete');
          }
          break;

        case 'map':
          const mapPath = parsed.repoPath || currentContext.repoPath;
          const mapSpec = parsed.spec || currentContext.paperSpec || 'rmsnorm';
          if (!mapPath) console.log('Error: Provide path');
          else {
            console.log(`Mapping ${mapSpec} to: ${mapPath}`);
            await orchestrator.runIntegration({ repoUrl: mapPath, mode: 'autonomous' });
          }
          break;

        case 'generate':
          const genPath = parsed.repoPath || currentContext.repoPath;
          const genSpec = parsed.spec || currentContext.paperSpec || 'rmsnorm';
          if (!genPath) console.log('Error: Provide path');
          else {
            console.log(`Generating patches for: ${genPath}`);
            await orchestrator.runIntegration({ repoUrl: genPath, mode: 'autonomous' });
          }
          break;

        case 'validate':
          const valPath = parsed.repoPath || currentContext.repoPath;
          if (!valPath) console.log('Error: Provide path');
          else {
            console.log(`Validating: ${valPath}`);
            await orchestrator.runIntegration({ repoUrl: valPath, mode: 'autonomous' });
          }
          break;

        case 'specs':
          console.log('Available specs:');
          console.log('  rmsnorm        - Root Mean Square Layer Normalization');
          console.log('  flashattention - Flash Attention');
          console.log('  swiglu         - SwiGLU Activation');
          console.log('  geglu          - GEGLU Activation');
          console.log('  gqa            - Grouped Query Attention');
          console.log('  rope           - Rotary Position Embedding');
          console.log('  preln          - Pre-Layer Normalization');
          console.log('  alibi          - ALiBi Position Embedding');
          console.log('  qknorm         - Query-Key Normalization');
          break;

        default:
          console.log(`Unknown: ${input}. Type "help" for commands.`);
      }
    } catch (error) {
      console.log(`Error: ${error instanceof Error ? error.message : String(error)}`);
    }

    console.log('');
    rl.prompt();
  });

  rl.on('close', () => process.exit(0));
}

async function run(): Promise<void> {
  const command = process.argv[2] || 'heartbeat';
  
  if (command === '--repl' || command === 'repl') {
    await runRepl();
    return;
  }

  if (command === 'tui' || command === '--tui') {
    // Delegate to OpenTUI-based terminal interface
    const { main: runOpenTui } = await import('./tui/opentui-app.js');
    await runOpenTui();
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

    if (!repoUrl) throw new Error('Missing required argument: --repo <path-or-url>');
    if (!paperUrl && !paperPdfPath) throw new Error('Provide: --paper-url <url> or --paper-pdf <path>');

    await orchestrator.runIntegration({ repoUrl, paperUrl, paperPdfPath, mode });
    logger.info('Run completed');
    return;
  }

  if (command === 'resume') {
    const runId = getArg('--run-id');
    if (!runId) throw new Error('Missing required argument: --run-id <run-id>');
    const resumed = await orchestrator.resumeRun(runId);
    if (!resumed) throw new Error(`Resume failed for run id: ${runId}`);
    logger.info('Resume completed', { runId });
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
