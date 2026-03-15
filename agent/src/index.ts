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
  paperUrl?: string;
}

function parseNaturalInput(input: string): ParsedCommand {
  const lower = input.toLowerCase().trim();
  
  // Default to help
  const result: ParsedCommand = { command: 'help' };
  
  // Extract repo path - look for paths starting with /, ~/, or ./ 
  const pathMatch = input.match(/(?:\s|^)([\/~.][^\s]+|\/[a-zA-Z]:[^\s]+)/);
  if (pathMatch) {
    result.repoPath = pathMatch[1];
  }
  
  // Extract spec name
  const specs = ['rmsnorm', 'flashattention', 'swiglu', 'geglu', 'gqa', 'rope', 'preln', 'alibi', 'qknorm'];
  for (const spec of specs) {
    if (lower.includes(spec)) {
      result.spec = spec;
      break;
    }
  }
  
  // Detect command intent
  if (lower.includes('analyze') || lower.includes('scan') || lower.includes('inspect')) {
    result.command = 'analyze';
  } else if (lower.includes('suggest') || lower.includes('recommend') || lower.includes('ideas')) {
    result.command = 'suggest';
  } else if (lower.includes('integrate') || lower.includes('apply') || lower.includes('implement') || lower.includes('add')) {
    result.command = 'integrate';
  } else if (lower.includes('search') || lower.includes('find') || lower.includes('look for')) {
    result.command = 'search';
    // Extract query
    const queryMatch = lower.match(/(?:search|find|look for)\s+(?:for\s+)?["']?(.+?)(?:["']|$)/);
    if (queryMatch) {
      result.query = queryMatch[1].trim();
    } else if (!result.query) {
      // Use remaining text as query
      let q = lower.replace(/(?:search|find|look for|for)/g, '').trim();
      if (q) result.query = q;
    }
  } else if (lower.includes('map') || lower.includes('connect')) {
    result.command = 'map';
  } else if (lower.includes('generate') || lower.includes('create patch')) {
    result.command = 'generate';
  } else if (lower.includes('validate') || lower.includes('test')) {
    result.command = 'validate';
  } else if (lower.includes('specs') || lower.includes('list') || lower.includes('show')) {
    result.command = 'specs';
  } else if (lower.startsWith('set ')) {
    result.command = 'set';
  } else if (lower === 'context' || lower === 'status') {
    result.command = 'context';
  }
  
  // If no explicit command but has repo path, assume analyze
  if (result.command === 'help' && result.repoPath) {
    result.command = 'analyze';
  }
  
  return result;
}

async function runRepl(): Promise<void> {
  console.log('🤖 ScholarDevClaw Agent - Interactive Mode');
  console.log('═'.repeat(50));
  console.log('💡 Type naturally! Examples:');
  console.log('   → "analyze /path/to/repo"');
  console.log('   → "apply rmsnorm to my project"');
  console.log('   → "suggest improvements for /repo"');
  console.log('   → "search for transformer attention"');
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

  // Banner
  console.log('✅ Agent ready! Type your request or "help" for commands.\n');

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
      console.log('📖 Commands:');
      console.log('  analyze <path>         - Analyze a repository');
      console.log('  suggest <path>         - Get improvement suggestions');
      console.log('  search <query>        - Search research papers');
      console.log('  integrate <path> [spec] - Run full integration');
      console.log('  map <path> <spec>     - Map spec to repository');
      console.log('  generate <path> <spec> - Generate patch');
      console.log('  validate <path>      - Validate repository');
      console.log('  specs                 - List available specs');
      console.log('  set repo <path>       - Set default repository');
      console.log('  set spec <name>       - Set default spec');
      console.log('  context               - Show current context');
      console.log('');
      console.log('💡 Natural language works too!');
      console.log('   "apply rmsnorm to /path/to/repo"');
      console.log('   "what improvements can you suggest for my project?"\n');
      rl.prompt();
      return;
    }

    if (input.toLowerCase() === 'context' || input.toLowerCase() === 'status') {
      console.log('📋 Current context:');
      console.log(`  Repository: ${currentContext.repoPath || '— (not set)'}`);
      console.log(`  Spec: ${currentContext.paperSpec || '— (not set)'}`);
      console.log('');
      rl.prompt();
      return;
    }

    // Parse natural language
    const parsed = parseNaturalInput(input);
    
    // Override with explicit values from input if present
    const parts = input.split(/\s+/);
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
        case 'set': {
          if (parsed.repoPath) {
            currentContext.repoPath = parsed.repoPath;
            console.log(`✅ Set repository to: ${parsed.repoPath}`);
          } else if (parsed.spec) {
            currentContext.paperSpec = parsed.spec;
            console.log(`✅ Set spec to: ${parsed.spec}`);
          } else {
            console.log('Usage: set repo <path> | set spec <name>');
          }
          break;
        }

        case 'analyze': {
          const path = parsed.repoPath || currentContext.repoPath;
          if (!path) {
            console.log('❌ Please provide a path: analyze <path>');
            console.log('   Or set default: set repo /path/to/repo');
          } else {
            console.log(`🔍 Analyzing: ${path}`);
            console.log('⏳ This may take a moment...');
            await orchestrator.runIntegration({ repoUrl: path, mode: 'autonomous' });
            console.log('✅ Analysis complete!');
          }
          break;
        }

        case 'suggest': {
          const path = parsed.repoPath || currentContext.repoPath;
          if (!path) {
            console.log('❌ Please provide a path');
          } else {
            console.log(`💡 Getting suggestions for: ${path}`);
            await orchestrator.runIntegration({ repoUrl: path, mode: 'autonomous' });
            console.log('✅ Suggestions ready!');
          }
          break;
        }

        case 'search': {
          const query = parsed.query || input.replace(/^(search|find)\s+/i, '').trim() || 'machine learning';
          console.log(`🔎 Searching for: ${query}`);
          console.log('(Search via REPL coming soon - use TUI for now)');
          break;
        }

        case 'integrate': {
          const path = parsed.repoPath || currentContext.repoPath;
          const spec = parsed.spec || currentContext.paperSpec || 'rmsnorm';
          if (!path) {
            console.log('❌ Please provide a path');
          } else {
            console.log(`🚀 Integrating ${spec} into: ${path}`);
            console.log('⏳ Running full workflow...');
            await orchestrator.runIntegration({
              repoUrl: path,
              paperUrl: `spec:${spec}`,
              mode: 'autonomous',
            });
            console.log('✅ Integration complete!');
          }
          break;
        }

        case 'map': {
          const path = parsed.repoPath || currentContext.repoPath;
          const spec = parsed.spec || currentContext.paperSpec || 'rmsnorm';
          if (!path) {
            console.log('❌ Please provide a path');
          } else {
            console.log(`🗺 Mapping ${spec} to: ${path}`);
            await orchestrator.runIntegration({ repoUrl: path, mode: 'autonomous' });
            console.log('✅ Mapping complete!');
          }
          break;
        }

        case 'generate': {
          const path = parsed.repoPath || currentContext.repoPath;
          const spec = parsed.spec || currentContext.paperSpec || 'rmsnorm';
          if (!path) {
            console.log('❌ Please provide a path');
          } else {
            console.log(`⚡ Generating patches for: ${path}`);
            await orchestrator.runIntegration({ repoUrl: path, mode: 'autonomous' });
            console.log('✅ Generation complete!');
          }
          break;
        }

        case 'validate': {
          const path = parsed.repoPath || currentContext.repoPath;
          if (!path) {
            console.log('❌ Please provide a path');
          } else {
            console.log(`✅ Validating: ${path}`);
            await orchestrator.runIntegration({ repoUrl: path, mode: 'autonomous' });
            console.log('✅ Validation complete!');
          }
          break;
        }

        case 'specs': {
          console.log('📋 Available specs:');
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
        }

        default: {
          console.log(`❓ "${input}" - I didn\'t understand that.`);
          console.log('   Type "help" for available commands.');
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
