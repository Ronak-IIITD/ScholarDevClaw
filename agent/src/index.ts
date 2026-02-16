import { logger } from './utils/logger.js';
import { ScholarDevClawOrchestrator } from './orchestrator.js';

function getArg(flag: string): string | undefined {
  const index = process.argv.indexOf(flag);
  if (index < 0 || index + 1 >= process.argv.length) {
    return undefined;
  }
  return process.argv[index + 1];
}

async function run(): Promise<void> {
  const command = process.argv[2] || 'heartbeat';
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
