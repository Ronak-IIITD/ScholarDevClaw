import { ScholarDevClawOrchestrator } from './orchestrator.js';
import { logger } from './utils/logger.js';

async function main() {
  logger.info('ScholarDevClaw Agent Starting...');

  const orchestrator = new ScholarDevClawOrchestrator();
  
  await orchestrator.initialize();

  logger.info('Agent ready');
}

main().catch((err) => {
  logger.error('Fatal error', { error: err.message });
  process.exit(1);
});
