import { logger } from '../utils/logger.js';
import type { PythonBridge } from '../bridges/python-bridge.js';
import type { Phase3Context, PhaseResult } from './types.js';
import type { MappingResult } from '../bridges/python-subprocess.js';

export async function executePhase3(
  bridge: PythonBridge,
  context: Phase3Context
): Promise<PhaseResult<MappingResult>> {
  logger.info('=== Phase 3: Mapping Engine ===');

  try {
    const result = await bridge.mapArchitecture(
      context.repoAnalysis,
      context.researchSpec
    );

    if (!result.success) {
      logger.error('Phase 3 failed', { error: result.error });
      return { success: false, error: result.error };
    }

    const mapping = result.data as MappingResult;
    
    logger.info('Phase 3 completed', {
      targetsFound: mapping.targets.length,
      strategy: mapping.strategy,
      confidence: mapping.confidence,
    });

    return {
      success: true,
      data: mapping,
      confidence: mapping.confidence,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    logger.error('Phase 3 error', { error: message });
    return { success: false, error: message };
  }
}

export function validatePhase3Input(context: Phase3Context): string | null {
  if (!context.repoAnalysis) {
    return 'Repository analysis is required (complete Phase 1)';
  }

  if (!context.researchSpec) {
    return 'Research specification is required (complete Phase 2)';
  }

  return null;
}
