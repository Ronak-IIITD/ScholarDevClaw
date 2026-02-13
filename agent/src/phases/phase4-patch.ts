import { logger } from '../utils/logger.js';
import type { PythonBridge } from '../bridges/python-bridge.js';
import type { Phase4Context, PhaseResult } from './types.js';
import type { PatchResult } from '../bridges/python-subprocess.js';

export async function executePhase4(
  bridge: PythonBridge,
  context: Phase4Context
): Promise<PhaseResult<PatchResult>> {
  logger.info('=== Phase 4: Patch Generation ===');

  try {
    const result = await bridge.generatePatch(context.mapping);

    if (!result.success) {
      logger.error('Phase 4 failed', { error: result.error });
      return { success: false, error: result.error };
    }

    const patch = result.data as PatchResult;
    
    logger.info('Phase 4 completed', {
      newFiles: patch.newFiles.length,
      transformations: patch.transformations.length,
      branchName: patch.branchName,
    });

    return {
      success: true,
      data: patch,
      confidence: 90,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    logger.error('Phase 4 error', { error: message });
    return { success: false, error: message };
  }
}

export function validatePhase4Input(context: Phase4Context): string | null {
  if (!context.mapping) {
    return 'Mapping result is required (complete Phase 3)';
  }

  if (context.mapping.targets.length === 0) {
    return 'No insertion targets found - cannot generate patch';
  }

  return null;
}
