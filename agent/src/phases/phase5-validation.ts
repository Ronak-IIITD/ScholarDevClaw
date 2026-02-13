import { logger } from '../utils/logger.js';
import type { PythonBridge } from '../bridges/python-bridge.js';
import type { Phase5Context, PhaseResult } from './types.js';
import type { ValidationResult } from '../bridges/python-subprocess.js';

export async function executePhase5(
  bridge: PythonBridge,
  context: Phase5Context,
  repoPath: string
): Promise<PhaseResult<ValidationResult>> {
  logger.info('=== Phase 5: Validation Engine ===');

  try {
    const result = await bridge.validate(context.patch, repoPath);

    if (!result.success) {
      logger.error('Phase 5 failed', { error: result.error });
      return { success: false, error: result.error };
    }

    const validation = result.data as ValidationResult;
    
    logger.info('Phase 5 completed', {
      passed: validation.passed,
      stage: validation.stage,
      speedup: validation.comparison?.speedup,
    });

    return {
      success: true,
      data: validation,
      confidence: validation.passed ? 95 : 50,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    logger.error('Phase 5 error', { error: message });
    return { success: false, error: message };
  }
}

export function validatePhase5Input(context: Phase5Context): string | null {
  if (!context.patch) {
    return 'Patch is required (complete Phase 4)';
  }

  return null;
}
