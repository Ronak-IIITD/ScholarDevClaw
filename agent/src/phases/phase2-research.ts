import { logger } from '../utils/logger.js';
import type { PythonBridge } from '../bridges/python-bridge.js';
import type { Phase2Context, PhaseResult } from './types.js';
import type { ResearchSpecResult } from '../bridges/python-subprocess.js';

export async function executePhase2(
  bridge: PythonBridge,
  context: Phase2Context
): Promise<PhaseResult<ResearchSpecResult>> {
  logger.info('=== Phase 2: Research Intelligence ===', { 
    paperSource: context.paperSource,
    sourceType: context.sourceType,
  });

  try {
    const result = await bridge.extractResearch(context.paperSource, context.sourceType);

    if (!result.success) {
      logger.error('Phase 2 failed', { error: result.error });
      return { success: false, error: result.error };
    }

    const spec = result.data as ResearchSpecResult;
    
    const confidence = calculateResearchConfidence(spec);
    
    logger.info('Phase 2 completed', {
      algorithmName: spec.algorithm.name,
      confidence,
    });

    return {
      success: true,
      data: spec,
      confidence,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    logger.error('Phase 2 error', { error: message });
    return { success: false, error: message };
  }
}

function calculateResearchConfidence(spec: ResearchSpecResult): number {
  let confidence = 60;

  if (spec.paper.arxiv) {
    confidence += 15;
  }

  if (spec.implementation.codeTemplate) {
    confidence += 15;
  }

  if (spec.changes.targetPattern) {
    confidence += 10;
  }

  return Math.min(confidence, 100);
}

export function validatePhase2Input(context: Phase2Context): string | null {
  if (!context.paperSource) {
    return 'Paper source is required';
  }

  if (!context.sourceType) {
    return 'Source type (pdf or arxiv) is required';
  }

  return null;
}
