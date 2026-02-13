import { logger } from '../utils/logger.js';
import type { PythonBridge } from '../bridges/python-bridge.js';
import type { Phase1Context, PhaseResult } from './types.js';
import type { RepoAnalysisResult } from '../bridges/python-subprocess.js';

export async function executePhase1(
  bridge: PythonBridge,
  context: Phase1Context
): Promise<PhaseResult<RepoAnalysisResult>> {
  logger.info('=== Phase 1: Repository Intelligence ===', { repoPath: context.repoPath });

  try {
    const result = await bridge.analyzeRepo(context.repoPath);

    if (!result.success) {
      logger.error('Phase 1 failed', { error: result.error });
      return { success: false, error: result.error };
    }

    const analysis = result.data as RepoAnalysisResult;
    
    const confidence = calculateRepoConfidence(analysis);
    
    logger.info('Phase 1 completed', { 
      repoName: analysis.repoName,
      confidence,
      modelsFound: analysis.architecture.models.length,
    });

    return {
      success: true,
      data: analysis,
      confidence,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    logger.error('Phase 1 error', { error: message });
    return { success: false, error: message };
  }
}

function calculateRepoConfidence(analysis: RepoAnalysisResult): number {
  let confidence = 50;

  if (analysis.architecture.models.length > 0) {
    confidence += 20;
  }

  if (analysis.architecture.trainingLoop) {
    confidence += 15;
  }

  if (analysis.testSuite.testFiles.length > 0) {
    confidence += 15;
  }

  return Math.min(confidence, 100);
}

export function validatePhase1Input(context: Phase1Context): string | null {
  if (!context.repoPath) {
    return 'Repository path is required';
  }

  if (!context.repoPath.endsWith('.git') && !context.repoPath.includes('/')) {
    return 'Invalid repository path format';
  }

  return null;
}
