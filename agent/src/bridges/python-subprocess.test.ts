import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { existsSync, readFileSync } from 'fs';
import { PythonSubprocessBridge } from './python-subprocess.js';

vi.mock('../utils/logger.js', () => ({
  logger: {
    info: vi.fn(),
    debug: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

describe('PythonSubprocessBridge', () => {
  let bridge: PythonSubprocessBridge;

  beforeEach(() => {
    bridge = new PythonSubprocessBridge('python3', '../core');
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('normalizes mapping context and snake_case research specs from CLI output', async () => {
    const runSpy = vi
      .spyOn(bridge as any, 'runPythonModule')
      .mockResolvedValue({
        success: true,
        data: {
          targets: [
            {
              file: 'model.py',
              line: 42,
              current_code: 'self.norm = LayerNorm(dim)',
              replacement_required: true,
              context: {
                replacement: 'RMSNorm',
                matched_pattern: 'LayerNorm',
              },
              original: 'LayerNorm',
              replacement: 'RMSNorm',
            },
          ],
          strategy: 'replace',
          confidence: 91,
          confidence_breakdown: {
            total: 91,
            counts: {
              targets: 1,
            },
          },
          research_spec: {
            paper: {
              title: 'Root Mean Square Layer Normalization',
              authors: ['Biao Zhang', 'Rico Sennrich'],
              arxiv: '1910.07467',
              year: 2019,
            },
            algorithm: {
              name: 'RMSNorm',
              replaces: 'LayerNorm',
              description: 'Norm without mean-centering.',
            },
            implementation: {
              module_name: 'RMSNorm',
              parent_class: 'nn.Module',
              parameters: ['ndim', 'eps'],
              code_template: 'class RMSNorm(nn.Module): ...',
            },
            changes: {
              type: 'replace',
              target_patterns: ['LayerNorm', 'nn.LayerNorm'],
              insertion_points: ['Block class'],
              replacement: 'RMSNorm',
              expected_benefits: ['Improved throughput'],
            },
            validation: {
              test_type: 'training_comparison',
              metrics: ['loss', 'tokens_per_second'],
            },
          },
        },
      });

    const result = await bridge.mapArchitecture(
      { root_path: '/tmp/repo' },
      { algorithm: { name: 'RMSNorm' } },
    );

    expect(runSpy).toHaveBeenCalledWith('scholardevclaw.cli', [
      'map',
      '/tmp/repo',
      'rmsnorm',
      '--output-json',
    ]);
    expect(result.success).toBe(true);
    expect(result.data).toEqual({
      targets: [
        {
          file: 'model.py',
          line: 42,
          currentCode: 'self.norm = LayerNorm(dim)',
          replacementRequired: true,
          context: {
            replacement: 'RMSNorm',
            matched_pattern: 'LayerNorm',
          },
          original: 'LayerNorm',
          replacement: 'RMSNorm',
        },
      ],
      strategy: 'replace',
      confidence: 91,
      confidence_breakdown: {
        total: 91,
        counts: {
          targets: 1,
        },
      },
      research_spec: {
        paper: {
          title: 'Root Mean Square Layer Normalization',
          authors: ['Biao Zhang', 'Rico Sennrich'],
          arxiv: '1910.07467',
          year: 2019,
        },
        algorithm: {
          name: 'RMSNorm',
          replaces: 'LayerNorm',
          description: 'Norm without mean-centering.',
          formula: undefined,
        },
        implementation: {
          moduleName: 'RMSNorm',
          parentClass: 'nn.Module',
          parameters: ['ndim', 'eps'],
          codeTemplate: 'class RMSNorm(nn.Module): ...',
        },
        changes: {
          type: 'replace',
          targetPattern: 'LayerNorm',
          targetPatterns: ['LayerNorm', 'nn.LayerNorm'],
          insertionPoints: ['Block class'],
          replacement: 'RMSNorm',
          expectedBenefits: ['Improved throughput'],
        },
        validation: {
          test_type: 'training_comparison',
          metrics: ['loss', 'tokens_per_second'],
        },
      },
      researchSpec: {
        paper: {
          title: 'Root Mean Square Layer Normalization',
          authors: ['Biao Zhang', 'Rico Sennrich'],
          arxiv: '1910.07467',
          year: 2019,
        },
        algorithm: {
          name: 'RMSNorm',
          replaces: 'LayerNorm',
          description: 'Norm without mean-centering.',
          formula: undefined,
        },
        implementation: {
          moduleName: 'RMSNorm',
          parentClass: 'nn.Module',
          parameters: ['ndim', 'eps'],
          codeTemplate: 'class RMSNorm(nn.Module): ...',
        },
        changes: {
          type: 'replace',
          targetPattern: 'LayerNorm',
          targetPatterns: ['LayerNorm', 'nn.LayerNorm'],
          insertionPoints: ['Block class'],
          replacement: 'RMSNorm',
          expectedBenefits: ['Improved throughput'],
        },
        validation: {
          test_type: 'training_comparison',
          metrics: ['loss', 'tokens_per_second'],
        },
      },
    });
  });

  it('passes the full mapping payload to CLI patch generation', async () => {
    const mappingPayload = {
      targets: [
        {
          file: 'model.py',
          line: 10,
          currentCode: 'self.norm = nn.LayerNorm(dim)',
          replacementRequired: true,
        },
      ],
      strategy: 'replace',
      confidence: 91,
      research_spec: {
        algorithm: { name: 'RMSNorm' },
        paper: { arxiv: '1910.07467' },
      },
    };

    const runSpy = vi
      .spyOn(bridge as any, 'runPythonModule')
      .mockImplementation(async (...callArgs: any[]) => {
        const [, args] = callArgs as [string, string[]];
        const payloadIndex = args.indexOf('--mapping-json');
        expect(payloadIndex).toBeGreaterThan(-1);
        const payloadPath = args[payloadIndex + 1];
        expect(JSON.parse(readFileSync(payloadPath, 'utf8'))).toEqual(mappingPayload);

        return {
          success: true,
          data: {
            new_files: [{ path: 'rmsnorm.py', content: 'class RMSNorm:\n    pass\n' }],
            transformations: [],
            branch_name: 'integration/rmsnorm',
            algorithm_name: 'RMSNorm',
            paper_reference: 'arXiv:1910.07467',
            research_spec: mappingPayload.research_spec,
          },
        };
      });

    const result = await bridge.generatePatch(mappingPayload, '/tmp/repo');

    expect(runSpy).toHaveBeenCalledWith('scholardevclaw.cli', [
      'generate',
      '/tmp/repo',
      '__mapping_payload__',
      '--mapping-json',
      expect.any(String),
      '--output-json',
    ]);
    const payloadPath = (runSpy.mock.calls[0] as [string, string[]] | undefined)?.[1][4];
    if (!payloadPath) {
      throw new Error('Expected --mapping-json payload path');
    }
    expect(existsSync(payloadPath)).toBe(false);
    expect(result.success).toBe(true);
    expect(result.data).toEqual({
      newFiles: [{ path: 'rmsnorm.py', content: 'class RMSNorm:\n    pass\n' }],
      transformations: [],
      branchName: 'integration/rmsnorm',
      algorithmName: 'RMSNorm',
      paperReference: 'arXiv:1910.07467',
      researchSpec: {
        paper: {
          title: 'Unknown',
          authors: [],
          arxiv: '1910.07467',
          year: 0,
        },
        algorithm: {
          name: 'RMSNorm',
          replaces: undefined,
          description: '',
          formula: undefined,
        },
        implementation: {
          moduleName: '',
          parentClass: '',
          parameters: [],
          codeTemplate: null,
        },
        changes: {
          type: 'replace',
          targetPattern: '',
          targetPatterns: [],
          insertionPoints: [],
          replacement: undefined,
          expectedBenefits: [],
        },
      },
    });
  });

  it('passes the patch payload to CLI validation and normalizes snake_case metrics', async () => {
    const patchPayload = {
      newFiles: [{ path: 'rmsnorm.py', content: 'class RMSNorm:\n    pass\n' }],
      transformations: [],
      branchName: 'integration/rmsnorm',
      algorithmName: 'RMSNorm',
      paperReference: 'arXiv:1910.07467',
    };

    const runSpy = vi
      .spyOn(bridge as any, 'runPythonModule')
      .mockImplementation(async (...callArgs: any[]) => {
        const [, args] = callArgs as [string, string[]];
        const payloadIndex = args.indexOf('--patch-json');
        expect(payloadIndex).toBeGreaterThan(-1);
        const payloadPath = args[payloadIndex + 1];
        expect(JSON.parse(readFileSync(payloadPath, 'utf8'))).toEqual(patchPayload);

        return {
          success: true,
          data: {
            passed: true,
            stage: 'benchmark',
            baseline_metrics: {
              loss: 1.0,
              perplexity: 2.0,
              tokens_per_second: 3.0,
              memory_mb: 4.0,
            },
            new_metrics: {
              loss: 0.9,
              perplexity: 1.9,
              tokens_per_second: 3.6,
              memory_mb: 3.5,
            },
            comparison: {
              speedup: 1.2,
              loss_change: -10,
              passed: true,
            },
            logs: 'ok',
            schemaVersion: '2026-05-25',
            payloadType: 'validation',
          },
        };
      });

    const result = await bridge.validate(patchPayload, '/tmp/repo');

    expect(runSpy).toHaveBeenCalledWith('scholardevclaw.cli', [
      'validate',
      '/tmp/repo',
      '--patch-json',
      expect.any(String),
      '--output-json',
    ]);
    const payloadPath = (runSpy.mock.calls[0] as [string, string[]] | undefined)?.[1][3];
    if (!payloadPath) {
      throw new Error('Expected --patch-json payload path');
    }
    expect(existsSync(payloadPath)).toBe(false);
    expect(result.success).toBe(true);
    expect(result.data).toEqual({
      passed: true,
      stage: 'benchmark',
      baselineMetrics: {
        loss: 1,
        perplexity: 2,
        tokensPerSecond: 3,
        memoryMb: 4,
      },
      newMetrics: {
        loss: 0.9,
        perplexity: 1.9,
        tokensPerSecond: 3.6,
        memoryMb: 3.5,
      },
      comparison: {
        speedup: 1.2,
        loss_change: -10,
        lossChange: -10,
        passed: true,
      },
      logs: 'ok',
      error: undefined,
      schemaVersion: '2026-05-25',
      payloadType: 'validation',
    });
  });
});
