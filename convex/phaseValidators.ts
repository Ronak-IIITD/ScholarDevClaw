import { v } from "convex/values";

const scalarValueSchema = v.union(v.string(), v.number(), v.boolean(), v.null());
const scalarArraySchema = v.union(
  v.array(v.string()),
  v.array(v.number()),
  v.array(v.boolean()),
  v.array(v.null()),
);
const flatValueSchema = v.union(scalarValueSchema, scalarArraySchema);
const flatObjectSchema = v.record(v.string(), flatValueSchema);

export const structuredValueSchema = v.union(flatValueSchema, flatObjectSchema);
export const structuredObjectSchema = v.record(v.string(), structuredValueSchema);
export const structuredObjectListSchema = v.array(structuredObjectSchema);

export const researchValidationSchema = structuredObjectSchema;

export const researchChangesSchema = v.object({
  type: v.string(),
  targetPattern: v.string(),
  targetPatterns: v.optional(v.array(v.string())),
  insertionPoints: v.array(v.string()),
  replacement: v.optional(v.union(v.string(), v.null())),
  expectedBenefits: v.optional(v.array(v.string())),
});

export const researchSpecSchema = v.object({
  paper: v.object({
    title: v.string(),
    authors: v.array(v.string()),
    arxiv: v.optional(v.string()),
    year: v.number(),
  }),
  algorithm: v.object({
    name: v.string(),
    replaces: v.optional(v.string()),
    description: v.string(),
    formula: v.optional(v.string()),
  }),
  implementation: v.object({
    moduleName: v.string(),
    parentClass: v.string(),
    parameters: v.array(v.string()),
    codeTemplate: v.optional(v.union(v.string(), v.null())),
  }),
  changes: researchChangesSchema,
  validation: v.optional(researchValidationSchema),
});

export const modelSchema = v.object({
  name: v.string(),
  file: v.string(),
  line: v.number(),
  parent: v.string(),
  components: structuredObjectSchema,
});

export const trainingLoopSchema = v.object({
  file: v.string(),
  line: v.number(),
  optimizer: v.string(),
  lossFn: v.string(),
});

export const phase1ResultSchema = v.object({
  repoName: v.string(),
  architecture: v.object({
    models: v.array(modelSchema),
    trainingLoop: v.optional(trainingLoopSchema),
  }),
  dependencies: structuredObjectSchema,
  testSuite: v.object({
    runner: v.string(),
    testFiles: v.array(v.string()),
  }),
  root_path: v.optional(v.string()),
});

export const phase2ResultSchema = researchSpecSchema;

export const mappingTargetSchema = v.object({
  file: v.string(),
  line: v.number(),
  currentCode: v.string(),
  replacementRequired: v.boolean(),
  context: v.optional(structuredObjectSchema),
  original: v.optional(v.string()),
  replacement: v.optional(v.union(v.string(), v.null())),
});

export const phase3ResultSchema = v.object({
  targets: v.array(mappingTargetSchema),
  strategy: v.string(),
  confidence: v.number(),
  confidence_breakdown: v.optional(structuredObjectSchema),
  research_spec: v.optional(researchSpecSchema),
  researchSpec: v.optional(researchSpecSchema),
});

export const newFileSchema = v.object({
  path: v.string(),
  content: v.string(),
});

export const transformationSchema = v.object({
  file: v.string(),
  original: v.string(),
  modified: v.string(),
  changes: structuredObjectListSchema,
});

export const phase4ResultSchema = v.object({
  newFiles: v.array(newFileSchema),
  transformations: v.array(transformationSchema),
  branchName: v.string(),
  algorithmName: v.optional(v.string()),
  paperReference: v.optional(v.string()),
  researchSpec: v.optional(researchSpecSchema),
});

export const metricsSchema = v.object({
  loss: v.number(),
  perplexity: v.number(),
  tokensPerSecond: v.number(),
  memoryMb: v.number(),
});

export const phase5ResultSchema = v.object({
  passed: v.boolean(),
  stage: v.string(),
  baselineMetrics: v.optional(metricsSchema),
  newMetrics: v.optional(metricsSchema),
  comparison: v.optional(structuredObjectSchema),
  logs: v.optional(v.string()),
  error: v.optional(v.string()),
  schemaVersion: v.optional(v.string()),
  payloadType: v.optional(v.string()),
});

const phase6MetadataSchema = v.object({
  integrationId: v.string(),
  repoUrl: v.string(),
  paper: v.string(),
  algorithm: v.string(),
  createdAt: v.string(),
});

const phase6SummarySchema = v.object({
  status: v.string(),
  confidence: v.number(),
  changesMade: v.number(),
  filesModified: v.array(v.string()),
  newFiles: v.array(v.string()),
});

const phase6RecommendationSchema = v.object({
  action: v.union(v.literal("approve"), v.literal("review"), v.literal("reject")),
  confidence: v.number(),
  notes: v.string(),
});

export const phase6ResultSchema = v.object({
  metadata: phase6MetadataSchema,
  summary: phase6SummarySchema,
  whatChanged: v.string(),
  why: v.string(),
  observedImpact: v.object({
    metricsComparison: structuredObjectSchema,
    meetsExpectations: v.boolean(),
  }),
  riskNotes: v.array(v.string()),
  diffPreview: v.string(),
  testResults: v.object({
    unitTestsPassed: v.boolean(),
    benchmarkResults: structuredObjectSchema,
  }),
  recommendation: phase6RecommendationSchema,
});

export const phaseResultFieldSchema = v.union(
  v.literal("phase1Result"),
  v.literal("phase2Result"),
  v.literal("phase3Result"),
  v.literal("phase4Result"),
  v.literal("phase5Result"),
  v.literal("phase6Result"),
);

export const phaseResultPayloadSchema = v.union(
  v.object({
    field: v.literal("phase1Result"),
    result: phase1ResultSchema,
  }),
  v.object({
    field: v.literal("phase2Result"),
    result: phase2ResultSchema,
  }),
  v.object({
    field: v.literal("phase3Result"),
    result: phase3ResultSchema,
  }),
  v.object({
    field: v.literal("phase4Result"),
    result: phase4ResultSchema,
  }),
  v.object({
    field: v.literal("phase5Result"),
    result: phase5ResultSchema,
  }),
  v.object({
    field: v.literal("phase6Result"),
    result: phase6ResultSchema,
  }),
);
