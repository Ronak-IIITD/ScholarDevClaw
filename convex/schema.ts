import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

// --- Phase 1: Repo Analysis ---
const modelSchema = v.object({
  name: v.string(),
  file: v.string(),
  line: v.number(),
  parent: v.string(),
  components: v.any(), // flexible — architecture-specific
});

const trainingLoopSchema = v.object({
  file: v.string(),
  line: v.number(),
  optimizer: v.string(),
  lossFn: v.string(),
});

const phase1ResultSchema = v.object({
  repoName: v.string(),
  architecture: v.object({
    models: v.array(modelSchema),
    trainingLoop: v.optional(trainingLoopSchema),
  }),
  dependencies: v.any(), // flexible — language-specific
  testSuite: v.object({
    runner: v.string(),
    testFiles: v.array(v.string()),
  }),
});

// --- Phase 2: Research Spec ---
const phase2ResultSchema = v.object({
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
    codeTemplate: v.string(),
  }),
  changes: v.object({
    type: v.string(),
    targetPattern: v.string(),
    insertionPoints: v.array(v.string()),
  }),
});

// --- Phase 3: Mapping ---
const mappingTargetSchema = v.object({
  file: v.string(),
  line: v.number(),
  currentCode: v.string(),
  replacementRequired: v.boolean(),
});

const phase3ResultSchema = v.object({
  targets: v.array(mappingTargetSchema),
  strategy: v.string(),
  confidence: v.number(),
});

// --- Phase 4: Patch ---
const newFileSchema = v.object({
  path: v.string(),
  content: v.string(),
});

const transformationSchema = v.object({
  file: v.string(),
  original: v.string(),
  modified: v.string(),
  changes: v.array(v.any()),
});

const phase4ResultSchema = v.object({
  newFiles: v.array(newFileSchema),
  transformations: v.array(transformationSchema),
  branchName: v.string(),
});

// --- Phase 5: Validation ---
const metricsSchema = v.object({
  loss: v.number(),
  perplexity: v.number(),
  tokensPerSecond: v.number(),
  memoryMb: v.number(),
});

const comparisonSchema = v.object({
  lossChange: v.number(),
  speedup: v.number(),
  passed: v.boolean(),
});

const phase5ResultSchema = v.object({
  passed: v.boolean(),
  stage: v.string(),
  baselineMetrics: v.optional(metricsSchema),
  newMetrics: v.optional(metricsSchema),
  comparison: v.optional(comparisonSchema),
  logs: v.optional(v.string()),
  error: v.optional(v.string()),
});

// --- Phase 6: Report ---
const phase6ResultSchema = v.object({
  summary: v.any(), // flexible — report format varies
  diff: v.string(),
  metadata: v.any(), // flexible
});

export default defineSchema({
  integrations: defineTable({
    repoUrl: v.string(),
    paperUrl: v.optional(v.string()),
    paperPdfPath: v.optional(v.string()),
    status: v.union(
      v.literal("pending"),
      v.literal("phase1_analyzing"),
      v.literal("phase2_extracting"),
      v.literal("phase3_mapping"),
      v.literal("phase4_patching"),
      v.literal("phase5_validating"),
      v.literal("phase6_reporting"),
      v.literal("awaiting_approval"),
      v.literal("completed"),
      v.literal("failed")
    ),
    mode: v.union(v.literal("step_approval"), v.literal("autonomous")),
    currentPhase: v.number(),
    phase1Result: v.optional(phase1ResultSchema),
    phase2Result: v.optional(phase2ResultSchema),
    phase3Result: v.optional(phase3ResultSchema),
    phase4Result: v.optional(phase4ResultSchema),
    phase5Result: v.optional(phase5ResultSchema),
    phase6Result: v.optional(phase6ResultSchema),
    confidence: v.optional(v.number()),
    createdAt: v.number(),
    updatedAt: v.number(),
    errorMessage: v.optional(v.string()),
    awaitingReason: v.optional(v.string()),
    guardrailReasons: v.optional(v.array(v.string())),
    retryCount: v.number(),
    branchName: v.optional(v.string()),
  })
    .index("by_status", ["status"])
    .index("by_created", ["createdAt"]),

  validationRuns: defineTable({
    integrationId: v.id("integrations"),
    baselineMetrics: v.optional(metricsSchema),
    newMetrics: v.optional(metricsSchema),
    comparison: v.optional(comparisonSchema),
    passed: v.boolean(),
    logs: v.string(),
    createdAt: v.number(),
  })
    .index("by_integration", ["integrationId"]),

  approvals: defineTable({
    integrationId: v.id("integrations"),
    phase: v.number(),
    action: v.union(v.literal("approved"), v.literal("rejected")),
    notes: v.optional(v.string()),
    createdAt: v.number(),
  })
    .index("by_integration", ["integrationId"]),

  integrationLogs: defineTable({
    integrationId: v.id("integrations"),
    message: v.string(),
    level: v.optional(v.string()),
    timestamp: v.number(),
  })
    .index("by_integration", ["integrationId"])
    .index("by_timestamp", ["integrationId", "timestamp"]),

  paperLibrary: defineTable({
    paperId: v.string(),
    paperTitle: v.string(),
    paperUrl: v.optional(v.string()),
    arxivId: v.optional(v.string()),
    algorithmName: v.optional(v.string()),
    analysisDate: v.number(),
    integrationCount: v.number(),
    lastUsed: v.number(),
    notes: v.optional(v.string()),
    repoPath: v.optional(v.string()),
  })
    .index("by_analysis_date", ["analysisDate"])
    .index("by_algo_name", ["algorithmName"]),

  sessionHistory: defineTable({
    integrationId: v.id("integrations"),
    action: v.string(),
    status: v.string(),
    repoPath: v.string(),
    specName: v.optional(v.string()),
    duration: v.number(),
    createdAt: v.number(),
    summary: v.optional(v.string()),
    errorMessage: v.optional(v.string()),
  })
    .index("by_integration", ["integrationId"])
    .index("by_created", ["createdAt"]),
});
