import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";
import {
  metricsSchema,
  phase1ResultSchema,
  phase2ResultSchema,
  phase3ResultSchema,
  phase4ResultSchema,
  phase5ResultSchema,
  phase6ResultSchema,
  structuredObjectSchema,
} from "./phaseValidators";

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
    comparison: v.optional(structuredObjectSchema),
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
