import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

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
    phase1Result: v.optional(v.any()),
    phase2Result: v.optional(v.any()),
    phase3Result: v.optional(v.any()),
    phase4Result: v.optional(v.any()),
    phase5Result: v.optional(v.any()),
    phase6Result: v.optional(v.any()),
    confidence: v.optional(v.number()),
    createdAt: v.number(),
    updatedAt: v.number(),
    errorMessage: v.optional(v.string()),
    retryCount: v.number(),
    branchName: v.optional(v.string()),
  })
    .index("by_status", ["status"])
    .index("by_created", ["createdAt"]),

  validationRuns: defineTable({
    integrationId: v.id("integrations"),
    baselineMetrics: v.optional(v.any()),
    newMetrics: v.optional(v.any()),
    comparison: v.optional(v.any()),
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
});
