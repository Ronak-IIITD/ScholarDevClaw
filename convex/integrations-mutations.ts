import { mutation } from "./_generated/server";
import { v } from "convex/values";

const AUTH_ENV_KEY = "SCHOLARDEVCLAW_CONVEX_AUTH_KEY";

// SECURITY: Allowed status values to prevent arbitrary state injection
const ALLOWED_STATUSES = new Set([
  "pending", "phase1_analyzing", "phase2_extracting", "phase3_mapping", "phase4_patching", "phase5_validating", "phase6_reporting", "awaiting_approval", "completed", "failed",
]);

// SECURITY: Whitelist allowed fields to prevent arbitrary field injection
const ALLOWED_PHASE_FIELDS = new Set([
  "phase1Result", "phase2Result", "phase3Result", "phase4Result", "phase5Result", "phase6Result",
]);

const APPROVAL_ACTIONS = new Set(["approved", "rejected"]);

const EXTERNAL_STATUS_VALIDATOR = v.union(
  v.literal("pending"),
  v.literal("phase1_analyzing"),
  v.literal("phase2_extracting"),
  v.literal("phase3_mapping"),
  v.literal("phase4_patching"),
  v.literal("phase5_validating"),
  v.literal("phase6_reporting"),
  v.literal("awaiting_approval"),
  v.literal("completed"),
  v.literal("failed"),
);

function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) {
    return false;
  }
  let diff = 0;
  for (let i = 0; i < a.length; i += 1) {
    diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return diff === 0;
}

function requireAuth(authKey: string): void {
  const expected = process.env[AUTH_ENV_KEY];
  if (!expected) {
    throw new Error(`${AUTH_ENV_KEY} is not configured`);
  }
  if (!authKey || !timingSafeEqual(authKey, expected)) {
    throw new Error("Unauthorized");
  }
}

export const create = mutation({
  args: {
    authKey: v.string(),
    repoUrl: v.string(),
    paperUrl: v.optional(v.string()),
    paperPdfPath: v.optional(v.string()),
    mode: v.optional(v.union(v.literal("step_approval"), v.literal("autonomous"))),
  },
  handler: async ({ db }, args) => {
    requireAuth(args.authKey);

    return await db.insert("integrations", {
      repoUrl: args.repoUrl,
      paperUrl: args.paperUrl,
      paperPdfPath: args.paperPdfPath,
      status: "pending",
      mode: args.mode || "step_approval",
      currentPhase: 0,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      retryCount: 0,
    });
  },
});

export const updateStatus = mutation({
  args: {
    authKey: v.string(),
    id: v.id("integrations"),
    status: EXTERNAL_STATUS_VALIDATOR,
    currentPhase: v.optional(v.number()),
    awaitingReason: v.optional(v.string()),
    guardrailReasons: v.optional(v.array(v.string())),
    updatedAt: v.number(),
  },
  handler: async ({ db }, args) => {
    requireAuth(args.authKey);

    if (!ALLOWED_STATUSES.has(args.status)) {
      throw new Error(`Invalid status: ${args.status}`);
    }
    await db.patch(args.id, {
      status: args.status,
      currentPhase: args.currentPhase,
      awaitingReason: args.awaitingReason,
      guardrailReasons: args.guardrailReasons,
      updatedAt: args.updatedAt,
    });
  },
});

export const savePhaseResult = mutation({
  args: {
    authKey: v.string(),
    id: v.id("integrations"),
    field: v.string(),
    result: v.any(),
    updatedAt: v.number(),
  },
  handler: async ({ db }, args) => {
    requireAuth(args.authKey);

    if (!ALLOWED_PHASE_FIELDS.has(args.field)) {
      throw new Error(`Disallowed field: ${args.field}`);
    }

    const update: Record<string, unknown> = {
      updatedAt: args.updatedAt,
    };
    update[args.field] = args.result;

    await db.patch(args.id, update);
  },
});

export const setConfidence = mutation({
  args: {
    authKey: v.string(),
    id: v.id("integrations"),
    confidence: v.number(),
    updatedAt: v.number(),
  },
  handler: async ({ db }, args) => {
    requireAuth(args.authKey);

    await db.patch(args.id, {
      confidence: args.confidence,
      updatedAt: args.updatedAt,
    });
  },
});

export const setError = mutation({
  args: {
    authKey: v.string(),
    id: v.id("integrations"),
    errorMessage: v.string(),
    updatedAt: v.number(),
  },
  handler: async ({ db }, args) => {
    requireAuth(args.authKey);

    await db.patch(args.id, {
      errorMessage: args.errorMessage,
      status: "failed",
      updatedAt: args.updatedAt,
    });
  },
});

export const incrementRetry = mutation({
  args: {
    authKey: v.string(),
    id: v.id("integrations"),
  },
  handler: async ({ db }, args) => {
    requireAuth(args.authKey);

    const doc = await db.get(args.id);
    if (!doc) return { retryCount: 0 };

    const retryCount = (doc.retryCount || 0) + 1;
    await db.patch(args.id, { retryCount });

    return { retryCount };
  },
});

export const createApproval = mutation({
  args: {
    authKey: v.string(),
    integrationId: v.id("integrations"),
    phase: v.number(),
    action: v.union(v.literal("approved"), v.literal("rejected")),
    notes: v.optional(v.string()),
  },
  handler: async ({ db }, args) => {
    requireAuth(args.authKey);

    if (!APPROVAL_ACTIONS.has(args.action)) {
      throw new Error(`Invalid approval action: ${args.action}`);
    }

    const action = args.action === "approved" ? "approved" : "rejected";

    await db.insert("approvals", {
      integrationId: args.integrationId,
      phase: args.phase,
      action,
      notes: args.notes,
      createdAt: Date.now(),
    });
  },
});
