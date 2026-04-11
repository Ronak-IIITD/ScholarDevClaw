import { query } from "./_generated/server";
import { v } from "convex/values";

const AUTH_ENV_KEY = "SCHOLARDEVCLAW_CONVEX_AUTH_KEY";

const INTEGRATION_STATUS_VALIDATOR = v.union(
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

export const list = query({
  args: {
    authKey: v.string(),
  },
  handler: async ({ db }, args) => {
    requireAuth(args.authKey);
    return await db.query("integrations").order("desc").take(100);
  },
});

export const listByStatus = query({
  args: {
    authKey: v.string(),
    status: INTEGRATION_STATUS_VALIDATOR,
  },
  handler: async ({ db }, args) => {
    requireAuth(args.authKey);
    return await db
      .query("integrations")
      .withIndex("by_status", (q) => q.eq("status", args.status))
      .take(100);
  },
});

export const get = query({
  args: {
    authKey: v.string(),
    id: v.id("integrations"),
  },
  handler: async ({ db }, args) => {
    requireAuth(args.authKey);
    return await db.get(args.id);
  },
});

export const listApprovals = query({
  args: {
    authKey: v.string(),
    integrationId: v.id("integrations"),
  },
  handler: async ({ db }, args) => {
    requireAuth(args.authKey);
    return await db
      .query("approvals")
      .withIndex("by_integration", (q) => q.eq("integrationId", args.integrationId))
      .collect();
  },
});
