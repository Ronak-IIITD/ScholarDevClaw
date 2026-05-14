/**
 * Dynamic system prompt injection for ScholarDevClaw agent.
 *
 * Injects the current execution context (yoloMode, etc.) into the
 * system prompt so that the orchestrator and LLM are always aware
 * of the active safety configuration.
 */

export interface SessionHeader {
  yoloMode: boolean;
}

/**
 * Build the session header string that will be injected at the top
 * of the system prompt whenever a phase executes.
 *
 * The returned string is empty when no special overrides are active
 * and non-empty when a condition (e.g. YOLO mode) is enabled.
 */
export function buildSessionHeader(config: SessionHeader): string {
  const lines: string[] = [];

  if (config.yoloMode) {
    lines.push(
      '========================================',
      '  YOLO MODE ACTIVE: Destructive checks disabled',
      '  ',
      '  - Check A (destructive operation detection) is SKIPPED',
      '  - Check B (normal approval gate) is SKIPPED',
      '  - All non-destructive operations are auto-approved',
      '========================================',
    );
  }

  return lines.length > 0 ? lines.join('\n') : '';
}

/**
 * Get the current yoloMode from the environment variable set by
 * the orchestrator.
 */
export function getYoloModeFromEnv(): boolean {
  return (
    (typeof process !== 'undefined' &&
      process.env?.SCHOLARDEVCLAW_YOLO_MODE?.toLowerCase()) === 'true' ||
    false
  );
}

/**
 * Build the full system prompt for a given phase by injecting the
 * session header into the base system prompt template.
 */
export function buildSystemPrompt(
  basePrompt: string,
  headerConfig?: SessionHeader,
): string {
  const config = headerConfig ?? { yoloMode: getYoloModeFromEnv() };
  const header = buildSessionHeader(config);
  if (!header) {
    return basePrompt;
  }
  return `${header}\n\n${basePrompt}`;
}
