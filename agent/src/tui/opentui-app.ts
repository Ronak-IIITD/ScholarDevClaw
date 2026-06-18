/**
 * ScholarDevClaw OpenTUI — keyboard-first terminal shell.
 *
 * Architecture:
 * - OpenTUI (Zig core + TS bindings) renders the UI
 * - Python FastAPI pipeline runs in background (auto-started)
 * - HTTP bridge connects UI → pipeline
 * - Full 6-phase orchestrator with live progress and approval gates
 * - Local run history via RunStore
 */

import {
  Box,
  Text,
  Input,
  InputRenderableEvents,
  ScrollBox,
  stringToStyledText,
  createCliRenderer,
  type KeyEvent,
} from "@opentui/core";
import { PythonHttpBridge } from "../bridges/python-http.js";
import type {
  RepoAnalysisResult,
  ResearchSpecResult,
  MappingResult,
  PatchResult,
  ValidationResult,
} from "../bridges/python-subprocess.js";
import { RunStore, type RunSnapshot } from "../utils/run-store.js";
import { spawn, type ChildProcess } from "child_process";
import { existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const CORE_ROOT = join(__dirname, "..", "..", "..", "core");
const DEFAULT_PROVIDER = "openrouter";
const DEFAULT_MODEL = "anthropic/claude-sonnet-4";
const DEFAULT_PORT = 8000;

const PHASE_NAMES = [
  "Repo Analysis",
  "Research Extraction",
  "Mapping",
  "Patch Generation",
  "Validation",
  "Report",
];

const KNOWN_SPECS = [
  "rmsnorm",
  "flashattention",
  "swiglu",
  "geglu",
  "gqa",
  "rope",
  "preln",
  "alibi",
  "qknorm",
];

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type ExecutionMode = "autonomous" | "step_approval";

interface PhaseState {
  phase: number;
  name: string;
  status: "pending" | "running" | "completed" | "failed";
  durationMs?: number;
  error?: string;
}

interface ApprovalPrompt {
  phase: number;
  phaseName: string;
  resolve: (approved: boolean) => void;
}

interface AppState {
  repoPath: string;
  spec: string;
  mode: ExecutionMode;
  provider: string;
  model: string;
  running: boolean;
  cancellationRequested: boolean;
  phases: PhaseState[];
  approvalPrompt: ApprovalPrompt | null;
  commandHistory: string[];
  historyIndex: number;
  // Accumulated phase results for multi-step commands
  repoAnalysis: RepoAnalysisResult | null;
  researchSpec: ResearchSpecResult | null;
  mapping: MappingResult | null;
  patch: PatchResult | null;
  validation: ValidationResult | null;
  lastRunId: string | null;
}

// ---------------------------------------------------------------------------
// Colors (Catppuccin Mocha inspired)
// ---------------------------------------------------------------------------

const C = {
  bg: "#1e1e2e",
  surface: "#181825",
  border: "#313244",
  text: "#cdd6f4",
  muted: "#6c7086",
  accent: "#89b4fa",
  success: "#a6e3a1",
  error: "#f38ba8",
  warning: "#f9e2af",
  yellow: "#f9e2af",
  green: "#a6e3a1",
  red: "#f38ba8",
  blue: "#89b4fa",
  mauve: "#cba6f7",
  teal: "#94e2d5",
} as const;

// ---------------------------------------------------------------------------
// Python subprocess manager
// ---------------------------------------------------------------------------

class PythonServer {
  private process: ChildProcess | null = null;
  private port: number;

  constructor(port: number = DEFAULT_PORT) {
    this.port = port;
  }

  async start(): Promise<boolean> {
    try {
      const res = await fetch(`http://localhost:${this.port}/health`);
      if (res.ok) return true;
    } catch {
      // not running
    }

    const venvPython = join(CORE_ROOT, ".venv", "bin", "python");
    const py = existsSync(venvPython) ? venvPython : "python3";

    return new Promise((resolve) => {
      this.process = spawn(
        py,
        ["-m", "uvicorn", "scholardevclaw.api.server:app", "--port", String(this.port)],
        { cwd: CORE_ROOT, stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, PYTHONUNBUFFERED: "1" } },
      );

      this.process.stdout?.on("data", () => {});
      this.process.stderr?.on("data", () => {});

      let attempts = 0;
      const check = setInterval(async () => {
        attempts++;
        try {
          const res = await fetch(`http://localhost:${this.port}/health`);
          if (res.ok) {
            clearInterval(check);
            resolve(true);
          }
        } catch {
          if (attempts > 30) {
            clearInterval(check);
            resolve(false);
          }
        }
      }, 500);
    });
  }

  stop() {
    if (this.process) {
      this.process.kill("SIGTERM");
      this.process = null;
    }
  }
}

// ---------------------------------------------------------------------------
// Phase progress helpers
// ---------------------------------------------------------------------------

function initPhases(): PhaseState[] {
  return PHASE_NAMES.map((name, i) => ({
    phase: i + 1,
    name,
    status: "pending" as const,
  }));
}

function renderPhaseProgress(phases: PhaseState[]): string {
  return phases
    .map((p) => {
      const icon = p.status === "completed" ? "✓" : p.status === "running" ? "▶" : p.status === "failed" ? "✗" : "○";
      const time = p.durationMs != null ? ` (${(p.durationMs / 1000).toFixed(1)}s)` : "";
      const err = p.error ? ` — ${p.error}` : "";
      return `  ${p.phase}. ${icon} ${p.name}${time}${err}`;
    })
    .join("\n");
}

function phaseIcon(status: PhaseState["status"]): string {
  return status === "completed" ? "✓" : status === "running" ? "▶" : status === "failed" ? "✗" : "○";
}

// ---------------------------------------------------------------------------
// Main TUI App
// ---------------------------------------------------------------------------

export async function main() {
  const state: AppState = {
    repoPath: process.argv[2] || ".",
    spec: "rmsnorm",
    mode: (process.env.DEFAULT_MODE as ExecutionMode) || "autonomous",
    provider: process.env.SCHOLARDEVCLAW_API_PROVIDER || DEFAULT_PROVIDER,
    model: process.env.SCHOLARDEVCLAW_API_MODEL || DEFAULT_MODEL,
    running: false,
    cancellationRequested: false,
    phases: initPhases(),
    approvalPrompt: null,
    commandHistory: [],
    historyIndex: -1,
    repoAnalysis: null,
    researchSpec: null,
    mapping: null,
    patch: null,
    validation: null,
    lastRunId: null,
  };

  const server = new PythonServer();
  const bridge = new PythonHttpBridge(`http://localhost:${DEFAULT_PORT}`);
  const runStore = new RunStore();

  // Start Python server
  const serverReady = await server.start();

  const renderer = await createCliRenderer({ exitOnCtrlC: false });

  // --- Build UI components ---

  const headerText = Text({ content: " ScholarDevClaw", fg: C.accent });

  const statusText = Text({
    content: `REPO: ${state.repoPath}  SPEC: ${state.spec}  MODE: ${state.mode}`,
    fg: C.muted,
  });

  const serverStatusText = Text({
    content: serverReady ? "● Pipeline ready" : "○ Pipeline offline",
    fg: serverReady ? C.green : C.red,
  });

  const phaseProgressText = Text({
    content: renderPhaseProgress(state.phases),
    fg: C.text,
  });

  const header = Box(
    { flexDirection: "column", gap: 0, marginBottom: 1 },
    headerText,
    statusText,
    serverStatusText,
    phaseProgressText,
  );

  const separator = Text({ content: "─".repeat(80), fg: C.border });

  const logBox = Box(
    { flexDirection: "column", gap: 0, height: 16, borderStyle: "rounded", padding: 1 },
    Text({ content: "Output", fg: C.muted }),
    Text({ content: "─".repeat(76), fg: C.border }),
  );

  const logScroll = ScrollBox({ width: 76, height: 13 });
  logBox.add(logScroll);

  const hintText = Text({
    content: `Tip: integrate ${state.repoPath} ${state.spec} — or: analyze, map, generate, validate, runs`,
    fg: C.muted,
  });

  const promptInput = Input({
    placeholder: `> integrate ${state.repoPath} ${state.spec} ...`,
    width: 78,
    backgroundColor: C.surface,
    focusedBackgroundColor: "#2a2a3e",
    textColor: C.text,
    cursorColor: C.accent,
  });

  const keyHints = Text({
    content: "Enter: run  Tab: autocomplete  ↑↓: history  Ctrl+C: cancel  Ctrl+K: clear  Esc: quit",
    fg: C.muted,
  });

  renderer.root.add(
    Box(
      { flexDirection: "column", gap: 0, padding: 1 },
      header,
      separator,
      logBox,
      hintText,
      promptInput,
      separator,
      keyHints,
    ),
  );

  promptInput.focus();

  // --- Log helpers ---

  function addLog(text: string, level: string = "info") {
    const color =
      level === "error"
        ? C.red
        : level === "success"
          ? C.green
          : level === "warning"
            ? C.yellow
            : level === "accent"
              ? C.accent
              : C.text;
    const line = Text({ content: text, fg: color });
    logScroll.add(line);
  }

  function clearLogs() {
    const children = [...(logScroll.children ?? [])];
    for (const child of children) {
      if (child && typeof child === "object" && "id" in child && typeof child.id === "string") {
        logScroll.remove(child.id);
      }
    }
  }

  function updateStatus() {
    statusText.content = stringToStyledText(
      `REPO: ${state.repoPath}  SPEC: ${state.spec}  MODE: ${state.mode}`,
    );
    hintText.content = stringToStyledText(
      `Tip: integrate ${state.repoPath} ${state.spec} — or: analyze, map, generate, validate, runs`,
    );
    promptInput.placeholder = `> integrate ${state.repoPath} ${state.spec} ...`;
  }

  function updatePhaseProgress() {
    phaseProgressText.content = stringToStyledText(renderPhaseProgress(state.phases));
  }

  function resetPhases() {
    state.phases = initPhases();
    updatePhaseProgress();
  }

  // --- Approval flow ---

  function promptApproval(phase: number, phaseName: string): Promise<boolean> {
    return new Promise((resolve) => {
      state.approvalPrompt = { phase, phaseName, resolve };
      addLog(`⏸  Approve Phase ${phase} (${phaseName})? Type y or n`, "warning");
    });
  }

  // --- Phase execution helpers ---

  async function runPhase<T>(
    phase: number,
    fn: () => Promise<T>,
  ): Promise<{ ok: boolean; data?: T; error?: string }> {
    const phaseState = state.phases[phase - 1];
    phaseState.status = "running";
    phaseState.durationMs = undefined;
    phaseState.error = undefined;
    updatePhaseProgress();

    const start = Date.now();
    try {
      if (state.cancellationRequested) {
        throw new Error("Cancelled by user");
      }
      const data = await fn();
      phaseState.status = "completed";
      phaseState.durationMs = Date.now() - start;
      updatePhaseProgress();
      return { ok: true, data };
    } catch (err) {
      phaseState.status = "failed";
      phaseState.durationMs = Date.now() - start;
      phaseState.error = err instanceof Error ? err.message : String(err);
      updatePhaseProgress();
      return { ok: false, error: phaseState.error };
    }
  }

  async function maybeApprove(phase: number, phaseName: string): Promise<boolean> {
    if (state.mode !== "step_approval") return true;
    return promptApproval(phase, phaseName);
  }

  // --- Integration runner (full 6-phase pipeline) ---

  async function runIntegration(path: string, spec: string) {
    state.running = true;
    state.cancellationRequested = false;
    state.repoAnalysis = null;
    state.researchSpec = null;
    state.mapping = null;
    state.patch = null;
    state.validation = null;
    resetPhases();

    const runId = `tui-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    state.lastRunId = runId;
    addLog(`🚀 Integration started: ${path} + ${spec}`, "accent");
    addLog(`   Run ID: ${runId}  Mode: ${state.mode}`, "muted");
    addLog("", "info");

    // --- Phase 1: Repo Analysis ---
    addLog("[Phase 1] Analyzing repository...", "accent");
    const p1 = await runPhase(1, () => bridge.analyzeRepo(path));
    if (!p1.ok || !p1.data) {
      addLog(`[Phase 1] ✗ Failed: ${p1.error}`, "error");
      await saveRunSnapshot(runId, path, spec, "failed");
      state.running = false;
      return;
    }
    state.repoAnalysis = p1.data as unknown as RepoAnalysisResult;
    const models = state.repoAnalysis.architecture?.models?.length || 0;
    addLog(`[Phase 1] ✓ Found ${models} model(s), repo: ${state.repoAnalysis.repoName}`, "success");

    if (!(await maybeApprove(1, "Repo Analysis"))) {
      addLog("Integration stopped — Phase 1 rejected", "error");
      await saveRunSnapshot(runId, path, spec, "failed");
      state.running = false;
      return;
    }

    // --- Phase 2: Research Extraction ---
    addLog("[Phase 2] Extracting research spec...", "accent");
    const source = spec.startsWith("spec:") ? spec : `spec:${spec}`;
    const p2 = await runPhase(2, () => bridge.extractResearch(source, "arxiv"));
    if (!p2.ok || !p2.data) {
      addLog(`[Phase 2] ✗ Failed: ${p2.error}`, "error");
      await saveRunSnapshot(runId, path, spec, "failed");
      state.running = false;
      return;
    }
    state.researchSpec = p2.data as unknown as ResearchSpecResult;
    addLog(`[Phase 2] ✓ Algorithm: ${state.researchSpec.algorithm.name}`, "success");

    if (!(await maybeApprove(2, "Research Extraction"))) {
      addLog("Integration stopped — Phase 2 rejected", "error");
      await saveRunSnapshot(runId, path, spec, "failed");
      state.running = false;
      return;
    }

    // --- Phase 3: Mapping ---
    addLog("[Phase 3] Mapping spec to code...", "accent");
    const p3 = await runPhase(3, () =>
      bridge.mapArchitecture(state.repoAnalysis!, state.researchSpec!),
    );
    if (!p3.ok || !p3.data) {
      addLog(`[Phase 3] ✗ Failed: ${p3.error}`, "error");
      await saveRunSnapshot(runId, path, spec, "failed");
      state.running = false;
      return;
    }
    state.mapping = p3.data as unknown as MappingResult;
    addLog(
      `[Phase 3] ✓ ${state.mapping.targets.length} target(s), confidence: ${state.mapping.confidence}%`,
      "success",
    );

    if (!(await maybeApprove(3, "Mapping"))) {
      addLog("Integration stopped — Phase 3 rejected", "error");
      await saveRunSnapshot(runId, path, spec, "failed");
      state.running = false;
      return;
    }

    // --- Phase 4: Patch Generation ---
    addLog("[Phase 4] Generating patches...", "accent");
    const p4 = await runPhase(4, () =>
      bridge.generatePatch(state.mapping!, path),
    );
    if (!p4.ok || !p4.data) {
      addLog(`[Phase 4] ✗ Failed: ${p4.error}`, "error");
      await saveRunSnapshot(runId, path, spec, "failed");
      state.running = false;
      return;
    }
    state.patch = p4.data as unknown as PatchResult;
    addLog(
      `[Phase 4] ✓ ${state.patch.newFiles.length} new file(s), ${state.patch.transformations.length} transformation(s), branch: ${state.patch.branchName}`,
      "success",
    );

    if (!(await maybeApprove(4, "Patch Generation"))) {
      addLog("Integration stopped — Phase 4 rejected", "error");
      await saveRunSnapshot(runId, path, spec, "failed");
      state.running = false;
      return;
    }

    // --- Phase 5: Validation ---
    addLog("[Phase 5] Running validation...", "accent");
    const p5 = await runPhase(5, () => bridge.validate(state.patch!, path));
    if (!p5.ok || !p5.data) {
      addLog(`[Phase 5] ✗ Failed: ${p5.error}`, "error");
      await saveRunSnapshot(runId, path, spec, "failed");
      state.running = false;
      return;
    }
    state.validation = p5.data as unknown as ValidationResult;
    const speedup = state.validation.comparison?.speedup;
    const speedStr = typeof speedup === "number" ? ` (${speedup.toFixed(2)}x)` : "";
    addLog(
      `[Phase 5] ✓ ${state.validation.passed ? "PASSED" : "FAILED"} at ${state.validation.stage}${speedStr}`,
      state.validation.passed ? "success" : "warning",
    );

    if (!(await maybeApprove(5, "Validation"))) {
      addLog("Integration stopped — Phase 5 rejected", "error");
      await saveRunSnapshot(runId, path, spec, "failed");
      state.running = false;
      return;
    }

    // --- Phase 6: Report ---
    addLog("[Phase 6] Generating report...", "accent");
    const p6 = await runPhase(6, async () => {
      return {
        algorithm: state.researchSpec!.algorithm.name,
        paper: state.researchSpec!.paper.title,
        passed: state.validation!.passed,
        speedup: state.validation!.comparison?.speedup,
        filesModified: state.patch!.transformations.map((t) => t.file),
        newFiles: state.patch!.newFiles.map((f) => f.path),
        branchName: state.patch!.branchName,
        confidence: state.mapping!.confidence,
      };
    });
    if (!p6.ok) {
      addLog(`[Phase 6] ✗ Report generation failed: ${p6.error}`, "error");
    } else {
      addLog("[Phase 6] ✓ Report generated", "success");
    }

    // Summary
    addLog("", "info");
    addLog("═".repeat(60), "muted");
    const allPassed = state.phases.every((p) => p.status === "completed");
    if (allPassed) {
      addLog("🎉 Integration complete — all 6 phases passed", "success");
    } else {
      const failed = state.phases.filter((p) => p.status === "failed");
      addLog(`⚠  Integration finished with ${failed.length} failed phase(s)`, "warning");
    }
    addLog(`   Algorithm: ${state.researchSpec?.algorithm.name || spec}`, "info");
    addLog(`   Branch: ${state.patch?.branchName || "n/a"}`, "info");
    addLog(`   Total time: ${state.phases.reduce((s, p) => s + (p.durationMs || 0), 0) / 1000}s`, "info");
    addLog("═".repeat(60), "muted");

    await saveRunSnapshot(runId, path, spec, allPassed ? "completed" : "failed");
    state.running = false;
  }

  // --- Save run snapshot ---

  async function saveRunSnapshot(
    runId: string,
    path: string,
    spec: string,
    status: RunSnapshot["status"],
  ) {
    try {
      await runStore.save({
        runId,
        repoUrl: path,
        paperUrl: `spec:${spec}`,
        mode: state.mode,
        status,
        currentPhase: (() => { const last = state.phases.filter((p: PhaseState) => p.status === "completed").pop(); return last?.phase || 0; })(),
        phaseResults: {
          ...(state.repoAnalysis ? { 1: state.repoAnalysis } : {}),
          ...(state.researchSpec ? { 2: state.researchSpec } : {}),
          ...(state.mapping ? { 3: state.mapping } : {}),
          ...(state.patch ? { 4: state.patch } : {}),
          ...(state.validation ? { 5: state.validation } : {}),
        },
        context: { repoPath: path, spec, sourceType: "arxiv" },
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        retryCount: 0,
        approvals: [],
      });
    } catch (err) {
      addLog(`  (Could not save run: ${err instanceof Error ? err.message : err})`, "muted");
    }
  }

  // --- Command parsing ---

  function parseCommand(input: string): { action: string; args: Record<string, string> } {
    const trimmed = input.trim();

    // Mode shorthand: :autonomous, :step_approval
    if (trimmed.startsWith(":")) {
      const mode = trimmed.slice(1).trim();
      if (mode === "autonomous" || mode === "step_approval") {
        return { action: "set_mode", args: { mode } };
      }
    }

    const setProviderMatch = trimmed.match(/^set\s+provider\s+(\S+)/i);
    if (setProviderMatch) return { action: "set_provider", args: { provider: setProviderMatch[1] } };

    const setModelMatch = trimmed.match(/^set\s+model\s+(.+)/i);
    if (setModelMatch) return { action: "set_model", args: { model: setModelMatch[1].trim() } };

    const setRepoMatch = trimmed.match(/^set\s+repo\s+(.+)/i);
    if (setRepoMatch) return { action: "set_repo", args: { repo: setRepoMatch[1].trim() } };

    const setSpecMatch = trimmed.match(/^set\s+spec\s+(\S+)/i);
    if (setSpecMatch) return { action: "set_spec", args: { spec: setSpecMatch[1] } };

    const setModeMatch = trimmed.match(/^set\s+mode\s+(autonomous|step_approval)/i);
    if (setModeMatch) return { action: "set_mode", args: { mode: setModeMatch[1] } };

    if (trimmed === "setup") return { action: "setup", args: {} };
    if (trimmed === "status") return { action: "status", args: {} };
    if (trimmed === "clear") return { action: "clear", args: {} };
    if (trimmed === "help") return { action: "help", args: {} };
    if (trimmed === "quit" || trimmed === "exit") return { action: "quit", args: {} };
    if (trimmed === "runs" || trimmed === "history") return { action: "runs", args: {} };

    // Resume: resume <run-id>
    const resumeMatch = trimmed.match(/^resume\s+(\S+)/i);
    if (resumeMatch) return { action: "resume", args: { runId: resumeMatch[1] } };

    // Pipeline commands with path+spec: integrate <path> <spec>, map <path> <spec>, etc.
    const pipelineWords = ["integrate", "map", "generate", "validate", "analyze", "search", "suggest"];
    for (const action of pipelineWords) {
      if (trimmed.toLowerCase().startsWith(action + " ") || trimmed.toLowerCase() === action) {
        const rest = trimmed.slice(action.length).trim();
        const parts = rest.split(/\s+/);
        const path = parts[0] || "";
        const specOrQuery = parts[1] || "";
        return { action, args: { path, spec: specOrQuery, query: rest } };
      }
    }

    return { action: "chat", args: { query: trimmed } };
  }

  // --- Command execution ---

  async function executeCommand(input: string) {
    // Handle approval prompt first
    if (state.approvalPrompt) {
      const answer = input.trim().toLowerCase();
      const approved = answer === "y" || answer === "yes";
      state.approvalPrompt.resolve(approved);
      state.approvalPrompt = null;
      if (approved) {
        addLog("  → Approved", "success");
      } else {
        addLog("  → Rejected", "error");
      }
      return;
    }

    const { action, args } = parseCommand(input);
    addLog(`$ ${input}`, "accent");

    switch (action) {
      case "set_mode": {
        state.mode = args.mode as ExecutionMode;
        updateStatus();
        addLog(`Mode: ${state.mode}`, "success");
        break;
      }

      case "set_provider": {
        state.provider = args.provider;
        updateStatus();
        addLog(`Provider: ${state.provider}`, "success");
        break;
      }

      case "set_model": {
        state.model = args.model;
        updateStatus();
        addLog(`Model: ${state.model}`, "success");
        break;
      }

      case "set_repo": {
        state.repoPath = args.repo;
        updateStatus();
        addLog(`Repository: ${state.repoPath}`, "success");
        break;
      }

      case "set_spec": {
        state.spec = args.spec;
        updateStatus();
        addLog(`Spec: ${state.spec}`, "success");
        break;
      }

      case "setup": {
        addLog("Configuration:", "accent");
        addLog(`  Provider: ${state.provider}`, "info");
        addLog(`  Model: ${state.model}`, "info");
        addLog(`  Repository: ${state.repoPath}`, "info");
        addLog(`  Spec: ${state.spec}`, "info");
        addLog(`  Mode: ${state.mode}`, "info");
        addLog(`  Pipeline: ${serverReady ? "ready" : "offline"}`, serverReady ? "success" : "error");
        addLog("", "info");
        addLog("  Change with: set repo/spec/mode/provider/model <value>", "muted");
        break;
      }

      case "status": {
        addLog("Status:", "accent");
        addLog(`  Mode: ${state.mode}`, "info");
        addLog(`  Provider: ${state.provider} / ${state.model}`, "info");
        addLog(`  Repository: ${state.repoPath}`, "info");
        addLog(`  Spec: ${state.spec}`, "info");
        addLog(`  Pipeline: ${serverReady ? "ready" : "offline"}`, serverReady ? "success" : "error");
        const completed = state.phases.filter((p) => p.status === "completed").length;
        addLog(`  Phases: ${completed}/6 completed`, "info");
        if (state.lastRunId) addLog(`  Last run: ${state.lastRunId}`, "info");
        break;
      }

      case "clear": {
        clearLogs();
        break;
      }

      case "help": {
        addLog("Commands:", "accent");
        addLog("  integrate <path> <spec>    — full 6-phase pipeline", "info");
        addLog("  analyze <path>             — phase 1: repo analysis", "info");
        addLog("  search <query>             — phase 2: research extraction", "info");
        addLog("  map <path> <spec>          — phases 1-3: mapping", "info");
        addLog("  generate <path> <spec>     — phases 1-4: patch generation", "info");
        addLog("  validate <path>            — phases 1-5: validation", "info");
        addLog("  runs / history             — list recent runs", "info");
        addLog("  resume <run-id>            — resume a previous run", "info");
        addLog("", "info");
        addLog("  set repo <path>            — set default repository", "info");
        addLog("  set spec <name>            — set default spec", "info");
        addLog("  set mode <autonomous|step_approval>", "info");
        addLog("  set provider <name>        — set LLM provider", "info");
        addLog("  set model <id>             — set model", "info");
        addLog("  setup | status | clear | help | quit", "info");
        addLog("", "info");
        addLog("  Available specs: " + KNOWN_SPECS.join(", "), "muted");
        break;
      }

      case "quit": {
        addLog("Goodbye!", "warning");
        server.stop();
        setTimeout(() => process.exit(0), 300);
        break;
      }

      case "runs": {
        try {
          const runs = await runStore.list();
          if (runs.length === 0) {
            addLog("No runs found", "info");
          } else {
            addLog("Recent runs:", "accent");
            for (const run of runs.slice(0, 15)) {
              const icon = run.status === "completed" ? "✓" : run.status === "failed" ? "✗" : "○";
              const age = formatAge(run.updatedAt);
              addLog(
                `  ${icon} ${run.runId}  ${run.repoUrl}  ${run.status}  ${age}`,
                run.status === "completed" ? "success" : run.status === "failed" ? "error" : "info",
              );
            }
          }
        } catch (err) {
          addLog(`Error listing runs: ${err instanceof Error ? err.message : err}`, "error");
        }
        break;
      }

      case "resume": {
        const runId = args.runId;
        if (!runId) {
          addLog("Usage: resume <run-id>", "error");
          break;
        }
        try {
          const snapshot = await runStore.get(runId);
          if (!snapshot) {
            addLog(`Run not found: ${runId}`, "error");
            break;
          }
          if (snapshot.status === "completed") {
            addLog(`Run ${runId} already completed`, "info");
            break;
          }
          addLog(`Resuming run ${runId} from phase ${(snapshot.currentPhase || 0) + 1}...`, "accent");
          state.repoPath = snapshot.repoUrl;
          const specStr = snapshot.paperUrl?.replace("spec:", "") || state.spec;
          state.spec = specStr;
          updateStatus();
          await runIntegration(snapshot.repoUrl, specStr);
        } catch (err) {
          addLog(`Error resuming: ${err instanceof Error ? err.message : err}`, "error");
        }
        break;
      }

      case "integrate": {
        const path = args.path || state.repoPath;
        const spec = args.spec || state.spec;
        if (!path) {
          addLog("Usage: integrate <path> <spec>", "error");
          break;
        }
        await runIntegration(path, spec);
        break;
      }

      case "analyze": {
        const path = args.path || state.repoPath;
        if (!path) {
          addLog("Usage: analyze <path>", "error");
          break;
        }
        if (!serverReady) {
          addLog("Error: Pipeline server not running", "error");
          break;
        }
        state.running = true;
        state.cancellationRequested = false;
        resetPhases();
        addLog(`[Phase 1] Analyzing ${path}...`, "accent");
        const result = await runPhase(1, () => bridge.analyzeRepo(path));
        if (result.ok && result.data) {
          state.repoAnalysis = result.data as unknown as RepoAnalysisResult;
          const models = state.repoAnalysis.architecture?.models?.length || 0;
          addLog(`[Phase 1] ✓ ${models} model(s), repo: ${state.repoAnalysis.repoName}`, "success");
        } else {
          addLog(`[Phase 1] ✗ ${result.error}`, "error");
        }
        state.running = false;
        break;
      }

      case "search": {
        const query = args.spec || args.query || "";
        if (!query) {
          addLog("Usage: search <query>", "error");
          break;
        }
        if (!serverReady) {
          addLog("Error: Pipeline server not running", "error");
          break;
        }
        state.running = true;
        state.cancellationRequested = false;
        resetPhases();
        addLog(`[Phase 2] Searching: ${query}...`, "accent");
        const result = await runPhase(2, () => bridge.extractResearch(query, "arxiv"));
        if (result.ok && result.data) {
          state.researchSpec = result.data as unknown as ResearchSpecResult;
          addLog(`[Phase 2] ✓ ${state.researchSpec.algorithm.name}`, "success");
          addLog(`   Paper: ${state.researchSpec.paper.title}`, "info");
        } else {
          addLog(`[Phase 2] ✗ ${result.error}`, "error");
        }
        state.running = false;
        break;
      }

      case "suggest": {
        const path = args.path || state.repoPath;
        if (!path) {
          addLog("Usage: suggest <path>", "error");
          break;
        }
        if (!serverReady) {
          addLog("Error: Pipeline server not running", "error");
          break;
        }
        state.running = true;
        state.cancellationRequested = false;
        resetPhases();
        addLog(`[Phase 1] Analyzing ${path} for suggestions...`, "accent");
        const result = await runPhase(1, () => bridge.analyzeRepo(path));
        if (result.ok && result.data) {
          state.repoAnalysis = result.data as unknown as RepoAnalysisResult;
          addLog(`[Phase 1] ✓ Analysis complete`, "success");
          addLog(`   Models: ${state.repoAnalysis.architecture?.models?.length || 0}`, "info");
          addLog(`   Suggestion: Run "integrate ${path} <spec>" to apply an improvement`, "muted");
        } else {
          addLog(`[Phase 1] ✗ ${result.error}`, "error");
        }
        state.running = false;
        break;
      }

      case "map": {
        const path = args.path || state.repoPath;
        const spec = args.spec || state.spec;
        if (!path) {
          addLog("Usage: map <path> <spec>", "error");
          break;
        }
        if (!serverReady) {
          addLog("Error: Pipeline server not running", "error");
          break;
        }
        state.running = true;
        state.cancellationRequested = false;
        resetPhases();

        // Ensure we have repoAnalysis
        if (!state.repoAnalysis) {
          addLog("[Phase 1] Analyzing repository...", "accent");
          const p1 = await runPhase(1, () => bridge.analyzeRepo(path));
          if (!p1.ok || !p1.data) {
            addLog(`[Phase 1] ✗ ${p1.error}`, "error");
            state.running = false;
            break;
          }
          state.repoAnalysis = p1.data as unknown as RepoAnalysisResult;
          addLog(`[Phase 1] ✓ ${state.repoAnalysis.repoName}`, "success");
        }

        // Ensure we have researchSpec
        if (!state.researchSpec) {
          addLog("[Phase 2] Extracting research spec...", "accent");
          const source = spec.startsWith("spec:") ? spec : `spec:${spec}`;
          const p2 = await runPhase(2, () => bridge.extractResearch(source, "arxiv"));
          if (!p2.ok || !p2.data) {
            addLog(`[Phase 2] ✗ ${p2.error}`, "error");
            state.running = false;
            break;
          }
          state.researchSpec = p2.data as unknown as ResearchSpecResult;
          addLog(`[Phase 2] ✓ ${state.researchSpec.algorithm.name}`, "success");
        }

        addLog("[Phase 3] Mapping...", "accent");
        const p3 = await runPhase(3, () =>
          bridge.mapArchitecture(state.repoAnalysis!, state.researchSpec!),
        );
        if (p3.ok && p3.data) {
          state.mapping = p3.data as unknown as MappingResult;
          addLog(
            `[Phase 3] ✓ ${state.mapping.targets.length} target(s), confidence: ${state.mapping.confidence}%`,
            "success",
          );
        } else {
          addLog(`[Phase 3] ✗ ${p3.error}`, "error");
        }
        state.running = false;
        break;
      }

      case "generate": {
        const path = args.path || state.repoPath;
        const spec = args.spec || state.spec;
        if (!path) {
          addLog("Usage: generate <path> <spec>", "error");
          break;
        }
        if (!serverReady) {
          addLog("Error: Pipeline server not running", "error");
          break;
        }
        state.running = true;
        state.cancellationRequested = false;
        resetPhases();

        // Build up to mapping if needed
        if (!state.repoAnalysis) {
          addLog("[Phase 1] Analyzing repository...", "accent");
          const p1 = await runPhase(1, () => bridge.analyzeRepo(path));
          if (!p1.ok || !p1.data) { addLog(`[Phase 1] ✗ ${p1.error}`, "error"); state.running = false; break; }
          state.repoAnalysis = p1.data as unknown as RepoAnalysisResult;
          addLog(`[Phase 1] ✓ ${state.repoAnalysis.repoName}`, "success");
        }
        if (!state.researchSpec) {
          addLog("[Phase 2] Extracting research spec...", "accent");
          const source = spec.startsWith("spec:") ? spec : `spec:${spec}`;
          const p2 = await runPhase(2, () => bridge.extractResearch(source, "arxiv"));
          if (!p2.ok || !p2.data) { addLog(`[Phase 2] ✗ ${p2.error}`, "error"); state.running = false; break; }
          state.researchSpec = p2.data as unknown as ResearchSpecResult;
          addLog(`[Phase 2] ✓ ${state.researchSpec.algorithm.name}`, "success");
        }
        if (!state.mapping) {
          addLog("[Phase 3] Mapping...", "accent");
          const p3 = await runPhase(3, () => bridge.mapArchitecture(state.repoAnalysis!, state.researchSpec!));
          if (!p3.ok || !p3.data) { addLog(`[Phase 3] ✗ ${p3.error}`, "error"); state.running = false; break; }
          state.mapping = p3.data as unknown as MappingResult;
          addLog(`[Phase 3] ✓ ${state.mapping.targets.length} target(s)`, "success");
        }

        addLog("[Phase 4] Generating patches...", "accent");
        const p4 = await runPhase(4, () => bridge.generatePatch(state.mapping!, path));
        if (p4.ok && p4.data) {
          state.patch = p4.data as unknown as PatchResult;
          addLog(
            `[Phase 4] ✓ ${state.patch.newFiles.length} new, ${state.patch.transformations.length} modified, branch: ${state.patch.branchName}`,
            "success",
          );
        } else {
          addLog(`[Phase 4] ✗ ${p4.error}`, "error");
        }
        state.running = false;
        break;
      }

      case "validate": {
        const path = args.path || state.repoPath;
        if (!path) {
          addLog("Usage: validate <path>", "error");
          break;
        }
        if (!serverReady) {
          addLog("Error: Pipeline server not running", "error");
          break;
        }
        state.running = true;
        state.cancellationRequested = false;
        resetPhases();

        // Build up to patch if needed
        if (!state.repoAnalysis) {
          addLog("[Phase 1] Analyzing repository...", "accent");
          const p1 = await runPhase(1, () => bridge.analyzeRepo(path));
          if (!p1.ok || !p1.data) { addLog(`[Phase 1] ✗ ${p1.error}`, "error"); state.running = false; break; }
          state.repoAnalysis = p1.data as unknown as RepoAnalysisResult;
          addLog(`[Phase 1] ✓ ${state.repoAnalysis.repoName}`, "success");
        }
        if (!state.researchSpec) {
          const spec = args.spec || state.spec;
          addLog("[Phase 2] Extracting research spec...", "accent");
          const source = spec.startsWith("spec:") ? spec : `spec:${spec}`;
          const p2 = await runPhase(2, () => bridge.extractResearch(source, "arxiv"));
          if (!p2.ok || !p2.data) { addLog(`[Phase 2] ✗ ${p2.error}`, "error"); state.running = false; break; }
          state.researchSpec = p2.data as unknown as ResearchSpecResult;
          addLog(`[Phase 2] ✓ ${state.researchSpec.algorithm.name}`, "success");
        }
        if (!state.mapping) {
          addLog("[Phase 3] Mapping...", "accent");
          const p3 = await runPhase(3, () => bridge.mapArchitecture(state.repoAnalysis!, state.researchSpec!));
          if (!p3.ok || !p3.data) { addLog(`[Phase 3] ✗ ${p3.error}`, "error"); state.running = false; break; }
          state.mapping = p3.data as unknown as MappingResult;
          addLog(`[Phase 3] ✓ ${state.mapping.targets.length} target(s)`, "success");
        }
        if (!state.patch) {
          addLog("[Phase 4] Generating patches...", "accent");
          const p4 = await runPhase(4, () => bridge.generatePatch(state.mapping!, path));
          if (!p4.ok || !p4.data) { addLog(`[Phase 4] ✗ ${p4.error}`, "error"); state.running = false; break; }
          state.patch = p4.data as unknown as PatchResult;
          addLog(`[Phase 4] ✓ branch: ${state.patch.branchName}`, "success");
        }

        addLog("[Phase 5] Running validation...", "accent");
        const p5 = await runPhase(5, () => bridge.validate(state.patch!, path));
        if (p5.ok && p5.data) {
          state.validation = p5.data as unknown as ValidationResult;
          const speedup = state.validation.comparison?.speedup;
          const speedStr = typeof speedup === "number" ? ` (${speedup.toFixed(2)}x)` : "";
          addLog(
            `[Phase 5] ✓ ${state.validation.passed ? "PASSED" : "FAILED"}${speedStr}`,
            state.validation.passed ? "success" : "warning",
          );
        } else {
          addLog(`[Phase 5] ✗ ${p5.error}`, "error");
        }
        state.running = false;
        break;
      }

      default: {
        addLog(`Unknown command: ${action}. Type 'help' for commands.`, "warning");
      }
    }
  }

  // --- Utility ---

  function formatAge(isoDate: string): string {
    const diff = Date.now() - new Date(isoDate).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  }

  // --- Event wiring ---

  promptInput.on(InputRenderableEvents.ENTER, (value: string) => {
    if (!value.trim()) return;
    if (state.running && !state.approvalPrompt) return;

    state.commandHistory.push(value.trim());
    state.historyIndex = state.commandHistory.length;
    promptInput.value = "";
    executeCommand(value.trim());
  });

  renderer.keyInput.on("keypress", (key: KeyEvent) => {
    // Ctrl+C: cancel or quit
    if (key.ctrl && key.name === "c") {
      if (state.running) {
        state.cancellationRequested = true;
        if (state.approvalPrompt) {
          state.approvalPrompt.resolve(false);
          state.approvalPrompt = null;
        }
        addLog("Task cancelled", "warning");
      } else {
        addLog("Goodbye!", "warning");
        server.stop();
        setTimeout(() => process.exit(0), 300);
      }
      return;
    }

    // Ctrl+K: clear logs
    if (key.ctrl && key.name === "k") {
      clearLogs();
      addLog("Output cleared", "info");
      return;
    }

    // Escape: quit
    if (key.name === "escape" && !state.approvalPrompt) {
      addLog("Goodbye!", "warning");
      server.stop();
      setTimeout(() => process.exit(0), 300);
      return;
    }

    // Tab: autocomplete
    if (key.name === "tab" && !state.running) {
      const current = promptInput.value;
      if (!current) {
        promptInput.value = `integrate ${state.repoPath} ${state.spec}`;
      } else {
        // Try to autocomplete spec names
        const lower = current.toLowerCase();
        const match = KNOWN_SPECS.find((s) => s.startsWith(lower.split(/\s+/).pop() || ""));
        if (match) {
          const parts = current.split(/\s+/);
          parts[parts.length - 1] = match;
          promptInput.value = parts.join(" ");
        }
      }
      return;
    }

    // Up: history prev
    if (key.name === "up" && state.commandHistory.length > 0 && !state.approvalPrompt) {
      if (state.historyIndex <= 0) state.historyIndex = state.commandHistory.length;
      state.historyIndex--;
      promptInput.value = state.commandHistory[state.historyIndex] || "";
      return;
    }

    // Down: history next
    if (key.name === "down" && !state.approvalPrompt) {
      if (state.historyIndex < state.commandHistory.length - 1) {
        state.historyIndex++;
        promptInput.value = state.commandHistory[state.historyIndex] || "";
      } else {
        state.historyIndex = state.commandHistory.length;
        promptInput.value = "";
      }
      return;
    }
  });

  // Welcome message
  addLog("ScholarDevClaw OpenTUI — keyboard-first research shell", "accent");
  addLog(`Provider: ${state.provider} | Model: ${state.model} | Mode: ${state.mode}`, "info");
  addLog(`Repository: ${state.repoPath} | Spec: ${state.spec}`, "info");
  addLog("Type 'help' for commands, 'integrate <path> <spec>' to start", "info");
  addLog("─".repeat(76), "muted");
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
