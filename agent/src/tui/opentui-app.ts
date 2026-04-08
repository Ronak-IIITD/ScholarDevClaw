/**
 * ScholarDevClaw OpenTUI — keyboard-first terminal shell.
 *
 * Architecture:
 * - OpenTUI (Zig core + TS bindings) renders the UI
 * - Python FastAPI pipeline runs in background (auto-started)
 * - HTTP bridge connects UI → pipeline
 * - Provider setup (OpenRouter/Ollama) persisted via auth store
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

const MODES = ["analyze", "search", "edit", "chat"] as const;
type Mode = (typeof MODES)[number];

interface AppState {
  mode: Mode;
  provider: string;
  model: string;
  directory: string;
  history: string[];
  historyIndex: number;
  running: boolean;
  logLines: string[];
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

  constructor(port: number = 8000) {
    this.port = port;
  }

  async start(): Promise<boolean> {
    // Check if already running
    try {
      const res = await fetch(`http://localhost:${this.port}/health`);
      if (res.ok) return true;
    } catch {
      // not running
    }

    const venvPython = join(CORE_ROOT, ".venv", "bin", "python");
    const py = existsSync(venvPython) ? venvPython : "python3";

    return new Promise((resolve) => {
      this.process = spawn(py, ["-m", "uvicorn", "scholardevclaw.api.server:app", "--port", String(this.port)], {
        cwd: CORE_ROOT,
        stdio: ["ignore", "pipe", "pipe"],
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
      });

      // Suppress server output — we show status in UI instead
      this.process.stdout?.on("data", () => {});
      this.process.stderr?.on("data", () => {});

      // Wait for server to be ready
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
// Main TUI App
// ---------------------------------------------------------------------------

export async function main() {
  const state: AppState = {
    mode: "analyze",
    provider: process.env.SCHOLARDEVCLAW_API_PROVIDER || DEFAULT_PROVIDER,
    model: process.env.SCHOLARDEVCLAW_API_MODEL || DEFAULT_MODEL,
    directory: process.argv[2] || ".",
    history: [],
    historyIndex: -1,
    running: false,
    logLines: [],
  };

  const server = new PythonServer();
  const bridge = new PythonHttpBridge(`http://localhost:${8000}`);

  // Start Python server
  const serverReady = await server.start();

  const renderer = await createCliRenderer({
    exitOnCtrlC: false,
  });

  // --- Build UI components ---

  // Header
  const headerText = Text({
    content: ` ScholarDevClaw`,
    fg: C.accent,
  });

  const statusText = Text({
    content: `MODE: ${state.mode}   PROVIDER: ${state.provider}   MODEL: ${state.model}   DIR: ${state.directory}`,
    fg: C.muted,
  });

  const serverStatusText = Text({
    content: serverReady ? `● Pipeline ready` : `○ Pipeline offline`,
    fg: serverReady ? C.green : C.red,
  });

  const header = Box(
    {
      flexDirection: "column",
      gap: 0,
      marginBottom: 1,
    },
    headerText,
    statusText,
    serverStatusText,
  );

  // Separator
  const separator = Text({
    content: "─".repeat(80),
    fg: C.border,
  });

  // Log / output area
  const logBox = Box(
    {
      flexDirection: "column",
      gap: 0,
      height: 12,
      borderStyle: "rounded",
      padding: 1,
    },
    Text({ content: "Workflow output", fg: C.muted }),
    Text({ content: "─".repeat(76), fg: C.border }),
  );

  const logScroll = ScrollBox({
    width: 76,
    height: 9,
  });

  logBox.add(logScroll);

  // Hint line
  const hintText = Text({
    content: `Hint → ${state.mode} ${state.directory}`,
    fg: C.muted,
  });

  // Command input
  const promptInput = Input({
    placeholder: `> ${state.mode} ${state.directory} ...`,
    width: 78,
    backgroundColor: C.surface,
    focusedBackgroundColor: "#2a2a3e",
    textColor: C.text,
    cursorColor: C.accent,
  });

  // Key hints
  const keyHints = Text({
    content: "Tab: autocomplete  ↑↓: history  Enter: run  Ctrl+C: cancel  Ctrl+K: clear  Esc: quit",
    fg: C.muted,
  });

  // --- Assemble root ---

  renderer.root.add(
    Box(
      {
        flexDirection: "column",
        gap: 0,
        padding: 1,
      },
      header,
      separator,
      logBox,
      hintText,
      promptInput,
      separator,
      keyHints,
    ),
  );

  // Focus prompt
  promptInput.focus();

  // --- Log helpers ---

  function addLog(text: string, level: string = "info") {
    const color = level === "error" ? C.red : level === "success" ? C.green : level === "warning" ? C.yellow : C.text;
    const line = Text({ content: text, fg: color });
    logScroll.add(line);
    state.logLines.push(text);
  }

  function clearLogs() {
    // Remove all children except the header + separator
    const children = [...(logScroll.children ?? [])];
    for (const child of children) {
      if (child && typeof child === "object" && "id" in child && typeof child.id === "string") {
        logScroll.remove(child.id);
      }
    }
    state.logLines = [];
  }

  function updateStatus() {
    statusText.content = stringToStyledText(
      `MODE: ${state.mode}   PROVIDER: ${state.provider}   MODEL: ${state.model}   DIR: ${state.directory}`,
    );
    hintText.content = stringToStyledText(`Hint → ${state.mode} ${state.directory}`);
    promptInput.placeholder = `> ${state.mode} ${state.directory} ...`;
  }

  // --- Command parsing ---

  function parseCommand(input: string): { action: string; args: Record<string, string> } {
    const trimmed = input.trim();

    // Mode shorthand: :analyze, :search, :edit
    if (trimmed.startsWith(":")) {
      const mode = trimmed.slice(1).trim() as Mode;
      if (MODES.includes(mode)) {
        return { action: "set_mode", args: { mode } };
      }
    }

    // set provider X
    const setProviderMatch = trimmed.match(/^set\s+provider\s+(\S+)/i);
    if (setProviderMatch) {
      return { action: "set_provider", args: { provider: setProviderMatch[1] } };
    }

    // set model X
    const setModelMatch = trimmed.match(/^set\s+model\s+(.+)/i);
    if (setModelMatch) {
      return { action: "set_model", args: { model: setModelMatch[1].trim() } };
    }

    // set dir X
    const setDirMatch = trimmed.match(/^set\s+dir\s+(.+)/i);
    if (setDirMatch) {
      return { action: "set_dir", args: { directory: setDirMatch[1].trim() } };
    }

    // setup
    if (trimmed === "setup") {
      return { action: "setup", args: {} };
    }

    // status
    if (trimmed === "status") {
      return { action: "status", args: {} };
    }

    // clear
    if (trimmed === "clear") {
      return { action: "clear", args: {} };
    }

    // help
    if (trimmed === "help") {
      return { action: "help", args: {} };
    }

    // quit / exit
    if (trimmed === "quit" || trimmed === "exit") {
      return { action: "quit", args: {} };
    }

    // Natural language: "analyze ./repo", "search layer normalization"
    const actionWords = ["analyze", "search", "suggest", "map", "generate", "validate", "integrate", "chat"];
    for (const action of actionWords) {
      if (trimmed.toLowerCase().startsWith(action)) {
        const rest = trimmed.slice(action.length).trim();
        return { action, args: { query: rest } };
      }
    }

    // Default: chat
    return { action: "chat", args: { query: trimmed } };
  }

  // --- Command execution ---

  async function executeCommand(input: string) {
    const { action, args } = parseCommand(input);

    addLog(`$ ${input}`, "accent");

    switch (action) {
      case "set_mode": {
        state.mode = args.mode as Mode;
        updateStatus();
        addLog(`Mode set to: ${state.mode}`, "success");
        break;
      }

      case "set_provider": {
        state.provider = args.provider;
        updateStatus();
        addLog(`Provider set to: ${state.provider}`, "success");
        break;
      }

      case "set_model": {
        state.model = args.model;
        updateStatus();
        addLog(`Model set to: ${state.model}`, "success");
        break;
      }

      case "set_dir": {
        state.directory = args.directory;
        updateStatus();
        addLog(`Directory set to: ${state.directory}`, "success");
        break;
      }

      case "setup": {
        addLog("Provider setup:", "accent");
        addLog(`  Current: ${state.provider} / ${state.model}`, "info");
        addLog(`  Change with: set provider <name>`, "info");
        addLog(`  Change with: set model <model-id>`, "info");
        break;
      }

      case "status": {
        addLog(`Mode: ${state.mode}`, "info");
        addLog(`Provider: ${state.provider}`, "info");
        addLog(`Model: ${state.model}`, "info");
        addLog(`Directory: ${state.directory}`, "info");
        addLog(`Pipeline: ${serverReady ? "ready" : "offline"}`, serverReady ? "success" : "error");
        break;
      }

      case "clear": {
        clearLogs();
        break;
      }

      case "help": {
        addLog("Commands:", "accent");
        addLog("  :analyze | :search | :edit | :chat  — set mode", "info");
        addLog("  set provider <openrouter|ollama>     — set LLM provider", "info");
        addLog("  set model <model-id>                 — set model (e.g. anthropic/claude-sonnet-4)", "info");
        addLog("  set dir <path>                       — set target directory", "info");
        addLog("  analyze ./repo                       — analyze repository", "info");
        addLog("  search <query>                       — research search", "info");
        addLog("  suggest ./repo                       — suggest improvements", "info");
        addLog("  map ./repo <spec>                    — map paper to code", "info");
        addLog("  generate ./repo <spec>               — generate patch", "info");
        addLog("  validate ./repo                      — validate changes", "info");
        addLog("  integrate ./repo <spec>              — integrate paper", "info");
        addLog("  chat <message>                       — chat with LLM", "info");
        addLog("  setup | status | clear | help | quit", "info");
        break;
      }

      case "quit": {
        addLog("Goodbye!", "warning");
        setTimeout(() => {
          server.stop();
          process.exit(0);
        }, 500);
        break;
      }

      case "chat": {
        if (!serverReady) {
          addLog("Error: Pipeline server not running", "error");
          break;
        }
        state.running = true;
        addLog("Chatting...", "info");
        try {
          const result = await bridge.healthCheck();
          addLog(`Pipeline health: ${result ? "OK" : "DEGRADED"}`, result ? "success" : "warning");
          addLog(`Chat: "${args.query}"`, "info");
          addLog("(Chat streaming requires LLM key — set provider + model first)", "warning");
        } catch (err) {
          addLog(`Error: ${err instanceof Error ? err.message : String(err)}`, "error");
        }
        state.running = false;
        break;
      }

      default: {
        // Pipeline commands: analyze, search, suggest, map, generate, validate, integrate
        if (!serverReady) {
          addLog("Error: Pipeline server not running", "error");
          break;
        }

        state.running = true;
        addLog(`Running: ${action}...`, "info");

        try {
          let result;
          const query = args.query || "";

          switch (action) {
            case "analyze":
              result = await bridge.analyzeRepo(state.directory);
              break;
            case "search":
              result = await bridge.extractResearch(query, "arxiv");
              break;
            case "suggest":
              result = await bridge.analyzeRepo(state.directory);
              break;
            case "map":
            case "generate":
            case "validate":
            case "integrate":
              addLog(`Note: ${action} requires prior analyze + suggest steps`, "warning");
              result = { success: false, error: "Prerequisites not met" };
              break;
            default:
              result = { success: false, error: `Unknown action: ${action}` };
          }

          if (result.success) {
            addLog(`${action} complete ✓`, "success");
          } else {
            addLog(`${action} failed: ${result.error || "unknown error"}`, "error");
          }
        } catch (err) {
          addLog(`Error: ${err instanceof Error ? err.message : String(err)}`, "error");
        }

        state.running = false;
      }
    }
  }

  // --- Event wiring ---

  // Input submit
  promptInput.on(InputRenderableEvents.ENTER, (value: string) => {
    if (!value.trim() || state.running) return;

    // Add to history
    state.history.push(value.trim());
    state.historyIndex = state.history.length;

    // Clear input
    promptInput.value = "";

    // Execute
    executeCommand(value.trim());
  });

  // Global key handling
  renderer.keyInput.on("keypress", (key: KeyEvent) => {
    // Ctrl+C: cancel or quit
    if (key.ctrl && key.name === "c") {
      if (state.running) {
        state.running = false;
        addLog("Task cancelled", "warning");
      } else {
        addLog("Goodbye!", "warning");
        setTimeout(() => {
          server.stop();
          process.exit(0);
        }, 500);
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
    if (key.name === "escape") {
      addLog("Goodbye!", "warning");
      setTimeout(() => {
        server.stop();
        process.exit(0);
      }, 500);
      return;
    }

    // Tab: autocomplete (simple mode/provider suggestions)
    if (key.name === "tab" && !state.running) {
      const current = promptInput.value;
      if (!current) {
        promptInput.value = `${state.mode} ${state.directory}`;
      }
      return;
    }

    // Up: history prev
    if (key.name === "up" && state.history.length > 0) {
      if (state.historyIndex <= 0) {
        state.historyIndex = state.history.length;
      }
      state.historyIndex--;
      promptInput.value = state.history[state.historyIndex] || "";
      return;
    }

    // Down: history next
    if (key.name === "down") {
      if (state.historyIndex < state.history.length - 1) {
        state.historyIndex++;
        promptInput.value = state.history[state.historyIndex] || "";
      } else {
        state.historyIndex = state.history.length;
        promptInput.value = "";
      }
      return;
    }
  });

  // Welcome message
  addLog("ScholarDevClaw OpenTUI — keyboard-first research shell", "accent");
  addLog(`Provider: ${state.provider} | Model: ${state.model}`, "info");
  addLog("Type 'help' for commands, 'quit' to exit", "info");
  addLog("─".repeat(76), "muted");
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
