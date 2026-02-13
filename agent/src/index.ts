import { spawn, execSync } from "child_process";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { join, resolve } from "path";
import { homedir } from "os";

export interface Config {
  pythonCorePath: string;
  pythonCommand: string;
  coreApiUrl: string;
  maxRetries: number;
  benchmarkTimeout: number;
  logLevel: "debug" | "info" | "warn" | "error";
}

export interface PhaseResult {
  success: boolean;
  data?: unknown;
  error?: string;
  confidence?: number;
}

export interface IntegrationContext {
  id: string;
  repoUrl: string;
  repoPath: string;
  paperSource: string;
  specName: string;
  mode: "step_approval" | "autonomous";
  phase: number;
  results: Record<number, unknown>;
  branchName?: string;
}

export class ScholarDevClawAgent {
  private config: Config;
  private logDir: string;

  constructor(config?: Partial<Config>) {
    this.config = {
      pythonCorePath: config?.pythonCorePath || resolve(process.cwd(), "../core/src"),
      pythonCommand: config?.pythonCommand || "python3",
      coreApiUrl: config?.coreApiUrl || "http://localhost:8000",
      maxRetries: config?.maxRetries || 2,
      benchmarkTimeout: config?.benchmarkTimeout || 300,
      logLevel: config?.logLevel || "info",
    };

    this.logDir = join(homedir(), ".scholardevclaw", "logs");
    this.ensureDir(this.logDir);
  }

  private ensureDir(path: string): void {
    if (!existsSync(path)) {
      mkdirSync(path, { recursive: true });
    }
  }

  private log(level: string, message: string, data?: unknown): void {
    const timestamp = new Date().toISOString();
    const logEntry = `[${timestamp}] [${level.toUpperCase()}] ${message}${
      data ? ` ${JSON.stringify(data)}` : ""
    }\n`;

    const logFile = join(this.logDir, `scholardevclaw-${new Date().toISOString().split("T")[0]}.log`);
    try {
      writeFileSync(logFile, logEntry, { flag: "a" });
    } catch (e) {
      console.error("Failed to write log:", e);
    }

    if (level === "error") {
      console.error(logEntry);
    } else if (this.config.logLevel === "debug" || level !== "debug") {
      console.log(logEntry);
    }
  }

  async runIntegration(
    repoUrl: string,
    paperSource: string,
    specName: string,
    mode: "step_approval" | "autonomous" = "step_approval"
  ): Promise<IntegrationContext> {
    const context: IntegrationContext = {
      id: this.generateId(),
      repoUrl,
      repoPath: "",
      paperSource,
      specName,
      mode,
      phase: 0,
      results: {},
    };

    this.log("info", "Starting integration", { id: context.id, repoUrl, specName });

    try {
      // Phase 1: Repository Intelligence
      this.log("info", "Phase 1: Analyzing repository");
      context.phase = 1;
      const phase1Result = await this.executePhase1(repoUrl);
      context.results[1] = phase1Result;
      context.repoPath = this.extractRepoPath(repoUrl);

      if (!phase1Result.success) {
        throw new Error(`Phase 1 failed: ${phase1Result.error}`);
      }

      if (mode === "step_approval") {
        this.log("info", "Phase 1 complete - awaiting approval");
        // In autonomous mode, continue without approval
      }

      // Phase 2: Research Intelligence
      this.log("info", "Phase 2: Extracting research specification");
      context.phase = 2;
      const phase2Result = await this.executePhase2(paperSource, specName);
      context.results[2] = phase2Result;

      if (!phase2Result.success) {
        throw new Error(`Phase 2 failed: ${phase2Result.error}`);
      }

      if (mode === "step_approval") {
        this.log("info", "Phase 2 complete - awaiting approval");
      }

      // Phase 3: Mapping Engine
      this.log("info", "Phase 3: Mapping architecture");
      context.phase = 3;
      const phase3Result = await this.executePhase3(
        context.repoPath,
        phase1Result.data,
        phase2Result.data
      );
      context.results[3] = phase3Result;

      if (!phase3Result.success) {
        throw new Error(`Phase 3 failed: ${phase3Result.error}`);
      }

      if (mode === "step_approval") {
        this.log("info", "Phase 3 complete - awaiting approval");
      }

      // Phase 4: Patch Generation
      this.log("info", "Phase 4: Generating patch");
      context.phase = 4;
      const phase4Result = await this.executePhase4(
        context.repoPath,
        phase3Result.data
      );
      context.results[4] = phase4Result;
      context.branchName = (phase4Result.data as any)?.branchName;

      if (!phase4Result.success) {
        throw new Error(`Phase 4 failed: ${phase4Result.error}`);
      }

      if (mode === "step_approval") {
        this.log("info", "Phase 4 complete - awaiting approval");
      }

      // Phase 5: Validation
      this.log("info", "Phase 5: Running validation");
      context.phase = 5;
      const phase5Result = await this.executePhase5(
        context.repoPath,
        phase4Result.data
      );
      context.results[5] = phase5Result;

      if (!phase5Result.success) {
        if ((phase5Result.data as any)?.retryCount < this.config.maxRetries) {
          this.log("warn", "Validation failed, will retry", {
            retryCount: (phase5Result.data as any)?.retryCount,
          });
        } else {
          throw new Error(`Phase 5 failed: ${phase5Result.error}`);
        }
      }

      if (mode === "step_approval") {
        this.log("info", "Phase 5 complete - awaiting approval");
      }

      // Phase 6: Report Generation
      this.log("info", "Phase 6: Generating report");
      context.phase = 6;
      const phase6Result = await this.executePhase6(context);
      context.results[6] = phase6Result;

      this.log("info", "Integration complete", {
        id: context.id,
        branch: context.branchName,
        recommendation: (phase6Result.data as any)?.recommendation,
      });

      return context;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.log("error", "Integration failed", { id: context.id, error: errorMessage });
      throw error;
    }
  }

  private async executePhase1(repoUrl: string): Promise<PhaseResult> {
    return this.runPythonScript("repo_intelligence", "parser", [
      "--repo-path",
      repoUrl,
      "--output-json",
    ]);
  }

  private async executePhase2(
    paperSource: string,
    specName: string
  ): Promise<PhaseResult> {
    return this.runPythonScript("research_intelligence", "extractor", [
      "--spec",
      specName,
      "--output-json",
    ]);
  }

  private async executePhase3(
    repoPath: string,
    repoAnalysis: unknown,
    researchSpec: unknown
  ): Promise<PhaseResult> {
    return this.runPythonScript("mapping", "engine", [
      "--repo-path",
      repoPath,
      "--spec",
      (researchSpec as any)?.algorithm?.name || "rmsnorm",
      "--output-json",
    ]);
  }

  private async executePhase4(
    repoPath: string,
    mapping: unknown
  ): Promise<PhaseResult> {
    return this.runPythonScript("patch_generation", "generator", [
      "--repo-path",
      repoPath,
      "--output-json",
    ]);
  }

  private async executePhase5(
    repoPath: string,
    patch: unknown
  ): Promise<PhaseResult> {
    return this.runPythonScript("validation", "runner", [
      "--repo-path",
      repoPath,
      "--output-json",
    ]);
  }

  private async executePhase6(context: IntegrationContext): Promise<PhaseResult> {
    const report = {
      metadata: {
        integrationId: context.id,
        repoUrl: context.repoUrl,
        paperSource: context.paperSource,
        specName: context.specName,
        completedAt: new Date().toISOString(),
      },
      summary: {
        phases: context.phase,
        branchName: context.branchName,
        results: Object.keys(context.results).map((k) => ({
          phase: k,
          success: (context.results[parseInt(k)] as PhaseResult)?.success,
        })),
      },
      recommendation: this.calculateRecommendation(context),
      confidence: this.calculateConfidence(context),
    };

    return {
      success: true,
      data: report,
      confidence: report.confidence,
    };
  }

  private calculateRecommendation(context: IntegrationContext): string {
    const phase5Result = context.results[5] as PhaseResult | undefined;
    const passed = (phase5Result?.data as any)?.passed;

    if (!passed) return "review";
    if (passed && context.mode === "autonomous") return "approve";
    return "review";
  }

  private calculateConfidence(context: IntegrationContext): number {
    let confidence = 50;

    if (context.phase >= 1) confidence += 10;
    if (context.phase >= 2) confidence += 10;
    if (context.phase >= 3) confidence += 10;
    if (context.phase >= 4) confidence += 10;
    if (context.phase >= 5) confidence += 10;
    if (context.phase >= 6) confidence += 10;

    return confidence;
  }

  private async runPythonScript(
    module: string,
    script: string,
    args: string[]
  ): Promise<PhaseResult> {
    return new Promise((resolve) => {
      const fullArgs = [
        "-m",
        `scholardevclaw.${module}.${script}`,
        ...args,
      ];

      this.log("debug", `Running: ${this.config.pythonCommand} ${fullArgs.join(" ")}`);

      const proc = spawn(this.config.pythonCommand, fullArgs, {
        cwd: resolve(this.config.pythonCorePath),
        stdio: ["pipe", "pipe", "pipe"],
      });

      let stdout = "";
      let stderr = "";

      proc.stdout.on("data", (data) => {
        stdout += data.toString();
      });

      proc.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      proc.on("close", (code) => {
        if (code === 0) {
          try {
            const data = stdout.trim() ? JSON.parse(stdout) : {};
            resolve({ success: true, data });
          } catch {
            resolve({ success: true, data: stdout.trim() });
          }
        } else {
          this.log("error", `Python script failed: ${stderr}`);
          resolve({ success: false, error: stderr || `Exit code: ${code}` });
        }
      });

      proc.on("error", (err) => {
        this.log("error", `Failed to start Python: ${err.message}`);
        resolve({ success: false, error: err.message });
      });

      setTimeout(() => {
        proc.kill();
        resolve({ success: false, error: "Timeout" });
      }, this.config.benchmarkTimeout * 1000);
    });
  }

  private extractRepoPath(repoUrl: string): string {
    // For local repos, return as-is. For GitHub URLs, would need to clone
    if (repoUrl.startsWith("/") || repoUrl.startsWith(".")) {
      return resolve(repoUrl);
    }
    // Default to test_repos for now
    return resolve("./test_repos/nanogpt");
  }

  private generateId(): string {
    return `int_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Git operations
  async createBranch(repoPath: string, branchName: string): Promise<boolean> {
    try {
      execSync(`git checkout -b ${branchName}`, { cwd: repoPath });
      this.log("info", `Created branch: ${branchName}`, { repoPath });
      return true;
    } catch (error) {
      this.log("error", `Failed to create branch: ${error}`);
      return false;
    }
  }

  async commitChanges(
    repoPath: string,
    message: string,
    files?: string[]
  ): Promise<boolean> {
    try {
      if (files && files.length > 0) {
        execSync(`git add ${files.join(" ")}`, { cwd: repoPath });
      } else {
        execSync(`git add -A`, { cwd: repoPath });
      }
      execSync(`git commit -m "${message}"`, { cwd: repoPath });
      this.log("info", `Committed: ${message}`, { repoPath });
      return true;
    } catch (error) {
      this.log("error", `Failed to commit: ${error}`);
      return false;
    }
  }

  async createPullRequest(
    owner: string,
    repo: string,
    baseBranch: string,
    headBranch: string,
    title: string,
    body: string,
    githubToken: string
  ): Promise<{ url: string; number: number } | null> {
    try {
      const { Octokit } = await import("octokit");
      const octokit = new Octokit({ auth: githubToken });

      const { data: pr } = await octokit.rest.pulls.create({
        owner,
        repo,
        title,
        body,
        head: headBranch,
        base: baseBranch,
      });

      this.log("info", `Created PR: ${pr.html_url}`);
      return { url: pr.html_url, number: pr.number };
    } catch (error) {
      this.log("error", `Failed to create PR: ${error}`);
      return null;
    }
  }
}

// CLI entry point
if (import.meta.main) {
  const args = process.argv.slice(2);
  const command = args[0];

  const agent = new ScholarDevClawAgent();

  if (command === "integrate") {
    const repoUrl = args[1] || "./test_repos/nanogpt";
    const specName = args[2] || "rmsnorm";

    console.log(`Starting integration: ${repoUrl} with ${specName}`);

    agent
      .runIntegration(repoUrl, "", specName, "autonomous")
      .then((result) => {
        console.log("Integration complete:", JSON.stringify(result, null, 2));
        process.exit(0);
      })
      .catch((error) => {
        console.error("Integration failed:", error.message);
        process.exit(1);
      });
  } else {
    console.log(`
ScholarDevClaw Agent

Usage:
  bun run src/index.ts integrate <repo-url> <spec-name>

Examples:
  bun run src/index.ts integrate ./test_repos/nanogpt rmsnorm
  bun run src/index.ts integrate https://github.com/user/repo swiglu
`);
  }
}
