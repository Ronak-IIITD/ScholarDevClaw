#!/usr/bin/env bun
/**
 * ScholarDevClaw Agent - OpenClaw Integration Wrapper
 * 
 * This is the main entry point for running ScholarDevClaw as an OpenClaw agent.
 * It provides full orchestration with OpenClaw's heartbeat, workspace, and state management.
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { join, resolve } from "path";
import { homedir } from "os";

interface IntegrationRequest {
  id: string;
  repoUrl: string;
  repoPath: string;
  focus?: string;
  specName?: string;
  mode: "autonomous" | "step_approval";
  status: IntegrationStatus;
  phase: number;
  createdAt: string;
  updatedAt: string;
}

type IntegrationStatus = 
  | "pending"
  | "analyzing"
  | "researching"
  | "mapping"
  | "generating"
  | "validating"
  | "reporting"
  | "awaiting_approval"
  | "completed"
  | "failed";

interface PhaseResult {
  success: boolean;
  data?: unknown;
  error?: string;
  confidence?: number;
}

class ScholarDevClawAgent {
  private workspacePath: string;
  private logPath: string;
  private pythonCorePath: string;
  private convexClient: any; // Would be typed Convex client

  constructor() {
    this.workspacePath = join(homedir(), ".scholardevclaw", "workspace");
    this.logPath = join(homedir(), ".scholardevclaw", "logs");
    this.pythonCorePath = resolve(process.cwd(), "../core/src");
    
    this.ensureDirectories();
  }

  private ensureDirectories(): void {
    if (!existsSync(this.workspacePath)) {
      mkdirSync(this.workspacePath, { recursive: true });
    }
    if (!existsSync(this.logPath)) {
      mkdirSync(this.logPath, { recursive: true });
    }
  }

  private log(level: string, message: string, data?: unknown): void {
    const timestamp = new Date().toISOString();
    const logEntry = `[${timestamp}] [${level.toUpperCase()}] ${message}`;
    
    console.log(logEntry);
    if (data) {
      console.log(JSON.stringify(data, null, 2));
    }

    // Write to log file
    const logFile = join(this.logPath, `agent-${new Date().toISOString().split("T")[0]}.log`);
    try {
      const line = logEntry + (data ? ` ${JSON.stringify(data)}` : "") + "\n";
      writeFileSync(logFile, line, { flag: "a" });
    } catch (e) {
      console.error("Failed to write log:", e);
    }
  }

  /**
   * Main entry point - called by OpenClaw heartbeat
   */
  async runHeartbeat(): Promise<void> {
    this.log("info", "Starting heartbeat check");

    // Check for pending integrations (would query Convex in production)
    const pendingRequests = await this.getPendingRequests();

    for (const request of pendingRequests) {
      await this.processIntegration(request);
    }

    this.log("info", "Heartbeat check complete", { 
      processed: pendingRequests.length 
    });
  }

  /**
   * Process a single integration request through all phases
   */
  private async processIntegration(request: IntegrationRequest): Promise<void> {
    this.log("info", `Processing integration ${request.id}`, { 
      repo: request.repoUrl,
      mode: request.mode 
    });

    try {
      // Phase 1: Analyze
      await this.updateStatus(request, "analyzing", 1);
      const analysisResult = await this.executePhase1(request);
      
      if (!analysisResult.success) {
        throw new Error(`Phase 1 failed: ${analysisResult.error}`);
      }

      if (request.mode === "step_approval") {
        await this.waitForApproval(request, 1);
      }

      // Phase 2: Research
      await this.updateStatus(request, "researching", 2);
      const researchResult = await this.executePhase2(request, analysisResult.data);
      
      if (!researchResult.success) {
        throw new Error(`Phase 2 failed: ${researchResult.error}`);
      }

      if (request.mode === "step_approval") {
        await this.waitForApproval(request, 2);
      }

      // Phase 3: Map
      await this.updateStatus(request, "mapping", 3);
      const mappingResult = await this.executePhase3(request, analysisResult.data, researchResult.data);
      
      if (!mappingResult.success) {
        throw new Error(`Phase 3 failed: ${mappingResult.error}`);
      }

      if (request.mode === "step_approval") {
        await this.waitForApproval(request, 3);
      }

      // Phase 4: Generate
      await this.updateStatus(request, "generating", 4);
      const generationResult = await this.executePhase4(request, mappingResult.data);
      
      if (!generationResult.success) {
        throw new Error(`Phase 4 failed: ${generationResult.error}`);
      }

      if (request.mode === "step_approval") {
        await this.waitForApproval(request, 4);
      }

      // Phase 5: Validate
      await this.updateStatus(request, "validating", 5);
      const validationResult = await this.executePhase5(request, generationResult.data);
      
      if (!validationResult.success) {
        // Retry up to 2 times
        if (request.phase < 2) {
          this.log("warn", "Validation failed, retrying", { attempt: request.phase + 1 });
          await this.processIntegration({ ...request, phase: request.phase + 1 });
          return;
        }
        throw new Error(`Phase 5 failed: ${validationResult.error}`);
      }

      if (request.mode === "step_approval") {
        await this.waitForApproval(request, 5);
      }

      // Phase 6: Report
      await this.updateStatus(request, "reporting", 6);
      const reportResult = await this.executePhase6(request, {
        analysis: analysisResult.data,
        research: researchResult.data,
        mapping: mappingResult.data,
        generation: generationResult.data,
        validation: validationResult.data,
      });

      // Final approval before PR
      await this.updateStatus(request, "awaiting_approval", 6);
      
      if (request.mode === "autonomous") {
        // In autonomous mode, we still require final approval
        this.log("info", "Integration complete - awaiting final approval for PR");
      }

      this.log("info", `Integration ${request.id} completed successfully`);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.log("error", `Integration ${request.id} failed`, { error: errorMessage });
      await this.updateStatus(request, "failed", request.phase, errorMessage);
    }
  }

  /**
   * Phase 1: Analyze repository structure
   */
  private async executePhase1(request: IntegrationRequest): Promise<PhaseResult> {
    this.log("info", "Phase 1: Analyzing repository", { repo: request.repoPath });

    const result = await this.runPythonScript("scholardevclaw.repo_intelligence.tree_sitter_analyzer", [
      "--repo-path", request.repoPath,
      "--output-json",
    ]);

    return result;
  }

  /**
   * Phase 2: Research relevant papers
   */
  private async executePhase2(request: IntegrationRequest, analysis: unknown): Promise<PhaseResult> {
    this.log("info", "Phase 2: Researching improvements");

    // If spec name provided, use it
    if (request.specName) {
      const result = await this.runPythonScript("scholardevclaw.research_intelligence.extractor", [
        "--spec", request.specName,
        "--output-json",
      ]);
      return result;
    }

    // Otherwise, search based on code patterns
    const focus = request.focus || "general";
    
    const result = await this.runPythonScript("scholardevclaw.research_intelligence.web_research", [
      "--query", focus,
      "--arxiv",
      "--web",
      "--output-json",
    ]);

    return result;
  }

  /**
   * Phase 3: Map research to code
   */
  private async executePhase3(
    request: IntegrationRequest, 
    analysis: unknown, 
    research: unknown
  ): Promise<PhaseResult> {
    this.log("info", "Phase 3: Mapping research to code");

    const result = await this.runPythonScript("scholardevclaw.mapping.engine", [
      "--repo-path", request.repoPath,
      "--output-json",
    ]);

    return result;
  }

  /**
   * Phase 4: Generate implementation
   */
  private async executePhase4(request: IntegrationRequest, mapping: unknown): Promise<PhaseResult> {
    this.log("info", "Phase 4: Generating implementation");

    const result = await this.runPythonScript("scholardevclaw.patch_generation.generator", [
      "--repo-path", request.repoPath,
      "--output-json",
    ]);

    return result;
  }

  /**
   * Phase 5: Validate implementation
   */
  private async executePhase5(request: IntegrationRequest, patch: unknown): Promise<PhaseResult> {
    this.log("info", "Phase 5: Validating implementation");

    const result = await this.runPythonScript("scholardevclaw.validation.runner", [
      "--repo-path", request.repoPath,
      "--output-json",
    ]);

    return result;
  }

  /**
   * Phase 6: Generate report
   */
  private async executePhase6(request: IntegrationRequest, results: unknown): Promise<PhaseResult> {
    this.log("info", "Phase 6: Generating report");

    const report = {
      integrationId: request.id,
      repoUrl: request.repoUrl,
      completedAt: new Date().toISOString(),
      phases: results,
      recommendation: this.generateRecommendation(results),
    };

    // Save report to workspace
    const reportPath = join(this.workspacePath, `report-${request.id}.json`);
    writeFileSync(reportPath, JSON.stringify(report, null, 2));

    return {
      success: true,
      data: report,
    };
  }

  /**
   * Run a Python script and capture output
   */
  private async runPythonScript(module: string, args: string[]): Promise<PhaseResult> {
    const { spawn } = await import("child_process");
    
    return new Promise((resolve) => {
      const proc = spawn("python3", ["-m", module, ...args], {
        cwd: this.pythonCorePath,
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
          this.log("error", `Python script failed`, { error: stderr });
          resolve({ success: false, error: stderr || `Exit code: ${code}` });
        }
      });

      proc.on("error", (err) => {
        this.log("error", `Failed to start Python`, { error: err.message });
        resolve({ success: false, error: err.message });
      });

      // Timeout after 5 minutes
      setTimeout(() => {
        proc.kill();
        resolve({ success: false, error: "Timeout" });
      }, 300000);
    });
  }

  /**
   * Get pending integration requests from Convex
   */
  private async getPendingRequests(): Promise<IntegrationRequest[]> {
    // In production, this would query Convex
    // For now, check workspace for request files
    
    const requests: IntegrationRequest[] = [];
    
    try {
      // Look for request files in workspace
      // This is a simplified version - real implementation would use Convex
    } catch (e) {
      this.log("error", "Failed to get pending requests", { error: e });
    }

    return requests;
  }

  /**
   * Update integration status in Convex
   */
  private async updateStatus(
    request: IntegrationRequest, 
    status: IntegrationStatus, 
    phase: number,
    error?: string
  ): Promise<void> {
    request.status = status;
    request.phase = phase;
    request.updatedAt = new Date().toISOString();

    // Save to workspace
    const requestPath = join(this.workspacePath, `request-${request.id}.json`);
    writeFileSync(requestPath, JSON.stringify(request, null, 2));

    this.log("info", `Status updated`, { id: request.id, status, phase });
  }

  /**
   * Wait for user approval (in step_approval mode)
   */
  private async waitForApproval(request: IntegrationRequest, phase: number): Promise<void> {
    this.log("info", `Waiting for approval after phase ${phase}`, { id: request.id });
    
    // In production, this would poll Convex for approval status
    // For now, simulate with a delay or manual check
    
    // Check workspace for approval file
    const approvalPath = join(this.workspacePath, `approval-${request.id}-phase${phase}.json`);
    
    // Wait for approval (with timeout)
    const maxWait = 7 * 24 * 60 * 60 * 1000; // 7 days
    const startTime = Date.now();
    
    while (Date.now() - startTime < maxWait) {
      if (existsSync(approvalPath)) {
        const approval = JSON.parse(readFileSync(approvalPath, "utf-8"));
        if (approval.approved) {
          this.log("info", "Approval received", { id: request.id, phase });
          return;
        } else {
          throw new Error("Integration rejected by user");
        }
      }
      
      // Wait 5 seconds before checking again
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
    
    throw new Error("Approval timeout");
  }

  /**
   * Generate final recommendation
   */
  private generateRecommendation(results: unknown): string {
    // Analyze results and generate recommendation
    return "approve"; // or "review" or "reject"
  }

  /**
   * Create a new integration request
   */
  async createIntegrationRequest(
    repoUrl: string,
    options: {
      focus?: string;
      specName?: string;
      mode?: "autonomous" | "step_approval";
    } = {}
  ): Promise<string> {
    const id = `int_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const request: IntegrationRequest = {
      id,
      repoUrl,
      repoPath: resolve("./test_repos/temp"), // Would be determined from repoUrl
      focus: options.focus,
      specName: options.specName,
      mode: options.mode || "step_approval",
      status: "pending",
      phase: 0,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    // Save request
    const requestPath = join(this.workspacePath, `request-${id}.json`);
    writeFileSync(requestPath, JSON.stringify(request, null, 2));

    this.log("info", "Integration request created", { id, repoUrl });

    return id;
  }
}

// Main execution
async function main() {
  const agent = new ScholarDevClawAgent();

  // Check command line arguments
  const args = process.argv.slice(2);
  const command = args[0];

  if (command === "heartbeat") {
    // Run heartbeat (called by OpenClaw)
    await agent.runHeartbeat();
  } else if (command === "create") {
    // Create new integration request
    const repoUrl = args[1];
    if (!repoUrl) {
      console.error("Usage: scholardevclaw-agent create <repo-url>");
      process.exit(1);
    }
    
    const id = await agent.createIntegrationRequest(repoUrl, {
      focus: args[2],
      mode: (args[3] as any) || "step_approval",
    });
    
    console.log(`Integration request created: ${id}`);
  } else {
    console.log(`
ScholarDevClaw Agent - OpenClaw Integration

Usage:
  bun run src/index.ts heartbeat          # Run heartbeat check
  bun run src/index.ts create <repo-url>  # Create integration request

Environment Variables:
  SCHOLARDEVCLAW_WORKSPACE    # Workspace directory (default: ~/.scholardevclaw/workspace)
  SCHOLARDEVCLAW_LOG_PATH     # Log directory (default: ~/.scholardevclaw/logs)
  SCHOLARDEVCLAW_CORE_PATH    # Python core path (default: ../core/src)
`);
  }
}

// Run if called directly
if (import.meta.main) {
  main().catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
  });
}

export { ScholarDevClawAgent };
