import { Octokit } from 'octokit';
import { logger } from '../utils/logger.js';
import { config } from '../utils/config.js';

export interface CreatePRInput {
  owner: string;
  repo: string;
  baseBranch: string;
  headBranch: string;
  title: string;
  body: string;
  draft?: boolean;
  labels?: string[];
  assignees?: string[];
  reviewers?: string[];
}

export interface PRResult {
  url: string;
  number: number;
  id: string;
}

export interface PRUpdateInput {
  owner: string;
  repo: string;
  pullNumber: number;
  title?: string;
  body?: string;
  state?: 'open' | 'closed';
  baseBranch?: string;
}

export interface CheckRunInput {
  owner: string;
  repo: string;
  headSha: string;
  name: string;
  status: 'queued' | 'in_progress' | 'completed';
  conclusion?: 'success' | 'failure' | 'neutral' | 'cancelled' | 'skipped' | 'timed_out' | 'action_required';
  output?: {
    title: string;
    summary: string;
    text?: string;
  };
}

export interface WorkflowRun {
  id: number;
  name: string;
  status: 'queued' | 'in_progress' | 'completed';
  conclusion?: 'success' | 'failure' | 'neutral' | 'cancelled' | 'skipped' | 'timed_out' | 'action_required';
  htmlUrl: string;
  createdAt: string;
  updatedAt: string;
}

export class GitHubClient {
  private octokit: Octokit | null = null;

  constructor(token?: string) {
    const githubToken = token || config.github.token;
    if (githubToken) {
      this.octokit = new Octokit({ auth: githubToken });
      logger.info('GitHub client initialized');
    } else {
      logger.warn('GitHub token not configured - PR creation disabled');
    }
  }

  /**
   * Create a new pull request with optional draft, labels, assignees, reviewers.
   */
  async createPullRequest(input: CreatePRInput): Promise<PRResult | null> {
    if (!this.octokit) {
      logger.error('GitHub client not initialized');
      return null;
    }

    try {
      const { data: pr } = await this.octokit.rest.pulls.create({
        owner: input.owner,
        repo: input.repo,
        title: input.title,
        body: input.body,
        head: input.headBranch,
        base: input.baseBranch,
        draft: input.draft ?? false,
      });

      // Add labels if provided
      if (input.labels?.length) {
        await this.octokit.rest.issues.addLabels({
          owner: input.owner,
          repo: input.repo,
          issue_number: pr.number,
          labels: input.labels,
        });
      }

      // Request reviewers if provided
      if (input.reviewers?.length) {
        await this.octokit.rest.pulls.requestReviewers({
          owner: input.owner,
          repo: input.repo,
          pull_number: pr.number,
          reviewers: input.reviewers,
        });
      }

      // Assign assignees if provided
      if (input.assignees?.length) {
        await this.octokit.rest.issues.addAssignees({
          owner: input.owner,
          repo: input.repo,
          issue_number: pr.number,
          assignees: input.assignees,
        });
      }

      logger.info('Created pull request', { url: pr.html_url, number: pr.number });

      return {
        url: pr.html_url,
        number: pr.number,
        id: pr.node_id,
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      logger.error(`Failed to create PR: ${message}`);
      return null;
    }
  }

  /**
   * Update an existing pull request.
   */
  async updatePullRequest(input: PRUpdateInput): Promise<PRResult | null> {
    if (!this.octokit) {
      logger.error('GitHub client not initialized');
      return null;
    }

    try {
      const { data: pr } = await this.octokit.rest.pulls.update({
        owner: input.owner,
        repo: input.repo,
        pull_number: input.pullNumber,
        title: input.title,
        body: input.body,
        state: input.state,
        base: input.baseBranch,
      });

      logger.info('Updated pull request', { url: pr.html_url, number: pr.number });

      return {
        url: pr.html_url,
        number: pr.number,
        id: pr.node_id,
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      logger.error(`Failed to update PR: ${message}`);
      return null;
    }
  }

  /**
   * Get pull request details.
   */
  async getPullRequest(owner: string, repo: string, pullNumber: number): Promise<any | null> {
    if (!this.octokit) return null;

    try {
      const { data: pr } = await this.octokit.rest.pulls.get({
        owner,
        repo,
        pull_number: pullNumber,
      });
      return pr;
    } catch (err) {
      logger.error(`Failed to get PR: ${err}`);
      return null;
    }
  }

  /**
   * List pull requests for a repo.
   */
  async listPullRequests(owner: string, repo: string, state: 'open' | 'closed' | 'all' = 'open'): Promise<any[]> {
    if (!this.octokit) return [];

    try {
      const { data } = await this.octokit.rest.pulls.list({
        owner,
        repo,
        state,
        per_page: 100,
      });
      return data;
    } catch (err) {
      logger.error(`Failed to list PRs: ${err}`);
      return [];
    }
  }

  /**
   * Create or update a check run (for CI integration).
   */
  async createCheckRun(input: CheckRunInput): Promise<string | null> {
    if (!this.octokit) {
      logger.error('GitHub client not initialized');
      return null;
    }

    try {
      const { data } = await this.octokit.rest.checks.create({
        owner: input.owner,
        repo: input.repo,
        name: input.name,
        head_sha: input.headSha,
        status: input.status,
        conclusion: input.conclusion,
        output: input.output,
      });

      logger.info('Created check run', { id: data.id, name: input.name });
      return data.id.toString();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      logger.error(`Failed to create check run: ${message}`);
      return null;
    }
  }

  /**
   * Update an existing check run.
   */
  async updateCheckRun(
    owner: string,
    repo: string,
    checkRunId: string,
    status: 'queued' | 'in_progress' | 'completed',
    conclusion?: 'success' | 'failure' | 'neutral' | 'cancelled' | 'skipped' | 'timed_out' | 'action_required',
    output?: { title: string; summary: string; text?: string }
  ): Promise<boolean> {
    if (!this.octokit) return false;

    try {
      await this.octokit.rest.checks.update({
        owner,
        repo,
        check_run_id: parseInt(checkRunId, 10),
        status,
        conclusion,
        output,
      });
      return true;
    } catch (err) {
      logger.error(`Failed to update check run: ${err}`);
      return false;
    }
  }

  /**
   * Get workflow runs for a repo (CI/CD status).
   */
  async getWorkflowRuns(owner: string, repo: string, branch?: string): Promise<WorkflowRun[]> {
    if (!this.octokit) return [];

    try {
      const { data } = await this.octokit.rest.actions.listWorkflowRunsForRepo({
        owner,
        repo,
        branch,
        per_page: 20,
      });

      return data.workflow_runs.map((run) => ({
        id: run.id,
        name: run.name ?? '',
        status: run.status as WorkflowRun['status'],
        conclusion: run.conclusion as WorkflowRun['conclusion'] | undefined,
        htmlUrl: run.html_url,
        createdAt: run.created_at,
        updatedAt: run.updated_at,
      }));
    } catch (err) {
      logger.error(`Failed to get workflow runs: ${err}`);
      return [];
    }
  }

  /**
   * Get the latest workflow run for a specific workflow file.
   */
  async getLatestWorkflowRun(owner: string, repo: string, workflowFile: string, branch?: string): Promise<WorkflowRun | null> {
    if (!this.octokit) return null;

    try {
      const { data } = await this.octokit.rest.actions.listWorkflowRuns({
        owner,
        repo,
        workflow_id: workflowFile,
        branch,
        per_page: 1,
      });

      if (data.workflow_runs.length === 0) return null;

      const run = data.workflow_runs[0];
      return {
        id: run.id,
        name: run.name ?? '',
        status: run.status as WorkflowRun['status'],
        conclusion: run.conclusion as WorkflowRun['conclusion'] | undefined,
        htmlUrl: run.html_url,
        createdAt: run.created_at,
        updatedAt: run.updated_at,
      };
    } catch (err) {
      logger.error(`Failed to get latest workflow run: ${err}`);
      return null;
    }
  }

  /**
   * Add a comment to a PR.
   */
  async addPRComment(owner: string, repo: string, pullNumber: number, body: string): Promise<boolean> {
    if (!this.octokit) return false;

    try {
      await this.octokit.rest.issues.createComment({
        owner,
        repo,
        issue_number: pullNumber,
        body,
      });
      return true;
    } catch (err) {
      logger.error(`Failed to add PR comment: ${err}`);
      return false;
    }
  }

  /**
   * Get PR comments.
   */
  async getPRComments(owner: string, repo: string, pullNumber: number): Promise<any[]> {
    if (!this.octokit) return [];

    try {
      const { data } = await this.octokit.rest.issues.listComments({
        owner,
        repo,
        issue_number: pullNumber,
      });
      return data;
    } catch (err) {
      logger.error(`Failed to get PR comments: ${err}`);
      return [];
    }
  }

  /**
   * Merge a pull request.
   */
  async mergePullRequest(
    owner: string,
    repo: string,
    pullNumber: number,
    mergeMethod: 'merge' | 'squash' | 'rebase' = 'squash'
  ): Promise<boolean> {
    if (!this.octokit) return false;

    try {
      await this.octokit.rest.pulls.merge({
        owner,
        repo,
        pull_number: pullNumber,
        merge_method: mergeMethod,
      });
      logger.info('Merged pull request', { owner, repo, pullNumber });
      return true;
    } catch (err) {
      logger.error(`Failed to merge PR: ${err}`);
      return false;
    }
  }

  async getBranchSha(owner: string, repo: string, branch: string): Promise<string | null> {
    if (!this.octokit) return null;

    try {
      const { data } = await this.octokit.rest.repos.getBranch({
        owner,
        repo,
        branch,
      });
      return data.commit.sha;
    } catch (err) {
      logger.error(`Failed to get branch SHA: ${err}`);
      return null;
    }
  }

  async branchExists(owner: string, repo: string, branch: string): Promise<boolean> {
    if (!this.octokit) return false;

    try {
      await this.octokit.rest.repos.getBranch({
        owner,
        repo,
        branch,
      });
      return true;
    } catch {
      return false;
    }
  }

  parseRepoUrl(url: string): { owner: string; repo: string } | null {
    const match = url.match(/github\.com[/:]([^/]+)\/([^/.]+)/);
    if (!match) return null;
    return { owner: match[1], repo: match[2] };
  }

  /**
   * Check if the client is properly configured.
   */
  isConfigured(): boolean {
    return this.octokit !== null;
  }
}
