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
}

export interface PRResult {
  url: string;
  number: number;
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
      });

      logger.info('Created pull request', { url: pr.html_url, number: pr.number });
      
      return {
        url: pr.html_url,
        number: pr.number,
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      logger.error(`Failed to create PR: ${message}`);
      return null;
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
}
