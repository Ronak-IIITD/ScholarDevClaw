import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// Use vi.hoisted to ensure these are initialized before the vi.mock factory runs
const {
  mockPullsCreate,
  mockGetBranch,
  mockPullsUpdate,
  mockPullsGet,
  mockPullsList,
  mockChecksCreate,
  mockChecksUpdate,
  mockListWorkflowRunsForRepo,
  mockListWorkflowRuns,
  mockIssuesCreateComment,
  mockIssuesListComments,
  mockPullsMerge,
} = vi.hoisted(() => ({
  mockPullsCreate: vi.fn(),
  mockGetBranch: vi.fn(),
  mockPullsUpdate: vi.fn(),
  mockPullsGet: vi.fn(),
  mockPullsList: vi.fn(),
  mockChecksCreate: vi.fn(),
  mockChecksUpdate: vi.fn(),
  mockListWorkflowRunsForRepo: vi.fn(),
  mockListWorkflowRuns: vi.fn(),
  mockIssuesCreateComment: vi.fn(),
  mockIssuesListComments: vi.fn(),
  mockPullsMerge: vi.fn(),
}));

// Note: Using a plain function instead of vi.fn() for Octokit because
// vi.clearAllMocks() in beforeEach resets vi.fn() mock state, breaking
// subsequent calls in the same test suite.
vi.mock('octokit', () => ({
  Octokit: function () {
    return {
      rest: {
        pulls: {
          create: mockPullsCreate,
          update: mockPullsUpdate,
          get: mockPullsGet,
          list: mockPullsList,
          merge: mockPullsMerge,
        },
        repos: { getBranch: mockGetBranch },
        checks: { create: mockChecksCreate, update: mockChecksUpdate },
        actions: {
          listWorkflowRunsForRepo: mockListWorkflowRunsForRepo,
          listWorkflowRuns: mockListWorkflowRuns,
        },
        issues: {
          createComment: mockIssuesCreateComment,
          listComments: mockIssuesListComments,
        },
      },
    };
  },
}));

import { GitHubClient } from './github.js';

describe('GitHubClient', () => {
  let client: GitHubClient;

  beforeEach(() => {
    vi.clearAllMocks();
    client = new GitHubClient('mock-token');
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
  });

  describe('constructor', () => {
    it('creates client successfully', () => {
      expect(client).toBeInstanceOf(GitHubClient);
    });

    it('creates client without token when none provided', () => {
      vi.stubEnv('GITHUB_TOKEN', '');
      const noTokenClient = new GitHubClient();
      expect(noTokenClient).toBeInstanceOf(GitHubClient);
    });
  });

  describe('createPullRequest', () => {
    it('creates a PR and returns URL and number', async () => {
      mockPullsCreate.mockResolvedValue({
        data: { html_url: 'https://github.com/owner/repo/pull/42', number: 42, node_id: 'PR_node' },
      });

      const result = await client.createPullRequest({
        owner: 'owner',
        repo: 'repo',
        baseBranch: 'main',
        headBranch: 'integration/feature',
        title: 'My PR',
        body: 'Description',
      });

      expect(result).toEqual({
        url: 'https://github.com/owner/repo/pull/42',
        number: 42,
        id: 'PR_node',
      });
    });

    it('returns null when PR creation fails', async () => {
      mockPullsCreate.mockRejectedValue(new Error('API error'));

      const result = await client.createPullRequest({
        owner: 'owner',
        repo: 'repo',
        baseBranch: 'main',
        headBranch: 'integration/feature',
        title: 'My PR',
        body: 'Description',
      });

      expect(result).toBeNull();
    });

    it('returns null when octokit is not initialized', async () => {
      vi.stubEnv('GITHUB_TOKEN', '');
      const noTokenClient = new GitHubClient();

      const result = await noTokenClient.createPullRequest({
        owner: 'owner',
        repo: 'repo',
        baseBranch: 'main',
        headBranch: 'integration/feature',
        title: 'My PR',
        body: 'Description',
      });

      expect(result).toBeNull();
    });
  });

  describe('updatePullRequest', () => {
    it('updates a PR and returns updated data', async () => {
      mockPullsUpdate.mockResolvedValue({
        data: { html_url: 'https://github.com/owner/repo/pull/42', number: 42, node_id: 'updated' },
      });

      const result = await client.updatePullRequest({
        owner: 'owner',
        repo: 'repo',
        pullNumber: 42,
        title: 'New title',
        body: 'New body',
      });

      expect(result).toEqual({
        url: 'https://github.com/owner/repo/pull/42',
        number: 42,
        id: 'updated',
      });
    });

    it('returns null when update fails', async () => {
      mockPullsUpdate.mockRejectedValue(new Error('forbidden'));

      const result = await client.updatePullRequest({
        owner: 'owner',
        repo: 'repo',
        pullNumber: 42,
      });

      expect(result).toBeNull();
    });

    it('returns null when not configured', async () => {
      vi.stubEnv('GITHUB_TOKEN', '');
      const noTokenClient = new GitHubClient();

      const result = await noTokenClient.updatePullRequest({
        owner: 'owner',
        repo: 'repo',
        pullNumber: 42,
      });

      expect(result).toBeNull();
    });
  });

  describe('getPullRequest', () => {
    it('returns PR data when found', async () => {
      const prData = { id: 1, number: 42, title: 'Test PR', state: 'open' };
      mockPullsGet.mockResolvedValue({ data: prData });

      const result = await client.getPullRequest('owner', 'repo', 42);
      expect(result).toEqual(prData);
    });

    it('returns null when not found', async () => {
      mockPullsGet.mockRejectedValue(new Error('Not Found'));
      const result = await client.getPullRequest('owner', 'repo', 9999);
      expect(result).toBeNull();
    });

    it('returns null when not configured', async () => {
      vi.stubEnv('GITHUB_TOKEN', '');
      const noTokenClient = new GitHubClient();
      const result = await noTokenClient.getPullRequest('owner', 'repo', 1);
      expect(result).toBeNull();
    });
  });

  describe('listPullRequests', () => {
    it('returns a list of PRs', async () => {
      const prs = [{ number: 1 }, { number: 2 }];
      mockPullsList.mockResolvedValue({ data: prs });

      const result = await client.listPullRequests('owner', 'repo', 'open');
      expect(result).toEqual(prs);
    });

    it('defaults to open state', async () => {
      mockPullsList.mockResolvedValue({ data: [] });
      await client.listPullRequests('owner', 'repo');
      expect(mockPullsList).toHaveBeenCalledWith(
        expect.objectContaining({ state: 'open' })
      );
    });

    it('returns empty array on failure', async () => {
      mockPullsList.mockRejectedValue(new Error('API error'));
      const result = await client.listPullRequests('owner', 'repo');
      expect(result).toEqual([]);
    });
  });

  describe('createCheckRun', () => {
    it('creates a check run and returns the ID', async () => {
      mockChecksCreate.mockResolvedValue({ data: { id: 12345 } });

      const result = await client.createCheckRun({
        owner: 'owner',
        repo: 'repo',
        headSha: 'abc123',
        name: 'scholardevclaw/validate',
        status: 'in_progress',
      });

      expect(result).toBe('12345');
    });

    it('returns null on failure', async () => {
      mockChecksCreate.mockRejectedValue(new Error('forbidden'));
      const result = await client.createCheckRun({
        owner: 'owner',
        repo: 'repo',
        headSha: 'abc123',
        name: 'check',
        status: 'queued',
      });
      expect(result).toBeNull();
    });

    it('returns null when not configured', async () => {
      vi.stubEnv('GITHUB_TOKEN', '');
      const noTokenClient = new GitHubClient();
      const result = await noTokenClient.createCheckRun({
        owner: 'o',
        repo: 'r',
        headSha: 's',
        name: 'n',
        status: 'queued',
      });
      expect(result).toBeNull();
    });
  });

  describe('updateCheckRun', () => {
    it('returns true on success', async () => {
      mockChecksUpdate.mockResolvedValue({ data: {} });
      const result = await client.updateCheckRun('owner', 'repo', '12345', 'completed', 'success');
      expect(result).toBe(true);
    });

    it('returns false on failure', async () => {
      mockChecksUpdate.mockRejectedValue(new Error('expired'));
      const result = await client.updateCheckRun('owner', 'repo', '12345', 'completed', 'failure');
      expect(result).toBe(false);
    });

    it('returns false when not configured', async () => {
      vi.stubEnv('GITHUB_TOKEN', '');
      const noTokenClient = new GitHubClient();
      const result = await noTokenClient.updateCheckRun('o', 'r', '1', 'completed', 'success');
      expect(result).toBe(false);
    });
  });

  describe('getWorkflowRuns', () => {
    it('returns mapped workflow runs', async () => {
      mockListWorkflowRunsForRepo.mockResolvedValue({
        data: {
          workflow_runs: [
            {
              id: 1,
              name: 'CI',
              status: 'completed',
              conclusion: 'success',
              html_url: 'https://github.com/owner/repo/actions/runs/1',
              created_at: '2026-01-01T00:00:00Z',
              updated_at: '2026-01-01T00:01:00Z',
            },
            {
              id: 2,
              name: null,
              status: 'in_progress',
              conclusion: null,
              html_url: 'https://github.com/owner/repo/actions/runs/2',
              created_at: '2026-01-01T00:00:00Z',
              updated_at: '2026-01-01T00:01:00Z',
            },
          ],
        },
      });

      const result = await client.getWorkflowRuns('owner', 'repo', 'main');
      expect(result).toHaveLength(2);
      expect(result[0]).toEqual({
        id: 1,
        name: 'CI',
        status: 'completed',
        conclusion: 'success',
        htmlUrl: 'https://github.com/owner/repo/actions/runs/1',
        createdAt: '2026-01-01T00:00:00Z',
        updatedAt: '2026-01-01T00:01:00Z',
      });
      expect(result[1].name).toBe(''); // null name defaults to empty string
    });

    it('returns empty array on failure', async () => {
      mockListWorkflowRunsForRepo.mockRejectedValue(new Error('api'));
      const result = await client.getWorkflowRuns('owner', 'repo');
      expect(result).toEqual([]);
    });

    it('returns empty array when not configured', async () => {
      vi.stubEnv('GITHUB_TOKEN', '');
      const noTokenClient = new GitHubClient();
      const result = await noTokenClient.getWorkflowRuns('o', 'r');
      expect(result).toEqual([]);
    });
  });

  describe('getLatestWorkflowRun', () => {
    it('returns the most recent run for a workflow', async () => {
      mockListWorkflowRuns.mockResolvedValue({
        data: {
          workflow_runs: [
            {
              id: 99,
              name: 'CI',
              status: 'completed',
              conclusion: 'success',
              html_url: 'https://github.com/x',
              created_at: '2026-01-01',
              updated_at: '2026-01-01',
            },
          ],
        },
      });

      const result = await client.getLatestWorkflowRun('owner', 'repo', 'ci.yml');
      expect(result?.id).toBe(99);
    });

    it('returns null when no runs exist', async () => {
      mockListWorkflowRuns.mockResolvedValue({ data: { workflow_runs: [] } });
      const result = await client.getLatestWorkflowRun('owner', 'repo', 'ci.yml');
      expect(result).toBeNull();
    });

    it('returns null on failure', async () => {
      mockListWorkflowRuns.mockRejectedValue(new Error('api'));
      const result = await client.getLatestWorkflowRun('owner', 'repo', 'ci.yml');
      expect(result).toBeNull();
    });

    it('returns null when not configured', async () => {
      vi.stubEnv('GITHUB_TOKEN', '');
      const noTokenClient = new GitHubClient();
      const result = await noTokenClient.getLatestWorkflowRun('o', 'r', 'ci.yml');
      expect(result).toBeNull();
    });
  });

  describe('addPRComment', () => {
    it('returns true on success', async () => {
      mockIssuesCreateComment.mockResolvedValue({ data: { id: 1 } });
      const result = await client.addPRComment('owner', 'repo', 1, 'hello');
      expect(result).toBe(true);
    });

    it('returns false on failure', async () => {
      mockIssuesCreateComment.mockRejectedValue(new Error('forbidden'));
      const result = await client.addPRComment('owner', 'repo', 1, 'hello');
      expect(result).toBe(false);
    });

    it('returns false when not configured', async () => {
      vi.stubEnv('GITHUB_TOKEN', '');
      const noTokenClient = new GitHubClient();
      const result = await noTokenClient.addPRComment('o', 'r', 1, 'hi');
      expect(result).toBe(false);
    });
  });

  describe('getPRComments', () => {
    it('returns list of comments', async () => {
      const comments = [{ id: 1, body: 'a' }];
      mockIssuesListComments.mockResolvedValue({ data: comments });
      const result = await client.getPRComments('owner', 'repo', 1);
      expect(result).toEqual(comments);
    });

    it('returns empty array on failure', async () => {
      mockIssuesListComments.mockRejectedValue(new Error('api'));
      const result = await client.getPRComments('owner', 'repo', 1);
      expect(result).toEqual([]);
    });
  });

  describe('mergePullRequest', () => {
    it('merges with default squash', async () => {
      mockPullsMerge.mockResolvedValue({ data: { merged: true } });
      const result = await client.mergePullRequest('owner', 'repo', 1);
      expect(result).toBe(true);
      expect(mockPullsMerge).toHaveBeenCalledWith(
        expect.objectContaining({ merge_method: 'squash' })
      );
    });

    it('supports custom merge method', async () => {
      mockPullsMerge.mockResolvedValue({ data: { merged: true } });
      await client.mergePullRequest('owner', 'repo', 1, 'rebase');
      expect(mockPullsMerge).toHaveBeenCalledWith(
        expect.objectContaining({ merge_method: 'rebase' })
      );
    });

    it('returns false on failure', async () => {
      mockPullsMerge.mockRejectedValue(new Error('conflict'));
      const result = await client.mergePullRequest('owner', 'repo', 1);
      expect(result).toBe(false);
    });

    it('returns false when not configured', async () => {
      vi.stubEnv('GITHUB_TOKEN', '');
      const noTokenClient = new GitHubClient();
      const result = await noTokenClient.mergePullRequest('o', 'r', 1);
      expect(result).toBe(false);
    });
  });

  describe('getBranchSha', () => {
    it('returns commit SHA for existing branch', async () => {
      mockGetBranch.mockResolvedValue({
        data: { commit: { sha: 'abc123' } },
      });

      const sha = await client.getBranchSha('owner', 'repo', 'main');
      expect(sha).toBe('abc123');
    });

    it('returns null when branch fetch fails', async () => {
      mockGetBranch.mockRejectedValue(new Error('not found'));

      const sha = await client.getBranchSha('owner', 'repo', 'nonexistent');
      expect(sha).toBeNull();
    });
  });

  describe('branchExists', () => {
    it('returns true when branch exists', async () => {
      mockGetBranch.mockResolvedValue({ data: {} });

      const exists = await client.branchExists('owner', 'repo', 'main');
      expect(exists).toBe(true);
    });

    it('returns false when branch does not exist', async () => {
      mockGetBranch.mockRejectedValue(new Error('Branch not found'));

      const exists = await client.branchExists('owner', 'repo', 'nonexistent');
      expect(exists).toBe(false);
    });
  });

  describe('isConfigured', () => {
    it('returns true when token is provided', () => {
      expect(client.isConfigured()).toBe(true);
    });

    it('returns false when no token available', () => {
      vi.stubEnv('GITHUB_TOKEN', '');
      const noTokenClient = new GitHubClient();
      expect(noTokenClient.isConfigured()).toBe(false);
    });
  });

  describe('parseRepoUrl', () => {
    it('parses github.com/owner/repo URL', () => {
      const result = client.parseRepoUrl('https://github.com/my-org/my-repo');
      expect(result).toEqual({ owner: 'my-org', repo: 'my-repo' });
    });

    it('parses github.com:owner/repo SSH URL', () => {
      const result = client.parseRepoUrl('git@github.com:my-org/my-repo.git');
      expect(result).toEqual({ owner: 'my-org', repo: 'my-repo' });
    });

    it('parses URL with .git suffix', () => {
      const result = client.parseRepoUrl('https://github.com/owner/repo.git');
      expect(result).toEqual({ owner: 'owner', repo: 'repo' });
    });

    it('returns null for non-github URLs', () => {
      const result = client.parseRepoUrl('https://gitlab.com/owner/repo');
      expect(result).toBeNull();
    });

    it('returns null for invalid URLs', () => {
      const result = client.parseRepoUrl('not-a-url');
      expect(result).toBeNull();
    });

    it('returns null for empty string', () => {
      const result = client.parseRepoUrl('');
      expect(result).toBeNull();
    });
  });
});
