import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// Use vi.hoisted to ensure these are initialized before the vi.mock factory runs
const { mockPullsCreate, mockGetBranch } = vi.hoisted(() => ({
  mockPullsCreate: vi.fn(),
  mockGetBranch: vi.fn(),
}));

// Note: Using a plain function instead of vi.fn() for Octokit because
// vi.clearAllMocks() in beforeEach resets vi.fn() mock state, breaking
// subsequent calls in the same test suite.
vi.mock('octokit', () => ({
  Octokit: function () {
    return {
      rest: {
        pulls: { create: mockPullsCreate },
        repos: { getBranch: mockGetBranch },
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
        data: { html_url: 'https://github.com/owner/repo/pull/42', number: 42 },
      });

      const result = await client.createPullRequest({
        owner: 'owner',
        repo: 'repo',
        baseBranch: 'main',
        headBranch: 'integration/feature',
        title: 'My PR',
        body: 'Description',
      });

      expect(result).toEqual({ url: 'https://github.com/owner/repo/pull/42', number: 42 });
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
