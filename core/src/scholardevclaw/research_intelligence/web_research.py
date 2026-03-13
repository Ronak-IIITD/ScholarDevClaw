"""
Web research module for searching blogs, GitHub, and other sources.

This module searches for relevant implementation examples and research
from multiple sources including arXiv, GitHub, technical blogs, and Papers with Code.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scholardevclaw.llm.research_assistant import LLMResearchAssistant

logger = logging.getLogger(__name__)

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from bs4 import BeautifulSoup

    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


@dataclass
class WebResource:
    """Represents a web resource (blog post, GitHub repo, etc.)"""

    title: str
    url: str
    source: str  # github, blog, papers_with_code, stackoverflow, etc.
    description: str = ""
    author: str = ""
    date: str = ""
    relevance_score: float = 0.0
    code_snippets: list[str] = field(default_factory=list)
    language: str = ""


@dataclass
class GitHubRepo:
    """Represents a GitHub repository"""

    name: str
    owner: str
    url: str
    description: str = ""
    stars: int = 0
    language: str = ""
    topics: list[str] = field(default_factory=list)
    relevance_score: float = 0.0


@dataclass
class BlogPost:
    """Represents a technical blog post"""

    title: str
    url: str
    author: str = ""
    site: str = ""
    description: str = ""
    date: str = ""
    relevance_score: float = 0.0


class WebResearchEngine:
    """Searches for implementations and research across the web.

    Parameters
    ----------
    llm_assistant : LLMResearchAssistant | None
        Optional LLM assistant for enriching analysis of search results
        and GitHub repositories.
    """

    def __init__(
        self,
        llm_assistant: LLMResearchAssistant | None = None,
    ) -> None:
        self.client: Any = None
        if HAS_HTTPX:
            self.client = httpx.AsyncClient(timeout=30.0)  # type: ignore[possibly-undefined]  # noqa: F821
        self.github_token: str | None = None
        self._llm = llm_assistant

    def set_github_token(self, token: str) -> None:
        """Set GitHub API token for authenticated requests"""
        self.github_token = token

    async def search_all_sources(
        self, query: str, language: str = "python", max_results: int = 10
    ) -> dict[str, list[Any]]:
        """Search across all available sources"""
        results: dict[str, list[Any]] = {
            "github_repos": [],
            "blog_posts": [],
            "stackoverflow": [],
            "papers_with_code": [],
        }

        # Search GitHub
        if self.client:
            results["github_repos"] = await self.search_github(query, language, max_results)

        # Search Papers with Code
        results["papers_with_code"] = await self.search_papers_with_code(query, max_results)

        # Search blogs (requires search engine API)
        # results["blog_posts"] = await self.search_blogs(query, max_results)

        return results

    async def search_github(
        self, query: str, language: str = "python", max_results: int = 10
    ) -> list[GitHubRepo]:
        """Search GitHub for relevant repositories"""
        if not self.client:
            return []

        try:
            # Build search query
            search_query = f"{query} language:{language}"

            headers: dict[str, str] = {
                "Accept": "application/vnd.github.v3+json",
            }

            if self.github_token:
                headers["Authorization"] = f"token {self.github_token}"

            url = "https://api.github.com/search/repositories"
            params = {
                "q": search_query,
                "sort": "stars",
                "order": "desc",
                "per_page": max_results,
            }

            response = await self.client.get(url, params=params, headers=headers)

            if response.status_code != 200:
                logger.warning("GitHub API error: %s", response.status_code)
                return []

            data = response.json()
            repos = []

            for item in data.get("items", []):
                repo = GitHubRepo(
                    name=item["name"],
                    owner=item["owner"]["login"],
                    url=item["html_url"],
                    description=item.get("description", "") or "",
                    stars=item.get("stargazers_count", 0),
                    language=item.get("language", language) or language,
                    topics=item.get("topics", []),
                    relevance_score=self._calculate_github_relevance(item, query),
                )
                repos.append(repo)

            return sorted(repos, key=lambda x: x.relevance_score, reverse=True)

        except Exception as e:
            logger.warning("GitHub search error: %s", e)
            return []

    def _calculate_github_relevance(self, item: dict, query: str) -> float:
        """Calculate relevance score for a GitHub repo"""
        score = 0.0

        # Stars boost
        stars = item.get("stargazers_count", 0)
        if stars > 1000:
            score += 30
        elif stars > 100:
            score += 20
        elif stars > 10:
            score += 10

        # Name match boost
        name = item.get("name", "").lower()
        query_lower = query.lower()
        if query_lower in name:
            score += 25

        # Description match
        description = (item.get("description") or "").lower()
        if query_lower in description:
            score += 15

        # Topics match
        topics = item.get("topics", [])
        for topic in topics:
            if query_lower in topic.lower():
                score += 10
                break

        # Recently updated
        updated = item.get("updated_at", "")
        if updated and ("2024" in updated or "2025" in updated or "2026" in updated):
            score += 10

        return min(score, 100.0)

    async def search_papers_with_code(self, query: str, max_results: int = 10) -> list[WebResource]:
        """Search Papers with Code for implementations"""
        if not self.client:
            return []

        try:
            url = "https://paperswithcode.com/api/v1/search/"
            params = {
                "q": query,
                "items_per_page": max_results,
            }

            response = await self.client.get(url, params=params)

            if response.status_code != 200:
                return []

            data = response.json()
            resources = []

            for item in data.get("results", []):
                resource = WebResource(
                    title=item.get("title", ""),
                    url=f"https://paperswithcode.com/paper/{item.get('slug', '')}",
                    source="papers_with_code",
                    description=(item.get("abstract") or "")[:200],
                    relevance_score=50.0,  # Base score
                )
                resources.append(resource)

            return resources

        except Exception as e:
            logger.warning("Papers with Code search error: %s", e)
            return []

    async def search_stackoverflow(self, query: str, max_results: int = 5) -> list[WebResource]:
        """Search Stack Overflow for relevant discussions"""
        if not self.client:
            return []

        try:
            # Stack Overflow API requires an app key for high quota
            # Using basic search without key
            url = "https://api.stackexchange.com/2.3/search/advanced"
            params = {
                "q": query,
                "site": "stackoverflow",
                "pagesize": max_results,
                "order": "desc",
                "sort": "relevance",
            }

            response = await self.client.get(url, params=params)

            if response.status_code != 200:
                return []

            data = response.json()
            resources = []

            for item in data.get("items", []):
                resource = WebResource(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    source="stackoverflow",
                    description=f"Score: {item.get('score', 0)}, Answers: {item.get('answer_count', 0)}",
                    relevance_score=min(item.get("score", 0) * 2, 100),
                )
                resources.append(resource)

            return resources

        except Exception as e:
            logger.warning("Stack Overflow search error: %s", e)
            return []

    # ------------------------------------------------------------------
    # Implementation reference finder (was a stub)
    # ------------------------------------------------------------------

    async def find_implementation_references(
        self,
        paper_title: str,
        algorithm_name: str,
        *,
        language: str = "python",
        max_results: int = 10,
    ) -> list[WebResource]:
        """Find reference implementations for a paper/algorithm.

        Searches GitHub, Papers with Code, and Stack Overflow in parallel
        for implementations of the specified algorithm.
        """
        references: list[WebResource] = []

        search_queries = [
            f"{algorithm_name} implementation",
            f"{algorithm_name} {language}",
            f"{paper_title} code",
        ]

        for query in search_queries:
            # GitHub repos → convert to WebResource
            repos = await self.search_github(query, language, max_results=max_results // 2)
            for repo in repos:
                if not any(r.url == repo.url for r in references):
                    references.append(
                        WebResource(
                            title=f"{repo.owner}/{repo.name}",
                            url=repo.url,
                            source="github",
                            description=repo.description,
                            relevance_score=repo.relevance_score,
                            language=repo.language,
                        )
                    )

            # Papers with Code
            pwc_results = await self.search_papers_with_code(query, max_results=max_results // 2)
            for res in pwc_results:
                if not any(r.url == res.url for r in references):
                    references.append(res)

            if len(references) >= max_results:
                break

        # Sort by relevance and cap
        references.sort(key=lambda r: r.relevance_score, reverse=True)
        return references[:max_results]

    # Also provide synchronous compat name for backward compatibility
    def find_implementation_references_sync(
        self,
        paper_title: str,
        algorithm_name: str,
    ) -> list[WebResource]:
        """Synchronous backward-compatible wrapper (returns empty list).

        For real results use the async ``find_implementation_references``
        method or the ``SyncWebResearchEngine`` wrapper.
        """
        return []

    # ------------------------------------------------------------------
    # Code extraction from URLs (fixed async/sync bug)
    # ------------------------------------------------------------------

    async def extract_code_from_url(self, url: str) -> str | None:
        """Extract code from a GitHub URL or gist.

        This method is now properly async (previously called synchronous
        ``.get()`` on the async client).
        """
        if not self.client:
            return None

        if "github.com" in url and "/blob/" in url:
            # Convert blob URL to raw content URL
            raw_url = url.replace("github.com", "raw.githubusercontent.com")
            raw_url = raw_url.replace("/blob/", "/")

            try:
                response = await self.client.get(raw_url)
                if response.status_code == 200:
                    return response.text
            except Exception as exc:
                logger.debug("Failed to fetch code from %s: %s", url, exc)

        elif "gist.github.com" in url:
            # Handle gist URLs
            try:
                response = await self.client.get(url, headers={"Accept": "application/json"})
                if response.status_code == 200 and HAS_BS4:
                    soup = BeautifulSoup(response.text, "html.parser")  # type: ignore[possibly-undefined]  # noqa: F821
                    code_blocks = soup.find_all("td", class_="blob-code")
                    if code_blocks:
                        return "\n".join(block.get_text() for block in code_blocks)
            except Exception as exc:
                logger.debug("Failed to fetch gist from %s: %s", url, exc)

        return None

    # Keep the old synchronous method name for backward compatibility
    def extract_code_from_url_sync(self, url: str) -> str | None:
        """Synchronous wrapper for extract_code_from_url.

        For real results, prefer the async version.
        """
        if not HAS_HTTPX:
            return None

        if "github.com" in url and "/blob/" in url:
            raw_url = url.replace("github.com", "raw.githubusercontent.com")
            raw_url = raw_url.replace("/blob/", "/")
            try:
                with httpx.Client(timeout=15.0) as sync_client:  # type: ignore[possibly-undefined]  # noqa: F821
                    response = sync_client.get(raw_url)
                    if response.status_code == 200:
                        return response.text
            except Exception as exc:
                logger.debug("Sync fetch failed for %s: %s", url, exc)

        return None

    # ------------------------------------------------------------------
    # GitHub repo analysis (was a stub)
    # ------------------------------------------------------------------

    async def analyze_github_repo(self, repo_url: str) -> dict | None:
        """Analyse a GitHub repository for relevant code.

        Fetches the repo tree via the GitHub API and optionally uses the
        LLM assistant to analyse key files.
        """
        match = re.match(r"https://github\.com/([^/]+)/([^/]+)", repo_url)
        if not match:
            return None

        owner, repo = match.groups()
        repo = repo.rstrip("/")

        if not self.client:
            return {"owner": owner, "repo": repo, "url": repo_url, "structure": None}

        headers: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"

        # Fetch repo metadata
        repo_info: dict[str, Any] = {
            "owner": owner,
            "repo": repo,
            "url": repo_url,
            "description": "",
            "stars": 0,
            "language": "",
            "topics": [],
            "structure": [],
            "key_files": [],
        }

        try:
            meta_resp = await self.client.get(
                f"https://api.github.com/repos/{owner}/{repo}",
                headers=headers,
            )
            if meta_resp.status_code == 200:
                meta = meta_resp.json()
                repo_info["description"] = meta.get("description") or ""
                repo_info["stars"] = meta.get("stargazers_count", 0)
                repo_info["language"] = meta.get("language") or ""
                repo_info["topics"] = meta.get("topics", [])
        except Exception as exc:
            logger.debug("Failed to fetch repo metadata for %s/%s: %s", owner, repo, exc)

        # Fetch repo tree (first level)
        try:
            tree_resp = await self.client.get(
                f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD",
                headers=headers,
                params={"recursive": "1"},
            )
            if tree_resp.status_code == 200:
                tree_data = tree_resp.json()
                items = tree_data.get("tree", [])
                # Build structure list (files and directories)
                structure = []
                for item in items[:500]:  # Cap to avoid huge repos
                    structure.append(
                        {
                            "path": item.get("path", ""),
                            "type": item.get("type", ""),
                            "size": item.get("size", 0),
                        }
                    )
                repo_info["structure"] = structure

                # Identify key files (Python, model files, configs)
                key_extensions = {".py", ".yaml", ".yml", ".toml", ".json", ".cfg"}
                key_names = {"model", "train", "config", "main", "setup", "requirements"}
                key_files = []
                for item in items:
                    path = item.get("path", "")
                    name_lower = path.lower()
                    ext = "." + name_lower.rsplit(".", 1)[-1] if "." in name_lower else ""
                    base_name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()

                    if (
                        item.get("type") == "blob"
                        and ext in key_extensions
                        and (base_name in key_names or ext == ".py")
                    ):
                        key_files.append(path)

                repo_info["key_files"] = key_files[:50]
        except Exception as exc:
            logger.debug("Failed to fetch repo tree for %s/%s: %s", owner, repo, exc)

        # Optional: LLM-powered analysis of key files
        if self._llm is not None and self._llm.is_available and repo_info.get("key_files"):
            file_contents: dict[str, str] = {}
            # Fetch up to 5 key files for LLM analysis
            for file_path in repo_info["key_files"][:5]:
                try:
                    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{file_path}"
                    file_resp = await self.client.get(raw_url)
                    if file_resp.status_code == 200:
                        file_contents[file_path] = file_resp.text[:4000]
                except Exception:
                    continue

            if file_contents:
                try:
                    llm_analysis = self._llm.analyse_github_repo_content(repo_info, file_contents)
                    if llm_analysis is not None:
                        repo_info["llm_analysis"] = {
                            k: v for k, v in llm_analysis.items() if k != "raw_response"
                        }
                except Exception as exc:
                    logger.debug("LLM repo analysis failed for %s/%s: %s", owner, repo, exc)

        return repo_info

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self.client is not None:
            await self.client.aclose()


# ---------------------------------------------------------------------------
# Synchronous wrapper
# ---------------------------------------------------------------------------


class SyncWebResearchEngine:
    """Synchronous wrapper for WebResearchEngine"""

    def __init__(
        self,
        llm_assistant: LLMResearchAssistant | None = None,
    ) -> None:
        import asyncio

        self.async_engine = WebResearchEngine(llm_assistant=llm_assistant)
        self.loop = asyncio.new_event_loop()

    def search_github(
        self, query: str, language: str = "python", max_results: int = 10
    ) -> list[GitHubRepo]:
        return self.loop.run_until_complete(
            self.async_engine.search_github(query, language, max_results)
        )

    def search_papers_with_code(self, query: str, max_results: int = 10) -> list[WebResource]:
        return self.loop.run_until_complete(
            self.async_engine.search_papers_with_code(query, max_results)
        )

    def search_all(
        self, query: str, language: str = "python", max_results: int = 10
    ) -> dict[str, list[Any]]:
        return self.loop.run_until_complete(
            self.async_engine.search_all_sources(query, language, max_results)
        )

    def find_implementation_references(
        self,
        paper_title: str,
        algorithm_name: str,
        *,
        language: str = "python",
        max_results: int = 10,
    ) -> list[WebResource]:
        return self.loop.run_until_complete(
            self.async_engine.find_implementation_references(
                paper_title, algorithm_name, language=language, max_results=max_results
            )
        )

    def analyze_github_repo(self, repo_url: str) -> dict | None:
        return self.loop.run_until_complete(self.async_engine.analyze_github_repo(repo_url))

    def extract_code_from_url(self, url: str) -> str | None:
        return self.loop.run_until_complete(self.async_engine.extract_code_from_url(url))


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def find_implementations(
    algorithm_name: str, language: str = "python", max_results: int = 10
) -> dict[str, list[Any]]:
    """Find implementations of an algorithm from multiple sources"""
    engine = SyncWebResearchEngine()

    # Search for the algorithm
    results = engine.search_all(algorithm_name, language, max_results)

    # Also search with "implementation" suffix
    impl_results = engine.search_all(f"{algorithm_name} implementation", language, max_results // 2)

    # Merge results
    for key in results:
        if key in impl_results:
            existing_urls = {getattr(r, "url", None) for r in results[key]}
            for item in impl_results[key]:
                if getattr(item, "url", None) not in existing_urls:
                    results[key].append(item)

    return results
