"""
Web research module for searching blogs, GitHub, and other sources

This module searches for relevant implementation examples and research
from multiple sources including arXiv, GitHub, technical blogs, and Papers with Code.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import re
import json


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
    code_snippets: List[str] = field(default_factory=list)
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
    topics: List[str] = field(default_factory=list)
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
    """Searches for implementations and research across the web"""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0) if HAS_HTTPX else None
        self.github_token = None

    def set_github_token(self, token: str):
        """Set GitHub API token for authenticated requests"""
        self.github_token = token

    async def search_all_sources(
        self, query: str, language: str = "python", max_results: int = 10
    ) -> Dict[str, List[Any]]:
        """Search across all available sources"""
        results = {
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
    ) -> List[GitHubRepo]:
        """Search GitHub for relevant repositories"""
        if not self.client:
            return []

        try:
            # Build search query
            search_query = f"{query} language:{language}"

            headers = {
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
                print(f"GitHub API error: {response.status_code}")
                return []

            data = response.json()
            repos = []

            for item in data.get("items", []):
                repo = GitHubRepo(
                    name=item["name"],
                    owner=item["owner"]["login"],
                    url=item["html_url"],
                    description=item.get("description", ""),
                    stars=item.get("stargazers_count", 0),
                    language=item.get("language", language),
                    topics=item.get("topics", []),
                    relevance_score=self._calculate_github_relevance(item, query),
                )
                repos.append(repo)

            return sorted(repos, key=lambda x: x.relevance_score, reverse=True)

        except Exception as e:
            print(f"GitHub search error: {e}")
            return []

    def _calculate_github_relevance(self, item: Dict, query: str) -> float:
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
        description = item.get("description", "").lower()
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
        if updated and "2024" in updated or "2023" in updated:
            score += 10

        return min(score, 100.0)

    async def search_papers_with_code(self, query: str, max_results: int = 10) -> List[WebResource]:
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
                    description=item.get("abstract", "")[:200],
                    relevance_score=50.0,  # Base score
                )
                resources.append(resource)

            return resources

        except Exception as e:
            print(f"Papers with Code search error: {e}")
            return []

    async def search_stackoverflow(self, query: str, max_results: int = 5) -> List[WebResource]:
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
            print(f"Stack Overflow search error: {e}")
            return []

    def find_implementation_references(
        self, paper_title: str, algorithm_name: str
    ) -> List[WebResource]:
        """Find reference implementations for a paper/algorithm"""
        references = []

        # Search for known implementation patterns
        search_terms = [
            f"{algorithm_name} implementation",
            f"{algorithm_name} github",
            f"{paper_title} code",
        ]

        # This would make async calls to search_github and other sources
        # For now, return structure for later implementation

        return references

    def extract_code_from_url(self, url: str) -> Optional[str]:
        """Extract code from a GitHub URL or gist"""
        if "github.com" in url and "/blob/" in url:
            # Convert blob URL to raw
            raw_url = url.replace("github.com", "raw.githubusercontent.com")
            raw_url = raw_url.replace("/blob/", "/")

            try:
                if not self.client:
                    return None

                response = self.client.get(raw_url)
                if response.status_code == 200:
                    return response.text
            except:
                pass

        return None

    def analyze_github_repo(self, repo_url: str) -> Optional[Dict]:
        """Analyze a GitHub repository for relevant code"""
        # Extract owner and repo name
        match = re.match(r"https://github.com/([^/]+)/([^/]+)", repo_url)
        if not match:
            return None

        owner, repo = match.groups()

        # This would fetch repo structure and analyze
        # Return basic info for now
        return {
            "owner": owner,
            "repo": repo,
            "url": repo_url,
            "structure": None,  # Would be populated with tree API call
        }


# Synchronous wrapper for convenience
class SyncWebResearchEngine:
    """Synchronous wrapper for WebResearchEngine"""

    def __init__(self):
        import asyncio

        self.async_engine = WebResearchEngine()
        self.loop = asyncio.new_event_loop()

    def search_github(
        self, query: str, language: str = "python", max_results: int = 10
    ) -> List[GitHubRepo]:
        import asyncio

        return self.loop.run_until_complete(
            self.async_engine.search_github(query, language, max_results)
        )

    def search_papers_with_code(self, query: str, max_results: int = 10) -> List[WebResource]:
        import asyncio

        return self.loop.run_until_complete(
            self.async_engine.search_papers_with_code(query, max_results)
        )

    def search_all(
        self, query: str, language: str = "python", max_results: int = 10
    ) -> Dict[str, List[Any]]:
        import asyncio

        return self.loop.run_until_complete(
            self.async_engine.search_all_sources(query, language, max_results)
        )


# Convenience function
def find_implementations(
    algorithm_name: str, language: str = "python", max_results: int = 10
) -> Dict[str, List[Any]]:
    """Find implementations of an algorithm from multiple sources"""
    engine = SyncWebResearchEngine()

    # Search for the algorithm
    results = engine.search_all(algorithm_name, language, max_results)

    # Also search with "implementation" suffix
    impl_results = engine.search_all(f"{algorithm_name} implementation", language, max_results // 2)

    # Merge results
    for key in results:
        if key in impl_results:
            existing_urls = {r.url for r in results[key]}
            for item in impl_results[key]:
                if item.url not in existing_urls:
                    results[key].append(item)

    return results
