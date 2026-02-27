"""
Citation graph analysis for understanding paper relationships.

Tracks:
- Paper citations (what papers reference this paper)
- Paper references (what papers this paper references)
- Citation chains and paths
- Influence metrics
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CitationNode:
    """A node in the citation graph"""

    paper_id: str
    title: str
    year: int
    citations: set[str] = field(default_factory=set)
    references: set[str] = field(default_factory=set)


@dataclass
class CitationPath:
    """A path between two papers through citations"""

    source_id: str
    target_id: str
    path: list[str]
    length: int


class CitationGraph:
    """Graph of paper citations"""

    def __init__(self):
        self.nodes: dict[str, CitationNode] = {}
        self._in_degree: dict[str, int] = defaultdict(int)
        self._out_degree: dict[str, int] = defaultdict(int)

    def add_paper(
        self,
        paper_id: str,
        title: str,
        year: int,
        citations: list[str] | None = None,
        references: list[str] | None = None,
    ) -> CitationNode:
        """Add a paper to the graph"""
        if paper_id in self.nodes:
            node = self.nodes[paper_id]
            node.title = title
            node.year = year
        else:
            node = CitationNode(
                paper_id=paper_id,
                title=title,
                year=year,
            )
            self.nodes[paper_id] = node

        if citations:
            node.citations = set(citations)
            for cited_id in citations:
                self._in_degree[cited_id] += 1
                self._out_degree[paper_id] += 1
                if cited_id not in self.nodes:
                    self.nodes[cited_id] = CitationNode(
                        paper_id=cited_id,
                        title="",
                        year=0,
                    )

        if references:
            node.references = set(references)
            for ref_id in references:
                self._out_degree[ref_id] += 1
                self._in_degree[paper_id] += 1
                if ref_id not in self.nodes:
                    self.nodes[ref_id] = CitationNode(
                        paper_id=ref_id,
                        title="",
                        year=0,
                    )

        return node

    def get_citations(self, paper_id: str) -> set[str]:
        """Get papers that cite this paper"""
        if paper_id not in self.nodes:
            return set()
        return self.nodes[paper_id].citations

    def get_references(self, paper_id: str) -> set[str]:
        """Get papers that this paper references"""
        if paper_id not in self.nodes:
            return set()
        return self.nodes[paper_id].references

    def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> CitationPath | None:
        """Find shortest citation path between two papers"""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None

        if source_id == target_id:
            return CitationPath(source_id, target_id, [source_id], 0)

        from collections import deque

        queue = deque([(source_id, [source_id])])
        visited = {source_id}

        while queue:
            current, path = queue.popleft()

            if len(path) > max_depth:
                continue

            for neighbor in self.nodes[current].references:
                if neighbor == target_id:
                    return CitationPath(source_id, target_id, path + [neighbor], len(path))

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def find_all_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 4,
    ) -> list[CitationPath]:
        """Find all citation paths between two papers (up to max_depth)"""
        if source_id not in self.nodes or target_id not in self.nodes:
            return []

        all_paths = []

        def dfs(current: str, path: list[str]):
            if len(path) > max_depth:
                return
            if current == target_id and len(path) > 1:
                all_paths.append(
                    CitationPath(
                        source_id,
                        target_id,
                        path,
                        len(path) - 1,
                    )
                )
                return

            for neighbor in self.nodes[current].references:
                if neighbor not in path:
                    dfs(neighbor, path + [neighbor])

        dfs(source_id, [source_id])
        return sorted(all_paths, key=lambda p: p.length)

    def get_common_ancestors(self, paper_id1: str, paper_id2: str) -> set[str]:
        """Find common ancestor papers (papers that cite both)"""
        if paper_id1 not in self.nodes or paper_id2 not in self.nodes:
            return set()

        citations1 = self.get_citations(paper_id1)
        citations2 = self.get_citations(paper_id2)
        return citations1 & citations2

    def get_common_descendants(self, paper_id1: str, paper_id2: str) -> set[str]:
        """Find common descendant papers (papers referenced by both)"""
        if paper_id1 not in self.nodes or paper_id2 not in self.nodes:
            return set()

        refs1 = self.get_references(paper_id1)
        refs2 = self.get_references(paper_id2)
        return refs1 & refs2

    def get_pagerank(self, damping: float = 0.85, iterations: int = 100) -> dict[str, float]:
        """Calculate PageRank for papers"""
        if not self.nodes:
            return {}

        n = len(self.nodes)
        ranks = {node_id: 1.0 / n for node_id in self.nodes}

        for _ in range(iterations):
            new_ranks = {}
            for node_id in self.nodes:
                rank_sum = 0.0
                for other_id in self.nodes:
                    if node_id in self.nodes[other_id].references:
                        out_degree = len(self.nodes[other_id].references)
                        if out_degree > 0:
                            rank_sum += ranks[other_id] / out_degree
                new_ranks[node_id] = (1 - damping) / n + damping * rank_sum

            ranks = new_ranks

        return ranks

    def get_influence_score(self, paper_id: str) -> float:
        """Calculate influence score based on citations and PageRank"""
        if paper_id not in self.nodes:
            return 0.0

        node = self.nodes[paper_id]
        citations = len(node.citations)
        refs = len(node.references)

        if citations == 0 and refs == 0:
            return 0.0

        year_weight = 1.0
        if node.year > 0:
            current_year = 2026
            years_old = current_year - node.year
            year_weight = 1.0 / (1.0 + 0.1 * years_old)

        score = math.sqrt(citations * refs + 1) * year_weight
        return score

    def get_related_papers(
        self,
        paper_id: str,
        max_results: int = 10,
    ) -> list[tuple[str, float]]:
        """Find related papers based on citation overlap"""
        if paper_id not in self.nodes:
            return []

        node = self.nodes[paper_id]
        related_scores: dict[str, float] = defaultdict(float)

        for other_id, other_node in self.nodes.items():
            if other_id == paper_id:
                continue

            common_citations = len(node.citations & other_node.citations)
            common_refs = len(node.references & other_node.references)

            if common_citations > 0 or common_refs > 0:
                similarity = common_citations + common_refs
                related_scores[other_id] = similarity

        sorted_papers = sorted(
            related_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_papers[:max_results]

    def to_dict(self) -> dict:
        """Export graph to dictionary"""
        return {
            "nodes": {
                pid: {
                    "title": node.title,
                    "year": node.year,
                    "citations": list(node.citations),
                    "references": list(node.references),
                }
                for pid, node in self.nodes.items()
            }
        }

    def save(self, path: Path):
        """Save graph to file"""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> CitationGraph:
        """Load graph from file"""
        with open(path) as f:
            data = json.load(f)

        graph = cls()
        for pid, node_data in data.get("nodes", {}).items():
            graph.add_paper(
                paper_id=pid,
                title=node_data.get("title", ""),
                year=node_data.get("year", 0),
                citations=node_data.get("citations", []),
                references=node_data.get("references", []),
            )
        return graph


class CitationAnalyzer:
    """Analyze citation patterns and relationships"""

    def __init__(self, graph: CitationGraph):
        self.graph = graph
        self._pagerank_cache: dict[str, float] | None = None

    def analyze_influence(self, paper_id: str) -> dict:
        """Get detailed influence analysis for a paper"""
        if paper_id not in self.graph.nodes:
            return {}

        node = self.graph.nodes[paper_id]
        in_degree = len(node.citations)
        out_degree = len(node.references)

        if self._pagerank_cache is None:
            self._pagerank_cache = self.graph.get_pagerank()

        pagerank = self._pagerank_cache.get(paper_id, 0)
        influence = self.graph.get_influence_score(paper_id)

        related = self.graph.get_related_papers(paper_id, 5)

        return {
            "paper_id": paper_id,
            "title": node.title,
            "year": node.year,
            "citations_count": in_degree,
            "references_count": out_degree,
            "pagerank": pagerank,
            "influence_score": influence,
            "related_papers": [{"paper_id": pid, "score": score} for pid, score in related],
        }

    def compare_papers(self, paper_id1: str, paper_id2: str) -> dict:
        """Compare two papers"""
        if paper_id1 not in self.graph.nodes or paper_id2 not in self.graph.nodes:
            return {}

        node1 = self.graph.nodes[paper_id1]
        node2 = self.graph.nodes[paper_id2]

        common_ancestors = self.graph.get_common_ancestors(paper_id1, paper_id2)
        common_descendants = self.graph.get_common_descendants(paper_id1, paper_id2)

        path = self.graph.find_shortest_path(paper_id1, paper_id2)

        return {
            "paper1": {
                "paper_id": paper_id1,
                "title": node1.title,
                "year": node1.year,
                "citations": len(node1.citations),
                "references": len(node1.references),
            },
            "paper2": {
                "paper_id": paper_id2,
                "title": node2.title,
                "year": node2.year,
                "citations": len(node2.citations),
                "references": len(node2.references),
            },
            "common_ancestors": list(common_ancestors),
            "common_descendants": list(common_descendants),
            "shortest_path": path.path if path else None,
            "path_length": path.length if path else -1,
        }

    def get_citation_trends(self, min_year: int = 2000) -> dict:
        """Get citation trends over years"""
        yearly_citations: dict[int, int] = defaultdict(int)
        yearly_papers: dict[int, int] = defaultdict(int)

        for node in self.graph.nodes.values():
            if node.year >= min_year:
                yearly_papers[node.year] += 1
                yearly_citations[node.year] += len(node.citations)

        return {
            "papers_per_year": dict(yearly_papers),
            "citations_per_year": dict(yearly_citations),
        }
