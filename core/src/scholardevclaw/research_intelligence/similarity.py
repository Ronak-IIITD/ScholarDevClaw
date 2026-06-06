"""
Research similarity search to find related papers.

Uses multiple strategies:
- Keyword overlap
- Citation/-reference overlap
- TF-IDF similarity on abstracts (vectorized with NumPy)
- Year proximity
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class SimilarPaper:
    """A paper with similarity score"""

    paper_id: str
    title: str
    year: int
    source: str
    similarity_score: float
    match_reasons: list[str]


class ResearchSimilaritySearch:
    """Find similar research papers"""

    def __init__(self):
        self._stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "been",
            "be",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "we",
            "our",
            "they",
            "their",
            "them",
            "he",
            "she",
            "him",
            "her",
            "his",
            "hers",
            "which",
            "who",
            "whom",
            "whose",
            "what",
            "where",
            "when",
            "why",
            "how",
            "not",
            "no",
            "nor",
            "yet",
            "so",
            "than",
            "too",
            "very",
            "just",
            "also",
            "only",
            "more",
            "most",
            "some",
            "any",
            "all",
            "each",
            "every",
            "both",
            "few",
            "several",
            "other",
            "such",
            "into",
            "over",
        }

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words"""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = text.split()
        return [t for t in tokens if t not in self._stop_words and len(t) > 2]

    def _compute_tf(self, tokens: list[str]) -> dict[str, float]:
        """Compute term frequency"""
        if not tokens:
            return {}
        counter = Counter(tokens)
        total = len(tokens)
        return {word: count / total for word, count in counter.items()}

    def _compute_idf(self, documents: list[list[str]]) -> dict[str, float]:
        """Compute inverse document frequency across all documents (vectorized).

        Uses NumPy for the batch IDF computation instead of per-term Python loops.
        """
        n_docs = len(documents)
        df: dict[str, int] = Counter()

        for doc in documents:
            for term in set(doc):
                df[term] += 1

        if not df:
            return {}

        terms = list(df.keys())
        doc_freqs = np.array([df[t] for t in terms], dtype=np.float64)
        idf_values = np.log(n_docs / (1.0 + doc_freqs)) + 1.0
        return dict(zip(terms, idf_values.tolist()))

    def _tfidf_similarity(
        self,
        tokens1: list[str],
        tokens2: list[str],
        idf: dict[str, float],
    ) -> float:
        """Compute TF-IDF cosine similarity using vectorized NumPy operations.

        Builds sparse-like vectors only over the union of terms present in the
        two documents, then computes cosine similarity in bulk with NumPy.
        """
        tf1 = self._compute_tf(tokens1)
        tf2 = self._compute_tf(tokens2)

        if not tf1 and not tf2:
            return 0.0

        all_terms = sorted(set(tf1) | set(tf2))
        if not all_terms:
            return 0.0

        idf_arr = np.array([idf.get(t, 1.0) for t in all_terms], dtype=np.float64)
        vec1 = np.array([tf1.get(t, 0.0) for t in all_terms], dtype=np.float64) * idf_arr
        vec2 = np.array([tf2.get(t, 0.0) for t in all_terms], dtype=np.float64) * idf_arr

        magnitude1 = float(np.linalg.norm(vec1))
        magnitude2 = float(np.linalg.norm(vec2))

        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0

        return float(np.dot(vec1, vec2) / (magnitude1 * magnitude2))

    def keyword_similarity(
        self,
        query: str,
        paper_title: str,
        paper_abstract: str = "",
    ) -> float:
        """Compute keyword overlap similarity"""
        query_tokens = set(self._tokenize(query))
        paper_tokens = set(self._tokenize(paper_title + " " + paper_abstract))

        if not query_tokens or not paper_tokens:
            return 0.0

        intersection = query_tokens & paper_tokens
        return len(intersection) / len(query_tokens)

    def find_similar(
        self,
        query: str,
        papers: list[dict],
        max_results: int = 10,
        use_tfidf: bool = True,
    ) -> list[SimilarPaper]:
        """Find papers similar to a query"""
        query_tokens = self._tokenize(query)
        paper_tokens_list = [
            self._tokenize(p.get("title", "") + " " + p.get("abstract", "")) for p in papers
        ]

        idf = self._compute_idf([query_tokens] + paper_tokens_list)

        similarities: list[tuple[int, SimilarPaper]] = []
        query_set = set(query_tokens)
        query_len = len(query_set)

        for i, paper in enumerate(papers):
            title = paper.get("title", "")
            paper_id = paper.get("paper_id", "")
            year = paper.get("year", 0)
            source = paper.get("source", "")
            paper_tokens = paper_tokens_list[i]

            reasons = []

            paper_token_set = set(paper_tokens)
            keyword_score = len(query_set & paper_token_set) / query_len if query_len else 0.0
            if keyword_score > 0.3:
                reasons.append(f"keyword_match ({keyword_score:.0%})")

            tfidf_score = 0.0
            if use_tfidf:
                tfidf_score = self._tfidf_similarity(query_tokens, paper_tokens, idf)
                if tfidf_score > 0.2:
                    reasons.append(f"tfidf ({tfidf_score:.0%})")

            year_score = 0.0
            if year > 0:
                current_year = datetime.now().year
                years_diff = abs(current_year - year)
                year_score = 1.0 / (1.0 + 0.1 * years_diff)
                if year_score > 0.7:
                    reasons.append(f"recent ({year})")

            combined_score = 0.4 * keyword_score + 0.4 * tfidf_score + 0.2 * year_score

            if combined_score > 0.1:
                similarities.append(
                    (
                        i,
                        SimilarPaper(
                            paper_id=paper_id,
                            title=title,
                            year=year,
                            source=source,
                            similarity_score=combined_score,
                            match_reasons=reasons,
                        ),
                    )
                )

        similarities.sort(key=lambda x: x[1].similarity_score, reverse=True)
        return [s[1] for s in similarities[:max_results]]

    def find_related_by_papers(
        self,
        source_papers: list[dict],
        candidate_papers: list[dict],
        max_results: int = 10,
    ) -> list[SimilarPaper]:
        """Find papers related to a set of source papers"""
        if not source_papers:
            return []

        source_keywords: set[str] = set()
        for paper in source_papers:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            tokens = self._tokenize(title + " " + abstract)
            source_keywords.update(tokens)

        similarities: list[tuple[int, SimilarPaper]] = []

        for i, paper in enumerate(candidate_papers):
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            paper_id = paper.get("paper_id", "")
            year = paper.get("year", 0)
            source = paper.get("source", "")

            paper_tokens = set(self._tokenize(title + " " + abstract))
            overlap = len(source_keywords & paper_tokens)
            score = overlap / len(source_keywords) if source_keywords else 0

            if score > 0.05:
                similarities.append(
                    (
                        i,
                        SimilarPaper(
                            paper_id=paper_id,
                            title=title,
                            year=year,
                            source=source,
                            similarity_score=score,
                            match_reasons=[f"keyword_overlap ({overlap} terms)"],
                        ),
                    )
                )

        similarities.sort(key=lambda x: x[1].similarity_score, reverse=True)
        return [s[1] for s in similarities[:max_results]]


class ResearchRecommendationEngine:
    """Recommendation engine for research papers"""

    def __init__(self):
        self.similarity = ResearchSimilaritySearch()
        self._paper_index: dict[str, dict] = {}

    def index_papers(self, papers: list[dict]):
        """Index papers for faster lookup"""
        self._paper_index = {p.get("paper_id", ""): p for p in papers if p.get("paper_id")}

    def recommend(
        self,
        read_paper_ids: list[str],
        max_results: int = 10,
    ) -> list[SimilarPaper]:
        """Recommend papers based on what user has read"""
        read_papers = [self._paper_index[pid] for pid in read_paper_ids if pid in self._paper_index]

        if not read_papers:
            return []

        candidate_ids = set(self._paper_index.keys()) - set(read_paper_ids)
        candidate_papers = [self._paper_index[pid] for pid in candidate_ids]

        return self.similarity.find_related_by_papers(
            read_papers,
            candidate_papers,
            max_results,
        )

    def recommend_by_query(
        self,
        query: str,
        exclude_paper_ids: list[str] | None = None,
        max_results: int = 10,
    ) -> list[SimilarPaper]:
        """Recommend papers matching a query"""
        exclude_ids = set(exclude_paper_ids or [])
        papers = [p for p in self._paper_index.values() if p.get("paper_id") not in exclude_ids]

        return self.similarity.find_similar(query, papers, max_results)
