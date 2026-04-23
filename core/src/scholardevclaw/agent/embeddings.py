"""
Lightweight embedding engine for agent memory semantic search.

Supports three backends (in preference order):
1. sentence-transformers (best quality, ~80MB model)
2. numpy TF-IDF (no extra deps, decent quality)
3. keyword overlap (always available, baseline)

The engine auto-selects the best available backend at init time.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cosine similarity (works with plain Python lists and numpy arrays)
# ---------------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0

    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)

    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)

    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0

    return float(np.dot(arr_a, arr_b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class EmbeddingBackend:
    """Base class for embedding backends."""

    name: str = "base"
    dimension: int = 0

    def encode(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def encode_single(self, text: str) -> list[float]:
        return self.encode([text])[0]


# ---------------------------------------------------------------------------
# Backend 1: sentence-transformers (highest quality)
# ---------------------------------------------------------------------------


class SentenceTransformerBackend(EmbeddingBackend):
    """Uses sentence-transformers for high-quality embeddings."""

    name = "sentence-transformers"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Path | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self._cache_dir = cache_dir or Path.home() / ".scholardevclaw" / "models"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = SentenceTransformer(
            model_name,
            cache_folder=str(self._cache_dir),
        )
        self.dimension = self.model.get_sentence_embedding_dimension() or 384

    def encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [emb.tolist() for emb in embeddings]


# ---------------------------------------------------------------------------
# Backend 2: TF-IDF with numpy (no extra dependencies)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
        "am",
        "not",
        "no",
        "nor",
        "so",
        "if",
        "or",
        "and",
        "but",
        "for",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "only",
        "own",
        "same",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "he",
        "she",
        "they",
        "we",
        "you",
        "i",
        "me",
        "my",
        "our",
        "your",
        "his",
        "her",
        "their",
        "what",
        "which",
        "who",
        "whom",
    }
)

_TOKENIZE_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Simple lowercase tokenization with stop-word removal."""
    tokens = _TOKENIZE_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


class TFIDFBackend(EmbeddingBackend):
    """
    TF-IDF embedding backend using numpy.

    Builds a vocabulary from all texts seen so far and produces
    TF-IDF weighted vectors. The vocabulary grows incrementally.
    """

    name = "tfidf-numpy"

    def __init__(self, max_features: int = 2048):
        self.max_features = max_features
        self.dimension = max_features
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._doc_count = 0
        self._doc_freq: Counter[str] = Counter()
        self._lock = threading.Lock()

    def fit(self, corpus: list[str]) -> None:
        """Update vocabulary and IDF from a corpus of texts."""
        with self._lock:
            for text in corpus:
                tokens = set(_tokenize(text))
                self._doc_count += 1
                for token in tokens:
                    self._doc_freq[token] += 1

            # Rebuild vocab from most-common terms
            most_common = self._doc_freq.most_common(self.max_features)
            self._vocab = {term: idx for idx, (term, _) in enumerate(most_common)}

            # Recompute IDF
            self._idf = {}
            for term, idx in self._vocab.items():
                df = self._doc_freq.get(term, 0)
                self._idf[term] = math.log((1 + self._doc_count) / (1 + df)) + 1

    def encode(self, texts: list[str]) -> list[list[float]]:
        # Auto-fit on new texts
        self.fit(texts)

        results = []
        for text in texts:
            tokens = _tokenize(text)
            tf = Counter(tokens)
            total = len(tokens) or 1

            vec = [0.0] * self.max_features
            for token, count in tf.items():
                if token in self._vocab:
                    idx = self._vocab[token]
                    idf = self._idf.get(token, 1.0)
                    vec[idx] = (count / total) * idf

            # L2 normalize
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 1e-9:
                vec = [v / norm for v in vec]

            results.append(vec)

        return results


# ---------------------------------------------------------------------------
# Backend 3: Keyword overlap (always available, baseline)
# ---------------------------------------------------------------------------


class KeywordBackend(EmbeddingBackend):
    """
    Fallback: returns a bag-of-words hash vector.

    Uses feature hashing (hashing trick) so no vocabulary is needed.
    Quality is lower but it's always available with zero dependencies.
    """

    name = "keyword-hash"

    def __init__(self, dimension: int = 512):
        self.dimension = dimension

    def encode(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            tokens = _tokenize(text)
            vec = [0.0] * self.dimension

            for token in tokens:
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                idx = h % self.dimension
                sign = 1.0 if (h // self.dimension) % 2 == 0 else -1.0
                vec[idx] += sign

            # L2 normalize
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 1e-9:
                vec = [v / norm for v in vec]

            results.append(vec)

        return results


# ---------------------------------------------------------------------------
# EmbeddingEngine: auto-selects the best available backend
# ---------------------------------------------------------------------------


class EmbeddingEngine:
    """
    Unified embedding engine that auto-selects the best available backend.

    Usage:
        engine = EmbeddingEngine()
        vectors = engine.encode(["hello world", "research paper"])
        similarity = engine.similarity("query", "document")
    """

    def __init__(
        self,
        preferred_backend: str | None = None,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Path | None = None,
    ):
        self.backend = self._init_backend(preferred_backend, model_name, cache_dir)
        self.dimension = self.backend.dimension
        logger.info(
            "EmbeddingEngine initialized with backend=%s dim=%d",
            self.backend.name,
            self.dimension,
        )

    def _init_backend(
        self,
        preferred: str | None,
        model_name: str,
        cache_dir: Path | None,
    ) -> EmbeddingBackend:
        """Try backends in order of quality, return the first that works."""
        if preferred == "sentence-transformers" or preferred is None:
            try:
                return SentenceTransformerBackend(model_name, cache_dir)
            except Exception as exc:
                logger.debug("sentence-transformers unavailable: %s", exc)

        if preferred == "tfidf" or preferred is None:
            try:
                return TFIDFBackend()
            except Exception as exc:
                logger.debug("tfidf backend failed: %s", exc)

        # Always-available fallback
        return KeywordBackend()

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of texts into embedding vectors."""
        return self.backend.encode(texts)

    def encode_single(self, text: str) -> list[float]:
        """Encode a single text into an embedding vector."""
        return self.backend.encode_single(text)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        vecs = self.encode([text_a, text_b])
        return cosine_similarity(vecs[0], vecs[1])

    def find_most_similar(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Find the most similar candidates to a query.

        Returns list of (index, similarity_score) tuples, sorted by score desc.
        """
        if not candidates:
            return []

        all_texts = [query] + candidates
        all_vecs = self.encode(all_texts)
        query_vec = all_vecs[0]

        scores: list[tuple[int, float]] = []
        for i, cand_vec in enumerate(all_vecs[1:]):
            score = cosine_similarity(query_vec, cand_vec)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def batch_similarity_matrix(self, texts: list[str]) -> list[list[float]]:
        """Compute pairwise similarity matrix for a list of texts."""
        vecs = self.encode(texts)
        n = len(vecs)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 1.0
            for j in range(i + 1, n):
                sim = cosine_similarity(vecs[i], vecs[j])
                matrix[i][j] = sim
                matrix[j][i] = sim
        return matrix


# Convenience singleton (lazy-init)
_engine: EmbeddingEngine | None = None
_engine_lock = threading.Lock()


def get_embedding_engine(**kwargs: Any) -> EmbeddingEngine:
    """Get or create the global embedding engine singleton."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = EmbeddingEngine(**kwargs)
    return _engine
