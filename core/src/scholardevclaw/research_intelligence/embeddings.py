from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def _spec_identifier(spec: dict[str, Any], ordinal: int) -> str:
    paper = spec.get("paper", {})
    arxiv_id = str(paper.get("arxiv", "") or "").strip()
    if arxiv_id:
        return arxiv_id.replace("/", "_")
    title = str(paper.get("title", "") or "").strip()
    if title:
        digest = hashlib.sha256(title.encode("utf-8")).hexdigest()[:16]
        return f"title-{digest}"
    return f"spec-{ordinal}"


def _spec_text(spec: dict[str, Any]) -> str:
    paper = spec.get("paper", {})
    algorithm = spec.get("algorithm", {})
    changes = spec.get("changes", {})
    implementation = spec.get("implementation", {})
    parts = [
        str(paper.get("title", "") or ""),
        str(paper.get("abstract", "") or ""),
        str(algorithm.get("name", "") or ""),
        str(algorithm.get("category", "") or ""),
        str(algorithm.get("replaces", "") or ""),
        str(algorithm.get("description", "") or ""),
        str(algorithm.get("formula", "") or ""),
        " ".join(str(item) for item in changes.get("target_patterns", []) or []),
        str(changes.get("replacement", "") or ""),
        " ".join(str(item) for item in changes.get("expected_benefits", []) or []),
        " ".join(str(item) for item in implementation.get("parameters", []) or []),
    ]
    return " ".join(part for part in parts if part).strip()


class EmbeddingIndex:
    """Semantic index for paper specs with disk-backed embedding cache."""

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        cache_dir: Path | None = None,
        *,
        faiss_threshold: int = 1000,
        dimension: int = 768,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir or Path.home() / ".scholardevclaw" / "embedding_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_threshold = faiss_threshold
        self.dimension = dimension
        self._model: Any | None = None
        self._backend = "hash"
        self._records: list[dict[str, Any]] = []
        self._vectors = np.zeros((0, self.dimension), dtype=np.float32)
        self._faiss_index: Any | None = None

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name, cache_folder=str(self.cache_dir / "models"))
            self.dimension = self._model.get_sentence_embedding_dimension() or dimension
            self._backend = "sentence-transformers"
        except Exception as exc:
            logger.info("Semantic search model unavailable, falling back to hash embeddings: %s", exc)

    @property
    def backend(self) -> str:
        return self._backend

    def encode(self, text: str) -> np.ndarray:
        if self._model is not None:
            vector = self._model.encode(
                [text],
                show_progress_bar=False,
                normalize_embeddings=True,
            )[0]
            return np.asarray(vector, dtype=np.float32)

        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = _tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            vector[index] += 1.0

        norm = np.linalg.norm(vector)
        if norm > 1e-9:
            vector /= norm
        return vector

    def _cache_paths(self, spec_id: str) -> tuple[Path, Path]:
        return (
            self.cache_dir / f"{spec_id}.npy",
            self.cache_dir / f"{spec_id}.json",
        )

    def _load_cached_vector(self, spec_id: str, text_hash: str) -> np.ndarray | None:
        vector_path, meta_path = self._cache_paths(spec_id)
        if not vector_path.exists() or not meta_path.exists():
            return None

        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("text_hash") != text_hash or meta.get("dimension") != self.dimension:
                return None
            vector = np.load(vector_path)
            if vector.shape == (self.dimension,):
                return np.asarray(vector, dtype=np.float32)
        except Exception as exc:
            logger.debug("Failed to load cached embedding for %s: %s", spec_id, exc)
        return None

    def _save_cached_vector(self, spec_id: str, text_hash: str, vector: np.ndarray) -> None:
        vector_path, meta_path = self._cache_paths(spec_id)
        np.save(vector_path, vector)
        meta_path.write_text(
            json.dumps(
                {
                    "text_hash": text_hash,
                    "dimension": self.dimension,
                    "backend": self._backend,
                    "model_name": self.model_name,
                },
                indent=2,
            )
        )

    def index(self, papers: list[dict[str, Any]]) -> None:
        records: list[dict[str, Any]] = []
        vectors: list[np.ndarray] = []

        for ordinal, spec in enumerate(papers):
            text = _spec_text(spec)
            if not text:
                continue

            spec_id = _spec_identifier(spec, ordinal)
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            vector = self._load_cached_vector(spec_id, text_hash)
            if vector is None:
                vector = self.encode(text)
                self._save_cached_vector(spec_id, text_hash, vector)

            records.append(
                {
                    "id": spec_id,
                    "spec": spec,
                    "text": text,
                }
            )
            vectors.append(vector)

        self._records = records
        if vectors:
            self._vectors = np.vstack(vectors).astype(np.float32)
        else:
            self._vectors = np.zeros((0, self.dimension), dtype=np.float32)
        self._faiss_index = None

    def _ensure_faiss_index(self) -> Any | None:
        if self._faiss_index is not None:
            return self._faiss_index
        if len(self._records) < self.faiss_threshold:
            return None

        try:
            import faiss

            index = faiss.IndexFlatIP(self.dimension)
            index.add(self._vectors)
            self._faiss_index = index
            return index
        except Exception as exc:
            logger.info("FAISS unavailable, using numpy similarity search: %s", exc)
            return None

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not query.strip() or not self._records:
            return []

        query_vector = self.encode(query).astype(np.float32)
        faiss_index = self._ensure_faiss_index()

        scored: list[tuple[float, dict[str, Any]]] = []
        if faiss_index is not None:
            scores, indices = faiss_index.search(query_vector.reshape(1, -1), top_k)
            for score, index in zip(scores[0], indices[0], strict=False):
                if index < 0:
                    continue
                record = self._records[int(index)]
                if float(score) <= 0.0:
                    continue
                scored.append((float(score), record))
        else:
            similarities = self._vectors @ query_vector
            best_indices = np.argsort(similarities)[::-1][:top_k]
            for index in best_indices:
                score = float(similarities[int(index)])
                if score <= 0.0:
                    continue
                scored.append((score, self._records[int(index)]))

        results: list[dict[str, Any]] = []
        for score, record in scored:
            spec = dict(record["spec"])
            spec["_semantic_score"] = round(score, 6)
            results.append(spec)
        return results

    def similarity(self, left: str, right: str) -> float:
        return _cosine_similarity(self.encode(left), self.encode(right))
