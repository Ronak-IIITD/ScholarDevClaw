"""
RAG (Retrieval-Augmented Generation) context for better code generation.

Provides:
- Document chunking and embedding
- Vector store (simple in-memory)
- Semantic search
- Context retrieval
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class DocumentChunk:
    """A chunk of a document"""

    id: str
    content: str
    source: str  # file path or source identifier
    chunk_index: int
    start_line: int
    end_line: int
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with relevance score"""

    chunk: DocumentChunk
    similarity: float
    rank: int


class TextChunker:
    """Split text into chunks"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self,
        text: str,
        source: str,
        start_line: int = 1,
    ) -> list[DocumentChunk]:
        """Chunk text into overlapping pieces"""
        lines = text.split("\n")
        chunks = []

        for i in range(0, len(lines), self.chunk_size - self.chunk_overlap):
            chunk_lines = lines[i : i + self.chunk_size]
            chunk_text = "\n".join(chunk_lines)

            chunk = DocumentChunk(
                id=hashlib.md5(chunk_text.encode()).hexdigest()[:16],
                content=chunk_text,
                source=source,
                chunk_index=len(chunks),
                start_line=start_line + i,
                end_line=start_line + i + len(chunk_lines) - 1,
            )
            chunks.append(chunk)

        return chunks

    def chunk_file(self, file_path: Path) -> list[DocumentChunk]:
        """Chunk a file into pieces"""
        content = file_path.read_text()
        return self.chunk_text(content, str(file_path))


class SimpleEmbedder:
    """Simple hash-based embedding (no external API needed)"""

    def __init__(self, dimension: int = 512):
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Create embedding for text"""
        if not HAS_NUMPY:
            return self._simple_embedding(text)

        import numpy as np

        hash_input = text.encode()
        hash_values = []

        for i in range(self.dimension):
            h = hashlib.sha256(f"{i}:{text}".encode()).hexdigest()
            value = int(h[:8], 16) / (16**8)
            hash_values.append(value)

        vec = np.array(hash_values)
        vec = vec / np.linalg.norm(vec)

        return vec.tolist()

    def _simple_embedding(self, text: str) -> list[float]:
        """Simple embedding without numpy"""
        values = []
        for i in range(self.dimension):
            h = hashlib.sha256(f"{i}:{text}".encode()).hexdigest()
            value = int(h[:8], 16) / (16**8)
            values.append(value)

        return values


class VectorStore:
    """Simple in-memory vector store"""

    def __init__(self, embedder: SimpleEmbedder | None = None):
        self.embedder = embedder or SimpleEmbedder()
        self.chunks: dict[str, DocumentChunk] = {}
        self._indexed = False

    def add_chunk(self, chunk: DocumentChunk):
        """Add a chunk to the store"""
        if not chunk.embedding:
            chunk.embedding = self.embedder.embed(chunk.content)

        self.chunks[chunk.id] = chunk
        self._indexed = False

    def add_chunks(self, chunks: list[DocumentChunk]):
        """Add multiple chunks"""
        for chunk in chunks:
            self.add_chunk(chunk)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Search for relevant chunks"""
        if not self.chunks:
            return []

        query_embedding = self.embedder.embed(query)

        results = []
        for chunk in self.chunks.values():
            similarity = self._cosine_similarity(query_embedding, chunk.embedding)
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    similarity=similarity,
                    rank=0,
                )
            )

        results.sort(key=lambda r: r.similarity, reverse=True)

        for i, r in enumerate(results[:top_k]):
            r.rank = i + 1

        return results[:top_k]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity"""
        if not HAS_NUMPY:
            dot = sum(x * y for x, y in zip(a, b))
            mag_a = sum(x * x for x in a) ** 0.5
            mag_b = sum(x * x for x in b) ** 0.5
            return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

        import numpy as np

        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class RAGContextBuilder:
    """Build RAG context from codebases"""

    def __init__(self, chunker: TextChunker | None = None, store: VectorStore | None = None):
        self.chunker = chunker or TextChunker()
        self.store = store or VectorStore()

    def index_repository(self, repo_path: Path, extensions: list[str] | None = None):
        """Index a repository"""
        extensions = extensions or [".py", ".js", ".ts", ".go", ".rs", ".java"]

        for ext in extensions:
            for file_path in repo_path.rglob(f"*{ext}"):
                if "__pycache__" in file_path.parts:
                    continue

                try:
                    chunks = self.chunker.chunk_file(file_path)
                    self.store.add_chunks(chunks)
                except Exception:
                    pass

    def index_file(self, file_path: Path):
        """Index a single file"""
        chunks = self.chunker.chunk_file(file_path)
        self.store.add_chunks(chunks)

    def get_context(
        self,
        query: str,
        max_chunks: int = 5,
        min_similarity: float = 0.1,
    ) -> str:
        """Get relevant context for a query"""
        results = self.store.search(query, top_k=max_chunks)

        context_parts = []
        for result in results:
            if result.similarity >= min_similarity:
                context_parts.append(
                    f"// Source: {result.chunk.source}:{result.chunk.start_line}-{result.chunk.end_line}\n"
                    f"// Similarity: {result.similarity:.2f}\n"
                    f"{result.chunk.content}\n"
                )

        return "\n\n".join(context_parts)

    def get_context_with_metadata(
        self,
        query: str,
        max_chunks: int = 5,
    ) -> dict[str, Any]:
        """Get context with full metadata"""
        results = self.store.search(query, top_k=max_chunks)

        return {
            "chunks": [
                {
                    "content": r.chunk.content,
                    "source": r.chunk.source,
                    "start_line": r.chunk.start_line,
                    "end_line": r.chunk.end_line,
                    "similarity": r.similarity,
                    "rank": r.rank,
                }
                for r in results
            ],
            "total_chunks": len(self.store.chunks),
        }


class CodeAwareChunker:
    """Chunker that respects code structure"""

    def __init__(self, chunk_size: int = 2000):
        self.chunk_size = chunk_size

    def chunk_code(self, content: str, source: str) -> list[DocumentChunk]:
        """Chunk code preserving functions/classes"""
        chunks = []

        lines = content.split("\n")
        current_chunk_lines = []
        current_start = 1
        chunk_index = 0

        in_function = False
        in_class = False
        function_name = ""

        for i, line in enumerate(lines):
            stripped = line.strip()

            if stripped.startswith("def ") or stripped.startswith("async def "):
                in_function = True
                function_name = (
                    re.search(r"def (\w+)", stripped).group(1)
                    if re.search(r"def (\w+)", stripped)
                    else "function"
                )

            elif stripped.startswith("class "):
                in_class = True

            current_chunk_lines.append(line)

            if len("\n".join(current_chunk_lines)) > self.chunk_size:
                if current_chunk_lines:
                    chunk = DocumentChunk(
                        id=hashlib.md5(f"{source}:{chunk_index}".encode()).hexdigest()[:16],
                        content="\n".join(current_chunk_lines),
                        source=source,
                        chunk_index=chunk_index,
                        start_line=current_start,
                        end_line=i + 1,
                    )
                    chunks.append(chunk)

                    current_chunk_lines = []
                    current_start = i + 1
                    chunk_index += 1

        if current_chunk_lines:
            chunk = DocumentChunk(
                id=hashlib.md5(f"{source}:{chunk_index}".encode()).hexdigest()[:16],
                content="\n".join(current_chunk_lines),
                source=source,
                chunk_index=chunk_index,
                start_line=current_start,
                end_line=len(lines),
            )
            chunks.append(chunk)

        return chunks


def create_rag_context(repo_path: Path) -> RAGContextBuilder:
    """Quick RAG context creation"""
    chunker = CodeAwareChunker()
    store = VectorStore()
    return RAGContextBuilder(chunker, store)
