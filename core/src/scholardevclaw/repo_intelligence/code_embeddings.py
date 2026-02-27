"""
Code embeddings for semantic similarity.

Provides:
- Local TF-IDF embeddings for code
- Optional sentence-transformers integration
- Fast hash-based similarity
- Code-aware tokenization
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class CodeEmbedding:
    """Embedding for a code element"""

    element_id: str
    name: str
    file: str
    embedding: list[float]
    token_count: int = 0


@dataclass
class SimilarCodeElement:
    """A code element with similarity score"""

    element_id: str
    name: str
    file: str
    similarity: float
    match_type: str  # name, semantic, structural


class CodeTokenizer:
    """Tokenizer for code that preserves semantic meaning"""

    KEYWORDS = {
        "python": {
            "def",
            "class",
            "return",
            "if",
            "else",
            "elif",
            "for",
            "while",
            "try",
            "except",
            "finally",
            "with",
            "import",
            "from",
            "as",
            "pass",
            "break",
            "continue",
            "and",
            "or",
            "not",
            "in",
            "is",
            "lambda",
            "yield",
            "async",
            "await",
        },
        "javascript": {
            "function",
            "class",
            "return",
            "if",
            "else",
            "for",
            "while",
            "try",
            "catch",
            "finally",
            "const",
            "let",
            "var",
            "import",
            "export",
            "async",
            "await",
        },
        "typescript": {
            "function",
            "class",
            "return",
            "if",
            "else",
            "for",
            "while",
            "try",
            "catch",
            "finally",
            "const",
            "let",
            "var",
            "import",
            "export",
            "async",
            "await",
            "interface",
            "type",
        },
    }

    def __init__(self, language: str = "python"):
        self.language = language
        self.keywords = self.KEYWORDS.get(language, self.KEYWORDS["python"])

    def tokenize(self, code: str) -> list[str]:
        """Tokenize code into semantic units"""
        code = self._normalize(code)
        tokens = []
        current = ""

        for char in code:
            if char.isspace():
                if current:
                    tokens.extend(self._split_token(current))
                    current = ""
            elif char in "(){}[].,;:+=*/<>!&|-":
                if current:
                    tokens.extend(self._split_token(current))
                    current = ""
                tokens.append(char)
            else:
                current += char

        if current:
            tokens.extend(self._split_token(current))

        return [t for t in tokens if t and len(t) > 0]

    def _normalize(self, code: str) -> str:
        """Normalize code for tokenization"""
        code = re.sub(r"#.*", "", code)
        code = re.sub(r'""".*?"""', "", code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", "", code, flags=re.DOTALL)
        code = re.sub(r"//.*", "", code)
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
        return code

    def _split_token(self, token: str) -> list[str]:
        """Split compound tokens"""
        parts = []
        current = ""

        for char in token:
            if char.isupper() and current and current[-1].islower():
                parts.append(current)
                current = char
            elif char.isdigit() and current and not current[-1].isdigit():
                parts.append(current)
                current = char
            else:
                current += char

        if current:
            parts.append(current)

        filtered = []
        for p in parts:
            if p in self.keywords:
                filtered.append(f"KEYWORD:{p}")
            elif p.isdigit():
                filtered.append("NUMBER")
            else:
                filtered.append(p.lower())

        return filtered


class CodeEmbeddingEngine:
    """Generate embeddings for code elements"""

    def __init__(self, language: str = "python", use_tfidf: bool = True):
        self.tokenizer = CodeTokenizer(language)
        self.use_tfidf = use_tfidf
        self._idf_cache: dict[str, float] = {}
        self._corpus_tokens: list[list[str]] = []

    def index_code(self, code: str, element_id: str, name: str, file: str) -> CodeEmbedding:
        """Index a code element and return its embedding"""
        tokens = self.tokenizer.tokenize(code)
        embedding = self._compute_embedding(tokens)

        return CodeEmbedding(
            element_id=element_id,
            name=name,
            file=file,
            embedding=embedding,
            token_count=len(tokens),
        )

    def _compute_embedding(self, tokens: list[str]) -> list[float]:
        """Compute embedding from tokens"""
        if not tokens:
            return [0.0] * 512

        if self.use_tfidf and self._corpus_tokens:
            return self._tfidf_embedding(tokens)
        else:
            return self._hash_embedding(tokens)

    def _tfidf_embedding(self, tokens: list[str]) -> list[float]:
        """Compute TF-IDF embedding"""
        tf = Counter(tokens)
        dim = min(512, len(self._idf_cache) or 512)
        embedding = [0.0] * dim

        for token, freq in tf.items():
            tfidf = freq * self._idf_cache.get(token, 1.0)
            idx = hash(token) % dim
            embedding[idx] += tfidf

        return self._normalize(embedding)

    def _hash_embedding(self, tokens: list[str]) -> list[float]:
        """Compute hash-based embedding (faster, no training needed)"""
        dim = 512
        embedding = [0.0] * dim

        for i, token in enumerate(tokens):
            h = hashlib.sha256(token.encode()).hexdigest()
            for j in range(8):
                idx = int(h[j * 8 : (j + 1) * 8], 16) % dim
                weight = (len(tokens) - i) / len(tokens)
                embedding[idx] += weight

        return self._normalize(embedding)

    def _normalize(self, vec: list[float]) -> list[float]:
        """L2 normalize a vector"""
        if not HAS_NUMPY:
            magnitude = math.sqrt(sum(x * x for x in vec))
            if magnitude > 0:
                return [x / magnitude for x in vec]
            return vec

        magnitude = math.sqrt(sum(x * x for x in vec))
        if magnitude > 0:
            return [x / magnitude for x in vec]
        return vec

    def build_index(self, code_elements: list[dict]):
        """Build TF-IDF index from multiple code elements"""
        self._corpus_tokens = []
        for elem in code_elements:
            code = elem.get("code", "")
            tokens = self.tokenizer.tokenize(code)
            self._corpus_tokens.append(tokens)

        if self._corpus_tokens:
            self._idf_cache = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        """Compute IDF for all tokens"""
        n_docs = len(self._corpus_tokens)
        df = Counter()

        for tokens in self._corpus_tokens:
            unique = set(tokens)
            for token in unique:
                df[token] += 1

        idf = {}
        for token, doc_freq in df.items():
            idf[token] = math.log(n_docs / (1 + doc_freq)) + 1

        return idf


class CodeSimilarityFinder:
    """Find similar code elements using embeddings"""

    def __init__(self, embedding_engine: CodeEmbeddingEngine):
        self.engine = embedding_engine
        self._index: dict[str, CodeEmbedding] = {}

    def add_to_index(self, embedding: CodeEmbedding):
        """Add embedding to index"""
        self._index[embedding.element_id] = embedding

    def find_similar(
        self,
        query_embedding: CodeEmbedding,
        top_k: int = 10,
    ) -> list[SimilarCodeElement]:
        """Find most similar code elements"""
        similarities = []

        for elem_id, embedding in self._index.items():
            if elem_id == query_embedding.element_id:
                continue

            similarity = self._cosine_similarity(
                query_embedding.embedding,
                embedding.embedding,
            )

            match_type = "semantic"
            if query_embedding.name == embedding.name:
                match_type = "name"
            elif self._structural_match(query_embedding, embedding):
                match_type = "structural"

            similarities.append(
                SimilarCodeElement(
                    element_id=elem_id,
                    name=embedding.name,
                    file=embedding.file,
                    similarity=similarity,
                    match_type=match_type,
                )
            )

        similarities.sort(key=lambda x: x.similarity, reverse=True)
        return similarities[:top_k]

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors"""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        return dot

    def _structural_match(self, e1: CodeEmbedding, e2: CodeEmbedding) -> bool:
        """Check if embeddings have similar structure"""
        return abs(e1.token_count - e2.token_count) < 5


class SemanticCodeMapper:
    """Map code elements semantically across a codebase"""

    def __init__(self, language: str = "python"):
        self.engine = CodeEmbeddingEngine(language)
        self.finder = CodeSimilarityFinder(self.engine)
        self._elements: list[dict] = []

    def index_repository(self, root_path: Path, languages: list[str] | None = None):
        """Index all code elements in a repository"""
        languages = languages or [".py"]
        self._elements = []

        for ext in languages:
            for file_path in root_path.rglob(f"*{ext}"):
                if "__pycache__" in file_path.parts:
                    continue
                self._index_file(file_path)

        self.engine.build_index(self._elements)

        for elem in self._elements:
            embedding = self.engine.index_code(
                code=elem["code"],
                element_id=elem["id"],
                name=elem["name"],
                file=elem["file"],
            )
            self.finder.add_to_index(embedding)

    def _index_file(self, file_path: Path):
        """Index functions and classes from a file"""
        try:
            content = file_path.read_text()
        except OSError:
            return

        import ast

        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError:
            return

        relative = file_path.name

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_code = ast.get_source_segment(content, node) or ""
                self._elements.append(
                    {
                        "id": f"{relative}::{node.name}",
                        "name": node.name,
                        "file": str(relative),
                        "code": func_code,
                        "type": "function",
                    }
                )

            elif isinstance(node, ast.ClassDef):
                class_code = ast.get_source_segment(content, node) or ""
                self._elements.append(
                    {
                        "id": f"{relative}::{node.name}",
                        "name": node.name,
                        "file": str(relative),
                        "code": class_code,
                        "type": "class",
                    }
                )

    def find_similar_to(
        self, code: str, name: str = "", file: str = ""
    ) -> list[SimilarCodeElement]:
        """Find similar code elements to the given code"""
        query_embedding = self.engine.index_code(
            code=code,
            element_id="query",
            name=name or "query",
            file=file or "query",
        )

        return self.finder.find_similar(query_embedding)

    def find_duplicates(self, threshold: float = 0.9) -> list[tuple[str, str, float]]:
        """Find duplicate or very similar code elements"""
        duplicates = []
        seen = set()

        for elem_id, embedding in self.finder._index.items():
            if elem_id in seen:
                continue

            similar = self.finder.find_similar(embedding, top_k=5)

            for s in similar:
                if s.similarity >= threshold and s.element_id not in seen:
                    duplicates.append((elem_id, s.element_id, s.similarity))
                    seen.add(s.element_id)

        return duplicates


def compute_code_hash(code: str) -> str:
    """Compute a simple hash of code for deduplication"""
    normalized = re.sub(r"\s+", "", code)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]
