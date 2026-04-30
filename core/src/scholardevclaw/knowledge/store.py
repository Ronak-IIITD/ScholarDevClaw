from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scholardevclaw.ingestion.models import PaperDocument
    from scholardevclaw.planning.models import CodeModule
    from scholardevclaw.understanding.models import PaperUnderstanding

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


class KnowledgeBase:
    def __init__(
        self,
        persist_dir: Path = Path.home() / ".scholardevclaw" / "kb",
        *,
        user_id: str | None = None,
    ) -> None:
        self.persist_dir = persist_dir.expanduser().resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        # SECURITY: user_id enables per-user collection namespacing in shared deployments
        # When set, collections are prefixed (e.g., papers_user123 instead of papers)
        self._user_id = user_id or "default"

        chromadb_module, settings_cls, embedder_cls = self._load_optional_dependencies()

        self.client = chromadb_module.PersistentClient(
            path=str(self.persist_dir),
            settings=settings_cls(anonymized_telemetry=False),
        )
        self.embedder = embedder_cls(EMBED_MODEL)

        # SECURITY: User-isolated collection names to prevent cross-user data access
        self.papers = self.client.get_or_create_collection(f"papers_{self._user_id}")
        self.implementations = self.client.get_or_create_collection(
            f"implementations_{self._user_id}"
        )
        self.patterns = self.client.get_or_create_collection(f"patterns_{self._user_id}")

    def store_paper(self, doc: PaperDocument, understanding: PaperUnderstanding) -> None:
        text = (
            f"{doc.title}. "
            f"{doc.abstract}. "
            f"{understanding.one_line_summary}. "
            f"{understanding.key_insight}"
        ).strip()
        embedding = self._embed(text)

        identifier = (
            (doc.arxiv_id or "").strip()
            or (doc.doi or "").strip()
            or self._stable_id(f"paper::{doc.title}")
        )

        metadata = {
            "title": doc.title,
            "domain": doc.domain,
            "complexity": understanding.complexity,
            "arxiv_id": doc.arxiv_id or "",
            "doi": doc.doi or "",
            "year": str(doc.year) if doc.year is not None else "",
        }

        self.papers.upsert(
            ids=[f"paper::{identifier}"],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )

    def store_implementation(
        self,
        module: CodeModule,
        code: str,
        understanding: PaperUnderstanding,
    ) -> None:
        code_preview = code.strip()[:1200]
        text = f"{module.name}: {module.description}\n{code_preview}".strip()
        embedding = self._embed(text)

        paper_id = self._stable_id(understanding.paper_title)
        module_id = module.id.strip() or self._stable_id(module.file_path)

        metadata = {
            "module_type": module.id,
            "tech_stack": module.tech_stack,
            "paper": understanding.paper_title,
            "file_path": module.file_path,
        }

        self.implementations.upsert(
            ids=[f"impl::{paper_id}::{module_id}"],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )

    def retrieve_similar_papers(
        self,
        query: str,
        n: int = 5,
        domain_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        embedding = self._embed(query)
        where = {"domain": domain_filter} if domain_filter else None
        results = self.papers.query(
            query_embeddings=[embedding],
            n_results=max(1, n),
            where=where,
        )

        documents = self._normalize_query_row(results.get("documents"))
        metadatas = self._normalize_query_row(results.get("metadatas"))

        items: list[dict[str, Any]] = []
        for index, document in enumerate(documents):
            metadata = (
                metadatas[index]
                if index < len(metadatas) and isinstance(metadatas[index], dict)
                else {}
            )
            items.append({"text": str(document), "metadata": metadata})
        return items

    def retrieve_similar_implementations(
        self,
        module_description: str,
        tech_stack: str,
        n: int = 3,
    ) -> list[str]:
        embedding = self._embed(module_description)
        results = self.implementations.query(
            query_embeddings=[embedding],
            n_results=max(1, n),
            where={"tech_stack": tech_stack},
        )
        documents = self._normalize_query_row(results.get("documents"))
        return [str(document) for document in documents if isinstance(document, str)]

    def stats(self) -> dict[str, Any]:
        return {
            "persist_dir": str(self.persist_dir),
            "papers": self._collection_count(self.papers),
            "implementations": self._collection_count(self.implementations),
            "patterns": self._collection_count(self.patterns),
        }

    def clear(self) -> None:
        delete_collection = getattr(self.client, "delete_collection", None)
        # SECURITY: Use user-isolated collection names
        collection_names = (
            f"papers_{self._user_id}",
            f"implementations_{self._user_id}",
            f"patterns_{self._user_id}",
        )

        if callable(delete_collection):
            for name in collection_names:
                try:
                    delete_collection(name)
                except Exception:
                    collection = getattr(self, name, None)
                    self._clear_collection(collection)
        else:
            for name in collection_names:
                collection = getattr(self, name, None)
                self._clear_collection(collection)

        # SECURITY: Re-create with user-isolated names
        self.papers = self.client.get_or_create_collection(f"papers_{self._user_id}")
        self.implementations = self.client.get_or_create_collection(
            f"implementations_{self._user_id}"
        )
        self.patterns = self.client.get_or_create_collection(f"patterns_{self._user_id}")

    def _load_optional_dependencies(self) -> tuple[Any, Any, Any]:
        chromadb_module: Any | None = None
        chromadb_config: Any | None = None
        sentence_transformers_module: Any | None = None
        missing: list[str] = []

        try:
            chromadb_module = importlib.import_module("chromadb")
            chromadb_config = importlib.import_module("chromadb.config")
        except ImportError:
            missing.append("chromadb>=0.5.0")

        try:
            sentence_transformers_module = importlib.import_module("sentence_transformers")
        except ImportError:
            missing.append("sentence-transformers>=3.0.0")

        if missing:
            unique_missing = ", ".join(sorted(set(missing)))
            raise ImportError(
                "Knowledge base dependencies are not installed. "
                f"Missing: {unique_missing}. "
                "Install with: pip install -e '.[knowledge]'"
            )

        settings_cls = getattr(chromadb_config, "Settings", None)
        embedder_cls = getattr(sentence_transformers_module, "SentenceTransformer", None)
        if settings_cls is None or embedder_cls is None:
            raise ImportError(
                "Knowledge base dependencies are incomplete. "
                "Install/upgrade with: pip install -e '.[knowledge]'"
            )

        return chromadb_module, settings_cls, embedder_cls

    def _embed(self, text: str) -> list[float]:
        encoded = self.embedder.encode([text])[0]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        return [float(value) for value in encoded]

    def _collection_count(self, collection: Any) -> int:
        count_fn = getattr(collection, "count", None)
        if not callable(count_fn):
            return 0
        try:
            count_value = count_fn()
        except Exception:
            return 0
        try:
            return int(count_value)
        except (TypeError, ValueError):
            return 0

    def _clear_collection(self, collection: Any) -> None:
        if collection is None:
            return
        delete_fn = getattr(collection, "delete", None)
        if not callable(delete_fn):
            return
        try:
            delete_fn(where={})
        except TypeError:
            delete_fn()

    def _normalize_query_row(self, value: Any) -> list[Any]:
        if not isinstance(value, list):
            return []
        if not value:
            return []
        first = value[0]
        if isinstance(first, list):
            return first
        return value

    def _stable_id(self, raw: str) -> str:
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
