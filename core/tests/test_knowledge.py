from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scholardevclaw.generation.module_agent import ModuleAgent
from scholardevclaw.ingestion.models import PaperDocument
from scholardevclaw.knowledge.store import KnowledgeBase
from scholardevclaw.planning.models import CodeModule, ImplementationPlan
from scholardevclaw.understanding.models import PaperUnderstanding


class _FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


class _FakeEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(self, texts: list[str]) -> list[_FakeVector]:
        vectors: list[_FakeVector] = []
        for text in texts:
            length = float(len(text))
            vectors.append(_FakeVector([length, length / 10.0, 1.0]))
        return vectors


class _FakeSettings:
    def __init__(self, anonymized_telemetry: bool = False) -> None:
        self.anonymized_telemetry = anonymized_telemetry


class _FakeCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, dict[str, object]] = {}

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, object]],
    ) -> None:
        for item_id, embedding, document, metadata in zip(ids, embeddings, documents, metadatas):
            self._items[item_id] = {
                "embedding": embedding,
                "document": document,
                "metadata": metadata,
            }

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, object] | None = None,
    ) -> dict[str, object]:
        del query_embeddings
        candidates: list[dict[str, object]] = []
        for item in self._items.values():
            metadata = item.get("metadata")
            if where and isinstance(metadata, dict):
                if not all(metadata.get(key) == value for key, value in where.items()):
                    continue
            candidates.append(item)

        docs = [str(item.get("document", "")) for item in candidates[:n_results]]
        metas = [
            item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
            for item in candidates[:n_results]
        ]
        return {"documents": [docs], "metadatas": [metas]}

    def count(self) -> int:
        return len(self._items)

    def delete(self, where: dict[str, object] | None = None) -> None:
        if where is None or not where:
            self._items.clear()
            return

        delete_ids: list[str] = []
        for item_id, item in self._items.items():
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                continue
            if all(metadata.get(key) == value for key, value in where.items()):
                delete_ids.append(item_id)
        for item_id in delete_ids:
            self._items.pop(item_id, None)


class _FakePersistentClient:
    def __init__(self, *, path: str, settings: _FakeSettings) -> None:
        self.path = path
        self.settings = settings
        self._collections: dict[str, _FakeCollection] = {}

    def get_or_create_collection(self, name: str) -> _FakeCollection:
        collection = self._collections.get(name)
        if collection is None:
            collection = _FakeCollection(name)
            self._collections[name] = collection
        return collection

    def delete_collection(self, name: str) -> None:
        self._collections.pop(name, None)


def _patch_fake_kb_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_import_module(name: str):
        if name == "chromadb":
            return SimpleNamespace(PersistentClient=_FakePersistentClient)
        if name == "chromadb.config":
            return SimpleNamespace(Settings=_FakeSettings)
        if name == "sentence_transformers":
            return SimpleNamespace(SentenceTransformer=_FakeEmbedder)
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)


def _build_doc() -> PaperDocument:
    return PaperDocument(
        title="Attention Is All You Need",
        authors=["A. Author"],
        arxiv_id="1706.03762",
        doi=None,
        year=2017,
        abstract="A transformer-based architecture.",
        sections=[],
        equations=[],
        algorithms=[],
        figures=[],
        full_text="",
        pdf_path=None,
        references=[],
        keywords=["transformer"],
        domain="nlp",
    )


def _build_understanding() -> PaperUnderstanding:
    return PaperUnderstanding(
        paper_title="Attention Is All You Need",
        one_line_summary="Transformer sequence modeling",
        key_insight="Self-attention replaces recurrence.",
        core_algorithm_description="Use multi-head attention with residuals.",
        complexity="medium",
    )


def test_knowledge_base_store_retrieve_stats_and_clear(monkeypatch, tmp_path):
    _patch_fake_kb_deps(monkeypatch)

    kb = KnowledgeBase(persist_dir=tmp_path / "kb")
    doc = _build_doc()
    understanding = _build_understanding()

    kb.store_paper(doc, understanding)
    module = CodeModule(
        id="multi_head_attention",
        name="Multi Head Attention",
        description="attention projection and split/merge heads",
        file_path="src/model/mha.py",
        depends_on=[],
        priority=1,
        estimated_lines=120,
        test_file_path="tests/test_mha.py",
        tech_stack="pytorch",
    )
    kb.store_implementation(module, "def run_attention(x):\n    return x\n", understanding)

    stats = kb.stats()
    assert stats["papers"] == 1
    assert stats["implementations"] == 1
    assert Path(stats["persist_dir"]).name == "kb"

    paper_results = kb.retrieve_similar_papers("attention", n=5)
    assert len(paper_results) == 1
    assert paper_results[0]["metadata"]["title"] == "Attention Is All You Need"

    filtered_results = kb.retrieve_similar_papers("attention", n=5, domain_filter="cv")
    assert filtered_results == []

    impl_results = kb.retrieve_similar_implementations(
        "attention module",
        "pytorch",
        n=3,
    )
    assert len(impl_results) == 1
    assert "Multi Head Attention" in impl_results[0]

    mismatched_impl_results = kb.retrieve_similar_implementations(
        "attention module",
        "jax",
        n=3,
    )
    assert mismatched_impl_results == []

    kb.clear()
    cleared_stats = kb.stats()
    assert cleared_stats["papers"] == 0
    assert cleared_stats["implementations"] == 0


def test_knowledge_base_import_error_is_helpful(monkeypatch, tmp_path):
    monkeypatch.setattr(
        importlib, "import_module", lambda _name: (_ for _ in ()).throw(ImportError)
    )

    with pytest.raises(ImportError, match=r"Install with: pip install -e '\.\[knowledge\]'"):
        KnowledgeBase(persist_dir=tmp_path / "kb")


def test_module_agent_prompt_includes_kb_snippets_when_available():
    class FakeKnowledgeBase:
        def retrieve_similar_implementations(
            self,
            module_description: str,
            tech_stack: str,
            n: int = 3,
        ) -> list[str]:
            del module_description, tech_stack, n
            return [
                "def cached_attention(x):\n    return x\n",
                "class LegacyAttention:\n    pass\n",
            ]

    module = CodeModule(
        id="attention",
        name="Attention",
        description="core attention block",
        file_path="src/attention.py",
        depends_on=[],
        priority=1,
        estimated_lines=50,
        test_file_path="tests/test_attention.py",
        tech_stack="pytorch",
    )
    plan = ImplementationPlan(
        project_name="demo",
        target_language="python",
        tech_stack="pytorch",
        modules=[module],
    )
    understanding = PaperUnderstanding(
        paper_title="Demo Paper",
        core_algorithm_description="Implement attention layers.",
    )

    agent = ModuleAgent(client=object(), model="fake", knowledge_base=FakeKnowledgeBase())
    prompt = agent._build_module_prompt(
        module=module,
        plan=plan,
        understanding=understanding,
        context_modules={},
        prior_errors=[],
    )

    assert "Similar implementations from knowledge base" in prompt
    assert "def cached_attention" in prompt
