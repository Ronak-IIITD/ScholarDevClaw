from __future__ import annotations

import json
from pathlib import Path

import pytest

from scholardevclaw.ingestion.models import Algorithm, Equation, PaperDocument, Section
from scholardevclaw.understanding.agent import UnderstandingAgent
from scholardevclaw.understanding.models import (
    ConceptEdge,
    ConceptNode,
    Contribution,
    PaperUnderstanding,
    Requirement,
)


def _sample_paper_document(tmp_path: Path) -> PaperDocument:
    return PaperDocument(
        title="Attention Is All You Need",
        authors=["A. Vaswani", "N. Shazeer"],
        arxiv_id="1706.03762",
        doi=None,
        year=2017,
        abstract="We propose the Transformer architecture based entirely on attention.",
        sections=[
            Section(title="Introduction", level=1, content="Intro text", page_start=1),
            Section(
                title="Conclusion",
                level=1,
                content=(
                    "We introduced a model based on multi-head attention and positional encoding. "
                    "It improves quality and parallelization."
                ),
                page_start=10,
            ),
        ],
        equations=[
            Equation(latex="Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V", description="core", page=4)
        ],
        algorithms=[
            Algorithm(
                name="Algorithm 1: Training Procedure",
                pseudocode="for step in range(T):\n  update(theta)",
                page=7,
                language_hint="python-like",
            )
        ],
        figures=[],
        full_text="full text",
        pdf_path=tmp_path / "paper.pdf",
        references=["[1] Bahdanau et al."],
        keywords=["transformer", "attention"],
        domain="nlp",
    )


def _sample_understanding_dict() -> dict[str, object]:
    return {
        "paper_title": "Attention Is All You Need",
        "one_line_summary": "Transformer replaces recurrence with attention",
        "problem_statement": "Sequence transduction is bottlenecked by recurrence",
        "key_insight": "Multi-head self-attention models dependencies in parallel.",
        "contributions": [
            {
                "claim": "Introduces Transformer",
                "novelty": "Removes recurrence entirely",
                "is_implementable": True,
            }
        ],
        "requirements": [
            {
                "name": "PyTorch",
                "type": "library",
                "is_optional": False,
                "notes": "Recommended implementation framework",
            }
        ],
        "concept_nodes": [
            {
                "id": "n1",
                "label": "Multi-Head Attention",
                "type": "operation",
                "description": "Core attention block",
            },
            {
                "id": "n2",
                "label": "Positional Encoding",
                "type": "operation",
                "description": "Adds sequence position information",
            },
        ],
        "concept_edges": [
            {
                "source_id": "n1",
                "target_id": "n2",
                "relation": "uses",
            }
        ],
        "core_algorithm_description": "Use stacked multi-head attention with positional encoding.",
        "input_output_spec": "Takes token ids and outputs logits over vocabulary.",
        "evaluation_protocol": "BLEU on WMT translation benchmarks.",
        "complexity": "medium",
        "estimated_impl_hours": 24,
        "confidence": 0.87,
    }


def test_understanding_models_roundtrip() -> None:
    understanding = PaperUnderstanding(
        paper_title="P",
        one_line_summary="S",
        problem_statement="Problem",
        key_insight="Insight",
        contributions=[Contribution("c", "n", True)],
        requirements=[Requirement("PyTorch", "library", False, "")],
        concept_nodes=[ConceptNode("n1", "Node", "operation", "desc")],
        concept_edges=[ConceptEdge("n1", "n1", "uses")],
        core_algorithm_description="core",
        input_output_spec="in->out",
        evaluation_protocol="eval",
        complexity="medium",
        estimated_impl_hours=5,
        confidence=0.5,
    )

    payload = understanding.to_dict()
    restored = PaperUnderstanding.from_dict(payload)

    assert restored == understanding
    assert restored.to_dict() == payload


def test_understanding_model_normalizes_out_of_range_values() -> None:
    restored = PaperUnderstanding.from_dict(
        {
            "complexity": "unsupported",
            "estimated_impl_hours": -10,
            "confidence": 2.5,
        }
    )

    assert restored.complexity == "research-only"
    assert restored.estimated_impl_hours == 0
    assert restored.confidence == 1.0


def test_parse_json_response_handles_fenced_json() -> None:
    agent = object.__new__(UnderstandingAgent)
    raw = '```json\n{"paper_title": "X", "confidence": 0.5}\n```'

    parsed = agent._parse_json_response(raw)
    assert parsed["paper_title"] == "X"


def test_parse_json_response_extracts_embedded_json() -> None:
    agent = object.__new__(UnderstandingAgent)
    raw = 'Result follows:\n{"paper_title": "Y", "confidence": 0.6}\nThanks'

    parsed = agent._parse_json_response(raw)
    assert parsed["paper_title"] == "Y"


def test_parse_json_response_raises_on_invalid_json() -> None:
    agent = object.__new__(UnderstandingAgent)
    with pytest.raises(ValueError):
        agent._parse_json_response("not json")


def test_build_prompt_truncates_algo_eq_but_keeps_abstract_and_conclusion(tmp_path: Path) -> None:
    agent = object.__new__(UnderstandingAgent)
    agent._MAX_PROMPT_CHARS = 1_500
    agent._MIN_EQUATION_CHARS = 200
    agent._MIN_ALGORITHM_CHARS = 300

    doc = _sample_paper_document(tmp_path)
    doc.abstract = "ABSTRACT_MARKER " + ("A" * 200)
    doc.sections.append(
        Section(
            title="Conclusion",
            level=1,
            content="CONCLUSION_MARKER " + ("C" * 200),
            page_start=11,
        )
    )
    doc.algorithms = [
        Algorithm(
            name=f"Algorithm {i}",
            pseudocode="P" * 800,
            page=1,
            language_hint="python-like",
        )
        for i in range(8)
    ]
    doc.equations = [Equation(latex=("E" * 400), description="desc", page=1) for _ in range(12)]

    prompt = agent._build_prompt(doc)

    assert "ABSTRACT_MARKER" in prompt
    assert "CONCLUSION_MARKER" in prompt
    assert "[truncated due to prompt budget]" in prompt


def test_graph_build_and_export() -> None:
    nx = pytest.importorskip("networkx")
    from scholardevclaw.understanding.graph import build_concept_graph, export_graph_json

    understanding = PaperUnderstanding.from_dict(_sample_understanding_dict())
    graph = build_concept_graph(understanding)

    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1

    exported = export_graph_json(graph)
    assert isinstance(exported, dict)
    assert "nodes" in exported
    assert "links" in exported


def test_understand_flow_with_mocked_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    doc = _sample_paper_document(tmp_path)
    expected_payload = _sample_understanding_dict()

    class FakeBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeMessages:
        def create(self, **_kwargs: object):
            return type("Resp", (), {"content": [FakeBlock(json.dumps(expected_payload))]})()

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.messages = FakeMessages()

    class FakeAnthropicModule:
        Anthropic = FakeClient

    import scholardevclaw.understanding.agent as agent_module

    monkeypatch.setattr(agent_module, "anthropic", FakeAnthropicModule)

    agent = UnderstandingAgent(api_key="test-key", model="fake-model")
    understanding = agent.understand(doc)

    assert understanding.paper_title == "Attention Is All You Need"
    assert understanding.complexity == "medium"
    assert any(req.name == "PyTorch" for req in understanding.requirements)
    assert "multi-head attention" in understanding.core_algorithm_description.lower()
