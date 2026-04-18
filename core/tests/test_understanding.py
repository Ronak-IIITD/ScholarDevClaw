from __future__ import annotations

import json
from pathlib import Path

import pytest

from scholardevclaw.ingestion.models import Algorithm, Equation, PaperDocument, Section
from scholardevclaw.understanding.agent import SYSTEM_PROMPT, UnderstandingAgent
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
        venue="NeurIPS 2017",
        sections=[
            Section(
                title="Method",
                level=1,
                content="Method text with multi-head attention and encoder decoder details.",
                page_start=3,
                section_type="method",
            ),
            Section(
                title="Experiments",
                level=1,
                content="Experiments on WMT14 with BLEU metrics and baselines.",
                page_start=8,
                section_type="experiments",
            ),
            Section(
                title="Conclusion",
                level=1,
                content="The model uses multi-head attention and positional encoding.",
                page_start=10,
                section_type="conclusion",
            ),
        ],
        equations=[
            Equation(
                latex="Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V",
                description="core attention equation",
                page=4,
                equation_type="model",
            )
        ],
        algorithms=[
            Algorithm(
                name="Algorithm 1: Training Procedure",
                pseudocode="Input: tokens, labels\nOutput: logits\nfor step in range(T):\n  update(theta)",
                page=7,
                language_hint="python-like",
                inputs=["tokens", "labels"],
                outputs=["logits"],
            )
        ],
        figures=[],
        full_text="full text",
        pdf_path=tmp_path / "paper.pdf",
        references=["[1] Bahdanau et al."],
        keywords=["transformer", "attention"],
        domain="nlp",
        subdomain="language-modeling",
    )


def _sample_understanding_dict() -> dict[str, object]:
    return {
        "paper_title": "Attention Is All You Need",
        "one_line_summary": "Transformer replaces recurrence with attention",
        "problem_statement": "Sequence transduction is bottlenecked by recurrence",
        "prior_state_of_art": "Recurrent seq2seq models dominated translation.",
        "key_insight": "Multi-head self-attention models dependencies in parallel.",
        "why_it_works": "Parallel attention lets the model see all tokens while positional encoding preserves order.",
        "contributions": [
            {
                "claim": "Introduces Transformer",
                "novelty": "Removes recurrence entirely",
                "is_implementable": True,
                "implementation_notes": "Use residual blocks and layer norm.",
            }
        ],
        "requirements": [
            {
                "name": "PyTorch",
                "requirement_type": "library",
                "is_optional": False,
                "version_constraint": ">=2.0",
                "acquisition_url": None,
                "notes": "Recommended implementation framework",
            }
        ],
        "concept_nodes": [
            {
                "id": "n1",
                "label": "Multi-Head Attention",
                "concept_type": "operation",
                "description": "Core attention block",
                "paper_section": "Method",
            },
            {
                "id": "n2",
                "label": "Positional Encoding",
                "concept_type": "operation",
                "description": "Adds sequence position information",
                "paper_section": "Method",
            },
        ],
        "concept_edges": [
            {
                "source_id": "n1",
                "target_id": "n2",
                "relation": "uses",
                "weight": 0.8,
            }
        ],
        "core_algorithm_description": "Use stacked multi-head attention with positional encoding.",
        "input_output_spec": "Takes token ids and outputs logits over vocabulary.",
        "hyperparameters": {"layers": 6, "heads": 8},
        "evaluation_protocol": "BLEU on WMT translation benchmarks.",
        "known_limitations": "Requires substantial parallel compute for full reproduction.",
        "complexity": "medium",
        "estimated_impl_hours": 24,
        "can_reproduce_without_compute": True,
        "confidence": 0.87,
        "confidence_notes": "Dataset preprocessing details are abbreviated in the paper.",
    }


def test_understanding_models_roundtrip() -> None:
    understanding = PaperUnderstanding(
        paper_title="P",
        one_line_summary="S",
        problem_statement="Problem",
        prior_state_of_art="Prior",
        key_insight="Insight",
        why_it_works="Mechanism",
        contributions=[Contribution("c", "n", True, "notes")],
        requirements=[Requirement("PyTorch", "library", False, "", ">=2.0", None)],
        concept_nodes=[ConceptNode("n1", "Node", "operation", "desc", "Method")],
        concept_edges=[ConceptEdge("n1", "n1", "uses", 0.9)],
        core_algorithm_description="core",
        input_output_spec="in->out",
        hyperparameters={"layers": 6},
        evaluation_protocol="eval",
        known_limitations="limit",
        complexity="medium",
        estimated_impl_hours=5,
        can_reproduce_without_compute=True,
        confidence=0.5,
        confidence_notes="uncertain",
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

    assert restored.complexity == "frontier-only"
    assert restored.estimated_impl_hours == 0
    assert restored.confidence == 1.0


def test_understanding_model_accepts_legacy_type_aliases() -> None:
    restored = PaperUnderstanding.from_dict(_sample_understanding_dict())

    assert restored.requirements[0].type == "library"
    assert restored.concept_nodes[0].type == "operation"
    assert restored.concept_edges[0].weight == pytest.approx(0.8)


def test_clean_json_response_handles_fenced_json() -> None:
    raw = '```json\n{"paper_title": "X", "confidence": 0.5}\n```'

    parsed = UnderstandingAgent.clean_json_response(raw)
    assert parsed["paper_title"] == "X"


def test_clean_json_response_extracts_embedded_json() -> None:
    raw = 'Result follows:\n{"paper_title": "Y", "confidence": 0.6}\nThanks'

    parsed = UnderstandingAgent.clean_json_response(raw)
    assert parsed["paper_title"] == "Y"


def test_parse_json_response_raises_on_invalid_json() -> None:
    agent = object.__new__(UnderstandingAgent)
    with pytest.raises(Exception):
        agent._parse_json_response("not json")


def test_build_prompt_contains_required_sections_and_prompt_contract(tmp_path: Path) -> None:
    agent = object.__new__(UnderstandingAgent)
    agent._MAX_PROMPT_CHARS = 4_000
    agent._MAX_SECTION_CHARS = 1_500

    doc = _sample_paper_document(tmp_path)
    prompt = agent._build_prompt(doc)

    assert "Method / Model Sections:" in prompt
    assert "Experiments / Evaluation Sections:" in prompt
    assert "Algorithm Blocks:" in prompt
    assert "Top Equations With Context:" in prompt
    assert '"confidence_notes": str' in prompt
    assert SYSTEM_PROMPT.startswith("You are a world-class AI researcher")


def test_build_prompt_truncates_large_sections(tmp_path: Path) -> None:
    agent = object.__new__(UnderstandingAgent)
    agent._MAX_PROMPT_CHARS = 1_500
    agent._MAX_SECTION_CHARS = 400

    doc = _sample_paper_document(tmp_path)
    doc.sections.append(
        Section(
            title="Method Details",
            level=2,
            content="M" * 2_000,
            page_start=4,
            section_type="method",
        )
    )

    prompt = agent._build_prompt(doc)

    assert "[truncated due to prompt budget]" in prompt


def test_merge_understandings_prefers_experiment_details() -> None:
    agent = object.__new__(UnderstandingAgent)
    architecture = PaperUnderstanding.from_dict(_sample_understanding_dict())
    experiments = PaperUnderstanding.from_dict(
        {
            "paper_title": "Attention Is All You Need",
            "evaluation_protocol": "BLEU on WMT14 En-De and En-Fr benchmarks.",
            "known_limitations": "Training is expensive on small hardware.",
            "confidence": 0.91,
            "confidence_notes": "Exact tokenizer details require appendix.",
            "requirements": [
                {
                    "name": "WMT14",
                    "requirement_type": "dataset",
                    "is_optional": False,
                    "notes": "Primary evaluation corpus",
                }
            ],
        }
    )

    merged = agent._merge_understandings(architecture, experiments)

    assert merged.evaluation_protocol.startswith("BLEU on WMT14")
    assert any(req.name == "WMT14" for req in merged.requirements)
    assert merged.confidence == pytest.approx(0.91)


def test_graph_build_and_export() -> None:
    nx = pytest.importorskip("networkx")
    from scholardevclaw.understanding.graph import build_concept_graph, export_graph_json

    understanding = PaperUnderstanding.from_dict(_sample_understanding_dict())
    graph = build_concept_graph(understanding)

    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert graph["n1"]["n2"]["weight"] == pytest.approx(0.8)

    exported = export_graph_json(graph)
    assert isinstance(exported, dict)
    assert "nodes" in exported
    assert "links" in exported
    assert "metrics" in exported
    assert "density" in exported["metrics"]
    assert "key_hubs" in exported["metrics"]
    assert "longest_path" in exported["metrics"]


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
    assert understanding.confidence_notes
