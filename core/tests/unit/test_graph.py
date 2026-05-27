"""Tests for understanding/graph.py"""

import networkx as nx

from scholardevclaw.understanding.graph import build_concept_graph, export_graph_json
from scholardevclaw.understanding.models import (
    ConceptEdge,
    ConceptNode,
    PaperUnderstanding,
)


class TestBuildConceptGraph:
    def test_empty_graph(self):
        u = PaperUnderstanding()
        graph = build_concept_graph(u)
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0

    def test_with_nodes(self):
        u = PaperUnderstanding(
            concept_nodes=[
                ConceptNode(id="n1", label="Attention", concept_type="mechanism", description="SA"),
                ConceptNode(id="n2", label="MLP", concept_type="layer", description="FFN"),
            ],
            concept_edges=[
                ConceptEdge(source_id="n1", target_id="n2", relation="feeds_into"),
            ],
        )
        graph = build_concept_graph(u)
        assert graph.number_of_nodes() == 2
        assert graph.has_node("n1")
        assert graph.has_node("n2")
        assert graph.nodes["n1"]["label"] == "Attention"
        assert graph.nodes["n2"]["concept_type"] == "layer"
        assert graph.has_edge("n1", "n2")
        assert graph.edges["n1", "n2"]["relation"] == "feeds_into"

    def test_cycle_detection(self):
        u = PaperUnderstanding(
            concept_nodes=[
                ConceptNode(id="a", label="A", concept_type="type", description=""),
                ConceptNode(id="b", label="B", concept_type="type", description=""),
                ConceptNode(id="c", label="C", concept_type="type", description=""),
            ],
            concept_edges=[
                ConceptEdge(source_id="a", target_id="b", relation="depends"),
                ConceptEdge(source_id="b", target_id="c", relation="depends"),
            ],
        )
        graph = build_concept_graph(u)
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 2


class TestExportGraphJson:
    def test_export_empty(self):
        graph = nx.DiGraph()
        result = export_graph_json(graph)
        assert "metrics" in result
        assert result["metrics"]["density"] == 0.0
        assert result["metrics"]["key_hubs"] == []
        assert result["metrics"]["longest_path"] == []

    def test_export_with_nodes(self):
        graph = nx.DiGraph()
        graph.add_node("n1", label="A", concept_type="method")
        graph.add_node("n2", label="B", concept_type="method")
        graph.add_edge("n1", "n2", relation="uses")

        result = export_graph_json(graph)
        assert result["metrics"]["density"] > 0
        assert len(result["metrics"]["key_hubs"]) > 0
        assert "nodes" in result
        assert "edges" in result or "links" in result

    def test_dag_longest_path(self):
        graph = nx.DiGraph()
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("a", "c")

        result = export_graph_json(graph)
        assert len(result["metrics"]["longest_path"]) >= 2

    def test_cycle_in_graph(self):
        graph = nx.DiGraph()
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("c", "a")

        result = export_graph_json(graph)
        assert "metrics" in result
