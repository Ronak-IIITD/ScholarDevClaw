from __future__ import annotations

from typing import Any

import networkx as nx

from scholardevclaw.understanding.models import PaperUnderstanding


def build_concept_graph(understanding: PaperUnderstanding) -> nx.DiGraph:
    """Build a directed concept graph from paper understanding payload."""

    graph = nx.DiGraph()
    for node in understanding.concept_nodes:
        graph.add_node(
            node.id,
            label=node.label,
            type=node.type,
            description=node.description,
        )
    for edge in understanding.concept_edges:
        graph.add_edge(edge.source_id, edge.target_id, relation=edge.relation)
    return graph


def export_graph_json(graph: nx.DiGraph) -> dict[str, Any]:
    """Export graph to node-link JSON data."""

    return dict(nx.node_link_data(graph))
