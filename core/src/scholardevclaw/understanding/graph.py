from __future__ import annotations

from typing import Any

import networkx as nx

from scholardevclaw.understanding.models import PaperUnderstanding


def build_concept_graph(understanding: PaperUnderstanding) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in understanding.concept_nodes:
        graph.add_node(
            node.id,
            label=node.label,
            concept_type=node.concept_type,
            description=node.description,
            paper_section=node.paper_section,
        )
    for edge in understanding.concept_edges:
        graph.add_edge(
            edge.source_id,
            edge.target_id,
            relation=edge.relation,
            weight=edge.weight,
        )
    return graph


def export_graph_json(graph: nx.DiGraph) -> dict[str, Any]:
    node_link = dict(nx.node_link_data(graph))
    hubs = sorted(graph.in_degree(), key=lambda item: item[1], reverse=True)
    density = float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0

    dag = nx.DiGraph(graph)
    if not nx.is_directed_acyclic_graph(dag):
        dag = nx.DiGraph()
        dag.add_nodes_from(graph.nodes(data=True))
        dag.add_edges_from((u, v, data) for u, v, data in graph.edges(data=True) if u != v)

    try:
        longest_path = nx.dag_longest_path(dag) if dag.number_of_edges() else []
    except nx.NetworkXUnfeasible:
        longest_path = []

    return {
        **node_link,
        "metrics": {
            "density": density,
            "key_hubs": [node_id for node_id, _ in hubs[:5]],
            "longest_path": longest_path,
        },
    }
