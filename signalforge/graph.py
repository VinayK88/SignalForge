from __future__ import annotations

import networkx as nx


def build_entity_graph(events: list[dict]) -> nx.Graph:
    """Build a shared-entity graph from synthetic accounts/devices/payments/addresses."""
    graph = nx.Graph()
    for event in events:
        account = event["account_id"]
        graph.add_node(account, kind="account", fraud_label=int(event.get("fraud_label", 0)))
        for key, kind in (("device_id", "device"), ("payment_id", "payment"), ("shipping_id", "shipping")):
            entity = event[key]
            graph.add_node(entity, kind=kind)
            graph.add_edge(account, entity, relation=f"uses_{kind}")
    return graph


def graph_summary(events: list[dict]) -> dict:
    graph = build_entity_graph(events)
    components = list(nx.connected_components(graph))
    account_nodes = [node for node, data in graph.nodes(data=True) if data.get("kind") == "account"]
    risky_components = 0
    for component in components:
        accounts = [node for node in component if graph.nodes[node].get("kind") == "account"]
        if accounts and sum(int(graph.nodes[node].get("fraud_label", 0)) for node in accounts) >= 2:
            risky_components += 1
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": len(components),
        "accounts": len(account_nodes),
        "components_with_multiple_synthetic_fraud_labels": risky_components,
        "boundary": "synthetic shared-entity graph only",
    }
