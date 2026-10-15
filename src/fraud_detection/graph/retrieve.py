"""Retrieval over the entity graph.

The explicit "retrieval" half of the neurosymbolic design: given a
transaction's device/IP, fetch the structured knowledge (connected
customers, their other devices/IPs) needed to ground both the fan-out
features fed to the model and the symbolic rule evaluation -- rather than
either relying on the transaction's own fields alone.
"""

from __future__ import annotations

import networkx as nx


def get_distinct_customers(graph: nx.Graph, node: str) -> set[str]:
    """Customer ids directly connected to the given ``device:<id>`` or
    ``ip:<id>`` node. Shared by the fan-out count below and by the rule
    engine in ``fraud_detection.graph.rules``, so both read the same graph
    query rather than two subtly different implementations of it.
    """
    if not graph.has_node(node):
        return set()
    return {n.split(":", 1)[1] for n in graph.neighbors(node) if n.startswith("customer:")}


def get_fanout_counts(graph: nx.Graph, device_id: str, ip_address: str) -> tuple[int, int]:
    """Distinct-customer fan-out for a device and an IP, read directly from
    the graph. A device/IP never seen before returns 1 (itself) -- same
    neutral convention as the data generator's cumulative counts, since the
    current transaction's own customer always counts as one.
    """
    device_count = len(get_distinct_customers(graph, f"device:{device_id}")) or 1
    ip_count = len(get_distinct_customers(graph, f"ip:{ip_address}")) or 1
    return device_count, ip_count


def get_related_entities(
    graph: nx.Graph, device_id: str | None = None, ip_address: str | None = None, k: int = 1
) -> dict[str, list[str]]:
    """K-hop neighbourhood fetch around a device and/or IP node: which
    customers are connected, and (at k=2) which other devices/IPs those
    customers also use. This is what grounds an investigation report in
    actual graph structure rather than just the one transaction under review.
    """
    seed_nodes = [n for n in (f"device:{device_id}" if device_id else None, f"ip:{ip_address}" if ip_address else None) if n]
    seed_nodes = [n for n in seed_nodes if graph.has_node(n)]
    if not seed_nodes:
        return {"customers": [], "devices": [], "ips": []}

    neighborhood: set[str] = set()
    for seed in seed_nodes:
        neighborhood |= set(nx.single_source_shortest_path_length(graph, seed, cutoff=k).keys())
    neighborhood -= set(seed_nodes)

    return {
        "customers": sorted(n.split(":", 1)[1] for n in neighborhood if n.startswith("customer:")),
        "devices": sorted(n.split(":", 1)[1] for n in neighborhood if n.startswith("device:")),
        "ips": sorted(n.split(":", 1)[1] for n in neighborhood if n.startswith("ip:")),
    }
