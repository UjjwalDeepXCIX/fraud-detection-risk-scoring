"""Knowledge graph construction over transaction entities.

A row-by-row classifier like the one in fraud_detection.models only ever
sees one transaction's own features -- it structurally cannot know that the
same device just transacted against four other accounts. This module builds
the entity graph (customers, devices, IPs, and how they're connected) that
the symbolic rule layer and the /investigate endpoint query to see that
relationship.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def build_entity_graph(df: pd.DataFrame) -> nx.Graph:
    """Build an undirected entity graph from a transaction history.

    Nodes are ``customer:<id>``, ``device:<id>``, ``ip:<id>``. Edges connect
    a customer to every device/IP it has transacted from, carrying the
    transaction count and first/last-seen timestamps as evidence a rule or
    an investigation report can cite directly.
    """
    graph = nx.Graph()
    for row in df.itertuples(index=False):
        customer_node = f"customer:{row.customer_id}"
        device_node = f"device:{row.device_id}"
        ip_node = f"ip:{row.ip_address}"

        graph.add_node(customer_node, type="customer")
        graph.add_node(device_node, type="device")
        graph.add_node(ip_node, type="ip")

        _add_or_update_edge(graph, customer_node, device_node, row.timestamp)
        _add_or_update_edge(graph, customer_node, ip_node, row.timestamp)

    logger.info(
        "Built entity graph: %d nodes, %d edges, from %d transactions",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        len(df),
    )
    return graph


def _add_or_update_edge(graph: nx.Graph, u: str, v: str, timestamp) -> None:
    if graph.has_edge(u, v):
        data = graph[u][v]
        data["transaction_count"] += 1
        data["first_seen"] = min(data["first_seen"], timestamp)
        data["last_seen"] = max(data["last_seen"], timestamp)
    else:
        graph.add_edge(u, v, transaction_count=1, first_seen=timestamp, last_seen=timestamp)


def save_graph(graph: nx.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graph, f)
    logger.info("Saved entity graph to %s", path)


def load_graph(path: Path) -> nx.Graph:
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    import argparse

    from fraud_detection.config import GRAPH_ARTIFACT_PATH, RAW_DATA_PATH

    parser = argparse.ArgumentParser(description="Build and persist the transaction entity graph.")
    parser.add_argument("--data-path", type=str, default=str(RAW_DATA_PATH))
    parser.add_argument("--output", type=str, default=str(GRAPH_ARTIFACT_PATH))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = pd.read_csv(args.data_path, parse_dates=["timestamp"])
    graph = build_entity_graph(df)
    save_graph(graph, Path(args.output))


if __name__ == "__main__":
    main()
