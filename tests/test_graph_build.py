import pandas as pd

from fraud_detection.graph.build import build_entity_graph


def test_graph_has_customer_device_and_ip_nodes(small_entity_graph):
    node_types = {data["type"] for _, data in small_entity_graph.nodes(data=True)}
    assert node_types == {"customer", "device", "ip"}


def test_graph_node_count_matches_distinct_entities(small_transactions_df, small_entity_graph):
    expected_customers = small_transactions_df["customer_id"].nunique()
    expected_devices = small_transactions_df["device_id"].nunique()
    expected_ips = small_transactions_df["ip_address"].nunique()

    actual_customers = sum(1 for _, d in small_entity_graph.nodes(data=True) if d["type"] == "customer")
    actual_devices = sum(1 for _, d in small_entity_graph.nodes(data=True) if d["type"] == "device")
    actual_ips = sum(1 for _, d in small_entity_graph.nodes(data=True) if d["type"] == "ip")

    assert actual_customers == expected_customers
    assert actual_devices == expected_devices
    assert actual_ips == expected_ips


def test_ring_device_connects_to_all_its_customers(small_transactions_df, small_entity_graph, ring_device_id):
    expected_customers = set(
        small_transactions_df.loc[small_transactions_df["device_id"] == ring_device_id, "customer_id"]
    )
    node = f"device:{ring_device_id}"
    connected_customers = {n.split(":", 1)[1] for n in small_entity_graph.neighbors(node) if n.startswith("customer:")}
    assert connected_customers == expected_customers


def test_edge_transaction_count_matches_row_count():
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C2"],
            "device_id": ["D1", "D1", "D1"],
            "ip_address": ["I1", "I1", "I2"],
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        }
    )
    graph = build_entity_graph(df)
    assert graph["customer:C1"]["device:D1"]["transaction_count"] == 2
    assert graph["customer:C2"]["device:D1"]["transaction_count"] == 1
    assert graph["customer:C1"]["ip:I1"]["transaction_count"] == 2
