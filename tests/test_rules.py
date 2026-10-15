from fraud_detection.graph.rules import RING_FANOUT_THRESHOLD, evaluate_rules, shared_device_ring, shared_ip_ring


def test_shared_device_ring_fires_on_real_ring(small_entity_graph, ring_device_id):
    result = shared_device_ring(small_entity_graph, ring_device_id)
    assert result.fired is True
    assert result.severity == "high"
    assert result.evidence["distinct_customer_count"] >= RING_FANOUT_THRESHOLD


def test_shared_device_ring_does_not_fire_on_legitimate_household(small_entity_graph, household_device_id):
    """The whole point of the household scenario in the data generator: a
    genuinely shared device that is NOT fraud must not trip the rule, or the
    rule would just be rediscovering the label rather than detecting a
    pattern.
    """
    result = shared_device_ring(small_entity_graph, household_device_id)
    assert result.fired is False
    assert result.severity == "none"
    assert result.evidence["distinct_customer_count"] < RING_FANOUT_THRESHOLD


def test_shared_device_ring_does_not_fire_on_unseen_device(small_entity_graph):
    result = shared_device_ring(small_entity_graph, "DEV-NEVER-SEEN")
    assert result.fired is False
    assert result.evidence["distinct_customer_count"] == 0


def test_shared_ip_ring_mirrors_device_rule(small_transactions_df, small_entity_graph, ring_device_id):
    ring_ip = small_transactions_df.loc[small_transactions_df["device_id"] == ring_device_id, "ip_address"].iloc[0]
    result = shared_ip_ring(small_entity_graph, ring_ip)
    assert result.fired is True


def test_evaluate_rules_returns_both_rules(small_transactions_df, small_entity_graph, ring_device_id):
    ring_ip = small_transactions_df.loc[small_transactions_df["device_id"] == ring_device_id, "ip_address"].iloc[0]
    results = evaluate_rules(small_entity_graph, ring_device_id, ring_ip)
    rule_names = {r.rule_name for r in results}
    assert rule_names == {"shared_device_ring", "shared_ip_ring"}
    assert all(r.fired for r in results)
