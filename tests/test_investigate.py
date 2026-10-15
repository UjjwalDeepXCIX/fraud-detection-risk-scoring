import pytest

from fraud_detection.api.inference import FraudModel
from fraud_detection.api.schemas import InvestigateRequest
from fraud_detection.graph.investigate import investigate

BASE_PAYLOAD = {
    "amount": 48.0,
    "merchant_category": "grocery",
    "channel": "in_store",
    "card_present": True,
    "is_foreign_transaction": False,
    "distance_from_home_km": 2.0,
    "account_age_days": 800,
    "customer_avg_amount": 50.0,
    "txn_count_last_1h": 0,
    "txn_count_last_24h": 1,
    "amount_sum_last_24h": 40.0,
}


@pytest.fixture(scope="module")
def loaded_model(trained_artifact_path) -> FraudModel:
    m = FraudModel()
    m.load(trained_artifact_path)
    return m


def test_investigate_escalates_on_ring_device_even_with_mundane_transaction(
    loaded_model, small_entity_graph, ring_device_id
):
    """The core neurosymbolic-fusion claim: an otherwise unremarkable-looking
    transaction (small amount, domestic, card present, grocery) on a device
    that is a known ring must still be escalated to high risk, because the
    symbolic rule is a hard override on top of the model's own score, not a
    blended average it can be outvoted in.
    """
    request = InvestigateRequest(**BASE_PAYLOAD, device_id=ring_device_id, ip_address="irrelevant-ip-not-in-graph")
    report = investigate(request, loaded_model, small_entity_graph)

    assert report.final_risk_level == "high"
    assert any(r.fired for r in report.matched_rules)
    assert report.escalated or report.model_risk_level == "high"


def test_investigate_does_not_escalate_on_household_device(loaded_model, small_entity_graph, household_device_id):
    """A legitimate shared device (household scale) must not trip the rule
    overlay -- final risk level should fall back to whatever the model
    itself says, same as it would without the graph layer at all.
    """
    request = InvestigateRequest(**BASE_PAYLOAD, device_id=household_device_id, ip_address="irrelevant-ip-not-in-graph")
    report = investigate(request, loaded_model, small_entity_graph)

    assert not any(r.fired for r in report.matched_rules)
    assert report.final_risk_level == report.model_risk_level
    assert report.escalated is False


def test_investigate_does_not_escalate_on_unseen_device(loaded_model, small_entity_graph):
    request = InvestigateRequest(**BASE_PAYLOAD, device_id="DEV-NEVER-SEEN", ip_address="IP-NEVER-SEEN")
    report = investigate(request, loaded_model, small_entity_graph)

    assert not any(r.fired for r in report.matched_rules)
    assert report.final_risk_level == report.model_risk_level


def test_investigate_scores_using_graph_fanout_not_caller_supplied_value(
    loaded_model, small_entity_graph, ring_device_id
):
    """The request's own device_distinct_customers_count must be ignored in
    favour of the graph's real value -- a caller can't under-report fan-out
    to get a lower score.
    """
    request = InvestigateRequest(
        **BASE_PAYLOAD,
        device_id=ring_device_id,
        ip_address="irrelevant-ip-not-in-graph",
        device_distinct_customers_count=1,  # caller under-reports; graph should override
    )
    report = investigate(request, loaded_model, small_entity_graph)
    # Regardless of the model's own probability, the rule layer must still
    # see the graph's real fan-out and fire.
    device_rule = next(r for r in report.matched_rules if r.rule_name == "shared_device_ring")
    assert device_rule.evidence["distinct_customer_count"] >= 4


def test_investigate_includes_related_entities(loaded_model, small_entity_graph, ring_device_id):
    request = InvestigateRequest(**BASE_PAYLOAD, device_id=ring_device_id, ip_address="irrelevant-ip-not-in-graph")
    report = investigate(request, loaded_model, small_entity_graph)
    assert len(report.related_entities["customers"]) >= 4
