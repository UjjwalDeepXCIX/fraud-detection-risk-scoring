"""Fuse the model's neural score with the symbolic graph-rule evaluation
into one investigation report.

The fusion policy is escalation, not a blended average: a fired hard rule
forces the final risk level to "high" regardless of what the model's own
score says. This mirrors how real fraud/compliance stacks work -- a
statistical model can be wrong in either direction, but a known typology
match (a confirmed ring pattern) is treated as a hard override that a
single probability number shouldn't be allowed to quietly out-vote.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import networkx as nx

from fraud_detection.api.inference import FraudModel
from fraud_detection.api.schemas import InvestigateRequest
from fraud_detection.graph.retrieve import get_fanout_counts, get_related_entities
from fraud_detection.graph.rules import RuleResult, evaluate_rules


@dataclass
class InvestigationReport:
    transaction_id: str | None
    fraud_probability: float
    model_risk_level: str
    final_risk_level: str
    escalated: bool
    threshold_used: float
    matched_rules: list[RuleResult]
    related_entities: dict[str, list[str]]
    top_factors: list | None
    latency_ms: float


def investigate(
    request: InvestigateRequest, model: FraudModel, graph: nx.Graph, explain: bool = False
) -> InvestigationReport:
    start = time.perf_counter()

    device_count, ip_count = get_fanout_counts(graph, request.device_id, request.ip_address)
    scored_request = request.model_copy(
        update={
            "device_distinct_customers_count": device_count,
            "ip_distinct_customers_count": ip_count,
        }
    )

    prediction = model.predict(scored_request, explain=explain)
    matched_rules = evaluate_rules(graph, request.device_id, request.ip_address)
    related = get_related_entities(graph, request.device_id, request.ip_address, k=1)

    any_rule_fired = any(r.fired for r in matched_rules)
    final_risk_level = "high" if any_rule_fired else prediction.risk_level

    return InvestigationReport(
        transaction_id=request.transaction_id,
        fraud_probability=prediction.fraud_probability,
        model_risk_level=prediction.risk_level,
        final_risk_level=final_risk_level,
        escalated=any_rule_fired and prediction.risk_level != "high",
        threshold_used=prediction.threshold_used,
        matched_rules=matched_rules,
        related_entities=related,
        top_factors=prediction.top_factors,
        latency_ms=(time.perf_counter() - start) * 1000,
    )
