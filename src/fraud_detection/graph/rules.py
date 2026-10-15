"""Symbolic fraud-typology rules evaluated over the entity graph.

Each rule is a pure function: graph + identifiers in, a structured,
auditable result out. Deliberately not learned -- the point of keeping this
layer symbolic is that every escalation it triggers can be explained in one
sentence to a compliance reviewer ("device D is linked to 5 other
accounts"), which a gradient-boosted probability alone cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from fraud_detection.graph.retrieve import get_distinct_customers

# Fan-out at or above this many distinct customers on one device/IP is
# treated as ring-scale, not household-scale. Calibrated directly against
# the data generator's own household size (2-3, HOUSEHOLD_SIZE_RANGE) and
# ring size (4-6, RING_SIZE_RANGE) -- see fraud_detection.data.generate --
# so the rule fires on the fraud pattern and not on the legitimate one by
# construction, not by coincidence.
RING_FANOUT_THRESHOLD = 4


@dataclass
class RuleResult:
    rule_name: str
    fired: bool
    evidence: dict = field(default_factory=dict)
    severity: str = "none"


def shared_device_ring(graph: nx.Graph, device_id: str) -> RuleResult:
    customers = get_distinct_customers(graph, f"device:{device_id}")
    fired = len(customers) >= RING_FANOUT_THRESHOLD
    return RuleResult(
        rule_name="shared_device_ring",
        fired=fired,
        evidence={
            "device_id": device_id,
            "distinct_customer_count": len(customers),
            "customers": sorted(customers),
        },
        severity="high" if fired else "none",
    )


def shared_ip_ring(graph: nx.Graph, ip_address: str) -> RuleResult:
    customers = get_distinct_customers(graph, f"ip:{ip_address}")
    fired = len(customers) >= RING_FANOUT_THRESHOLD
    return RuleResult(
        rule_name="shared_ip_ring",
        fired=fired,
        evidence={
            "ip_address": ip_address,
            "distinct_customer_count": len(customers),
            "customers": sorted(customers),
        },
        severity="high" if fired else "none",
    )


def evaluate_rules(graph: nx.Graph, device_id: str, ip_address: str) -> list[RuleResult]:
    return [shared_device_ring(graph, device_id), shared_ip_ring(graph, ip_address)]
