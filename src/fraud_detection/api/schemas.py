"""Pydantic request/response models for the scoring API.

Field-level validation here is the first line of defence against malformed
or adversarial input reaching the model -- amounts and counts are bounded to
sane ranges rather than trusted blindly.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from fraud_detection.config import CHANNELS, MERCHANT_CATEGORIES


class MerchantCategory(str, Enum):
    grocery = "grocery"
    restaurant = "restaurant"
    gas_station = "gas_station"
    electronics = "electronics"
    online_retail = "online_retail"
    travel = "travel"
    jewelry = "jewelry"
    utilities = "utilities"
    entertainment = "entertainment"
    healthcare = "healthcare"


class Channel(str, Enum):
    in_store = "in_store"
    online = "online"
    atm = "atm"


assert {c.value for c in MerchantCategory} == set(MERCHANT_CATEGORIES)
assert {c.value for c in Channel} == set(CHANNELS)


class TransactionRequest(BaseModel):
    transaction_id: Optional[str] = Field(default=None, description="Caller-supplied id, echoed back in the response")
    timestamp: Optional[datetime] = Field(default=None, description="Defaults to now if omitted")
    amount: float = Field(..., gt=0, le=1_000_000, description="Transaction amount in the account's currency")
    merchant_category: MerchantCategory
    channel: Channel
    card_present: bool
    is_foreign_transaction: bool
    distance_from_home_km: float = Field(..., ge=0, le=25_000)
    account_age_days: int = Field(..., ge=0, le=40_000)
    customer_avg_amount: float = Field(..., gt=0, le=1_000_000, description="Customer's historical average transaction amount")
    txn_count_last_1h: int = Field(..., ge=0, le=10_000)
    txn_count_last_24h: int = Field(..., ge=0, le=10_000)
    amount_sum_last_24h: float = Field(..., ge=0, le=10_000_000)
    device_distinct_customers_count: int = Field(
        default=1,
        ge=1,
        le=10_000,
        description=(
            "How many distinct customers have transacted on this device so far "
            "(graph-derived). Optional and defaults to 1 -- the neutral 'no known "
            "fan-out' assumption -- for callers without feature-store access to "
            "the real value; /investigate computes it directly from the entity graph."
        ),
    )
    ip_distinct_customers_count: int = Field(
        default=1, ge=1, le=10_000, description="Same as device_distinct_customers_count, for the IP instead."
    )

    @field_validator("txn_count_last_24h")
    @classmethod
    def validate_window_consistency(cls, v: int, info) -> int:
        count_1h = info.data.get("txn_count_last_1h")
        if count_1h is not None and v < count_1h:
            raise ValueError("txn_count_last_24h cannot be smaller than txn_count_last_1h")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": "TXN00000001",
                "timestamp": "2025-03-15T02:14:00Z",
                "amount": 1480.50,
                "merchant_category": "electronics",
                "channel": "online",
                "card_present": False,
                "is_foreign_transaction": True,
                "distance_from_home_km": 6200.0,
                "account_age_days": 42,
                "customer_avg_amount": 85.0,
                "txn_count_last_1h": 3,
                "txn_count_last_24h": 7,
                "amount_sum_last_24h": 2100.0,
            }
        }
    }


class ContributingFactor(BaseModel):
    feature: str
    impact: float


class PredictionResponse(BaseModel):
    transaction_id: Optional[str]
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    threshold_used: float
    top_factors: Optional[list[ContributingFactor]] = None
    model_trained_at: Optional[str] = None
    inference_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_trained_at: Optional[str] = None
    threshold: Optional[float] = None
