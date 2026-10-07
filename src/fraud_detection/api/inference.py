"""Model loading and single-transaction inference, shared by the API layer.

Loads the trained pipeline once and reuses it for every request, applying
the exact same feature-engineering code path used at training time.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from fraud_detection.api.schemas import PredictionResponse, TransactionRequest
from fraud_detection.config import MODEL_ARTIFACT_PATH
from fraud_detection.features.engineering import add_derived_features
from fraud_detection.models.explain import FraudExplainer

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    pass


class FraudModel:
    """Thin wrapper around the persisted artifact (pipeline + threshold +
    feature schema) that exposes a single-transaction predict() call.
    """

    def __init__(self) -> None:
        self.pipeline = None
        self.threshold: Optional[float] = None
        self.feature_columns: Optional[list[str]] = None
        self.trained_at: Optional[str] = None
        self._explainer: Optional[FraudExplainer] = None

    def load(self, model_path: Path = MODEL_ARTIFACT_PATH) -> None:
        logger.info("Loading model artifact from %s", model_path)
        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.threshold = artifact["threshold"]
        self.feature_columns = artifact["feature_columns"]
        self.trained_at = artifact.get("trained_at")
        self._explainer = None  # built lazily, only if an explanation is requested

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise ModelNotLoadedError("Model artifact has not been loaded yet")

    def _to_frame(self, transaction: TransactionRequest) -> pd.DataFrame:
        payload = transaction.model_dump()
        payload["timestamp"] = payload["timestamp"] or pd.Timestamp.now(tz="UTC")
        payload["merchant_category"] = transaction.merchant_category.value
        payload["channel"] = transaction.channel.value
        return pd.DataFrame([payload])

    def predict(self, transaction: TransactionRequest, explain: bool = False, top_k: int = 5) -> PredictionResponse:
        self._require_loaded()
        start = time.perf_counter()

        raw_frame = self._to_frame(transaction)
        engineered = add_derived_features(raw_frame)
        X = engineered[self.feature_columns]

        proba = float(self.pipeline.predict_proba(X)[0, 1])
        is_fraud = proba >= self.threshold

        if proba >= self.threshold:
            risk_level = "high"
        elif proba >= self.threshold * 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"

        top_factors = None
        if explain:
            if self._explainer is None:
                self._explainer = FraudExplainer(self.pipeline)
            top_factors = self._explainer.top_factors(X, top_k=top_k)

        latency_ms = (time.perf_counter() - start) * 1000
        return PredictionResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability=round(proba, 6),
            is_fraud=is_fraud,
            risk_level=risk_level,
            threshold_used=self.threshold,
            top_factors=top_factors,
            model_trained_at=self.trained_at,
            inference_latency_ms=round(latency_ms, 3),
        )


model = FraudModel()
