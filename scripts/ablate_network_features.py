#!/usr/bin/env python
"""Ablation: does the graph-derived network feature pair actually help?

Trains two otherwise-identical XGBoost pipelines -- one with
device_distinct_customers_count / ip_distinct_customers_count included, one
with them excluded -- and compares recall specifically on the synthetic
ring-fraud subset of the test split (not just overall metrics), since
that's the one pattern these two features exist to catch. Real numbers,
not an assumed improvement; see README for the result.

Usage: python scripts/ablate_network_features.py
"""
from __future__ import annotations

import argparse
import json
import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from fraud_detection.config import (
    ABLATION_METRICS_PATH,
    CATEGORICAL_FEATURES,
    DEFAULT_XGB_PARAMS,
    NETWORK_FEATURES,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
)
from fraud_detection.data.preprocessing import compute_scale_pos_weight, time_based_split
from fraud_detection.features.engineering import add_derived_features
from fraud_detection.models.evaluate import evaluate_at_threshold, tune_threshold

logger = logging.getLogger(__name__)


def _build_pipeline(numeric_features: list[str], scale_pos_weight: float) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("scale", StandardScaler())]), numeric_features),
            (
                "categorical",
                Pipeline([("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    params = {**DEFAULT_XGB_PARAMS, "scale_pos_weight": scale_pos_weight}
    return Pipeline([("preprocessor", preprocessor), ("classifier", XGBClassifier(**params))])


def _evaluate_variant(
    name: str,
    numeric_features: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ring_mask_test: pd.Series,
) -> dict:
    feature_cols = numeric_features + CATEGORICAL_FEATURES
    scale_pos_weight = compute_scale_pos_weight(train_df["is_fraud"])
    pipeline = _build_pipeline(numeric_features, scale_pos_weight)
    pipeline.fit(train_df[feature_cols], train_df["is_fraud"])

    val_proba = pipeline.predict_proba(val_df[feature_cols])[:, 1]
    threshold_info = tune_threshold(val_df["is_fraud"].to_numpy(), val_proba)
    threshold = threshold_info["f1_optimal_threshold"]

    test_proba = pipeline.predict_proba(test_df[feature_cols])[:, 1]
    overall_metrics = evaluate_at_threshold(test_df["is_fraud"].to_numpy(), test_proba, threshold)

    # The one number this ablation exists to produce: of the ring-fraud rows
    # in the test split specifically, what fraction does this variant catch
    # at its own tuned threshold?
    ring_test = test_df[ring_mask_test]
    if len(ring_test) > 0:
        ring_proba = pipeline.predict_proba(ring_test[feature_cols])[:, 1]
        ring_recall = float((ring_proba >= threshold).mean())
    else:
        ring_recall = None

    logger.info(
        "[%s] threshold=%.4f ring_fraud_recall=%s (n_ring_test=%d) overall=%s",
        name,
        threshold,
        ring_recall,
        len(ring_test),
        overall_metrics,
    )
    return {
        "threshold": threshold,
        "overall": overall_metrics,
        "ring_fraud_recall": ring_recall,
        "n_ring_fraud_in_test": int(len(ring_test)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=str, default=str(RAW_DATA_PATH))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw_df = pd.read_csv(args.data_path, parse_dates=["timestamp"])
    df = add_derived_features(raw_df)
    train_df, val_df, test_df = time_based_split(df)

    ring_mask_test = (test_df["is_fraud"] == 1) & test_df["device_id"].str.startswith("DEV-RING-")
    logger.info("Test split: %d rows, %d ring-fraud rows", len(test_df), ring_mask_test.sum())

    numeric_with = NUMERIC_FEATURES
    numeric_without = [f for f in NUMERIC_FEATURES if f not in NETWORK_FEATURES]

    results = {
        "with_network_features": _evaluate_variant(
            "with_network_features", numeric_with, train_df, val_df, test_df, ring_mask_test
        ),
        "without_network_features": _evaluate_variant(
            "without_network_features", numeric_without, train_df, val_df, test_df, ring_mask_test
        ),
    }

    ABLATION_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ABLATION_METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote ablation report to %s", ABLATION_METRICS_PATH)


if __name__ == "__main__":
    main()
