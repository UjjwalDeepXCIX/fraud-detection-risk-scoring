"""End-to-end training pipeline: load -> engineer -> split -> cross-validate
-> fit -> tune threshold -> evaluate -> explain -> persist.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraud_detection.config import (
    DEFAULT_XGB_PARAMS,
    METRICS_PATH,
    MODEL_ARTIFACT_PATH,
    RANDOM_SEED,
    RAW_DATA_PATH,
    SHAP_SUMMARY_PATH,
)
from fraud_detection.data.preprocessing import compute_scale_pos_weight, time_based_split
from fraud_detection.features.engineering import add_derived_features, build_preprocessor, get_feature_columns
from fraud_detection.models.evaluate import evaluate_at_threshold, tune_threshold
from fraud_detection.models.explain import FraudExplainer

logger = logging.getLogger(__name__)


def build_pipeline(scale_pos_weight: float, xgb_overrides: dict | None = None) -> Pipeline:
    params = {**DEFAULT_XGB_PARAMS, "scale_pos_weight": scale_pos_weight}
    if xgb_overrides:
        params.update(xgb_overrides)
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", XGBClassifier(**params)),
        ]
    )


def run_cross_validation(train_df: pd.DataFrame, feature_columns: list[str], scale_pos_weight: float) -> dict:
    pipeline = build_pipeline(scale_pos_weight)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scoring = ["precision", "recall", "f1", "roc_auc"]
    results = cross_validate(
        pipeline,
        train_df[feature_columns],
        train_df["is_fraud"],
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )
    return {
        f"cv_{metric}_mean": float(np.mean(results[f"test_{metric}"]))
        for metric in scoring
    } | {
        f"cv_{metric}_std": float(np.std(results[f"test_{metric}"]))
        for metric in scoring
    }


def train(
    data_path=RAW_DATA_PATH,
    model_output=MODEL_ARTIFACT_PATH,
    metrics_output=METRICS_PATH,
    skip_shap: bool = False,
) -> dict:
    logger.info("Loading raw transactions from %s", data_path)
    raw_df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df = add_derived_features(raw_df)
    feature_columns = get_feature_columns()

    train_df, val_df, test_df = time_based_split(df)
    logger.info(
        "Split sizes -> train=%d val=%d test=%d (fraud rates: %.3f%% / %.3f%% / %.3f%%)",
        len(train_df),
        len(val_df),
        len(test_df),
        train_df["is_fraud"].mean() * 100,
        val_df["is_fraud"].mean() * 100,
        test_df["is_fraud"].mean() * 100,
    )

    scale_pos_weight = compute_scale_pos_weight(train_df["is_fraud"])
    logger.info("Using scale_pos_weight=%.2f to counter class imbalance", scale_pos_weight)

    logger.info("Running 5-fold stratified cross-validation on the training split")
    cv_metrics = run_cross_validation(train_df, feature_columns, scale_pos_weight)
    logger.info("CV metrics: %s", cv_metrics)

    logger.info("Fitting final pipeline on the full training split")
    pipeline = build_pipeline(scale_pos_weight)
    pipeline.fit(train_df[feature_columns], train_df["is_fraud"])

    val_proba = pipeline.predict_proba(val_df[feature_columns])[:, 1]
    threshold_info = tune_threshold(val_df["is_fraud"].to_numpy(), val_proba)
    chosen_threshold = threshold_info["f1_optimal_threshold"]
    logger.info("Tuned decision threshold on validation split: %.4f", chosen_threshold)

    test_proba = pipeline.predict_proba(test_df[feature_columns])[:, 1]
    test_metrics = evaluate_at_threshold(test_df["is_fraud"].to_numpy(), test_proba, chosen_threshold)
    logger.info("Held-out test metrics @ threshold=%.4f: %s", chosen_threshold, test_metrics)

    latency_ms = _benchmark_single_row_latency(pipeline, test_df[feature_columns])
    logger.info("Median single-row inference latency: %.2f ms", latency_ms)

    if not skip_shap:
        logger.info("Generating SHAP global summary plot")
        explainer = FraudExplainer(pipeline)
        sample = test_df[feature_columns].sample(
            n=min(2000, len(test_df)), random_state=RANDOM_SEED
        )
        explainer.save_summary_plot(sample, SHAP_SUMMARY_PATH)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "threshold": chosen_threshold,
        "feature_columns": feature_columns,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(artifact, model_output)
    logger.info("Saved model artifact to %s", model_output)

    metrics_report = {
        "cross_validation": cv_metrics,
        "threshold_tuning": threshold_info,
        "test_set": test_metrics,
        "chosen_threshold": chosen_threshold,
        "median_inference_latency_ms": latency_ms,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "fraud_rate_overall": float(df["is_fraud"].mean()),
        "scale_pos_weight": scale_pos_weight,
    }
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output, "w") as f:
        json.dump(metrics_report, f, indent=2)
    logger.info("Wrote metrics report to %s", metrics_output)

    return metrics_report


def _benchmark_single_row_latency(pipeline: Pipeline, X: pd.DataFrame, n_samples: int = 200) -> float:
    sample = X.sample(n=min(n_samples, len(X)), random_state=RANDOM_SEED)
    timings = []
    for i in range(len(sample)):
        row = sample.iloc[[i]]
        start = time.perf_counter()
        pipeline.predict_proba(row)
        timings.append((time.perf_counter() - start) * 1000)
    return float(np.median(timings))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=str, default=str(RAW_DATA_PATH))
    parser.add_argument("--model-output", type=str, default=str(MODEL_ARTIFACT_PATH))
    parser.add_argument("--metrics-output", type=str, default=str(METRICS_PATH))
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP plot generation (faster iteration)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from pathlib import Path

    train(
        data_path=Path(args.data_path),
        model_output=Path(args.model_output),
        metrics_output=Path(args.metrics_output),
        skip_shap=args.skip_shap,
    )


if __name__ == "__main__":
    main()
