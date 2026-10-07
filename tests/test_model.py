import numpy as np
import pandas as pd

from fraud_detection.data.preprocessing import compute_scale_pos_weight, time_based_split
from fraud_detection.models.evaluate import evaluate_at_threshold, tune_threshold


def test_time_based_split_preserves_order_and_size(small_transactions_df):
    train_df, val_df, test_df = time_based_split(small_transactions_df)
    assert len(train_df) + len(val_df) + len(test_df) == len(small_transactions_df)
    assert train_df["timestamp"].max() <= val_df["timestamp"].min()
    assert val_df["timestamp"].max() <= test_df["timestamp"].min()


def test_compute_scale_pos_weight_matches_manual_ratio():
    y = pd.Series([0] * 97 + [1] * 3)
    weight = compute_scale_pos_weight(y)
    assert weight == 97 / 3


def test_tune_threshold_prefers_separable_scores():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 900 + [1] * 100)
    # Fraud cases score systematically higher -- a well-separated toy case.
    y_proba = np.concatenate(
        [rng.uniform(0, 0.4, 900), rng.uniform(0.6, 1.0, 100)]
    )
    result = tune_threshold(y_true, y_proba)
    assert 0.0 < result["f1_optimal_threshold"] < 1.0
    assert result["f1_optimal_precision"] > 0.8
    assert result["f1_optimal_recall"] > 0.8


def test_evaluate_at_threshold_confusion_matrix_sums_to_n():
    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.6, 0.9, 0.3])
    metrics = evaluate_at_threshold(y_true, y_proba, threshold=0.5)
    cm = metrics["confusion_matrix"]
    assert sum(cm.values()) == len(y_true)
    # Predictions @0.5: [0, 0, 1, 1, 0] vs true [0, 0, 0, 1, 1] -> TP=1, FP=1, FN=1, TN=2
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


def test_full_training_pipeline_produces_valid_metrics(trained_artifact_path):
    import joblib

    artifact = joblib.load(trained_artifact_path)
    assert "pipeline" in artifact
    assert 0.0 < artifact["threshold"] < 1.0
    assert artifact["feature_columns"]

    metrics_path = trained_artifact_path.parent / "metrics.json"
    assert metrics_path.exists()


def test_trained_pipeline_predicts_valid_probabilities(trained_artifact_path):
    import joblib

    artifact = joblib.load(trained_artifact_path)
    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]

    from fraud_detection.data.generate import generate_transactions
    from fraud_detection.features.engineering import add_derived_features

    fresh = generate_transactions(n_customers=50, n_transactions=100, days=10, seed=99)
    engineered = add_derived_features(fresh)
    proba = pipeline.predict_proba(engineered[feature_columns])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
