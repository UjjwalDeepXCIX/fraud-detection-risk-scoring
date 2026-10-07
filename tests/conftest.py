from pathlib import Path

import pandas as pd
import pytest

from fraud_detection.data.generate import generate_transactions
from fraud_detection.models.train import train


@pytest.fixture(scope="session")
def small_transactions_df() -> pd.DataFrame:
    """A small but structurally realistic dataset for fast unit tests."""
    return generate_transactions(n_customers=200, n_transactions=1500, days=30, seed=7)


@pytest.fixture(scope="session")
def trained_artifact_path(tmp_path_factory, small_transactions_df) -> Path:
    """Train a tiny model once per test session and return its artifact path.

    Shared across the model and API tests so the (relatively) expensive
    train/SHAP step only runs a single time.
    """
    tmp_dir = tmp_path_factory.mktemp("model_artifacts")
    data_path = tmp_dir / "transactions.csv"
    small_transactions_df.to_csv(data_path, index=False)

    model_path = tmp_dir / "fraud_model.joblib"
    metrics_path = tmp_dir / "metrics.json"
    train(data_path=data_path, model_output=model_path, metrics_output=metrics_path, skip_shap=False)
    return model_path
