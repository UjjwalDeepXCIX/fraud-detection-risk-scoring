from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from fraud_detection.data.generate import generate_transactions
from fraud_detection.graph.build import build_entity_graph, save_graph
from fraud_detection.models.train import train


@pytest.fixture(scope="session")
def small_transactions_df() -> pd.DataFrame:
    """A small but structurally realistic dataset for fast unit tests."""
    return generate_transactions(n_customers=200, n_transactions=1500, days=30, seed=7)


@pytest.fixture(scope="session")
def small_entity_graph(small_transactions_df) -> nx.Graph:
    """Entity graph built from the same small dataset the model fixtures
    use, so graph/rule tests and model tests are exercising consistent data.
    """
    return build_entity_graph(small_transactions_df)


@pytest.fixture(scope="session")
def ring_device_id(small_transactions_df) -> str:
    """A real device id from the fixture data known to carry the ring
    signal (fan-out >= 4 among fraud transactions) -- see
    tests/test_generate.py::test_ring_fraud_signal_exists, which asserts
    this signal actually exists in this exact fixture.
    """
    fraud = small_transactions_df[small_transactions_df["is_fraud"] == 1]
    fanout = fraud.groupby("device_id")["customer_id"].nunique()
    return fanout[fanout >= 4].index[0]


@pytest.fixture(scope="session")
def graph_artifact_path(tmp_path_factory, small_entity_graph) -> Path:
    """Persist the fixture entity graph once per session, for tests that
    exercise the API's graph-loading path the same way trained_artifact_path
    does for the model.
    """
    tmp_dir = tmp_path_factory.mktemp("graph_artifacts")
    path = tmp_dir / "entity_graph.pkl"
    save_graph(small_entity_graph, path)
    return path


@pytest.fixture(scope="session")
def household_device_id(small_transactions_df) -> str:
    """A real device id from the fixture data known to carry the
    legitimate household-sharing signal (2-3 distinct customers, all
    non-fraud) -- see
    tests/test_generate.py::test_legitimate_household_sharing_exists.
    """
    legit = small_transactions_df[small_transactions_df["is_fraud"] == 0]
    fanout = legit.groupby("device_id")["customer_id"].nunique()
    candidates = fanout[(fanout >= 2) & (fanout < 4)]
    return candidates.index[0]


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
