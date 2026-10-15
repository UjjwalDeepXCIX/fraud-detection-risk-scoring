"""Central configuration for paths, feature lists, and model defaults."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "transactions.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_ARTIFACT_PATH = MODELS_DIR / "fraud_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
GRAPH_ARTIFACT_PATH = MODELS_DIR / "entity_graph.pkl"
ABLATION_METRICS_PATH = MODELS_DIR / "ablation_metrics.json"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SHAP_SUMMARY_PATH = FIGURES_DIR / "shap_summary.png"

RANDOM_SEED = 42

MERCHANT_CATEGORIES = [
    "grocery",
    "restaurant",
    "gas_station",
    "electronics",
    "online_retail",
    "travel",
    "jewelry",
    "utilities",
    "entertainment",
    "healthcare",
]
# Categories more commonly abused in card-fraud schemes.
HIGH_RISK_CATEGORIES = {"electronics", "online_retail", "travel", "jewelry"}

CHANNELS = ["in_store", "online", "atm"]

# Graph-derived features: how many distinct customers have transacted on this
# device / IP so far (causal, cumulative-to-date). Kept as a named list, not
# folded directly into NUMERIC_FEATURES by value, so scripts/train_model.py
# can build a second pipeline with these excluded for an apples-to-apples
# ablation against the graph-aware model -- see scripts/ablate_network_features.py.
NETWORK_FEATURES = [
    "device_distinct_customers_count",
    "ip_distinct_customers_count",
]

NUMERIC_FEATURES = [
    "amount",
    "account_age_days",
    "customer_avg_amount",
    "amount_to_avg_ratio",
    "distance_from_home_km",
    "txn_count_last_1h",
    "txn_count_last_24h",
    "amount_sum_last_24h",
    "hour_sin",
    "hour_cos",
    *NETWORK_FEATURES,
]

CATEGORICAL_FEATURES = [
    "merchant_category",
    "channel",
    "card_present",
    "is_foreign_transaction",
]

TARGET = "is_fraud"

ID_COLUMNS = ["transaction_id", "customer_id", "timestamp"]

DEFAULT_XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}
