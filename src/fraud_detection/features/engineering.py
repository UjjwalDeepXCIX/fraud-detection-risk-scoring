"""Feature engineering and preprocessing pipeline construction.

Derived features and the preprocessing pipeline are shared between training
and inference so the exact same transformation is applied in both places --
the single most common source of training/serving skew in ML systems.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_detection.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns (ratios, cyclical time encoding) in place-safe fashion.

    Expects raw columns produced by ``fraud_detection.data.generate`` (or an
    equivalent real feature-store extract) and returns a copy with the
    derived columns appended.
    """
    df = df.copy()

    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["amount_to_avg_ratio"] = (df["amount"] / df["customer_avg_amount"].clip(lower=1)).clip(
        upper=100
    )

    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    for col in ["card_present", "is_foreign_transaction"]:
        if df[col].dtype != object:
            df[col] = df[col].astype(str)

    return df


def build_preprocessor() -> ColumnTransformer:
    """Return the sklearn ColumnTransformer used ahead of the classifier."""
    numeric_pipeline = Pipeline(steps=[("scale", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def get_feature_columns() -> list[str]:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES
