"""Train/validation/test splitting and class-imbalance handling."""

from __future__ import annotations

import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    time_column: str = "timestamp",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically: train on the earliest transactions, evaluate on the
    most recent ones. Fraud detection is a forecasting problem in disguise --
    a random shuffle split would leak future behaviour into training and
    overstate accuracy, so the split respects time order instead.
    """
    df = df.sort_values(time_column).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Ratio of negative to positive samples, used to rebalance XGBoost's loss
    without discarding or synthesising any rows.
    """
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0:
        raise ValueError("No positive (fraud) samples found; cannot compute class weight.")
    return negatives / positives
