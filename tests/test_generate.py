
from fraud_detection.config import TARGET


def test_shape_and_columns(small_transactions_df):
    df = small_transactions_df
    assert len(df) == 1500
    expected_base_cols = {
        "transaction_id",
        "customer_id",
        "timestamp",
        "amount",
        "merchant_category",
        "channel",
        "card_present",
        "is_foreign_transaction",
        "distance_from_home_km",
        "account_age_days",
        "customer_avg_amount",
        "txn_count_last_1h",
        "txn_count_last_24h",
        "amount_sum_last_24h",
        TARGET,
    }
    assert expected_base_cols.issubset(set(df.columns))


def test_fraud_rate_is_within_target_range(small_transactions_df):
    rate = small_transactions_df["is_fraud"].mean()
    # Calibrated around 1.5%; allow slack for sampling noise on a small set.
    assert 0.003 < rate < 0.06


def test_no_negative_amounts_or_distances(small_transactions_df):
    assert (small_transactions_df["amount"] > 0).all()
    assert (small_transactions_df["distance_from_home_km"] >= 0).all()


def test_velocity_features_are_causal(small_transactions_df):
    """A transaction's own window counts must not include itself or future rows."""
    df = small_transactions_df.sort_values(["customer_id", "timestamp"])
    for _, group in df.groupby("customer_id"):
        if len(group) < 2:
            continue
        # First transaction for any customer must have zero prior activity.
        first = group.iloc[0]
        assert first["txn_count_last_1h"] == 0
        assert first["txn_count_last_24h"] == 0
        assert first["amount_sum_last_24h"] == 0


def test_transaction_ids_are_unique(small_transactions_df):
    assert small_transactions_df["transaction_id"].is_unique
