
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
        "device_id",
        "ip_address",
        "txn_count_last_1h",
        "txn_count_last_24h",
        "amount_sum_last_24h",
        "device_distinct_customers_count",
        "ip_distinct_customers_count",
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


def test_ring_fraud_signal_exists(small_transactions_df):
    """At least one device/IP among fraud transactions is shared across
    several distinct customers -- the mule-network/card-testing pattern the
    graph-based features are built to catch. Without this, there would be
    nothing for that signal to learn from.
    """
    fraud = small_transactions_df[small_transactions_df["is_fraud"] == 1]
    device_fanout = fraud.groupby("device_id")["customer_id"].nunique()
    assert device_fanout.max() >= 4


def test_legitimate_household_sharing_exists(small_transactions_df):
    """At least one device among legitimate transactions is shared across a
    *small* number of distinct customers (a household), capped well below
    the fraud-ring fan-out. Without this, "shared device" would trivially
    equal "fraud" and the signal would be meaningless to learn.
    """
    legit = small_transactions_df[small_transactions_df["is_fraud"] == 0]
    device_fanout = legit.groupby("device_id")["customer_id"].nunique()
    assert device_fanout.max() >= 2
    assert device_fanout.max() < 4  # stays below ring scale by construction


def test_network_features_are_causal(small_transactions_df):
    """A device/IP's fan-out count must only reflect prior-or-current
    activity, never future rows -- the first transaction ever seen on a
    device must count exactly that one customer.
    """
    df = small_transactions_df.sort_values(["device_id", "timestamp"])
    for _, group in df.groupby("device_id"):
        first = group.iloc[0]
        assert first["device_distinct_customers_count"] == 1

    df = small_transactions_df.sort_values(["ip_address", "timestamp"])
    for _, group in df.groupby("ip_address"):
        first = group.iloc[0]
        assert first["ip_distinct_customers_count"] == 1


def test_network_fanout_counts_are_nondecreasing_over_time(small_transactions_df):
    """A device/IP's cumulative distinct-customer count can only grow (or
    stay flat) as time passes -- it must never decrease."""
    df = small_transactions_df.sort_values(["device_id", "timestamp"])
    for _, group in df.groupby("device_id"):
        counts = group["device_distinct_customers_count"].to_numpy()
        assert (counts[1:] >= counts[:-1]).all()
