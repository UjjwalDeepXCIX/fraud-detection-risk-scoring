"""Synthetic transaction data generator.

Produces a realistic, imbalanced card-transaction dataset with correlated
risk signals (velocity, geo-distance, account age, merchant category) so the
downstream feature engineering and modelling steps have genuine structure to
learn from rather than pure noise. This avoids depending on a licensed /
gated external dataset while keeping the pipeline fully reproducible.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from fraud_detection.config import (
    CHANNELS,
    HIGH_RISK_CATEGORIES,
    MERCHANT_CATEGORIES,
    RANDOM_SEED,
    RAW_DATA_PATH,
)

logger = logging.getLogger(__name__)


def _simulate_customers(n_customers: int, rng: np.random.Generator) -> pd.DataFrame:
    account_age_days = rng.exponential(scale=600, size=n_customers).clip(1, 4000).astype(int)
    avg_amount = rng.lognormal(mean=3.6, sigma=0.7, size=n_customers).clip(5, 2000)
    home_risk = rng.normal(0, 1, size=n_customers)  # latent customer-level risk factor
    return pd.DataFrame(
        {
            "customer_id": [f"CUST{idx:06d}" for idx in range(n_customers)],
            "account_age_days": account_age_days,
            "customer_avg_amount": avg_amount,
            "_home_risk": home_risk,
        }
    )


def _assign_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling transaction-count / amount-sum features per customer.

    Mirrors what a real-time feature store would serve at scoring time: for
    each transaction, look back at the same customer's prior activity only
    (no leakage from future transactions).
    """
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    counts_1h = np.zeros(len(df), dtype=int)
    counts_24h = np.zeros(len(df), dtype=int)
    sums_24h = np.zeros(len(df), dtype=float)

    for _, group in df.groupby("customer_id", sort=False):
        idx = group.index.to_numpy()
        times = group["timestamp"].to_numpy()
        amounts = group["amount"].to_numpy()
        start_1h = 0
        start_24h = 0
        running_sum_24h = 0.0
        for i in range(len(idx)):
            # advance the 1h window pointer
            while times[i] - times[start_1h] > np.timedelta64(1, "h"):
                start_1h += 1
            # advance the 24h window pointer, removing amounts leaving the window
            while times[i] - times[start_24h] > np.timedelta64(24, "h"):
                running_sum_24h -= amounts[start_24h]
                start_24h += 1
            counts_1h[idx[i]] = i - start_1h
            counts_24h[idx[i]] = i - start_24h
            sums_24h[idx[i]] = running_sum_24h
            running_sum_24h += amounts[i]

    df["txn_count_last_1h"] = counts_1h
    df["txn_count_last_24h"] = counts_24h
    df["amount_sum_last_24h"] = sums_24h
    return df


def _generate_legit_block(
    n_legit: int, customers: pd.DataFrame, days: int, start: pd.Timestamp, rng: np.random.Generator
) -> pd.DataFrame:
    customer_idx = rng.integers(0, len(customers), size=n_legit)
    block = customers.iloc[customer_idx].reset_index(drop=True)

    offsets_seconds = rng.uniform(0, days * 24 * 3600, size=n_legit)
    block["timestamp"] = start + pd.to_timedelta(offsets_seconds, unit="s")

    # Amount centred on the customer's personal baseline; a small fraction of
    # otherwise-ordinary large purchases keeps "high amount" from being a
    # perfect fraud tell on its own.
    amount_noise = rng.lognormal(mean=0.0, sigma=0.5, size=n_legit)
    spike = rng.random(n_legit) < 0.02
    spike_multiplier = np.where(spike, rng.uniform(2, 4, size=n_legit), 1.0)
    block["amount"] = (block["customer_avg_amount"] * amount_noise * spike_multiplier).round(2)

    block["merchant_category"] = rng.choice(MERCHANT_CATEGORIES, size=n_legit)
    block["channel"] = rng.choice(CHANNELS, size=n_legit, p=[0.55, 0.37, 0.08])
    card_present_prob = np.where(block["channel"] == "online", 0.03, 0.9)
    block["card_present"] = rng.random(n_legit) < card_present_prob

    is_foreign_prob = 0.05 + 0.1 * (block["merchant_category"] == "travel")
    block["is_foreign_transaction"] = rng.random(n_legit) < is_foreign_prob
    base_distance = rng.exponential(scale=8, size=n_legit)
    foreign_distance = rng.uniform(300, 9000, size=n_legit)
    block["distance_from_home_km"] = np.where(
        block["is_foreign_transaction"], foreign_distance, base_distance
    ).round(1)

    block["is_fraud"] = 0
    return block


def _generate_fraud_block(
    n_fraud: int, customers: pd.DataFrame, days: int, start: pd.Timestamp, rng: np.random.Generator
) -> pd.DataFrame:
    """Fraudulent transactions are drawn from a distinctly different
    generative process rather than a noisy shift of the legitimate one:
    higher amounts relative to the victim's baseline, skewed toward
    higher-risk merchant categories and online/card-not-present channels,
    more often foreign, and frequently occurring in short bursts (several
    transactions against the same compromised account within minutes) --
    the same patterns real card-testing and account-takeover fraud exhibit.
    Each signal still overlaps with the legitimate distribution so the
    problem stays learnable rather than trivially separable.
    """
    # Slightly favour newer accounts as fraud victims without making account
    # age a deterministic tell.
    age_weight = np.exp(-customers["account_age_days"].to_numpy() / 1500.0)
    age_weight = age_weight / age_weight.sum()

    # Build fraud "episodes": roughly half of fraud transactions are isolated,
    # the rest occur in bursts of 2-4 against the same account.
    episode_sizes: list[int] = []
    remaining = n_fraud
    while remaining > 0:
        if remaining == 1 or rng.random() < 0.5:
            size = 1
        else:
            size = int(rng.integers(2, 5))
        size = min(size, remaining)
        episode_sizes.append(size)
        remaining -= size

    n_episodes = len(episode_sizes)
    episode_customer_idx = rng.choice(len(customers), size=n_episodes, p=age_weight)

    fraud_customer_idx = np.repeat(episode_customer_idx, episode_sizes)
    block = customers.iloc[fraud_customer_idx].reset_index(drop=True)

    # Timestamps: one base time per episode, with burst transactions landing
    # within a tight window after it. ~40% of episodes are biased toward
    # late-night hours (a common fraud pattern), the rest spread normally.
    timestamps = np.empty(n_fraud, dtype="datetime64[ns]")
    cursor = 0
    for size in episode_sizes:
        day_offset = rng.integers(0, days)
        if rng.random() < 0.4:
            hour = rng.uniform(1, 5)
        else:
            hour = rng.uniform(0, 24)
        base_time = start + pd.Timedelta(days=int(day_offset)) + pd.Timedelta(hours=hour)
        if size == 1:
            offsets = np.array([0.0])
        else:
            offsets = np.sort(rng.uniform(0, 45 * 60, size=size))  # within a 45-minute window
        episode_times = base_time + pd.to_timedelta(offsets, unit="s")
        timestamps[cursor : cursor + size] = episode_times.to_numpy()
        cursor += size
    block["timestamp"] = timestamps

    # Amount: heavy-tailed multiplier on the victim's personal baseline --
    # fraud tends to cash out well above the account's usual spend, but with
    # enough spread to overlap the legitimate "big purchase" tail.
    amount_multiplier = rng.lognormal(mean=1.6, sigma=0.9, size=n_fraud).clip(1.0, 60)
    amount_noise = rng.lognormal(mean=0.0, sigma=0.3, size=n_fraud)
    block["amount"] = (block["customer_avg_amount"] * amount_multiplier * amount_noise).round(2)

    is_high_risk_cat = rng.random(n_fraud) < 0.55
    block["merchant_category"] = np.where(
        is_high_risk_cat,
        rng.choice(list(HIGH_RISK_CATEGORIES), size=n_fraud),
        rng.choice(MERCHANT_CATEGORIES, size=n_fraud),
    )
    block["channel"] = rng.choice(CHANNELS, size=n_fraud, p=[0.25, 0.65, 0.10])
    card_present_prob = np.where(block["channel"] == "online", 0.03, 0.5)
    block["card_present"] = rng.random(n_fraud) < card_present_prob

    block["is_foreign_transaction"] = rng.random(n_fraud) < 0.30
    base_distance = rng.exponential(scale=15, size=n_fraud)
    foreign_distance = rng.uniform(300, 9000, size=n_fraud)
    block["distance_from_home_km"] = np.where(
        block["is_foreign_transaction"], foreign_distance, base_distance
    ).round(1)

    block["is_fraud"] = 1
    return block


def generate_transactions(
    n_customers: int = 6000,
    n_transactions: int = 60000,
    days: int = 90,
    base_fraud_rate: float = 0.015,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a synthetic transaction dataset with a genuinely learnable
    fraud signal.

    Legitimate and fraudulent transactions are drawn from two distinct
    generative processes (see ``_generate_legit_block`` /
    ``_generate_fraud_block``) and interleaved, which is both more realistic
    than a single noisy shared formula and gives a model something concrete
    to learn: different amount distributions, channel/category mix,
    card-present rates, geo-distance and burst velocity patterns.
    """
    rng = np.random.default_rng(seed)
    customers = _simulate_customers(n_customers, rng)
    start = pd.Timestamp("2025-01-01")

    n_fraud = max(1, round(n_transactions * base_fraud_rate))
    n_legit = n_transactions - n_fraud

    legit_block = _generate_legit_block(n_legit, customers, days, start, rng)
    fraud_block = _generate_fraud_block(n_fraud, customers, days, start, rng)

    txns = pd.concat([legit_block, fraud_block], ignore_index=True)
    txns["transaction_id"] = [f"TXN{idx:08d}" for idx in range(len(txns))]

    txns = _assign_velocity_features(txns)
    txns = txns.drop(columns=["_home_risk"])

    ordered_cols = [
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
        "is_fraud",
    ]
    return txns[ordered_cols].sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-customers", type=int, default=6000)
    parser.add_argument("--n-transactions", type=int, default=60000)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--fraud-rate", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=str, default=str(RAW_DATA_PATH))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = generate_transactions(
        n_customers=args.n_customers,
        n_transactions=args.n_transactions,
        days=args.days,
        base_fraud_rate=args.fraud_rate,
        seed=args.seed,
    )
    out_path = RAW_DATA_PATH if args.output == str(RAW_DATA_PATH) else args.output
    from pathlib import Path

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    fraud_rate = df["is_fraud"].mean()
    logger.info("Wrote %d transactions to %s (fraud rate=%.3f%%)", len(df), out_path, fraud_rate * 100)


if __name__ == "__main__":
    main()
