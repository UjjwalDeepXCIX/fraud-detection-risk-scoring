"""Synthetic transaction data generator.

Produces a realistic, imbalanced card-transaction dataset with correlated
risk signals (velocity, geo-distance, account age, merchant category) so the
downstream feature engineering and modelling steps have genuine structure to
learn from rather than pure noise. This avoids depending on a licensed /
gated external dataset while keeping the pipeline fully reproducible.

Also includes device/IP linkage: most fraud is a single-account burst from an
unfamiliar device (no cross-account signal at all), a minority is a
shared-device ring spanning several distinct accounts, and a separate,
legitimate household scenario has ordinary customers genuinely sharing one
device -- so "this device is shared" is not by itself a fraud tell, only
sharing *combined with* the fraud block's other risk signals is. This is the
raw material the graph-based features and the knowledge-graph investigation
layer elsewhere in this project are built to use.
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

# Fraction of customers grouped into households of HOUSEHOLD_SIZE_RANGE members
# that genuinely share one device/IP for ordinary transactions. This is the
# legitimate positive-fan-out case: without it, any shared device would be a
# perfect (and therefore meaningless) fraud tell, since nothing innocent would
# ever produce one. See _generate_fraud_block for the fraud-side counterpart.
HOUSEHOLD_FRACTION = 0.06
HOUSEHOLD_SIZE_RANGE = (2, 4)  # upper bound exclusive, i.e. 2-3 members

# Fraction of fraud transactions generated as "ring" episodes -- several
# distinct compromised accounts transacting through one shared device/IP in a
# short window (card-testing / mule-network pattern). The remainder are
# "solo" episodes: a single account, a fresh device unlinked to anyone else.
RING_FRAUD_FRACTION = 0.25
RING_SIZE_RANGE = (4, 7)  # upper bound exclusive, i.e. 4-6 distinct customers


def _simulate_customers(n_customers: int, rng: np.random.Generator) -> pd.DataFrame:
    account_age_days = rng.exponential(scale=600, size=n_customers).clip(1, 4000).astype(int)
    avg_amount = rng.lognormal(mean=3.6, sigma=0.7, size=n_customers).clip(5, 2000)
    home_risk = rng.normal(0, 1, size=n_customers)  # latent customer-level risk factor

    device_id = np.array([f"DEV{idx:06d}" for idx in range(n_customers)], dtype=object)
    ip_address = np.array([f"IP{idx:06d}" for idx in range(n_customers)], dtype=object)

    # Household sharing: group a minority of customers so several genuinely
    # distinct accounts transact from the same device/IP for entirely
    # ordinary reasons (a shared family device), not fraud.
    n_household_members = round(n_customers * HOUSEHOLD_FRACTION)
    household_pool = rng.choice(n_customers, size=n_household_members, replace=False)
    cursor = 0
    while cursor < len(household_pool):
        size = min(int(rng.integers(*HOUSEHOLD_SIZE_RANGE)), len(household_pool) - cursor)
        if size < 2:
            break
        members = household_pool[cursor : cursor + size]
        device_id[members[1:]] = device_id[members[0]]
        ip_address[members[1:]] = ip_address[members[0]]
        cursor += size

    return pd.DataFrame(
        {
            "customer_id": [f"CUST{idx:06d}" for idx in range(n_customers)],
            "account_age_days": account_age_days,
            "customer_avg_amount": avg_amount,
            "device_id": device_id,
            "ip_address": ip_address,
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


def _assign_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal, cumulative-to-date count of distinct customers seen on the
    same device / IP, mirroring the backward-only discipline of the velocity
    features above: only prior-or-current activity is visible, never future
    rows. A higher count means more distinct accounts have funnelled through
    this device or IP by this point in time -- the fan-out signature a
    shared-device fraud ring produces (legitimate household sharing aside,
    which tops out at 2-3 by construction).
    """

    def _cumulative_distinct_customer_counts(frame: pd.DataFrame, key: str) -> np.ndarray:
        # Sort without dropping the original index, so counts computed in
        # (key, timestamp) order can be mapped back to the caller's row order
        # at the end via reindex -- reset_index(drop=True) here would discard
        # exactly the mapping needed to do that correctly.
        sorted_frame = frame.sort_values([key, "timestamp"])
        counts = pd.Series(0, index=sorted_frame.index, dtype=int)
        for _, group in sorted_frame.groupby(key, sort=False):
            seen: set = set()
            running = []
            for cust_id in group["customer_id"]:
                seen.add(cust_id)
                running.append(len(seen))
            counts.loc[group.index] = running
        return counts.reindex(frame.index).to_numpy()

    df = df.reset_index(drop=True)
    df["device_distinct_customers_count"] = _cumulative_distinct_customer_counts(df, "device_id")
    df["ip_distinct_customers_count"] = _cumulative_distinct_customer_counts(df, "ip_address")
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


def _build_solo_fraud_episodes(
    n_solo: int, customers: pd.DataFrame, days: int, start: pd.Timestamp, age_weight: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single-account fraud bursts (account-takeover style): roughly half are
    isolated transactions, the rest bursts of 2-4, each episode from a fresh
    device/IP unlinked to any other account -- an unfamiliar device used
    against one compromised account, not a shared-device ring.
    """
    episode_sizes: list[int] = []
    remaining = n_solo
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
    customer_idx = np.repeat(episode_customer_idx, episode_sizes)

    timestamps = np.empty(n_solo, dtype="datetime64[ns]")
    device_id = np.empty(n_solo, dtype=object)
    ip_address = np.empty(n_solo, dtype=object)
    cursor = 0
    for episode_num, size in enumerate(episode_sizes):
        day_offset = rng.integers(0, days)
        # ~40% of episodes are biased toward late-night hours (a common
        # fraud pattern), the rest spread normally.
        hour = rng.uniform(1, 5) if rng.random() < 0.4 else rng.uniform(0, 24)
        base_time = start + pd.Timedelta(days=int(day_offset)) + pd.Timedelta(hours=hour)
        offsets = np.array([0.0]) if size == 1 else np.sort(rng.uniform(0, 45 * 60, size=size))
        episode_times = base_time + pd.to_timedelta(offsets, unit="s")
        timestamps[cursor : cursor + size] = episode_times.to_numpy()
        device_id[cursor : cursor + size] = f"DEV-SOLO-{episode_num:06d}"
        ip_address[cursor : cursor + size] = f"IP-SOLO-{episode_num:06d}"
        cursor += size

    return customer_idx, timestamps, device_id, ip_address


def _build_ring_fraud_episodes(
    n_ring: int, customers: pd.DataFrame, days: int, start: pd.Timestamp, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared-device fraud rings: RING_SIZE_RANGE distinct customer accounts,
    one transaction each, all transacting through the same device/IP within
    a short window -- the card-testing / mule-network pattern that a
    row-by-row model cannot see, since the tell is in the relationship
    between transactions, not in any single transaction's own features.
    """
    if n_ring == 0:
        empty_obj = np.array([], dtype=object)
        return np.array([], dtype=int), np.array([], dtype="datetime64[ns]"), empty_obj, empty_obj

    ring_sizes: list[int] = []
    remaining = n_ring
    while remaining > 0:
        if remaining < RING_SIZE_RANGE[0]:
            # Too few transactions left for a standalone ring -- fold the
            # leftover into the previous ring rather than emit an
            # undersized one that would blur into the household signal.
            if ring_sizes:
                ring_sizes[-1] += remaining
            else:
                ring_sizes.append(remaining)
            remaining = 0
        else:
            size = min(int(rng.integers(*RING_SIZE_RANGE)), remaining)
            ring_sizes.append(size)
            remaining -= size

    customer_idx_parts = []
    timestamps = np.empty(n_ring, dtype="datetime64[ns]")
    device_id = np.empty(n_ring, dtype=object)
    ip_address = np.empty(n_ring, dtype=object)
    cursor = 0
    for ring_num, size in enumerate(ring_sizes):
        # Ring membership is unweighted -- a compromised device can land on
        # any account, not preferentially newer ones (unlike solo takeover).
        members = rng.choice(len(customers), size=size, replace=False)
        customer_idx_parts.append(members)

        day_offset = rng.integers(0, days)
        hour = rng.uniform(0, 24)
        base_time = start + pd.Timedelta(days=int(day_offset)) + pd.Timedelta(hours=hour)
        offsets = np.sort(rng.uniform(0, 60 * 60, size=size))  # within a 60-minute window
        episode_times = base_time + pd.to_timedelta(offsets, unit="s")
        timestamps[cursor : cursor + size] = episode_times.to_numpy()
        device_id[cursor : cursor + size] = f"DEV-RING-{ring_num:06d}"
        ip_address[cursor : cursor + size] = f"IP-RING-{ring_num:06d}"
        cursor += size

    customer_idx = np.concatenate(customer_idx_parts)
    return customer_idx, timestamps, device_id, ip_address


def _generate_fraud_block(
    n_fraud: int, customers: pd.DataFrame, days: int, start: pd.Timestamp, rng: np.random.Generator
) -> pd.DataFrame:
    """Fraudulent transactions are drawn from a distinctly different
    generative process rather than a noisy shift of the legitimate one:
    higher amounts relative to the victim's baseline, skewed toward
    higher-risk merchant categories and online/card-not-present channels,
    more often foreign, and frequently occurring in short bursts. Most
    fraud is a single-account burst from an unfamiliar device
    (``_build_solo_fraud_episodes``); a minority is a shared-device ring
    spanning several distinct accounts (``_build_ring_fraud_episodes``) --
    the pattern the graph-based features elsewhere in this project are
    specifically built to catch. Each signal still overlaps with the
    legitimate distribution so the problem stays learnable rather than
    trivially separable.
    """
    # Slightly favour newer accounts as solo-fraud victims without making
    # account age a deterministic tell. Ring membership is unweighted (see
    # _build_ring_fraud_episodes).
    age_weight = np.exp(-customers["account_age_days"].to_numpy() / 1500.0)
    age_weight = age_weight / age_weight.sum()

    n_ring = round(n_fraud * RING_FRAUD_FRACTION)
    n_solo = n_fraud - n_ring

    solo_customer_idx, solo_timestamps, solo_device_id, solo_ip = _build_solo_fraud_episodes(
        n_solo, customers, days, start, age_weight, rng
    )
    ring_customer_idx, ring_timestamps, ring_device_id, ring_ip = _build_ring_fraud_episodes(
        n_ring, customers, days, start, rng
    )

    fraud_customer_idx = np.concatenate([solo_customer_idx, ring_customer_idx])
    block = customers.iloc[fraud_customer_idx].reset_index(drop=True)
    block["timestamp"] = np.concatenate([solo_timestamps, ring_timestamps])
    block["device_id"] = np.concatenate([solo_device_id, ring_device_id])
    block["ip_address"] = np.concatenate([solo_ip, ring_ip])

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
    txns = _assign_network_features(txns)
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
        "device_id",
        "ip_address",
        "txn_count_last_1h",
        "txn_count_last_24h",
        "amount_sum_last_24h",
        "device_distinct_customers_count",
        "ip_distinct_customers_count",
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
