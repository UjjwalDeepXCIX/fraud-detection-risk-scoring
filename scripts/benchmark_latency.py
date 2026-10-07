#!/usr/bin/env python
"""Benchmark end-to-end /predict latency against a running API instance.

Validates the sub-250ms real-time scoring target under concurrent-ish
sequential load using the actual HTTP path (not an in-process call), so the
numbers reflect what a caller would actually observe.

Usage:
    uvicorn fraud_detection.api.main:app &
    python scripts/benchmark_latency.py --n-requests 200 --url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import statistics
import time

import httpx

SAMPLE_PAYLOAD = {
    "amount": 245.30,
    "merchant_category": "online_retail",
    "channel": "online",
    "card_present": False,
    "is_foreign_transaction": False,
    "distance_from_home_km": 12.4,
    "account_age_days": 540,
    "customer_avg_amount": 90.0,
    "txn_count_last_1h": 1,
    "txn_count_last_24h": 3,
    "amount_sum_last_24h": 310.0,
}


def run_benchmark(url: str, n_requests: int, explain: bool) -> None:
    latencies_ms = []
    with httpx.Client(base_url=url, timeout=10.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        print(f"Health check: {health.json()}")

        for _ in range(n_requests):
            start = time.perf_counter()
            response = client.post("/predict", params={"explain": explain}, json=SAMPLE_PAYLOAD)
            elapsed_ms = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            latencies_ms.append(elapsed_ms)

    latencies_ms.sort()
    p50 = statistics.median(latencies_ms)
    p95 = latencies_ms[int(len(latencies_ms) * 0.95) - 1]
    p99 = latencies_ms[int(len(latencies_ms) * 0.99) - 1]
    print(f"Requests: {n_requests} (explain={explain})")
    print(f"p50={p50:.2f}ms  p95={p95:.2f}ms  p99={p99:.2f}ms  max={max(latencies_ms):.2f}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n-requests", type=int, default=200)
    parser.add_argument("--explain", action="store_true", help="Request SHAP top factors on every call")
    args = parser.parse_args()
    run_benchmark(args.url, args.n_requests, args.explain)


if __name__ == "__main__":
    main()
