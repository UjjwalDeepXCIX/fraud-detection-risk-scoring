# Real-Time Fraud Detection & Transaction Risk Scoring

An end-to-end fraud detection system: synthetic transaction data generation, feature engineering, a cross-validated XGBoost classifier with tuned decision thresholds, SHAP explainability, a **knowledge-graph-grounded neurosymbolic layer** for patterns a row-by-row classifier cannot see, and a FastAPI service for real-time scoring.

No external dataset or API key is required — the pipeline generates its own realistic, imbalanced transaction data and is fully reproducible from a single command.

> **This is not just an XGBoost-plus-SHAP classifier.** A row-by-row model can score one transaction at a time, but it structurally cannot see that the same device just transacted against four other accounts in the last hour — the signal lives in the *relationship between* transactions. This project adds a knowledge graph of transaction entities and a symbolic rule layer on top of the model specifically to catch that, with a measured, real ablation showing a **+23.5 point recall lift** on exactly the fraud pattern it exists to catch. See [Beyond a classifier: the neurosymbolic investigation layer](#beyond-a-classifier-the-neurosymbolic-investigation-layer) below.

## Beyond a classifier: the neurosymbolic investigation layer

XGBoost scores each transaction independently — it structurally cannot know that five "normal-looking" transactions across five *different* customer accounts all used the same device in the last hour, which is a classic card-testing / mule-network signature. The signal lives in the relationship *between* transactions, not in any single transaction's own features. This project adds a small knowledge graph of transaction entities (customers, devices, IPs) so that pattern is actually visible, combining a feature the model learns from with an explicit, auditable rule layer on top. It's modeled on the core idea behind Knowledge-Augmented Generation (KAG) — ground a reasoning step in a retrieved, structured, symbolic knowledge representation rather than relying on an opaque score alone — adapted from KAG's usual LLM/RAG setting to what actually fits a tabular fraud domain: a knowledge graph and compliance-style rules, not a vector index and a document store.

### Design: hybrid, not either extreme

Two narrower designs were considered and rejected in favour of a hybrid:

- **Rule-only** (model untouched, a rule engine reads the graph as a second opinion) is simple and needs no retraining, but the model itself never learns anything from device/IP fan-out — a second opinion stapled on afterward, not integrated reasoning.
- **Learned-only** (fold graph-derived features into the model, no explicit rules) gives the model real signal, but loses the auditable "rule X fired because Y" explanation entirely — back to one opaque gradient-boosted number.
- **Hybrid (what's implemented):** two causal, graph-derived features (`device_distinct_customers_count`, `ip_distinct_customers_count`) are fed into XGBoost so it genuinely learns from fan-out, *and* a thin symbolic rule layer sits on top for hard, auditable escalation — mirroring how real fraud stacks combine network features feeding a statistical model with a compliance-style rule engine that can override it.

### Avoiding a tautological result

If only fraud transactions ever shared a device, "shared device" would trivially equal "fraud," and neither the model nor the rules would be learning a pattern — just rediscovering the label. The data generator (`fraud_detection.data.generate`) deliberately includes a **legitimate household scenario**: ~6% of customers are grouped into households of 2-3 that genuinely share one device for ordinary, low-value transactions. Fraud rings use a *larger* fan-out (4-6 distinct customers, `RING_FANOUT_THRESHOLD` in `graph/rules.py`) *combined with* the existing fraud block's elevated-amount/foreign/velocity profile — fan-out alone isn't the tell. The symbolic rule's threshold is set above household scale specifically so it does not fire on the legitimate case; `tests/test_rules.py::test_shared_device_ring_does_not_fire_on_legitimate_household` asserts exactly that.

### Ablation: does it actually help?

Two otherwise-identical pipelines were trained — one with the two network features, one without — and compared on recall specifically for the synthetic ring-fraud subset of the test split (`scripts/ablate_network_features.py`, results in `models/ablation_metrics.json`):

| | Ring-fraud recall | Overall precision | Overall recall | ROC-AUC |
|---|---|---|---|---|
| **With network features** | **71.9%** | 0.824 | 0.760 | 0.974 |
| Without network features | 48.4% | 0.785 | 0.662 | 0.958 |

A +23.5 point lift in ring-fraud recall from two features, on 64 held-out ring-fraud test transactions — and overall metrics improve too, since ring fraud is 25% of all fraud in this dataset. Real numbers from an actual run, not an assumed improvement.

### A real example: the fusion overriding the model's own score

```
POST /investigate   amount=$48, grocery, in-store, card present, domestic
                     device_id = a real device from the training data

Model's own score:    0.763 probability -> "medium" (below the 0.866 threshold)
Symbolic rule:         shared_device_ring FIRED -- 5 distinct customers on this device
Final decision:        "high" (escalated: true)
```

An otherwise unremarkable-looking transaction — small amount, domestic, card present, grocery — that the model alone would rate "medium" and let through, gets escalated to "high" because the graph layer sees something the single transaction's own features cannot: this device has touched five different accounts. This is the concrete case the hybrid design exists for. A matching legitimate case (the same mundane transaction, on a real 2-customer household device) does **not** escalate — `final_risk_level` falls back to whatever the model itself says, confirming the rule isn't just firing on any shared device.

### New endpoint

`POST /investigate` takes everything `/predict` does, plus raw `device_id`/`ip_address` (the two network-count fields are looked up live from the graph, not caller-supplied — a caller can't under-report fan-out to get a lower score). Returns the model's score, which symbolic rules fired and why, the related entities retrieved from the graph, and the fused `final_risk_level`. `/predict` itself is unchanged except for two new *optional*, default-1 fields, so existing callers are unaffected.

## Why synthetic data

Public fraud datasets (e.g. the Kaggle credit-card fraud set) are either PCA-anonymised, which defeats feature engineering, or gated behind licenses that don't belong in a public repo. Instead, `fraud_detection.data.generate` builds 60,000+ transactions from two distinct generative processes — legitimate behaviour and fraud (card-testing bursts, account-takeover patterns, elevated amounts, foreign/card-not-present skew) — so the resulting ~1.5% fraud rate carries genuine, learnable signal rather than being random noise with a label slapped on.

## Results

Trained via `scripts/train_model.py` on a chronological 70/15/15 train/validation/test split (train on the past, evaluate on the future — a random shuffle would leak information a real-time system would never have):

| Split | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 5-fold CV (train) | 0.583 ± 0.019 | 0.829 ± 0.012 | 0.685 ± 0.014 | 0.984 ± 0.003 |
| Held-out test | 0.824 | 0.760 | 0.791 | 0.974 |

- **Decision threshold** tuned on the validation split (not the default 0.5) by maximising F1 on the precision-recall curve: 0.866. A higher-recall operating point (0.81 recall at 0.75 precision) is also computed, for teams that would rather review more false positives than miss fraud.
- **PR-AUC (average precision):** 0.806 — the more honest summary statistic at a ~1.5% base rate, where ROC-AUC alone can look deceptively strong.
- **Inference latency:** 5.8ms median for a single prediction in-process; **p50 13.2ms / p95 14.9ms / p99 15.7ms** end-to-end over real HTTP requests (see [Latency benchmark](#latency-benchmark)) — comfortably inside a 250ms real-time scoring budget.

Exact numbers regenerate with the seeded data (`--seed 42` default) via the commands below, and are written to `models/metrics.json`.

## Architecture

```
            ┌─────────────────────┐
 generate   │  synthetic transactions │
 ─────────► │  (legit + fraud mix)    │──► data/raw/transactions.csv
            └─────────────────────┘
                        │
                        ▼
            ┌─────────────────────┐
            │  feature engineering   │  ratios, cyclical time encoding,
            │  (shared train/serve)  │  scaling, one-hot encoding
            └─────────────────────┘
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
  ┌──────────────────┐   ┌──────────────────────┐
  │ time-based split  │   │ class-imbalance       │
  │ train/val/test    │   │ handling (scale_pos_  │
  └──────────────────┘   │ weight)                │
            │             └──────────────────────┘
            ▼
  ┌──────────────────────┐     ┌─────────────────┐
  │ XGBoost + 5-fold CV   │────►│ SHAP explainer   │
  │ + threshold tuning    │     │ (global + local) │
  └──────────────────────┘     └─────────────────┘
            │
            ▼
   models/fraud_model.joblib
            │
            ▼
  ┌──────────────────────────────┐
  │ FastAPI  /predict  /health    │  Pydantic-validated input,
  │ (loaded once, reused per req) │  sub-250ms scoring
  └──────────────────────────────┘
```

The same `add_derived_features` + preprocessing pipeline is used at training and inference time (`src/fraud_detection/features/engineering.py`), which is the most common source of train/serve skew in real ML systems — there's only one code path. See [Beyond a classifier: the neurosymbolic investigation layer](#beyond-a-classifier-the-neurosymbolic-investigation-layer) above for the graph-based layer that sits on top of this pipeline.

## Project structure

```
src/fraud_detection/
  config.py              paths, feature lists, model defaults
  data/
    generate.py           synthetic transaction generator
    preprocessing.py      time-based split, class-imbalance weighting
  features/
    engineering.py         derived features + sklearn preprocessing pipeline
  models/
    train.py               training orchestration (CV, fit, threshold tuning, persist)
    evaluate.py             metrics + threshold tuning on the PR curve
    explain.py              SHAP wrapper (global summary plot + per-prediction factors)
  api/
    schemas.py              Pydantic request/response models
    inference.py             model loading + single-transaction prediction
    main.py                  FastAPI app (/predict, /investigate, /health)
  graph/
    build.py                 entity graph construction (customers, devices, IPs)
    retrieve.py               fan-out counts + k-hop related-entity retrieval
    rules.py                   symbolic fraud-typology rules (shared_device_ring, ...)
    investigate.py              fuses the model's score with the rule layer
scripts/                   thin CLI entrypoints, HTTP latency benchmark, network-feature ablation
tests/                     pytest suite (generator, features, model, graph, rules, API)
```

## Quickstart

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e . --no-deps

python scripts/generate_data.py          # writes data/raw/transactions.csv (60k rows)
python scripts/train_model.py            # trains, evaluates, saves models/fraud_model.joblib
python scripts/build_graph.py            # builds the entity graph for /investigate
pytest -v                                # 47 tests: generator, features, model, graph, rules, API

uvicorn fraud_detection.api.main:app --reload
# → http://127.0.0.1:8000/docs
```

### Scoring a transaction

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "amount": 1480.50,
  "merchant_category": "electronics",
  "channel": "online",
  "card_present": false,
  "is_foreign_transaction": true,
  "distance_from_home_km": 6200.0,
  "account_age_days": 42,
  "customer_avg_amount": 85.0,
  "txn_count_last_1h": 3,
  "txn_count_last_24h": 7,
  "amount_sum_last_24h": 2100.0
}'
```

Add `?explain=true` to also get the top SHAP-ranked contributing factors for that prediction — kept opt-in since computing SHAP values adds latency that a hot-path scoring call doesn't need, but a flagged-transaction review queue would want.

### Investigating a transaction (neurosymbolic layer)

```bash
curl -X POST http://127.0.0.1:8000/investigate -H "Content-Type: application/json" -d '{
  "amount": 48.0,
  "merchant_category": "grocery",
  "channel": "in_store",
  "card_present": true,
  "is_foreign_transaction": false,
  "distance_from_home_km": 2.0,
  "account_age_days": 800,
  "customer_avg_amount": 50.0,
  "txn_count_last_1h": 0,
  "txn_count_last_24h": 1,
  "amount_sum_last_24h": 40.0,
  "device_id": "<a device_id from data/raw/transactions.csv>",
  "ip_address": "<the matching ip_address for that row>"
}'
```

Returns the model's own score, which symbolic rules fired (with evidence -- which customers share that device), the related entities retrieved from the graph, and the fused `final_risk_level`. Reproduce the ablation comparing the model with and without the graph-derived features:

```bash
python scripts/ablate_network_features.py   # writes models/ablation_metrics.json
```

### Latency benchmark

```bash
uvicorn fraud_detection.api.main:app &
python scripts/benchmark_latency.py --n-requests 200
```

Measures actual round-trip HTTP latency against the running service, not an in-process shortcut.

## Docker

```bash
python scripts/train_model.py            # produces models/fraud_model.joblib first
docker compose up --build
curl http://127.0.0.1:8000/health
```

The compose file mounts `./models` read-only into the container so the image doesn't need rebuilding every time the model is retrained.

## Design notes

- **Time-based split, not random shuffle.** Fraud detection is a forecasting problem — training on a random shuffle would let the model see transactions chronologically after the ones it's evaluated on, which no real-time system ever has access to.
- **Class imbalance via `scale_pos_weight`**, not oversampling. At a ~1.5% fraud rate, reweighting XGBoost's loss keeps every real row intact and avoids the duplicate-aware overfitting that naive oversampling (and to a lesser extent SMOTE) can introduce on tree ensembles.
- **Threshold tuning on a held-out validation split**, evaluated on a separate, untouched test split — the 0.5 default cutoff has no special meaning for a model trained with class reweighting, and tuning and evaluating on the same data would overstate the result.
- **Input validation at the API boundary.** Pydantic enums for categorical fields, bounded numeric ranges, and a cross-field check (`txn_count_last_24h >= txn_count_last_1h`) reject malformed requests with a 422 before they reach the model.
- **Escalation, not a blended average.** The neurosymbolic fusion policy (`graph/investigate.py`) lets a fired symbolic rule force `final_risk_level="high"` outright, rather than averaging it into the model's probability. A known ring pattern is treated as a hard override a single number shouldn't be allowed to quietly out-vote — the real example above shows the model alone would have under-rated a ring transaction as "medium."
- **The caller can't under-report graph signal.** `/investigate` looks up `device_distinct_customers_count` / `ip_distinct_customers_count` from the graph itself rather than trusting whatever the request body says — the fields exist on `/predict`'s schema for callers with their own feature-store access, but `/investigate` always uses the graph's real value, confirmed by `test_investigate_scores_using_graph_fanout_not_caller_supplied_value`.
