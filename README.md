# Real-Time Fraud Detection & Transaction Risk Scoring

An end-to-end fraud detection system: synthetic transaction data generation, feature engineering, a cross-validated XGBoost classifier with tuned decision thresholds, SHAP explainability, and a FastAPI service for real-time scoring.

No external dataset or API key is required — the pipeline generates its own realistic, imbalanced transaction data and is fully reproducible from a single command.

## Why synthetic data

Public fraud datasets (e.g. the Kaggle credit-card fraud set) are either PCA-anonymised, which defeats feature engineering, or gated behind licenses that don't belong in a public repo. Instead, `fraud_detection.data.generate` builds 60,000+ transactions from two distinct generative processes — legitimate behaviour and fraud (card-testing bursts, account-takeover patterns, elevated amounts, foreign/card-not-present skew) — so the resulting ~1.5% fraud rate carries genuine, learnable signal rather than being random noise with a label slapped on.

## Results

Trained via `scripts/train_model.py` on a chronological 70/15/15 train/validation/test split (train on the past, evaluate on the future — a random shuffle would leak information a real-time system would never have):

| Split | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 5-fold CV (train) | 0.638 ± 0.006 | 0.828 ± 0.031 | 0.721 ± 0.014 | 0.983 ± 0.004 |
| Held-out test | 0.805 | 0.728 | 0.764 | 0.985 |

- **Decision threshold** tuned on the validation split (not the default 0.5) by maximising F1 on the precision-recall curve: 0.858. A higher-recall operating point (0.80 recall at 0.68 precision) is also computed, for teams that would rather review more false positives than miss fraud.
- **PR-AUC (average precision):** 0.829 — the more honest summary statistic at a ~1.5% base rate, where ROC-AUC alone can look deceptively strong.
- **Inference latency:** 5.6ms median for a single prediction in-process; **p50 12.5ms / p95 15.1ms / p99 23.5ms** end-to-end over real HTTP requests (see [Latency benchmark](#latency-benchmark)) — comfortably inside a 250ms real-time scoring budget.

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

The same `add_derived_features` + preprocessing pipeline is used at training and inference time (`src/fraud_detection/features/engineering.py`), which is the most common source of train/serve skew in real ML systems — there's only one code path.

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
    main.py                  FastAPI app (/predict, /health)
scripts/                   thin CLI entrypoints + HTTP latency benchmark
tests/                     pytest suite (generator, features, model, API)
```

## Quickstart

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e . --no-deps

python scripts/generate_data.py          # writes data/raw/transactions.csv (60k rows)
python scripts/train_model.py            # trains, evaluates, saves models/fraud_model.joblib
pytest -v                                # 23 tests: generator, features, model, API

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
