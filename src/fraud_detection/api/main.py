"""FastAPI application exposing the fraud-scoring model."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from fraud_detection.api.inference import ModelNotLoadedError, model
from fraud_detection.api.schemas import HealthResponse, PredictionResponse, TransactionRequest
from fraud_detection.config import MODEL_ARTIFACT_PATH

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model.load(MODEL_ARTIFACT_PATH)
        logger.info("Model loaded successfully; threshold=%.4f", model.threshold)
    except FileNotFoundError:
        logger.warning(
            "No model artifact found at %s. Run `python scripts/train_model.py` first; "
            "/predict will return 503 until a model is loaded.",
            MODEL_ARTIFACT_PATH,
        )
    yield


app = FastAPI(
    title="Transaction Risk Scoring API",
    description="Real-time fraud risk scoring for card transactions, backed by a gradient-boosted model with SHAP explainability.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model.is_loaded else "degraded",
        model_loaded=model.is_loaded,
        model_trained_at=model.trained_at,
        threshold=model.threshold,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(
    transaction: TransactionRequest,
    explain: bool = Query(
        default=False,
        description="Include top SHAP contributing factors (adds latency; typically used for flagged/high-risk review, not the hot path).",
    ),
) -> PredictionResponse:
    try:
        return model.predict(transaction, explain=explain)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
