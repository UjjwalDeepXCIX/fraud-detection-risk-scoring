"""FastAPI application exposing the fraud-scoring model."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import networkx as nx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from fraud_detection.api.inference import ModelNotLoadedError, model
from fraud_detection.api.schemas import (
    HealthResponse,
    InvestigateRequest,
    InvestigationResponse,
    PredictionResponse,
    RelatedEntitiesResponse,
    RuleResultResponse,
    TransactionRequest,
)
from fraud_detection.config import GRAPH_ARTIFACT_PATH, MODEL_ARTIFACT_PATH
from fraud_detection.graph.build import load_graph
from fraud_detection.graph.investigate import investigate as run_investigation

logger = logging.getLogger(__name__)

graph: nx.Graph | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    try:
        model.load(MODEL_ARTIFACT_PATH)
        logger.info("Model loaded successfully; threshold=%.4f", model.threshold)
    except FileNotFoundError:
        logger.warning(
            "No model artifact found at %s. Run `python scripts/train_model.py` first; "
            "/predict will return 503 until a model is loaded.",
            MODEL_ARTIFACT_PATH,
        )

    try:
        graph = load_graph(GRAPH_ARTIFACT_PATH)
        logger.info("Entity graph loaded: %d nodes", graph.number_of_nodes())
    except FileNotFoundError:
        logger.warning(
            "No entity graph found at %s. Run `python scripts/build_graph.py` first; "
            "/investigate will return 503 until a graph is loaded.",
            GRAPH_ARTIFACT_PATH,
        )
        graph = None

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
        graph_loaded=graph is not None,
        graph_entity_count=graph.number_of_nodes() if graph is not None else None,
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


@app.post("/investigate", response_model=InvestigationResponse, tags=["scoring"])
def investigate(
    request: InvestigateRequest,
    explain: bool = Query(default=False, description="Include top SHAP contributing factors alongside the rule findings."),
) -> InvestigationResponse:
    """Neurosymbolic investigation: scores the transaction with the same
    model as /predict, but with device/IP fan-out looked up live from the
    entity graph rather than caller-supplied, then evaluates symbolic
    fraud-typology rules over that graph and fuses the two into one
    escalation decision. See fraud_detection.graph.investigate for the
    fusion policy.
    """
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run scripts/train_model.py first.")
    if graph is None:
        raise HTTPException(status_code=503, detail="Entity graph not loaded. Run scripts/build_graph.py first.")

    report = run_investigation(request, model, graph, explain=explain)
    return InvestigationResponse(
        transaction_id=report.transaction_id,
        fraud_probability=report.fraud_probability,
        model_risk_level=report.model_risk_level,
        final_risk_level=report.final_risk_level,
        escalated=report.escalated,
        threshold_used=report.threshold_used,
        matched_rules=[RuleResultResponse(**r.__dict__) for r in report.matched_rules],
        related_entities=RelatedEntitiesResponse(**report.related_entities),
        top_factors=report.top_factors,
        latency_ms=round(report.latency_ms, 2),
    )
