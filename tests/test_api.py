import pytest
from fastapi.testclient import TestClient

from fraud_detection.api.inference import model as inference_model
from fraud_detection.api.main import app

VALID_PAYLOAD = {
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


@pytest.fixture()
def client(trained_artifact_path, monkeypatch):
    # Point the app's startup loader at the session's tiny test artifact
    # instead of the real default path, so these tests are deterministic
    # regardless of whether a full model happens to be trained locally.
    monkeypatch.setattr("fraud_detection.api.main.MODEL_ARTIFACT_PATH", trained_artifact_path)
    with TestClient(app) as test_client:
        yield test_client
    inference_model.pipeline = None


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert 0.0 < body["threshold"] < 1.0


def test_predict_returns_valid_response(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["top_factors"] is None  # explain defaults to False


def test_predict_with_explain_returns_top_factors(client):
    response = client.post("/predict", params={"explain": True}, json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["top_factors"] is not None
    assert len(body["top_factors"]) > 0
    for factor in body["top_factors"]:
        assert "feature" in factor and "impact" in factor


def test_predict_rejects_negative_amount(client):
    bad_payload = {**VALID_PAYLOAD, "amount": -10}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_unknown_merchant_category(client):
    bad_payload = {**VALID_PAYLOAD, "merchant_category": "not_a_real_category"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_inconsistent_velocity_windows(client):
    bad_payload = {**VALID_PAYLOAD, "txn_count_last_1h": 10, "txn_count_last_24h": 2}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_without_loaded_model_returns_503(monkeypatch, tmp_path):
    inference_model.pipeline = None
    # Point the app's startup loader at a path that is guaranteed not to
    # exist, regardless of whether a real artifact happens to be trained
    # locally -- otherwise this test would flake depending on repo state.
    monkeypatch.setattr("fraud_detection.api.main.MODEL_ARTIFACT_PATH", tmp_path / "nonexistent.joblib")
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503
