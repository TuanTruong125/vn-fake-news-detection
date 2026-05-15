from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app import api


# Build: Stub health service returning expected response structure for test mocking
class FakeService:
    def health(self):
        return {
            "status": "ok",
            "default_run_id": "run-1",
            "default_threshold_used": 0.5,
            "default_score_method": "predict_proba",
            "model_loaded": True,
            "cache_entries": 1,
            "cache_capacity": 3,
        }


# Build: Stub predict service returning schema-complete response for test assertions
class FakePredictService:
    def predict(self, **kwargs):
        return {
            "label_id": 1,
            "label_text": "fake",
            "model_family": kwargs["model_family"],
            "threshold_used": 0.5,
            "raw_score": 0.8,
            "raw_decision_score": None,
            "confidence": 0.8,
            "score_method": "predict_proba",
            "confidence_type": "probability",
            "is_probability": True,
            "run_id": kwargs["run_id"],
            "model_name": "dummy",
            "feature_set": "dummy",
            "text_variant": "text_clean",
            "original_length_chars": 3,
            "post_normalize_token_count": 1,
            "truncated": False,
            "truncated_tokens_from": 1,
            "truncated_tokens_to": 1,
            "segmentation_backend": "transformer",
            "segmentation_fallback_used": False,
            "preprocessing_mode": "normalize",
            "warnings": [],
            "explanation_available": False,
            "explanation_reason": None,
            "top_features_towards_fake": [],
            "top_features_towards_real": [],
            "explanation_decomposition": None,
            "processing_time_ms": 1.0,
        }


# Build: Stub registry returning model runs and best run flags for test data
class FakeRegistry:
    def list_runs_with_best(self):
        return [{"run_id": "run-1", "model_family": "ml", "is_best": True}]


# Test: Health endpoint returns degraded when service missing, ok when service available
def test_health_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(api, "service", None)
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "degraded"

    monkeypatch.setattr(api, "service", FakeService())
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"


# Test: Runs endpoint proxies registry output into response model structure
def test_list_runs_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(api, "model_registry", FakeRegistry())
    client = TestClient(api.app)

    response = client.get("/runs")
    assert response.status_code == 200
    assert response.json()["runs"][0]["run_id"] == "run-1"


# Test: Predict endpoint returns serialized service response with correct model family
def test_predict_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(api, "service", FakeService())
    monkeypatch.setattr(api, "predict_service", FakePredictService())
    client = TestClient(api.app)

    response = client.post(
        "/predict",
        json={"text": "abc", "model_family": "ml", "run_id": "run-1", "content_type": "news"},
    )

    assert response.status_code == 200
    assert response.json()["model_family"] == "ml"
