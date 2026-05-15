from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app import api
from src.app.inference_dl import DLInferenceEngine
from src.app.inference_ml import MLInferenceEngine
from src.app.services.predict_service import PredictServiceError


# Stable health-service stub for API contract tests.
class _DummyService:
    # Return a stable health payload for API contract tests.
    def health(self):
        return {
            "status": "ok",
            "default_run_id": "x",
            "default_threshold_used": 0.5,
            "default_score_method": "predict_proba",
            "model_loaded": True,
            "cache_entries": 1,
            "cache_capacity": 3,
        }


# Lightweight predict-service stub used to drive API contract branches.
class _DummyPredictService:
    # Simulate success and selected error modes for /predict contract coverage.
    def __init__(self, behavior: str = "ok"):
        self.behavior = behavior

    # Return a fixed prediction payload or raise the requested service error.
    def predict(self, **kwargs):
        if self.behavior == "not_found":
            raise PredictServiceError("run_id not found", status_code=404, error_code="RUN_ID_NOT_FOUND")
        if self.behavior == "mismatch":
            raise PredictServiceError("mismatch", status_code=422, error_code="RUN_ID_MODEL_FAMILY_MISMATCH")
        if self.behavior == "internal":
            raise PredictServiceError("artifact missing", status_code=500, error_code="ARTIFACT_MISSING")
        return {
            "label_id": 1,
            "label_text": "fake",
            "model_family": kwargs["model_family"],
            "threshold_used": 0.7,
            "raw_score": 0.8,
            "raw_decision_score": None,
            "confidence": 0.8,
            "score_method": "prob_fake" if kwargs["model_family"] == "dl" else "predict_proba",
            "confidence_type": "probability",
            "is_probability": True,
            "run_id": kwargs["run_id"],
            "model_name": "dummy",
            "feature_set": "dummy",
            "text_variant": "text_clean",
            "original_length_chars": 10,
            "post_normalize_token_count": 5,
            "truncated": False,
            "truncated_tokens_from": 5,
            "truncated_tokens_to": 5,
            "segmentation_backend": "transformer",
            "segmentation_fallback_used": False,
            "preprocessing_mode": "normalize",
            "warnings": [],
            "explanation_available": False,
            "explanation_reason": "n/a",
            "top_features_towards_fake": [],
            "top_features_towards_real": [],
            "explanation_decomposition": None,
            "processing_time_ms": 1.0,
        }


# Build a test client with injectable API service stubs.
def _client(behavior: str = "ok") -> TestClient:
    api.service = _DummyService()
    api.predict_service = _DummyPredictService(behavior=behavior)
    return TestClient(api.app)


# Validate the happy-path /predict response for the ML family.
def test_predict_happy_path_ml():
    c = _client("ok")
    r = c.post("/predict", json={"text": "abc", "model_family": "ml", "run_id": "r1", "content_type": "news"})
    assert r.status_code == 200
    assert r.json()["model_family"] == "ml"


# Validate the happy-path /predict response for the DL family.
def test_predict_happy_path_dl():
    c = _client("ok")
    r = c.post("/predict", json={"text": "abc", "model_family": "dl", "run_id": "r2", "content_type": "social"})
    assert r.status_code == 200
    assert r.json()["model_family"] == "dl"


# Validate request schema enforcement for missing required fields.
def test_predict_validation_missing_required_fields():
    c = _client("ok")
    r = c.post("/predict", json={"text": "abc"})
    assert r.status_code == 422


# Validate request validation for model family, content type, and top_k bounds.
def test_predict_validation_model_family_content_type_topk():
    c = _client("ok")
    r1 = c.post("/predict", json={"text": "abc", "model_family": "x", "run_id": "r1", "content_type": "news"})
    r2 = c.post("/predict", json={"text": "abc", "model_family": "ml", "run_id": "r1", "content_type": "x"})
    r3 = c.post("/predict", json={"text": "abc", "model_family": "ml", "run_id": "r1", "content_type": "news", "top_k": 31})
    assert r1.status_code == 422
    assert r2.status_code == 422
    assert r3.status_code == 422


# Validate API error mapping for not-found and family-mismatch cases.
def test_predict_error_code_mapping():
    r1 = _client("not_found").post(
        "/predict",
        json={"text": "abc", "model_family": "ml", "run_id": "r1", "content_type": "news"},
    )
    r2 = _client("mismatch").post(
        "/predict",
        json={"text": "abc", "model_family": "ml", "run_id": "r1", "content_type": "news"},
    )
    assert r1.status_code == 404
    assert r1.json()["error_code"] == "RUN_ID_NOT_FOUND"
    assert r2.status_code == 422
    assert r2.json()["error_code"] == "RUN_ID_MODEL_FAMILY_MISMATCH"


# Validate calibrated threshold resolution for both ML and DL engines.
def test_threshold_rule_ml_and_dl():
    ml = MLInferenceEngine.__new__(MLInferenceEngine)
    ml.use_calibrated_threshold = True
    ml.fallback_threshold = 0.5
    dl = DLInferenceEngine.__new__(DLInferenceEngine)
    dl.use_calibrated_threshold = True
    dl.fallback_threshold = 0.5
    assert ml._resolve_bundle_threshold({"threshold_calibration": {"recommended_threshold": 0.77}}) == 0.77
    assert dl._resolve_bundle_threshold({"threshold_calibration": {"recommended_threshold": 0.66}}) == 0.66
    assert ml._resolve_bundle_threshold({}) == 0.5
    assert dl._resolve_bundle_threshold({}) == 0.5
