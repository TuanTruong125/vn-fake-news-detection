from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

try:
    from src.app.inference import InferenceError, InferenceService
    from src.app.schemas import HealthResponse, PredictRequest, PredictResponse
except ModuleNotFoundError:
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from inference import InferenceError, InferenceService  # type: ignore
    from schemas import HealthResponse, PredictRequest, PredictResponse  # type: ignore


# FastAPI application instance and global inference service holder.
app = FastAPI(title="VN Fake News Inference API", version="1.0.0")
service: InferenceService | None = None


# Initialize inference service and preload default model at startup to avoid cold-start latency on the first request.
@app.on_event("startup")
def startup_event() -> None:
    global service
    service = InferenceService.from_config_path()
    service.preload_default_bundle()


# Health check endpoint returning service readiness and default model context.
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if service is None:
        return HealthResponse(
            status="degraded",
            default_run_id=None,
            default_threshold_used=None,
            default_score_method=None,
            model_loaded=False,
            cache_entries=0,
            cache_capacity=0,
        )
    return HealthResponse(**service.health())


# The /predict endpoint is defined to handle POST requests with a PredictRequest payload and return a PredictResponse.
@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="Inference service is not initialized.")

    try:
        result = service.predict(
            text=payload.text,
            run_id=payload.run_id,
            content_type=payload.content_type,
            top_k=payload.top_k,
            return_explanation=payload.return_explanation,
        )
        return PredictResponse(**result)
    except InferenceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Unexpected inference error: {exc}") from exc
