from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

try:
    from src.app.inference import InferenceService
    from src.app.services import PredictService, PredictServiceError
    from src.app.schemas import ErrorResponse, HealthResponse, PredictRequest, PredictResponse
except ModuleNotFoundError:
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from inference import InferenceService  # type: ignore
    from services import PredictService, PredictServiceError  # type: ignore
    from schemas import ErrorResponse, HealthResponse, PredictRequest, PredictResponse  # type: ignore


# FastAPI application instance and global inference service holder.
service: InferenceService | None = None
predict_service: PredictService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service, predict_service
    service = InferenceService.from_config_path()
    service.preload_default_bundle()
    predict_service = PredictService(
        inference_service=service,
        repo_root=Path(__file__).resolve().parents[2],
    )
    yield


app = FastAPI(title="VN Fake News Inference API", version="1.0.0", lifespan=lifespan)


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
    if service is None or predict_service is None:
        raise HTTPException(status_code=503, detail="Inference service is not initialized.")

    try:
        result = predict_service.predict(
            text=payload.text,
            model_family=payload.model_family,
            run_id=payload.run_id,
            content_type=payload.content_type,
            top_k=payload.top_k,
            return_explanation=payload.return_explanation,
        )
        return PredictResponse(**result)
    except PredictServiceError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error_code=exc.error_code, detail=str(exc)).model_dump(),
        )
    except Exception as exc:  # pragma: no cover
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error_code="INFERENCE_INTERNAL_ERROR",
                detail=f"Unexpected inference error: {exc}",
            ).model_dump(),
        )
