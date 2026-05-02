from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from src.app.inference import InferenceService
    from src.app.services.model_registry import ModelRegistry
    from src.app.services import PredictService, PredictServiceError
    from src.app.schemas import ErrorResponse, HealthResponse, PredictRequest, PredictResponse, RunsResponse
except ModuleNotFoundError:
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from inference import InferenceService  # type: ignore
    from services.model_registry import ModelRegistry  # type: ignore
    from services import PredictService, PredictServiceError  # type: ignore
    from schemas import ErrorResponse, HealthResponse, PredictRequest, PredictResponse, RunsResponse  # type: ignore


# FastAPI application instance and global inference service holder.
service: InferenceService | None = None
predict_service: PredictService | None = None
model_registry: ModelRegistry | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service, predict_service, model_registry
    service = InferenceService.from_config_path()
    service.preload_default_bundle()
    model_registry = ModelRegistry(repo_root=Path(__file__).resolve().parents[2])
    predict_service = PredictService(
        inference_service=service,
        repo_root=Path(__file__).resolve().parents[2],
    )
    yield


app = FastAPI(title="VN Fake News Inference API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://127.0.0.1:5501",
        "http://127.0.0.1:5502",
        "http://127.0.0.1:5503",
        "http://localhost:5500",
        "http://localhost:5501",
        "http://localhost:5502",
        "http://localhost:5503",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    
    # Log text fingerprint for cross-client payload debugging.
    text = payload.text
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    sha = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()[:16]
    newline_count = normalized_text.count("\n")
    non_ws_len = len("".join(normalized_text.split()))
    client_fp = request.headers.get("X-Client-Text-Fingerprint", "")
    logging.info(
        "predict_debug model_family=%s run_id=%s len=%d non_ws=%d nl=%d sha16=%s client_fp=%s",
        payload.model_family,
        payload.run_id,
        len(normalized_text),
        non_ws_len,
        newline_count,
        sha,
        client_fp,
    )
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


# Return run options for frontend selection.
@app.get("/runs", response_model=RunsResponse)
def list_runs() -> RunsResponse:
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry is not initialized.")
    runs = model_registry.list_runs_with_best()
    return RunsResponse(runs=runs)
