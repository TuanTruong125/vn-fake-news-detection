from __future__ import annotations

from pathlib import Path
from typing import Any

from src.app.inference import InferenceError, InferenceService

from .explain_service import ExplainService
from .model_registry import ModelRegistry
from .run_resolver import RunResolver, RunResolverError


# Custom exception for prediction request validation and service errors.
class PredictServiceError(Exception):
    def __init__(self, message: str, status_code: int = 422, error_code: str = "VALIDATION_ERROR") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


# Application-level orchestration for canonical /predict request handling.
class PredictService:

    # Initialize with an InferenceService instance, ModelRegistry for run_id validation, RunResolver for resolving run_id to model family, and ExplainService for handling explanation logic.
    def __init__(self, inference_service: InferenceService, repo_root: Path) -> None:
        self.inference_service = inference_service
        self.registry = ModelRegistry(repo_root)
        self.run_resolver = RunResolver(self.registry)
        self.explain_service = ExplainService()


    # Main entry point for handling prediction requests, which includes run_id resolution, routing to the appropriate inference engine, and error handling for validation and inference errors.
    def predict(
        self,
        *,
        text: str,
        model_family: str,
        run_id: str,
        content_type: str,
        top_k: int | None,
        return_explanation: bool,
    ) -> dict[str, Any]:
        try:
            resolved = self.run_resolver.resolve(run_id=run_id, model_family=model_family)
        except RunResolverError as exc:
            raise PredictServiceError(str(exc), status_code=exc.status_code, error_code=exc.error_code) from exc

        try:
            result = self.inference_service.predict(
                text=text,
                run_id=resolved.run_id,
                model_family=resolved.model_family,
                content_type=content_type,
                top_k=top_k,
                return_explanation=self.explain_service.should_return_explanation(return_explanation),
            )
        except InferenceError as exc:
            raise PredictServiceError(str(exc), status_code=exc.status_code, error_code="INFERENCE_INTERNAL_ERROR") from exc

        result["model_family"] = resolved.model_family
        return result
