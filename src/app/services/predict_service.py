from __future__ import annotations

import hashlib
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


    # Server-side text normalization to handle escaped/newline artifacts from clients.
    def _normalize_text(self, text: str) -> str:
        if text is None:
            return ""

        # Normalize newlines: handle CRLF (\r\n), CR (\r), and escaped versions (\\r\\n, \\n, \\r).
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        t = t.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
        t = t.replace("\\t", "\t")
        t = t.replace("\\\"", '"')
        return t


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
        
        # Server-side normalization (defensive layer)
        text = self._normalize_text(text)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        text_length = len(text)
        non_whitespace_count = len(text.split())
        text_preview = text[:100].replace("\n", "\\n")
        print(
            f"[PREDICT_DEBUG] text_hash={text_hash} | text_length={text_length} | "
            f"non_ws_count={non_whitespace_count} | preview={text_preview}"
        )
        print(
            f"[PREDICT_DEBUG] model_family={model_family} | run_id={run_id} | "
            f"content_type={content_type} | top_k={top_k}"
        )
        print(f"[PREDICT_DEBUG_FULL_TEXT] {repr(text)}")
        
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
