from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .model_registry import ModelRegistry


# Custom exception for run resolution errors, including validation and not-found scenarios.
class RunResolverError(Exception):
    def __init__(self, message: str, status_code: int = 422, error_code: str = "VALIDATION_ERROR") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


# Data class representing a resolved run with its ID and model family.
@dataclass(frozen=True)
class ResolvedRun:
    run_id: str
    model_family: Literal["ml", "dl"]


# Validate run_id existence and enforce run_id/model_family consistency.
class RunResolver:

    # Initialize the run resolver with a model registry.
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry


    # Resolve run_id to its corresponding model family, ensuring it exists and is not ambiguous across families, with detailed error handling for various validation scenarios.
    def resolve(self, run_id: str, model_family: str) -> ResolvedRun:
        normalized_run_id = run_id.strip()
        normalized_family = model_family.strip().lower()

        if not normalized_run_id:
            raise RunResolverError("run_id is required.", status_code=422)

        if normalized_family not in {"ml", "dl"}:
            raise RunResolverError("model_family must be one of: ml, dl.", status_code=422)

        ml_exists = self.registry.ml_run_exists(normalized_run_id)
        dl_exists = self.registry.dl_run_exists(normalized_run_id)
        if not ml_exists and not dl_exists:
            raise RunResolverError(
                f"run_id '{normalized_run_id}' was not found.",
                status_code=404,
                error_code="RUN_ID_NOT_FOUND",
            )
        if ml_exists and dl_exists:
            raise RunResolverError(
                f"run_id '{normalized_run_id}' is ambiguous across model families. Please use unique run_id naming.",
                status_code=422,
                error_code="RUN_ID_AMBIGUOUS",
            )

        actual_family = "ml" if ml_exists else "dl"
        if actual_family != normalized_family:
            raise RunResolverError(
                f"run_id '{normalized_run_id}' belongs to model_family='{actual_family}', not '{normalized_family}'.",
                status_code=422,
                error_code="RUN_ID_MODEL_FAMILY_MISMATCH",
            )

        return ResolvedRun(run_id=normalized_run_id, model_family=actual_family)
