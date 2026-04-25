from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# Schemas for API request and response models.
class PredictRequest(BaseModel):
    text: str = Field(..., description="Input text to classify.")
    run_id: str | None = Field(default=None, description="Optional run artifact override.")
    content_type: Literal["news", "social"] | None = Field(
        default=None,
        description="Optional content type to control preprocessing behavior.",
    )
    top_k: int | None = Field(default=None, ge=1, le=30, description="Top features per direction.")
    return_explanation: bool = Field(default=True, description="Whether to return explanation fields.")


# Single feature contribution item for explanation.
class FeatureContribution(BaseModel):
    feature: str
    value: float
    weight: float
    contribution: float
    direction: Literal["towards_fake", "towards_real"]


# Aggregated contribution breakdown for explainability diagnostics.
class ExplanationDecomposition(BaseModel):
    sum_positive_contrib: float
    sum_negative_contrib: float
    sum_total_contrib: float
    intercept: float
    estimated_decision_score: float
    raw_decision_score: float | None
    decision_score_gap: float | None
    approximate_score_alignment_note: str


# Response schema for prediction result and diagnostics.
class PredictResponse(BaseModel):
    label_id: Literal[0, 1]
    label_text: str

    threshold_used: float
    raw_score: float
    raw_decision_score: float | None

    confidence: float
    score_method: Literal["predict_proba", "decision_function_sigmoid"]
    confidence_type: Literal["probability", "pseudo_probability"]
    is_probability: bool

    run_id: str
    model_name: str
    feature_set: str
    text_variant: str

    original_length_chars: int
    post_normalize_token_count: int
    truncated: bool
    truncated_tokens_from: int
    truncated_tokens_to: int

    segmentation_backend: str
    segmentation_fallback_used: bool
    preprocessing_mode: str
    warnings: list[str]

    explanation_available: bool
    explanation_reason: str | None
    top_features_towards_fake: list[FeatureContribution]
    top_features_towards_real: list[FeatureContribution]
    explanation_decomposition: ExplanationDecomposition | None

    processing_time_ms: float


# Response schema for health check.
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    default_run_id: str | None
    default_threshold_used: float | None
    default_score_method: str | None
    model_loaded: bool
    cache_entries: int
    cache_capacity: int
