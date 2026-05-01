from __future__ import annotations

from typing import Any

import numpy as np


# Helper function to extract class 1 linear parameters from model.
def _resolve_class1_linear_params(model: Any) -> tuple[np.ndarray, float]:
    if not hasattr(model, "coef_"):
        raise ValueError("Model does not expose coef_ for explanation.")
    if not hasattr(model, "classes_"):
        raise ValueError("Model does not expose classes_ for explanation.")
    if not hasattr(model, "intercept_"):
        raise ValueError("Model does not expose intercept_ for explanation.")

    classes = [int(x) for x in list(model.classes_)]
    if sorted(classes) != [0, 1]:
        raise ValueError(f"Unsupported classes for binary fake/real explanation: {classes}")

    coef = np.asarray(model.coef_, dtype=float)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)

    intercept = np.asarray(model.intercept_, dtype=float).reshape(-1)

    if coef.shape[0] == 1:
        # In binary linear estimators, coef_[0] corresponds to classes_[1].
        if classes[1] == 1:
            intercept_value = float(intercept[0]) if intercept.size > 0 else 0.0
            return coef[0], intercept_value
        if classes[0] == 1:
            intercept_value = -float(intercept[0]) if intercept.size > 0 else 0.0
            return -coef[0], intercept_value
        raise ValueError(f"Cannot map class 1 weights from classes={classes}")

    if coef.shape[0] == 2:
        class_to_index = {c: i for i, c in enumerate(classes)}
        class1_index = class_to_index[1]
        intercept_value = float(intercept[class1_index]) if intercept.size > class1_index else 0.0
        return coef[class1_index], intercept_value

    raise ValueError(f"Unsupported coef_ shape for binary explanation: {coef.shape}")


# Explain linear model prediction by attributing contributions to input features.
def explain_linear_prediction(
    model: Any,
    vectorizer: Any,
    x_vec: Any,
    top_k_per_direction: int,
    include_decomposition: bool,
    raw_decision_score: float | None = None,
) -> dict[str, Any]:
    if top_k_per_direction < 1:
        raise ValueError("top_k_per_direction must be >= 1")

    if not hasattr(vectorizer, "get_feature_names_out"):
        return {
            "explanation_available": False,
            "explanation_reason": "Vectorizer does not expose get_feature_names_out.",
            "top_features_towards_fake": [],
            "top_features_towards_real": [],
            "explanation_decomposition": None,
        }

    try:
        class1_weights, intercept = _resolve_class1_linear_params(model)
    except Exception as exc:
        return {
            "explanation_available": False,
            "explanation_reason": str(exc),
            "top_features_towards_fake": [],
            "top_features_towards_real": [],
            "explanation_decomposition": None,
        }

    feature_names = vectorizer.get_feature_names_out()
    row = x_vec[0]
    if hasattr(row, "tocsr"):
        row = row.tocsr()

    indices = row.indices.tolist()
    values = row.data.tolist()

    towards_fake: list[dict[str, Any]] = []
    towards_real: list[dict[str, Any]] = []
    sum_positive = 0.0
    sum_negative = 0.0

    for idx, value in zip(indices, values):
        if idx >= len(feature_names) or idx >= len(class1_weights):
            continue
        feature = str(feature_names[idx])
        weight = float(class1_weights[idx])
        value_f = float(value)
        contribution = value_f * weight

        if contribution > 0:
            sum_positive += contribution
            towards_fake.append(
                {
                    "feature": feature,
                    "value": value_f,
                    "weight": weight,
                    "contribution": float(contribution),
                    "direction": "towards_fake",
                }
            )
        elif contribution < 0:
            sum_negative += contribution
            towards_real.append(
                {
                    "feature": feature,
                    "value": value_f,
                    "weight": weight,
                    "contribution": float(contribution),
                    "direction": "towards_real",
                }
            )

    towards_fake.sort(key=lambda item: item["contribution"], reverse=True)
    towards_real.sort(key=lambda item: abs(item["contribution"]), reverse=True)

    decomposition = None
    if include_decomposition:
        sum_total = float(sum_positive + sum_negative)
        estimated_decision_score = float(sum_total + intercept)
        decision_score_gap = (
            float(raw_decision_score - estimated_decision_score)
            if raw_decision_score is not None
            else None
        )
        decomposition = {
            "sum_positive_contrib": float(sum_positive),
            "sum_negative_contrib": float(sum_negative),
            "sum_total_contrib": sum_total,
            "intercept": float(intercept),
            "estimated_decision_score": estimated_decision_score,
            "raw_decision_score": float(raw_decision_score) if raw_decision_score is not None else None,
            "decision_score_gap": decision_score_gap,
            "approximate_score_alignment_note": (
                "Linear decision can be decomposed as sum(contribution) + intercept. "
                "Any residual gap is typically due to numeric precision or pipeline representation details."
            ),
        }

    return {
        "explanation_available": True,
        "explanation_reason": None,
        "top_features_towards_fake": towards_fake[:top_k_per_direction],
        "top_features_towards_real": towards_real[:top_k_per_direction],
        "explanation_decomposition": decomposition,
    }

