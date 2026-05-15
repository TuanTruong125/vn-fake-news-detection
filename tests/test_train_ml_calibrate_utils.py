from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.ml import calibrate as ml_calibrate


# Build one compact fake model used for calibration-score tests.
# Stand-in model exposing predict_proba for calibration coverage.
class PredictProbaModel:
    classes_ = [0, 1]

    def predict_proba(self, x_vec):
        return np.asarray([[0.2, 0.8], [0.6, 0.4]])


# Build one compact decision-function model used for calibration-score tests.
# Stand-in model exposing decision_function for calibration coverage.
class DecisionFunctionModel:
    classes_ = [0, 1]

    def decision_function(self, x_vec):
        return np.asarray([-1.0, 1.0])


# Validate threshold argument checks accept good bounds and reject invalid values.
def test_validate_threshold_args_guard() -> None:
    ml_calibrate.validate_threshold_args(0.05, 0.95, 0.01)

    with pytest.raises(ml_calibrate.CalibrationError):
        ml_calibrate.validate_threshold_args(0.6, 0.5, 0.01)

    with pytest.raises(ml_calibrate.CalibrationError):
        ml_calibrate.validate_threshold_args(0.05, 0.95, 0.0)


# Validate threshold grid includes both ends and rounds floats consistently.
def test_build_threshold_grid_inclusive() -> None:
    grid = ml_calibrate.build_threshold_grid(0.1, 0.3, 0.1)
    assert np.allclose(grid, np.array([0.1, 0.2, 0.3]))


# Validate score extraction uses class-1 probability when predict_proba is available.
def test_compute_scores_from_predict_proba() -> None:
    scores, method = ml_calibrate.compute_scores(PredictProbaModel(), x_vec=np.zeros((2, 1)))

    assert method == "predict_proba"
    assert np.allclose(scores, np.array([0.8, 0.4]))


# Validate score extraction falls back to sigmoid(decision_function) when probability is unavailable.
def test_compute_scores_from_decision_function() -> None:
    scores, method = ml_calibrate.compute_scores(DecisionFunctionModel(), x_vec=np.zeros((2, 1)))

    assert method == "decision_function_sigmoid"
    assert np.allclose(scores, np.array([1 / (1 + np.exp(1.0)), 1 / (1 + np.exp(-1.0))]))


# Validate evaluation metrics include the standard binary-class keys.
def test_compute_metrics_structure() -> None:
    metrics = ml_calibrate.compute_metrics(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))

    assert set(metrics) == {"f1_macro", "f1_fake", "precision_fake", "recall_fake", "accuracy"}


# Validate threshold sweep produces one row per threshold candidate.
def test_evaluate_threshold_grid_rows() -> None:
    sweep = ml_calibrate.evaluate_threshold_grid(
        y_true=np.array([0, 1, 1]),
        scores=np.array([0.2, 0.6, 0.9]),
        thresholds=np.array([0.3, 0.7]),
    )

    assert list(sweep["threshold"]) == [0.3, 0.7]
    assert set(sweep.columns) >= {"f1_macro", "f1_fake", "precision_fake", "recall_fake", "accuracy"}


# Validate best-threshold selection prefers objective, then tie-breaker, then distance to 0.5.
def test_select_best_threshold_tie_order() -> None:
    sweep_df = pd.DataFrame(
        [
            {"threshold": 0.2, "f1_macro": 0.9, "f1_fake": 0.8, "precision_fake": 0.7, "recall_fake": 0.85, "accuracy": 0.9},
            {"threshold": 0.6, "f1_macro": 0.9, "f1_fake": 0.8, "precision_fake": 0.72, "recall_fake": 0.82, "accuracy": 0.89},
            {"threshold": 0.4, "f1_macro": 0.9, "f1_fake": 0.8, "precision_fake": 0.73, "recall_fake": 0.81, "accuracy": 0.88},
        ]
    )

    best = ml_calibrate.select_best_threshold(sweep_df, "f1_macro", "f1_fake")
    assert float(best["threshold"]) == 0.4


# Validate before-threshold resolution prefers calibrated metadata and defaults to 0.5 otherwise.
def test_resolve_before_threshold_default_and_metadata() -> None:
    assert ml_calibrate.resolve_before_threshold({}) == 0.5
    assert ml_calibrate.resolve_before_threshold({"threshold_calibration": {"recommended_threshold": 0.62}}) == 0.62


# Validate score summary and markdown helpers keep the output compact and explicit.
def test_build_score_distribution_and_markdown_helpers() -> None:
    summary = ml_calibrate.build_score_distribution(np.array([0.2, 0.6, 0.9]))
    assert summary == {"min": 0.2, "max": 0.9, "mean": pytest.approx(0.5666666667)}

    assert "P(fake)" in ml_calibrate.describe_score_method("predict_proba")

    table = ml_calibrate.metrics_pair_to_markdown(
        0.5,
        {"f1_macro": 0.8, "f1_fake": 0.7, "precision_fake": 0.6, "recall_fake": 0.5, "accuracy": 0.9},
        0.6,
        {"f1_macro": 0.82, "f1_fake": 0.72, "precision_fake": 0.61, "recall_fake": 0.55, "accuracy": 0.91},
    )
    assert "before" in table and "after" in table
