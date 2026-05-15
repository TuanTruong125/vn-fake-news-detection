from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

calibrate_dl = pytest.importorskip("src.train.dl.calibrate_dl")


# Validate threshold args guard accepts valid values and rejects invalid bounds.
def test_validate_threshold_args_guard() -> None:
    calibrate_dl.validate_threshold_args(0.05, 0.95, 0.01)

    with pytest.raises(calibrate_dl.CalibrationError):
        calibrate_dl.validate_threshold_args(0.6, 0.5, 0.01)

    with pytest.raises(calibrate_dl.CalibrationError):
        calibrate_dl.validate_threshold_args(0.05, 0.95, 0.0)


# Validate threshold grid includes both ends with stable float rounding.
def test_build_threshold_grid_inclusive() -> None:
    grid = calibrate_dl.build_threshold_grid(0.1, 0.3, 0.1)
    assert np.allclose(grid, np.array([0.1, 0.2, 0.3]))


# Validate split extraction returns binary y_true and clipped probability scores.
def test_extract_scores_by_split_clip_probabilities() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "split": ["val", "val", "test"],
            "y_true": [1, 0, 1],
            "y_pred": [1, 0, 1],
            "prob_fake": [1.5, -0.2, 0.4],
            "is_error": [False, False, False],
        }
    )

    y_true, scores = calibrate_dl.extract_scores_by_split(df, "val")

    assert y_true.tolist() == [1, 0]
    assert scores.tolist() == [1.0, 0.0]


# Validate best-threshold selector honors objective, tie-breaker, then distance to 0.5.
def test_select_best_threshold_tie_order() -> None:
    sweep_df = pd.DataFrame(
        [
            {"threshold": 0.2, "f1_macro": 0.9, "f1_fake": 0.8, "precision_fake": 0.7, "recall_fake": 0.85, "accuracy": 0.9},
            {"threshold": 0.6, "f1_macro": 0.9, "f1_fake": 0.8, "precision_fake": 0.72, "recall_fake": 0.82, "accuracy": 0.89},
            {"threshold": 0.4, "f1_macro": 0.9, "f1_fake": 0.8, "precision_fake": 0.73, "recall_fake": 0.81, "accuracy": 0.88},
        ]
    )

    best = calibrate_dl.select_best_threshold(sweep_df, "f1_macro", "f1_fake")
    assert float(best["threshold"]) == 0.4


# Validate score method description documents prob_fake semantics.
def test_describe_score_method_prob_fake() -> None:
    message = calibrate_dl.describe_score_method("prob_fake")
    assert "P(fake)" in message
    assert "threshold" in message
