from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.ml import error_analysis as ml_error_analysis


# Build one compact raw frame used across ML error-analysis tests.
def build_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "label_binary": [0, 1, 1, 0],
            "source_file": ["s1", "s1", "s2", "s2"],
            "content_type": ["news", "news", "social", "social"],
            "text_length": [100, 300, 800, 1200],
            "text_clean": ["alpha", "beta", "gamma", "delta"],
        }
    )


# Validate confidence scoring uses the model-specific fallback path.
def test_compute_confidence_scores_from_probability_and_margin() -> None:
    class ProbaModel:
        def predict_proba(self, x_vec):
            return np.asarray([[0.1, 0.9], [0.7, 0.3]])

    class MarginModel:
        def decision_function(self, x_vec):
            return np.asarray([-2.0, 1.5])

    y_pred = pd.Series([1, 0], index=[0, 1])
    proba_conf, proba_method = ml_error_analysis.compute_confidence_scores(ProbaModel(), None, y_pred)
    margin_conf, margin_method = ml_error_analysis.compute_confidence_scores(MarginModel(), None, y_pred)

    assert proba_method == "predict_proba_max"
    assert list(proba_conf) == [0.9, 0.7]
    assert margin_method == "decision_function_abs"
    assert list(margin_conf) == [2.0, 1.5]


# Validate prediction dataframe construction adds the expected error labels.
def test_build_prediction_dataframe_and_aggregate_error() -> None:
    frame = build_raw_frame()
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([0, 0, 1, 1])
    confidence = pd.Series([0.9, 0.8, 0.7, 0.6])

    pred_df = ml_error_analysis.build_prediction_dataframe(frame, y_true, y_pred, confidence)
    assert pred_df["error_type"].tolist() == ["CORRECT", "FN", "CORRECT", "FP"]

    grouped = ml_error_analysis.aggregate_error(pred_df, "source_file")
    assert list(grouped["source_file"]) == ["s1", "s2"]
    assert grouped.iloc[0]["errors"] == 1


# Validate length binning and top-error selection keep the frame easy to inspect.
def test_add_length_bins_and_select_top_errors() -> None:
    frame = build_raw_frame()
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([1, 0, 1, 1])
    confidence = pd.Series([0.4, 0.8, 0.2, 0.6])

    pred_df = ml_error_analysis.build_prediction_dataframe(frame, y_true, y_pred, confidence)
    pred_df = ml_error_analysis.add_length_bins(pred_df)

    assert pred_df["length_bin"].astype(str).tolist() == ["0-200", "201-500", "501-1000", "1001-2000"]

    top_fp = ml_error_analysis.select_top_errors(pred_df, "FP", top_n=1)
    assert list(top_fp["sample_id"]) == ["d"]
    assert top_fp.iloc[0]["snippet"] == "delta"


# Validate snippet trimming and confidence-method descriptions remain explicit.
def test_build_snippet_and_describe_confidence_method() -> None:
    assert ml_error_analysis.build_snippet("a b c d e", max_len=5) == "a ..."
    assert "predicted-class probability" in ml_error_analysis.describe_confidence_method("predict_proba_max")


# Validate required-column checks and split-path resolution use the configured CSV location.
def test_validate_columns_and_resolve_paths(tmp_path: Path) -> None:
    frame = build_raw_frame()
    ml_error_analysis.validate_required_columns(frame)

    with pytest.raises(ml_error_analysis.ErrorAnalysisError):
        ml_error_analysis.validate_required_columns(frame.drop(columns=["text_clean"]))

    repo_root = tmp_path
    split_path = repo_root / "data" / "processed" / "test.csv"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(split_path, index=False, encoding="utf-8-sig")

    cfg = {"data": {"test_path": "data/processed/test.csv"}}
    assert ml_error_analysis.resolve_split_path(cfg, "test", repo_root) == split_path


# Validate best-config helpers read JSON payloads and resolve the requested run id.
def test_load_best_config_and_resolve_run_id(tmp_path: Path) -> None:
    best_config_path = tmp_path / "best_config.json"
    best_config_path.write_text(json.dumps({"best_run": {"run_id": "run-99"}}), encoding="utf-8")

    payload = ml_error_analysis.load_best_config(best_config_path)
    assert payload["best_run"]["run_id"] == "run-99"
    assert ml_error_analysis.resolve_run_id(payload, None) == "run-99"
    assert ml_error_analysis.resolve_run_id(payload, "manual-run") == "manual-run"
