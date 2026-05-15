from __future__ import annotations

from pathlib import Path
import json
import re
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.ml import train as ml_train


# Build one compact ML training config used across helper tests.
def build_train_config() -> dict[str, object]:
    return {
        "data": {"text_variants": ["text_ml_seg", "text_ml_seg_lower"]},
        "feature_sets": [
            {"name": "word"},
            {"name": "word_char"},
        ],
        "models": {
            "linear_svm": {"enabled": True, "params": {"C": [1.0], "class_weight": ["balanced"]}},
        },
        "run": {
            "default": {
                "text_variant": "text_ml_seg",
                "feature_set": "word_char",
                "model": "linear_svm",
                "params": {"C": 0.5},
            }
        },
    }


# Validate run id generation embeds the selected scope fields.
def test_generate_run_id_contains_scope() -> None:
    run_id = ml_train.generate_run_id("linear_svm", "word_char", "text_ml_seg")

    assert "linear_svm" in run_id
    assert "word_char" in run_id
    assert "text_ml_seg" in run_id
    assert re.match(r"^\d{8}_\d{6}_linear_svm_word_char_text_ml_seg_[0-9a-f]{8}$", run_id)


# Validate runs.csv bootstrap writes the expected header row.
def test_ensure_runs_csv_creates_header(tmp_path: Path) -> None:
    runs_path = tmp_path / "runs.csv"

    ml_train.ensure_runs_csv(runs_path)

    header = runs_path.read_text(encoding="utf-8-sig").splitlines()[0]
    assert header.startswith("run_id,run_timestamp,experiment_name")


# Validate appending a run record writes one CSV row with the given values.
def test_append_run_record_writes_row(tmp_path: Path) -> None:
    runs_path = tmp_path / "runs.csv"
    ml_train.ensure_runs_csv(runs_path)

    run_record = ml_train.init_run_record(
        run_id="run-1",
        run_timestamp="2026-05-01T00:00:00+07:00",
        experiment_name="ml_experiment",
        config_version=1,
        random_state=42,
        model_name="linear_svm",
        feature_set="word_char",
        text_variant="text_ml_seg",
        params_json="{\"C\":1.0}",
    )
    ml_train.append_run_record(runs_path, run_record)

    frame = pd.read_csv(runs_path, encoding="utf-8-sig")
    assert frame.loc[0, "run_id"] == "run-1"
    assert frame.loc[0, "status"] == "FAIL"


# Validate init_run_record seeds empty metrics with a FAIL status placeholder.
def test_init_run_record_defaults() -> None:
    record = ml_train.init_run_record(
        run_id="run-2",
        run_timestamp="2026-05-01T00:00:00+07:00",
        experiment_name="ml_experiment",
        config_version=1,
        random_state=42,
        model_name="linear_svm",
        feature_set="word_char",
        text_variant="text_ml_seg",
        params_json="{}",
    )

    assert record["status"] == "FAIL"
    assert record["val_f1_macro"] == ""
    assert record["notes"] == ""


# Validate JSON parsing accepts objects and rejects invalid payloads.
def test_parse_params_json_validation() -> None:
    parsed = ml_train.parse_params_json('{"C":1.0,"class_weight":"balanced"}')
    assert parsed == {"C": 1.0, "class_weight": "balanced"}

    with pytest.raises(ml_train.TrainRunError):
        ml_train.parse_params_json('[1, 2, 3]')


# Validate run.default extraction returns a mapping and rejects invalid types.
def test_get_run_default_config_validation() -> None:
    cfg = build_train_config()
    assert ml_train.get_run_default_config(cfg) == cfg["run"]["default"]

    with pytest.raises(ml_train.TrainRunError):
        ml_train.get_run_default_config({"run": [1, 2, 3]})


# Validate first parameter lookup selects the first matching model combo.
def test_get_first_params_for_model() -> None:
    combos = [("linear_svm", {"C": 1.0}), ("multinomial_nb", {"alpha": 0.1})]

    assert ml_train.get_first_params_for_model("multinomial_nb", combos) == {"alpha": 0.1}


# Validate single-run scope resolution honors config defaults and parameter overrides.
def test_resolve_single_run_scope_prefers_defaults() -> None:
    cfg = build_train_config()
    text_variant, feature_set, model_name, params = ml_train.resolve_single_run_scope(
        config=cfg,
        requested_text_variant=None,
        requested_feature_set=None,
        requested_model_name=None,
        params_override=None,
        max_combos_per_model=0,
    )

    assert (text_variant, feature_set, model_name) == ("text_ml_seg", "word_char", "linear_svm")
    assert params["C"] == 0.5
    assert params["class_weight"] == "balanced"


# Validate metric computation returns the standard ML scoring keys.
def test_compute_metrics_structure() -> None:
    metrics = ml_train.compute_metrics([0, 1, 1, 0], [0, 1, 0, 0])

    assert set(metrics) == {"f1_macro", "precision_macro", "recall_macro", "accuracy", "f1_fake"}
    assert metrics["accuracy"] == pytest.approx(0.75)


# Validate feature-space diagnostics aggregate configured max_features correctly.
def test_compute_feature_space_diagnostics() -> None:
    diagnostics = ml_train.compute_feature_space_diagnostics(
        {"parts": [{"max_features": 1000}, {"max_features": 500}]},
        observed_feature_count=900,
    )

    assert diagnostics == {
        "expected_max_features": 1500,
        "observed_feature_count": 900,
        "utilization_ratio": 0.6,
    }


# Validate artifact saving writes the run files and the best-run copies.
def test_save_artifacts_writes_expected_files(tmp_repo: Path, patch_module_repo_root) -> None:
    patch_module_repo_root(ml_train, "src/train/ml/train.py")

    artifact_paths = ml_train.save_artifacts(
        run_id="run-3",
        model={"kind": "model"},
        vectorizer={"kind": "vectorizer"},
        metadata={"run_id": "run-3"},
        save_as_best=True,
    )

    model_dir = tmp_repo / "models" / "ml"
    assert (model_dir / "run-3__model.joblib").exists()
    assert (model_dir / "run-3__vectorizer.joblib").exists()
    assert (model_dir / "run-3__metadata.json").exists()
    assert (model_dir / "best_model.joblib").exists()
    assert (model_dir / "best_vectorizer.joblib").exists()
    assert (model_dir / "best_metadata.json").exists()
    assert set(artifact_paths) >= {"model_path", "vectorizer_path", "metadata_path", "best_model_path"}
