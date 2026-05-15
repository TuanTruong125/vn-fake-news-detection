from __future__ import annotations

from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

dl_grid = pytest.importorskip("src.train.dl.grid_search_phobert")


# Build one compact grid-search config used across DL search-helper tests.
def build_grid_config() -> dict[str, object]:
    return {
        "experiment": {"name": "phobert", "random_state": 42},
        "selection": {"choose_best_by": "f1_macro", "tie_breaker": "f1_fake"},
        "search_space": {
            "learning_rate": [2e-5, 2e-5, 3e-5],
            "max_length": [128, 256],
            "train_batch_size": [8, 16],
        },
    }


# Validate selection fields are parsed from the dedicated selection section.
def test_ensure_selection_fields_and_experiment_meta() -> None:
    cfg = build_grid_config()
    primary, tie_breaker, val_primary, val_tie = dl_grid.ensure_selection_fields(cfg)
    experiment_name, random_state = dl_grid.parse_experiment_meta(cfg)

    assert (primary, tie_breaker, val_primary, val_tie) == ("f1_macro", "f1_fake", "val_f1_macro", "val_f1_fake")
    assert (experiment_name, random_state) == ("phobert", 42)


# Validate grid expansion deduplicates values while preserving cartesian order.
def test_build_grid_hyperparams_deduplicates_values() -> None:
    cfg = build_grid_config()
    candidates = dl_grid.build_grid_hyperparams(cfg["search_space"])

    assert candidates == [
        {"learning_rate": 2e-05, "max_length": 128, "train_batch_size": 8},
        {"learning_rate": 2e-05, "max_length": 128, "train_batch_size": 16},
        {"learning_rate": 2e-05, "max_length": 256, "train_batch_size": 8},
        {"learning_rate": 2e-05, "max_length": 256, "train_batch_size": 16},
        {"learning_rate": 3e-05, "max_length": 128, "train_batch_size": 8},
        {"learning_rate": 3e-05, "max_length": 128, "train_batch_size": 16},
        {"learning_rate": 3e-05, "max_length": 256, "train_batch_size": 8},
        {"learning_rate": 3e-05, "max_length": 256, "train_batch_size": 16},
    ]


# Validate fallback candidates step down in sequence and remove duplicates.
def test_build_fallback_candidates_sequence() -> None:
    fallback = dl_grid.build_fallback_candidates(
        initial_max_length=256,
        initial_batch_size=16,
        allowed_max_lengths=[128, 256],
        allowed_batch_sizes=[8, 16],
    )

    assert fallback == [(256, 16), (128, 16), (128, 8)]


# Validate safe float conversion and timestamp parsing stay stable for bad inputs.
def test_safe_float_and_timestamp_helpers() -> None:
    assert dl_grid.safe_float("1.25") == 1.25
    assert dl_grid.safe_float("bad") == float("-inf")
    assert dl_grid.parse_run_timestamp("not-a-timestamp") == dl_grid.datetime.min


# Validate CSV helpers preserve schema when writing and reading rows.
def test_write_and_load_runs_rows(tmp_path: Path) -> None:
    runs_path = tmp_path / "runs.csv"
    rows = [
        {"run_id": "r1", "run_timestamp": "2026-05-01T00:00:00+07:00", "status": "PASS"},
    ]

    dl_grid.write_runs_rows(runs_path, rows)
    loaded = dl_grid.load_runs_rows(runs_path)

    assert loaded[0]["run_id"] == "r1"
    assert loaded[0]["status"] == "PASS"


# Validate leaderboard and best-config payloads carry the selected run metadata.
def test_build_leaderboard_and_best_config_payload() -> None:
    run_row = {
        "run_id": "run-1",
        "run_timestamp": "2026-05-02T00:00:00+07:00",
        "experiment_name": "phobert",
        "config_version": 1,
        "model_name": "phobert",
        "feature_set": "transformer",
        "text_variant": "text_clean",
        "val_f1_macro": 0.9,
        "val_f1_fake": 0.8,
        "test_f1_macro": 0.88,
        "test_f1_fake": 0.77,
        "test_precision_macro": 0.86,
        "test_recall_macro": 0.87,
        "test_accuracy": 0.89,
        "params_json": '{"learning_rate":0.00003}',
        "notes": '{"run_dir":"C:/tmp/run-1"}',
    }

    leaderboard = dl_grid.build_leaderboard_row(run_row, "f1_macro", "f1_fake")
    payload = dl_grid.build_best_config_payload(run_row, "f1_macro", "f1_fake")

    assert leaderboard["primary_metric"] == "f1_macro"
    assert leaderboard["selection_metric"] == "f1_fake"
    assert payload["best_run"]["params"] == {"learning_rate": 3e-05}
    assert payload["best_run"]["notes"] == {"run_dir": "C:/tmp/run-1"}


# Validate best-run artifact promotion copies the checkpoint and metadata into models/dl.
def test_save_best_run_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "models" / "dl" / "run-1"
    checkpoint_dir = run_dir / "best_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "pytorch_model.bin").write_text("stub", encoding="utf-8")
    (run_dir / "metrics.json").write_text('{"val": {"f1_macro": 0.9}}', encoding="utf-8")
    (run_dir / "metadata.json").write_text('{"run_id": "run-1"}', encoding="utf-8")

    best_run_row = {
        "run_id": "run-1",
        "run_timestamp": "2026-05-02T00:00:00+07:00",
        "experiment_name": "phobert",
        "config_version": 1,
        "model_name": "phobert",
        "feature_set": "transformer",
        "text_variant": "text_clean",
        "params_json": '{"learning_rate":0.00003}',
        "notes": json.dumps({"run_dir": str(run_dir)}),
    }

    copied = dl_grid.save_best_run_artifacts(best_run_row, tmp_path)

    model_dir = tmp_path / "models" / "dl"
    assert (model_dir / "best_checkpoint" / "pytorch_model.bin").exists()
    assert (model_dir / "best_metrics.json").exists()
    assert (model_dir / "best_metadata.json").exists()
    assert (model_dir / "best_run.json").exists()
    assert (model_dir / "best_run_id.txt").read_text(encoding="utf-8").strip() == "run-1"
    assert copied["best_checkpoint_dir"].endswith("best_checkpoint")
