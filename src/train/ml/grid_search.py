from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from src.train.common.dataset_loader import DatasetLoaderError, get_xy_splits
    from src.train.common.split_utils import DatasetValidationError
    from src.train.ml.models import (
        ModelConfigError,
        ModelRuntimeError,
        build_model,
        fit_model,
        get_model_param_combinations,
        predict_model,
    )
    from src.train.ml.train import (
        RUNS_COLUMNS,
        TrainRunError,
        append_run_record,
        compute_metrics,
        compute_feature_space_diagnostics,
        generate_run_id,
        now_iso,
        save_artifacts,
    )
    from src.train.ml.vectorizers import (
        VectorizerConfigError,
        build_feature_pipeline,
        get_feature_set_names,
        get_text_variants,
        load_train_ml_config,
        resolve_config_path,
    )
except ModuleNotFoundError:
    CURRENT_DIR = Path(__file__).resolve().parent
    COMMON_DIR = CURRENT_DIR.parent / "common"
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    if str(COMMON_DIR) not in sys.path:
        sys.path.insert(0, str(COMMON_DIR))

    from dataset_loader import DatasetLoaderError, get_xy_splits  # type: ignore
    from split_utils import DatasetValidationError  # type: ignore
    from models import (  # type: ignore
        ModelConfigError,
        ModelRuntimeError,
        build_model,
        fit_model,
        get_model_param_combinations,
        predict_model,
    )
    from train import (  # type: ignore
        RUNS_COLUMNS,
        TrainRunError,
        append_run_record,
        compute_metrics,
        compute_feature_space_diagnostics,
        generate_run_id,
        now_iso,
        save_artifacts,
    )
    from vectorizers import (  # type: ignore
        VectorizerConfigError,
        build_feature_pipeline,
        get_feature_set_names,
        get_text_variants,
        load_train_ml_config,
        resolve_config_path,
    )


LEADERBOARD_COLUMNS = [
    "run_id",
    "run_timestamp",
    "experiment_name",
    "config_version",
    "model_name",
    "feature_set",
    "text_variant",
    "val_f1_macro",
    "val_f1_fake",
    "test_f1_macro",
    "test_f1_fake",
    "test_precision_macro",
    "test_recall_macro",
    "test_accuracy",
    "primary_metric",
    "selection_metric",
]


# Error class for grid search execution issues.
class GridSearchError(Exception):
    pass


# Resolve repository root path from current file location.
def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# Parse CLI arguments for ML grid-search execution.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML grid search and persist best config + leaderboard.")
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional path to train_ml.yaml. Default: configs/train_ml.yaml",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="If set, split overlap check becomes warning-only.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=0,
        help="Optional hard cap on total trials (0 means use config limit).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned trial count and exit without running training.",
    )
    return parser.parse_args()


# Build one base run record template compatible with experiments/ml/runs.csv.
def build_base_run_record(
    run_id: str,
    run_timestamp: str,
    experiment_name: str,
    config_version: int,
    random_state: int | str,
    model_name: str,
    feature_set: str,
    text_variant: str,
    params_json: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "experiment_name": experiment_name,
        "config_version": config_version,
        "random_state": random_state,
        "model_name": model_name,
        "feature_set": feature_set,
        "text_variant": text_variant,
        "params_json": params_json,
        "val_f1_macro": "",
        "val_precision_macro": "",
        "val_recall_macro": "",
        "val_accuracy": "",
        "val_f1_fake": "",
        "test_f1_macro": "",
        "test_precision_macro": "",
        "test_recall_macro": "",
        "test_accuracy": "",
        "test_f1_fake": "",
        "status": "FAIL",
        "notes": "",
    }


# Parse experiment metadata needed for run tracking and search controls.
def parse_experiment_meta(config: dict[str, Any], cli_max_trials: int) -> tuple[str, int, int, int]:
    experiment_cfg = config.get("experiment")
    if not isinstance(experiment_cfg, dict):
        raise GridSearchError("train_ml.yaml must define experiment section.")

    experiment_name = str(experiment_cfg.get("name", "ml_experiment")).strip() or "ml_experiment"
    random_state = experiment_cfg.get("random_state")
    if not isinstance(random_state, int):
        raise GridSearchError("experiment.random_state must be integer.")

    config_max_trials = experiment_cfg.get("max_trials")
    if not isinstance(config_max_trials, int) or config_max_trials <= 0:
        raise GridSearchError("experiment.max_trials must be a positive integer.")

    effective_max_trials = cli_max_trials if cli_max_trials > 0 else config_max_trials
    if effective_max_trials <= 0:
        raise GridSearchError("Effective max_trials must be positive.")

    return experiment_name, random_state, config_max_trials, effective_max_trials


# Parse selection metrics and return run fields used for ranking.
def parse_selection_fields(config: dict[str, Any]) -> tuple[str, str, str, str]:
    selection_cfg = config.get("selection")
    if not isinstance(selection_cfg, dict):
        raise GridSearchError("train_ml.yaml must define selection section.")

    primary_metric = str(selection_cfg.get("choose_best_by", "")).strip()
    tie_breaker = str(selection_cfg.get("tie_breaker", "")).strip()
    if not primary_metric or not tie_breaker:
        raise GridSearchError("selection.choose_best_by and selection.tie_breaker must be non-empty.")

    primary_field = f"val_{primary_metric}"
    tie_field = f"val_{tie_breaker}"
    return primary_metric, tie_breaker, primary_field, tie_field


# Return default float for metric fields when parsing fails.
def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("-inf")


# Build a leaderboard row from a PASS run record.
def build_leaderboard_row(run_row: dict[str, Any], primary_metric: str, selection_metric: str) -> dict[str, Any]:
    return {
        "run_id": run_row.get("run_id", ""),
        "run_timestamp": run_row.get("run_timestamp", ""),
        "experiment_name": run_row.get("experiment_name", ""),
        "config_version": run_row.get("config_version", ""),
        "model_name": run_row.get("model_name", ""),
        "feature_set": run_row.get("feature_set", ""),
        "text_variant": run_row.get("text_variant", ""),
        "val_f1_macro": run_row.get("val_f1_macro", ""),
        "val_f1_fake": run_row.get("val_f1_fake", ""),
        "test_f1_macro": run_row.get("test_f1_macro", ""),
        "test_f1_fake": run_row.get("test_f1_fake", ""),
        "test_precision_macro": run_row.get("test_precision_macro", ""),
        "test_recall_macro": run_row.get("test_recall_macro", ""),
        "test_accuracy": run_row.get("test_accuracy", ""),
        "primary_metric": primary_metric,
        "selection_metric": selection_metric,
    }


# Load all rows from runs.csv and validate header contract.
def load_runs_rows(runs_path: Path) -> list[dict[str, Any]]:
    if not runs_path.exists():
        raise GridSearchError(f"Missing runs file: {runs_path}")

    with runs_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if header != RUNS_COLUMNS:
            raise GridSearchError(
                "runs.csv header mismatch.\n"
                f"Expected: {RUNS_COLUMNS}\nFound:    {header}"
            )
        return list(reader)


# Rewrite full runs.csv rows while preserving header contract.
def write_runs_rows(runs_path: Path, rows: list[dict[str, Any]]) -> None:
    with runs_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in RUNS_COLUMNS})


# Update notes field for one run_id directly in runs.csv.
def update_run_notes_in_csv(runs_path: Path, run_id: str, notes_payload: dict[str, Any]) -> None:
    rows = load_runs_rows(runs_path)
    updated = False
    notes_json = json.dumps(notes_payload, ensure_ascii=False, separators=(",", ":"))
    for row in rows:
        if str(row.get("run_id", "")).strip() == run_id:
            row["notes"] = notes_json
            updated = True
            break
    if not updated:
        raise GridSearchError(f"Cannot update notes. run_id '{run_id}' not found in runs.csv.")
    write_runs_rows(runs_path, rows)


# Write leaderboard CSV with deterministic ordering.
def write_leaderboard_csv(leaderboard_path: Path, rows: list[dict[str, Any]]) -> None:
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    with leaderboard_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in LEADERBOARD_COLUMNS})


# Build best-config payload for JSON artifact.
def build_best_config_payload(
    best_run_row: dict[str, Any],
    selection_primary_metric: str,
    selection_tie_breaker: str,
) -> dict[str, Any]:
    run_timestamp = now_iso()
    params_raw = str(best_run_row.get("params_json", "")).strip()
    try:
        params_obj = json.loads(params_raw) if params_raw else {}
    except json.JSONDecodeError:
        params_obj = {"raw_params_json": params_raw}

    notes_raw = str(best_run_row.get("notes", "")).strip()
    notes_obj: Any = notes_raw
    try:
        notes_obj = json.loads(notes_raw) if notes_raw else {}
    except json.JSONDecodeError:
        pass

    return {
        "selected_at": run_timestamp,
        "selection_rule": {
            "primary_metric": selection_primary_metric,
            "tie_breaker": selection_tie_breaker,
            "selection_space": "model + params + feature_set + text_variant",
            "selection_scope": "validation metrics",
        },
        "best_run": {
            "run_id": best_run_row.get("run_id", ""),
            "run_timestamp": best_run_row.get("run_timestamp", ""),
            "experiment_name": best_run_row.get("experiment_name", ""),
            "config_version": best_run_row.get("config_version", ""),
            "model_name": best_run_row.get("model_name", ""),
            "feature_set": best_run_row.get("feature_set", ""),
            "text_variant": best_run_row.get("text_variant", ""),
            "params": params_obj,
            "metrics": {
                "val": {
                    "f1_macro": safe_float(best_run_row.get("val_f1_macro")),
                    "precision_macro": safe_float(best_run_row.get("val_precision_macro")),
                    "recall_macro": safe_float(best_run_row.get("val_recall_macro")),
                    "accuracy": safe_float(best_run_row.get("val_accuracy")),
                    "f1_fake": safe_float(best_run_row.get("val_f1_fake")),
                },
                "test": {
                    "f1_macro": safe_float(best_run_row.get("test_f1_macro")),
                    "precision_macro": safe_float(best_run_row.get("test_precision_macro")),
                    "recall_macro": safe_float(best_run_row.get("test_recall_macro")),
                    "accuracy": safe_float(best_run_row.get("test_accuracy")),
                    "f1_fake": safe_float(best_run_row.get("test_f1_fake")),
                },
            },
            "notes": notes_obj,
        },
    }


# Write best configuration JSON file for downstream training/inference.
def write_best_config(best_config_path: Path, payload: dict[str, Any]) -> None:
    best_config_path.parent.mkdir(parents=True, exist_ok=True)
    with best_config_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# Prepare and cache vectorized splits for one text-variant and feature-set pair.
def prepare_vectorized_cache_entry(
    config: dict[str, Any],
    config_path: Path,
    text_variant: str,
    feature_set: str,
    strict_overlap: bool,
) -> dict[str, Any]:
    xy_splits = get_xy_splits(
        text_variant=text_variant,
        config_path=config_path,
        strict_overlap=strict_overlap,
        print_summary=False,
    )
    x_train, y_train = xy_splits["train"]
    x_val, y_val = xy_splits["val"]
    x_test, y_test = xy_splits["test"]

    vectorizer, vectorizer_meta = build_feature_pipeline(feature_set_name=feature_set, config=config)
    vec_fit_start = time.perf_counter()
    x_train_vec = vectorizer.fit_transform(x_train)
    vec_fit_seconds = round(time.perf_counter() - vec_fit_start, 6)

    vec_transform_start = time.perf_counter()
    x_val_vec = vectorizer.transform(x_val)
    x_test_vec = vectorizer.transform(x_test)
    vec_transform_seconds = round(time.perf_counter() - vec_transform_start, 6)

    return {
        "vectorizer": vectorizer,
        "vectorizer_meta": vectorizer_meta,
        "x_train_vec": x_train_vec,
        "y_train": y_train,
        "x_val_vec": x_val_vec,
        "y_val": y_val,
        "x_test_vec": x_test_vec,
        "y_test": y_test,
        "vec_fit_seconds": vec_fit_seconds,
        "vec_transform_seconds": vec_transform_seconds,
    }


# Parse parameter JSON from one run row.
def parse_run_params(run_row: dict[str, Any]) -> dict[str, Any]:
    raw = str(run_row.get("params_json", "")).strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GridSearchError(f"Failed to parse params_json for run_id={run_row.get('run_id')}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise GridSearchError(f"params_json for run_id={run_row.get('run_id')} must decode to object.")
    return parsed


# Retrain best run from cached vectorized data and save model/vectorizer/metadata artifacts.
def save_best_run_artifacts(
    config: dict[str, Any],
    runs_path: Path,
    best_run_row: dict[str, Any],
    cache: dict[tuple[str, str], dict[str, Any]],
    config_path: Path,
    random_state: int,
) -> dict[str, Any]:
    run_id = str(best_run_row.get("run_id", "")).strip()
    model_name = str(best_run_row.get("model_name", "")).strip()
    feature_set = str(best_run_row.get("feature_set", "")).strip()
    text_variant = str(best_run_row.get("text_variant", "")).strip()
    experiment_name = str(best_run_row.get("experiment_name", "")).strip()
    config_version = best_run_row.get("config_version", config.get("version"))
    run_timestamp = str(best_run_row.get("run_timestamp", "")).strip() or now_iso()

    if not run_id or not model_name or not feature_set or not text_variant:
        raise GridSearchError("Best run row is missing required fields to save artifacts.")

    cache_key = (text_variant, feature_set)
    if cache_key not in cache:
        cache[cache_key] = prepare_vectorized_cache_entry(
            config=config,
            config_path=config_path,
            text_variant=text_variant,
            feature_set=feature_set,
            strict_overlap=True,
        )
    cache_entry = cache[cache_key]
    params = parse_run_params(best_run_row)

    model, final_params = build_model(model_name, params, random_state=random_state)
    model_fit_seconds = fit_model(model, cache_entry["x_train_vec"], cache_entry["y_train"])
    y_val_pred, val_predict_seconds = predict_model(model, cache_entry["x_val_vec"])
    y_test_pred, test_predict_seconds = predict_model(model, cache_entry["x_test_vec"])
    val_metrics = compute_metrics(cache_entry["y_val"], y_val_pred)
    test_metrics = compute_metrics(cache_entry["y_test"], y_test_pred)

    feature_space_diag = compute_feature_space_diagnostics(
        vectorizer_meta=cache_entry["vectorizer_meta"],
        observed_feature_count=int(cache_entry["x_train_vec"].shape[1]),
    )

    metadata = {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "experiment_name": experiment_name,
        "config_path": str(config_path.as_posix()),
        "config_version": config_version,
        "selection": {
            "text_variant": text_variant,
            "feature_set": feature_set,
            "model_name": model_name,
            "params": final_params,
            "random_state": random_state,
            "source": "ml_grid_search_best_retrain",
        },
        "shape": {
            "train_rows": int(cache_entry["x_train_vec"].shape[0]),
            "val_rows": int(cache_entry["x_val_vec"].shape[0]),
            "test_rows": int(cache_entry["x_test_vec"].shape[0]),
            "feature_count": int(cache_entry["x_train_vec"].shape[1]),
            "train_nnz": int(cache_entry["x_train_vec"].nnz),
            "val_nnz": int(cache_entry["x_val_vec"].nnz),
            "test_nnz": int(cache_entry["x_test_vec"].nnz),
        },
        "timing_seconds": {
            "vectorize_fit": cache_entry["vec_fit_seconds"],
            "vectorize_transform": cache_entry["vec_transform_seconds"],
            "model_fit": model_fit_seconds,
            "val_predict": val_predict_seconds,
            "test_predict": test_predict_seconds,
        },
        "metrics": {
            "val": val_metrics,
            "test": test_metrics,
        },
        "vectorizer_meta": cache_entry["vectorizer_meta"],
        "feature_space_diagnostics": feature_space_diag,
    }

    artifact_paths = save_artifacts(
        run_id=run_id,
        model=model,
        vectorizer=cache_entry["vectorizer"],
        metadata=metadata,
        save_as_best=True,
    )

    notes_payload = {
        "source": "ml_grid_search",
        "vectorize_fit_seconds": cache_entry["vec_fit_seconds"],
        "vectorize_transform_seconds": cache_entry["vec_transform_seconds"],
        "model_fit_seconds": model_fit_seconds,
        "val_predict_seconds": val_predict_seconds,
        "test_predict_seconds": test_predict_seconds,
        "artifacts": artifact_paths,
    }
    update_run_notes_in_csv(runs_path, run_id, notes_payload)
    best_run_row["notes"] = json.dumps(notes_payload, ensure_ascii=False, separators=(",", ":"))
    return artifact_paths


# Execute full grid search and persist runs, best config, and leaderboard.
def run_grid_search(
    config_path: str | Path | None = None,
    allow_overlap: bool = False,
    cli_max_trials: int = 0,
    dry_run: bool = False,
) -> tuple[int, int, Path, Path]:
    repo_root = get_repo_root()
    resolved_config_path = resolve_config_path(config_path)
    config = load_train_ml_config(resolved_config_path)

    text_variants = get_text_variants(config)
    feature_sets = get_feature_set_names(config)
    model_combos = get_model_param_combinations(config, max_combos_per_model=0)
    experiment_name, random_state, _, effective_max_trials = parse_experiment_meta(config, cli_max_trials)
    selection_primary_metric, selection_tie_breaker, primary_field, tie_field = parse_selection_fields(config)
    config_version = config.get("version")
    if not isinstance(config_version, int):
        raise GridSearchError("train_ml.yaml version must be integer.")

    planned_trials = len(text_variants) * len(feature_sets) * len(model_combos)
    run_trials = min(planned_trials, effective_max_trials)
    print(
        f"[ML GRID] Planned trials={planned_trials}, "
        f"effective_max_trials={effective_max_trials}, executing={run_trials}"
    )
    if dry_run:
        return 0, 0, repo_root / "experiments" / "ml" / "best_config.json", repo_root / "experiments" / "ml" / "leaderboard.csv"

    runs_path = repo_root / "experiments" / "ml" / "runs.csv"
    strict_overlap = not allow_overlap

    cache: dict[tuple[str, str], dict[str, Any]] = {}
    trial_index = 0
    pass_count = 0
    fail_count = 0

    for text_variant in text_variants:
        for feature_set in feature_sets:
            cache_key = (text_variant, feature_set)
            if trial_index >= run_trials:
                break

            print(f"[ML GRID] Preparing features for variant={text_variant}, feature_set={feature_set}")
            cache_entry = prepare_vectorized_cache_entry(
                config=config,
                config_path=resolved_config_path,
                text_variant=text_variant,
                feature_set=feature_set,
                strict_overlap=strict_overlap,
            )
            cache[cache_key] = cache_entry

            for model_name, params in model_combos:
                if trial_index >= run_trials:
                    break
                trial_index += 1
                print(
                    f"[ML GRID] Trial {trial_index}/{run_trials} | "
                    f"model={model_name} variant={text_variant} feature_set={feature_set} params={params}"
                )

                run_id = generate_run_id(model_name, feature_set, text_variant)
                run_timestamp = now_iso()
                params_json = json.dumps(params, ensure_ascii=False, separators=(",", ":"))
                run_record = build_base_run_record(
                    run_id=run_id,
                    run_timestamp=run_timestamp,
                    experiment_name=experiment_name,
                    config_version=config_version,
                    random_state=random_state,
                    model_name=model_name,
                    feature_set=feature_set,
                    text_variant=text_variant,
                    params_json=params_json,
                )

                try:
                    model, final_params = build_model(model_name, params, random_state=random_state)
                    model_fit_seconds = fit_model(model, cache_entry["x_train_vec"], cache_entry["y_train"])
                    y_val_pred, val_predict_seconds = predict_model(model, cache_entry["x_val_vec"])
                    y_test_pred, test_predict_seconds = predict_model(model, cache_entry["x_test_vec"])

                    val_metrics = compute_metrics(cache_entry["y_val"], y_val_pred)
                    test_metrics = compute_metrics(cache_entry["y_test"], y_test_pred)
                    notes_payload = {
                        "source": "ml_grid_search",
                        "vectorize_fit_seconds": cache_entry["vec_fit_seconds"],
                        "vectorize_transform_seconds": cache_entry["vec_transform_seconds"],
                        "model_fit_seconds": model_fit_seconds,
                        "val_predict_seconds": val_predict_seconds,
                        "test_predict_seconds": test_predict_seconds,
                    }

                    run_record.update(
                        {
                            "params_json": json.dumps(final_params, ensure_ascii=False, separators=(",", ":")),
                            "val_f1_macro": round(val_metrics["f1_macro"], 6),
                            "val_precision_macro": round(val_metrics["precision_macro"], 6),
                            "val_recall_macro": round(val_metrics["recall_macro"], 6),
                            "val_accuracy": round(val_metrics["accuracy"], 6),
                            "val_f1_fake": round(val_metrics["f1_fake"], 6),
                            "test_f1_macro": round(test_metrics["f1_macro"], 6),
                            "test_precision_macro": round(test_metrics["precision_macro"], 6),
                            "test_recall_macro": round(test_metrics["recall_macro"], 6),
                            "test_accuracy": round(test_metrics["accuracy"], 6),
                            "test_f1_fake": round(test_metrics["f1_fake"], 6),
                            "status": "PASS",
                            "notes": json.dumps(notes_payload, ensure_ascii=False, separators=(",", ":")),
                        }
                    )
                    pass_count += 1
                    print(
                        f"[ML GRID] PASS | val_{selection_primary_metric}={run_record.get(primary_field, '')} "
                        f"test_f1_macro={run_record.get('test_f1_macro', '')}"
                    )
                except Exception as exc:
                    run_record["status"] = "FAIL"
                    run_record["notes"] = str(exc)
                    fail_count += 1
                    print(f"[ML GRID] FAIL | {exc}")
                finally:
                    append_run_record(runs_path, run_record)
        if trial_index >= run_trials:
            break

    all_runs = load_runs_rows(runs_path)
    pass_rows = [
        row
        for row in all_runs
        if str(row.get("experiment_name", "")) == experiment_name
        and str(row.get("status", "")).upper() == "PASS"
    ]
    if not pass_rows:
        raise GridSearchError("Grid search finished with no PASS rows to select best config.")

    pass_rows.sort(
        key=lambda row: (
            safe_float(row.get(primary_field)),
            safe_float(row.get(tie_field)),
            safe_float(row.get("test_f1_macro")),
            parse_run_timestamp(str(row.get("run_timestamp", ""))),
        ),
        reverse=True,
    )
    best_run_row = pass_rows[0]
    print(
        f"[ML GRID] Best run selected | run_id={best_run_row.get('run_id','')} "
        f"model={best_run_row.get('model_name','')} "
        f"feature_set={best_run_row.get('feature_set','')} "
        f"text_variant={best_run_row.get('text_variant','')}"
    )

    artifact_paths = save_best_run_artifacts(
        config=config,
        runs_path=runs_path,
        best_run_row=best_run_row,
        cache=cache,
        config_path=resolved_config_path,
        random_state=random_state,
    )
    print(f"[ML GRID] Best artifacts saved: {artifact_paths}")

    leaderboard_rows = [
        build_leaderboard_row(
            run_row=row,
            primary_metric=selection_primary_metric,
            selection_metric=selection_primary_metric,
        )
        for row in pass_rows
    ]
    leaderboard_path = repo_root / "experiments" / "ml" / "leaderboard.csv"
    write_leaderboard_csv(leaderboard_path, leaderboard_rows)

    best_payload = build_best_config_payload(
        best_run_row=best_run_row,
        selection_primary_metric=selection_primary_metric,
        selection_tie_breaker=selection_tie_breaker,
    )
    best_config_path = repo_root / "experiments" / "ml" / "best_config.json"
    write_best_config(best_config_path, best_payload)

    print(
        "[ML GRID] Completed | "
        f"pass={pass_count} fail={fail_count} "
        f"best_run_id={best_run_row.get('run_id', '')}"
    )
    print(f"[ML GRID] best_config: {best_config_path.as_posix()}")
    print(f"[ML GRID] leaderboard: {leaderboard_path.as_posix()}")
    return pass_count, fail_count, best_config_path, leaderboard_path


# Parse run timestamp safely for ranking when timestamps have mixed formats.
def parse_run_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return datetime.min


# Execute CLI entrypoint and enforce fail-fast on trial failures.
def main() -> None:
    args = parse_args()
    pass_count, fail_count, _, _ = run_grid_search(
        config_path=args.config_path,
        allow_overlap=args.allow_overlap,
        cli_max_trials=args.max_trials,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print("[ML GRID] Dry-run done.")
        return
    if fail_count > 0:
        raise SystemExit(f"[ML GRID ERROR] Completed with failures. pass={pass_count}, fail={fail_count}")


# Expose grid search entrypoint with strict error mapping for CI and local runs.
if __name__ == "__main__":
    try:
        main()
    except (
        GridSearchError,
        TrainRunError,
        ModelConfigError,
        ModelRuntimeError,
        DatasetLoaderError,
        DatasetValidationError,
        VectorizerConfigError,
    ) as exc:
        raise SystemExit(f"[ML GRID ERROR] {exc}")
