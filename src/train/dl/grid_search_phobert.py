from __future__ import annotations

import argparse
import copy
import csv
import json
import itertools
import random
import sys
import tempfile
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
    from transformers import AutoConfig, AutoTokenizer
except ModuleNotFoundError as exc:
    raise SystemExit(
        "[DL GRID ERROR] Missing DL dependencies. Please install requirements including torch and transformers."
    ) from exc

try:
    from src.train.dl.train_phobert import (
        RUNS_COLUMNS,
        DlTrainError,
        get_repo_root,
        load_splits,
        load_yaml,
        now_iso,
        train_phobert,
    )
except ModuleNotFoundError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from train_phobert import (  # type: ignore
        RUNS_COLUMNS,
        DlTrainError,
        get_repo_root,
        load_splits,
        load_yaml,
        now_iso,
        train_phobert,
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

SEARCH_TRIAL_COLUMNS = [
    "trial_id",
    "learning_rate",
    "max_length",
    "train_batch_size",
    "val_primary_metric",
    "val_tie_breaker",
    "status",
    "error_message",
    "elapsed_time",
]

LEGACY_SEARCH_TRIAL_COLUMNS = [
    "trial_id",
    "learning_rate",
    "max_length",
    "train_batch_size",
    "val_f1_macro_epoch1",
    "val_f1_fake",
    "status",
    "error_message",
    "elapsed_time",
]


class DlHyperparamSearchError(Exception):
    pass


# Parse CLI arguments for DL grid search execution.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DL grid search and persist best config + leaderboard.")
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/train_dl.yaml",
        help="Path to train_dl.yaml",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=-1,   # 0 or negative means all combinations in the search space (= all candidates).
        help="Number of grid trials to execute (0 or negative means all).",
    )
    parser.add_argument(
        "--num-trials-exploit",
        type=int,
        default=3,
        help="Top-K configs selected for full training.",
    )
    parser.add_argument(
        "--early-epochs",
        type=int,
        default=2,
        help="Number of epochs used in explore phase.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for random_state; if omitted, config experiment.random_state is used.",
    )
    parser.add_argument(
        "--output-best-config",
        type=str,
        default="experiments/dl/best_config.json",
        help="Path to save best_config.json",
    )
    return parser.parse_args()


# Validate and extract selection metrics (primary + tie breaker) from config.
def ensure_selection_fields(config: dict[str, Any]) -> tuple[str, str, str, str]:
    selection_cfg = config.get("selection")
    if isinstance(selection_cfg, dict):
        primary_metric = str(selection_cfg.get("choose_best_by", "")).strip()
        tie_breaker = str(selection_cfg.get("tie_breaker", "")).strip()
    else:
        experiment_cfg = config.get("experiment", {})
        if not isinstance(experiment_cfg, dict):
            raise DlHyperparamSearchError("train_dl.yaml must define experiment or selection section.")
        primary_metric = str(experiment_cfg.get("primary_metric", "")).strip()
        tie_breaker = str(experiment_cfg.get("tie_breaker", "")).strip()

    if not primary_metric or not tie_breaker:
        raise DlHyperparamSearchError("selection.choose_best_by and selection.tie_breaker must be non-empty.")

    return primary_metric, tie_breaker, f"val_{primary_metric}", f"val_{tie_breaker}"


# Parse experiment metadata needed for run tracking and search controls.
def parse_experiment_meta(config: dict[str, Any]) -> tuple[str, int]:
    experiment_cfg = config.get("experiment")
    if not isinstance(experiment_cfg, dict):
        raise DlHyperparamSearchError("train_dl.yaml must define experiment section.")

    experiment_name = str(experiment_cfg.get("name", "phobert")).strip() or "phobert"
    random_state = experiment_cfg.get("random_state")
    if not isinstance(random_state, int):
        raise DlHyperparamSearchError("experiment.random_state must be integer.")

    return experiment_name, random_state


# Load base training config and validate required sections including search_space.
def load_base_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    repo_root = get_repo_root()
    resolved = Path(config_path)
    if not resolved.is_absolute():
        resolved = repo_root / resolved

    config = load_yaml(resolved)

    required_sections = ["data", "model", "training", "runtime", "experiment", "search_space"]
    for section in required_sections:
        if not isinstance(config.get(section), dict):
            raise DlHyperparamSearchError(f"train_dl.yaml missing required section: {section}")

    search_space = config["search_space"]
    for key in ["learning_rate", "max_length", "train_batch_size"]:
        values = search_space.get(key)
        if not isinstance(values, list) or not values:
            raise DlHyperparamSearchError(f"search_space.{key} must be a non-empty list.")

    ensure_selection_fields(config)
    return config, resolved


# Set global random seeds for reproducibility (random, numpy, torch, cuda).
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Build deterministic grid-search combinations from the search space.
def build_grid_hyperparams(search_space: dict[str, Any]) -> list[dict[str, Any]]:
    learning_rates = list(dict.fromkeys(float(v) for v in search_space["learning_rate"]))
    max_lengths = list(dict.fromkeys(int(v) for v in search_space["max_length"]))
    batch_sizes = list(dict.fromkeys(int(v) for v in search_space["train_batch_size"]))

    candidates: list[dict[str, Any]] = []
    for learning_rate, max_length, batch_size in itertools.product(learning_rates, max_lengths, batch_sizes):
        candidates.append(
            {
                "learning_rate": float(learning_rate),
                "max_length": int(max_length),
                "train_batch_size": int(batch_size),
            }
        )
    return candidates


# Ensure search_trials.csv exists with correct header.
def ensure_search_trials_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            if header == SEARCH_TRIAL_COLUMNS:
                return
            if header == LEGACY_SEARCH_TRIAL_COLUMNS:
                rows = list(reader)
        if header == LEGACY_SEARCH_TRIAL_COLUMNS:
            with path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=SEARCH_TRIAL_COLUMNS)
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            "trial_id": row.get("trial_id", ""),
                            "learning_rate": row.get("learning_rate", ""),
                            "max_length": row.get("max_length", ""),
                            "train_batch_size": row.get("train_batch_size", ""),
                            "val_primary_metric": row.get("val_f1_macro_epoch1", ""),
                            "val_tie_breaker": row.get("val_f1_fake", ""),
                            "status": row.get("status", ""),
                            "error_message": row.get("error_message", ""),
                            "elapsed_time": row.get("elapsed_time", ""),
                        }
                    )
            return
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SEARCH_TRIAL_COLUMNS)
        writer.writeheader()


# Append one trial result row into search_trials.csv.
def append_search_trial(path: Path, row: dict[str, Any]) -> None:
    ensure_search_trials_csv(path)
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SEARCH_TRIAL_COLUMNS)
        writer.writerow({k: row.get(k, "") for k in SEARCH_TRIAL_COLUMNS})


# Build fallback sequence for (max_length, batch_size) to handle OOM.
def build_fallback_candidates(
    initial_max_length: int,
    initial_batch_size: int,
    allowed_max_lengths: list[int],
    allowed_batch_sizes: list[int],
) -> list[tuple[int, int]]:
    max_candidates = sorted({int(v) for v in allowed_max_lengths}, reverse=True)
    batch_candidates = sorted({int(v) for v in allowed_batch_sizes}, reverse=True)

    lower_or_equal_max = [v for v in max_candidates if v <= initial_max_length]
    lower_batch = [v for v in batch_candidates if v < initial_batch_size]

    sequence: list[tuple[int, int]] = []
    for max_length in lower_or_equal_max:
        sequence.append((max_length, initial_batch_size))

    min_length = lower_or_equal_max[-1] if lower_or_equal_max else initial_max_length
    for batch_size in lower_batch:
        sequence.append((min_length, batch_size))

    deduped: list[tuple[int, int]] = []
    seen = set()
    for item in sequence:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


# Detect if an exception is related to CUDA/CPU out-of-memory.
def is_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or ("cuda error" in message and "alloc" in message)


# Safely convert value to float, fallback to -inf if invalid.
def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("-inf")


# Parse ISO timestamp safely for sorting runs.
def parse_run_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return datetime.min


# Load existing runs.csv rows and validate schema.
def load_runs_rows(runs_path: Path) -> list[dict[str, Any]]:
    if not runs_path.exists():
        return []
    with runs_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if header != RUNS_COLUMNS:
            raise DlHyperparamSearchError(
                "runs.csv header mismatch.\n"
                f"Expected: {RUNS_COLUMNS}\nFound:    {header}"
            )
        return list(reader)


# Write full runs.csv with provided rows.
def write_runs_rows(runs_path: Path, rows: list[dict[str, Any]]) -> None:
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with runs_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in RUNS_COLUMNS})


# Update notes field of a specific run_id inside runs.csv.
def update_run_notes_in_csv(runs_path: Path, run_id: str, notes_payload: dict[str, Any]) -> None:
    rows = load_runs_rows(runs_path)
    notes_json = json.dumps(notes_payload, ensure_ascii=False, separators=(",", ":"))
    for row in rows:
        if str(row.get("run_id", "")).strip() == run_id:
            row["notes"] = notes_json
            break
    write_runs_rows(runs_path, rows)


# Convert a run row into leaderboard row format.
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


# Write leaderboard.csv from sorted rows.
def write_leaderboard_csv(leaderboard_path: Path, rows: list[dict[str, Any]]) -> None:
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    with leaderboard_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in LEADERBOARD_COLUMNS})


# Build best_config.json payload from best run row.
def build_best_config_payload(best_run_row: dict[str, Any], primary_metric: str, tie_breaker: str) -> dict[str, Any]:
    params_raw = str(best_run_row.get("params_json", "")).strip()
    try:
        params_obj = json.loads(params_raw) if params_raw else {}
    except json.JSONDecodeError:
        params_obj = {"raw_params_json": params_raw}

    notes_raw = str(best_run_row.get("notes", "")).strip()
    try:
        notes_obj: Any = json.loads(notes_raw) if notes_raw else {}
    except json.JSONDecodeError:
        notes_obj = notes_raw

    return {
        "selected_at": now_iso(),
        "selection_rule": {
            "primary_metric": primary_metric,
            "tie_breaker": tie_breaker,
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


# Persist best_config.json to disk.
def write_best_config(best_config_path: Path, payload: dict[str, Any]) -> None:
    best_config_path.parent.mkdir(parents=True, exist_ok=True)
    with best_config_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# Resolve the physical run directory for a run row using notes metadata or fallback convention.
def resolve_run_dir(run_row: dict[str, Any], repo_root: Path) -> Path:
    run_id = str(run_row.get("run_id", "")).strip()
    if not run_id:
        raise DlHyperparamSearchError("run_id is missing from the best run row.")

    notes_raw = str(run_row.get("notes", "")).strip()
    if notes_raw:
        try:
            notes_obj = json.loads(notes_raw)
            if isinstance(notes_obj, dict):
                run_dir_value = str(notes_obj.get("run_dir", "")).strip()
                if run_dir_value:
                    return Path(run_dir_value)
        except json.JSONDecodeError:
            pass

    return repo_root / "models" / "dl" / run_id


# Promote the selected best run into top-level models/dl artifacts, similar to ML best artifact promotion.
def save_best_run_artifacts(best_run_row: dict[str, Any], repo_root: Path) -> dict[str, str]:
    run_dir = resolve_run_dir(best_run_row, repo_root)
    checkpoint_dir = run_dir / "best_checkpoint"
    metrics_path = run_dir / "metrics.json"
    metadata_path = run_dir / "metadata.json"

    if not checkpoint_dir.exists():
        raise DlHyperparamSearchError(f"Missing best checkpoint directory for best run: {checkpoint_dir}")
    if not metrics_path.exists():
        raise DlHyperparamSearchError(f"Missing metrics.json for best run: {metrics_path}")
    if not metadata_path.exists():
        raise DlHyperparamSearchError(f"Missing metadata.json for best run: {metadata_path}")

    model_dir = repo_root / "models" / "dl"
    model_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(best_run_row.get("run_id", "")).strip()
    best_checkpoint_copy = model_dir / "best_checkpoint"
    best_metrics_copy = model_dir / "best_metrics.json"
    best_metadata_copy = model_dir / "best_metadata.json"
    best_manifest_path = model_dir / "best_run.json"
    best_run_id_path = model_dir / "best_run_id.txt"

    if best_checkpoint_copy.exists():
        shutil.rmtree(best_checkpoint_copy)
    shutil.copytree(checkpoint_dir, best_checkpoint_copy)
    shutil.copy2(metrics_path, best_metrics_copy)
    shutil.copy2(metadata_path, best_metadata_copy)

    best_manifest = {
        "run_id": run_id,
        "run_timestamp": best_run_row.get("run_timestamp", ""),
        "experiment_name": best_run_row.get("experiment_name", ""),
        "config_version": best_run_row.get("config_version", ""),
        "model_name": best_run_row.get("model_name", ""),
        "feature_set": best_run_row.get("feature_set", ""),
        "text_variant": best_run_row.get("text_variant", ""),
        "params_json": best_run_row.get("params_json", ""),
        "source_run_dir": str(run_dir.as_posix()),
        "best_checkpoint_dir": str(best_checkpoint_copy.as_posix()),
        "best_metrics_path": str(best_metrics_copy.as_posix()),
        "best_metadata_path": str(best_metadata_copy.as_posix()),
        "saved_at": now_iso(),
    }
    with best_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(best_manifest, f, ensure_ascii=False, indent=2)
    with best_run_id_path.open("w", encoding="utf-8") as f:
        f.write(f"{run_id}\n")

    artifact_paths = {
        "best_checkpoint_dir": str(best_checkpoint_copy.as_posix()),
        "best_metrics_path": str(best_metrics_copy.as_posix()),
        "best_metadata_path": str(best_metadata_copy.as_posix()),
        "best_manifest_path": str(best_manifest_path.as_posix()),
        "best_run_id_path": str(best_run_id_path.as_posix()),
    }
    return artifact_paths


# Run one explore-phase trial with the configured early epochs and return metrics.
def run_explore_trial(
    base_config: dict[str, Any],
    splits: dict[str, Any],
    tokenizer: Any,
    encoded_cache: dict[tuple[int, str], dict[str, torch.Tensor]],
    params: dict[str, Any],
    early_epochs: int,
    seed: int,
    trial_id: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = copy.deepcopy(base_config)

    cfg["model"]["max_length"] = int(params["max_length"])
    cfg["training"]["train_batch_size"] = int(params["train_batch_size"])
    cfg["training"]["learning_rate"] = float(params["learning_rate"])
    cfg["training"]["num_epochs"] = int(early_epochs)
    cfg["training"]["early_stopping_patience"] = int(early_epochs)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)
            temp_path = Path(f.name)

        explore_artifacts = train_phobert(
            config_path=temp_path,
            mode="explore",
            early_epochs=int(early_epochs),
            track_runs=False,
            save_artifacts=False,
            splits_override=splits,
            tokenizer_override=tokenizer,
            encoded_cache=encoded_cache,
        )
        val_metrics = explore_artifacts.val_metrics
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)

    row = {
        "trial_id": trial_id,
        "learning_rate": float(params["learning_rate"]),
        "max_length": int(params["max_length"]),
        "train_batch_size": int(params["train_batch_size"]),
        "val_primary_metric": round(float(val_metrics.get("f1_macro", 0.0)), 6),
        "val_tie_breaker": round(float(val_metrics.get("f1_fake", 0.0)), 6),
        "status": "PASS",
        "error_message": "",
        "elapsed_time": "",
    }
    return row, val_metrics


# Run full training for one config with OOM fallback strategy.
def run_exploit_trial(
    base_config: dict[str, Any],
    params: dict[str, Any],
    trial_id: int,
    seed: int,
    allowed_max_lengths: list[int],
    allowed_batch_sizes: list[int],
) -> tuple[str | None, dict[str, Any] | None]:
    fallback_candidates = build_fallback_candidates(
        initial_max_length=int(params["max_length"]),
        initial_batch_size=int(params["train_batch_size"]),
        allowed_max_lengths=allowed_max_lengths,
        allowed_batch_sizes=allowed_batch_sizes,
    )

    repo_root = get_repo_root()
    last_error: str = ""

    for max_length, batch_size in fallback_candidates:
        cfg = copy.deepcopy(base_config)
        cfg["model"]["max_length"] = int(max_length)
        cfg["training"]["train_batch_size"] = int(batch_size)
        cfg["training"]["learning_rate"] = float(params["learning_rate"])
        cfg["experiment"]["random_state"] = int(seed)

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=False, sort_keys=False)
                temp_path = Path(f.name)

            artifacts = train_phobert(config_path=temp_path, output_root_override=cfg["runtime"].get("output_root"))

            runs_path = repo_root / "experiments" / "dl" / "runs.csv"
            notes_payload = {
                "source": "dl_hyperparam_search",
                "trial_id": int(trial_id),
                "used_params": {
                    "learning_rate": float(params["learning_rate"]),
                    "max_length": int(max_length),
                    "train_batch_size": int(batch_size),
                },
                "run_dir": str(artifacts.run_dir.as_posix()),
                "metrics_path": str((artifacts.run_dir / "metrics.json").as_posix()),
            }
            update_run_notes_in_csv(runs_path, artifacts.run_id, notes_payload)

            return artifacts.run_id, {
                "learning_rate": float(params["learning_rate"]),
                "max_length": int(max_length),
                "train_batch_size": int(batch_size),
            }
        except Exception as exc:
            last_error = str(exc)
            if is_oom_error(exc):
                print(
                    f"[DL GRID] OOM fallback | trial={trial_id} max_len={max_length} batch={batch_size}",
                    flush=True,
                )
                continue
            break
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    print(f"[DL GRID] FAIL | trial={trial_id} reason={last_error}", flush=True)
    return None, None


# Main orchestration: explore -> exploit -> ranking -> save artifacts.
def run_search(args: argparse.Namespace) -> tuple[int, int, Path, Path]:
    repo_root = get_repo_root()
    config, _ = load_base_config(args.config_path)
    primary_metric, tie_breaker, primary_field, tie_field = ensure_selection_fields(config)

    experiment_name, config_random_state = parse_experiment_meta(config)
    random_state = int(args.seed) if args.seed is not None else config_random_state

    set_seed(random_state)

    search_space = config["search_space"]

    pretrained_name = str(config["model"].get("pretrained_name", "")).strip()
    if not pretrained_name:
        raise DlHyperparamSearchError("model.pretrained_name must be non-empty")

    model_config = AutoConfig.from_pretrained(pretrained_name)
    model_max_positions = int(getattr(model_config, "max_position_embeddings", 0) or 0)
    if model_max_positions <= 0:
        raise DlHyperparamSearchError(
            f"model.pretrained_name={pretrained_name} has invalid max_position_embeddings={model_max_positions}."
        )
    
    original_lengths = [int(v) for v in search_space["max_length"]]
    allowed_lengths = sorted(set(int(v) for v in original_lengths if int(v) > 0 and int(v) <= model_max_positions))
    if not allowed_lengths:
        raise DlHyperparamSearchError(
            "search_space.max_length has no valid values in (0, "
            f"{model_max_positions}]. Valid values must be >= 1 and <= {model_max_positions}."
        )
    
    filtered_count = len(original_lengths) - len(allowed_lengths)
    if filtered_count > 0:
        print(
            f"[DL GRID] Filtered {filtered_count} max_length value(s) (max_position_embeddings={model_max_positions}): "
            f"original={original_lengths}, valid={allowed_lengths}",
            flush=True,
        )
    search_space["max_length"] = allowed_lengths

    all_candidates = build_grid_hyperparams(search_space=search_space)
    max_trials = int(args.num_trials)
    candidates = all_candidates if max_trials <= 0 else all_candidates[:max_trials]

    effective_max_trials = len(candidates)
    print(
        f"[DL GRID] Planned trials={len(all_candidates)}, "
        f"effective_max_trials={effective_max_trials}, executing={effective_max_trials}",
        flush=True,
    )

    search_trials_path = repo_root / "experiments" / "dl" / "search_trials.csv"
    ensure_search_trials_csv(search_trials_path)

    # Reuse split tensors and tokenizer across explore trials to reduce repeated overhead.
    explore_splits = load_splits(config, repo_root)
    explore_tokenizer = AutoTokenizer.from_pretrained(pretrained_name, use_fast=False)
    encoded_cache: dict[tuple[int, str], dict[str, torch.Tensor]] = {}

    explore_rows: list[dict[str, Any]] = []
    pass_count = 0
    fail_count = 0

    for idx, params in enumerate(candidates, start=1):
        print(
            f"[DL GRID] Trial {idx}/{effective_max_trials} | "
            f"model=phobert variant=text_clean feature_set=transformer params={params}",
            flush=True,
        )
        start = time.perf_counter()

        fallback_candidates = build_fallback_candidates(
            initial_max_length=int(params["max_length"]),
            initial_batch_size=int(params["train_batch_size"]),
            allowed_max_lengths=[int(v) for v in search_space["max_length"]],
            allowed_batch_sizes=[int(v) for v in search_space["train_batch_size"] + [4, 2]],
        )

        trial_result: dict[str, Any] | None = None
        for max_length, batch_size in fallback_candidates:
            attempt_params = dict(params)
            attempt_params["max_length"] = int(max_length)
            attempt_params["train_batch_size"] = int(batch_size)
            try:
                row, _ = run_explore_trial(
                    base_config=config,
                    splits=explore_splits,
                    tokenizer=explore_tokenizer,
                    encoded_cache=encoded_cache,
                    params=attempt_params,
                    early_epochs=int(args.early_epochs),
                    seed=random_state,
                    trial_id=idx,
                )
                elapsed = round(time.perf_counter() - start, 3)
                row["elapsed_time"] = elapsed
                append_search_trial(search_trials_path, row)
                explore_rows.append(row)
                pass_count += 1
                trial_result = row
                print(
                    f"[DL GRID] PASS | val_{primary_metric}={row['val_primary_metric']} "
                    f"val_{tie_breaker}={row['val_tie_breaker']}",
                    flush=True,
                )
                break
            except Exception as exc:
                if is_oom_error(exc):
                    print(f"[DL GRID] OOM fallback | trial={idx} max_len={max_length} batch={batch_size}", flush=True)
                    continue

                elapsed = round(time.perf_counter() - start, 3)
                row = {
                    "trial_id": idx,
                    "learning_rate": float(attempt_params["learning_rate"]),
                    "max_length": int(attempt_params["max_length"]),
                    "train_batch_size": int(attempt_params["train_batch_size"]),
                    "val_primary_metric": "",
                    "val_tie_breaker": "",
                    "status": "FAIL",
                    "error_message": str(exc),
                    "elapsed_time": elapsed,
                }
                append_search_trial(search_trials_path, row)
                explore_rows.append(row)
                fail_count += 1
                print(f"[DL GRID] FAIL | {exc}", flush=True)
                trial_result = row
                break

        if trial_result is None:
            elapsed = round(time.perf_counter() - start, 3)
            row = {
                "trial_id": idx,
                "learning_rate": float(params["learning_rate"]),
                "max_length": int(params["max_length"]),
                "train_batch_size": int(params["train_batch_size"]),
                "val_primary_metric": "",
                "val_tie_breaker": "",
                "status": "FAIL",
                "error_message": "OOM after all fallbacks",
                "elapsed_time": elapsed,
            }
            append_search_trial(search_trials_path, row)
            explore_rows.append(row)
            fail_count += 1
            print("[DL GRID] FAIL | OOM after all fallbacks", flush=True)

    print(f"[DL GRID] Explore phase completed ({effective_max_trials} trials)", flush=True)

    pass_explore_rows = [row for row in explore_rows if str(row.get("status", "")).upper() == "PASS"]
    if not pass_explore_rows:
        raise DlHyperparamSearchError("Explore phase finished with no PASS trials.")

    pass_explore_rows.sort(
        key=lambda row: (
            safe_float(row.get("val_primary_metric")),
            safe_float(row.get("val_tie_breaker")),
        ),
        reverse=True,
    )

    top_k = max(1, int(args.num_trials_exploit))
    top_configs = pass_explore_rows[:top_k]
    print(f"[DL GRID] Selected TOP-{len(top_configs)} configs for full training", flush=True)

    exploit_pass = 0
    exploit_fail = 0
    runs_path = repo_root / "experiments" / "dl" / "runs.csv"

    for idx, row in enumerate(top_configs, start=1):
        params = {
            "learning_rate": float(row["learning_rate"]),
            "max_length": int(row["max_length"]),
            "train_batch_size": int(row["train_batch_size"]),
        }

        run_id, used_params = run_exploit_trial(
            base_config=config,
            params=params,
            trial_id=int(row["trial_id"]),
            seed=random_state,
            allowed_max_lengths=[int(v) for v in search_space["max_length"]],
            allowed_batch_sizes=[int(v) for v in search_space["train_batch_size"] + [4, 2]],
        )

        if run_id and used_params:
            exploit_pass += 1
            all_rows = load_runs_rows(runs_path)
            target = next((r for r in all_rows if str(r.get("run_id", "")).strip() == run_id), None)
            if target is not None:
                print(
                    f"[DL GRID] Full-Train {idx}/{len(top_configs)} | "
                    f"val_f1_macro={target.get('val_f1_macro', '')} "
                    f"test_f1_macro={target.get('test_f1_macro', '')}",
                    flush=True,
                )
        else:
            exploit_fail += 1

    all_runs = load_runs_rows(runs_path)
    pass_rows = [
        row
        for row in all_runs
        if str(row.get("experiment_name", "")) == experiment_name and str(row.get("status", "")).upper() == "PASS"
    ]
    if not pass_rows:
        raise DlHyperparamSearchError("No PASS rows in runs.csv for this experiment after exploit phase.")

    pass_rows.sort(
        key=lambda row: (
            safe_float(row.get(primary_field)),
            safe_float(row.get(tie_field)),
            safe_float(row.get("test_f1_macro")),
            parse_run_timestamp(str(row.get("run_timestamp", ""))),
        ),
        reverse=True,
    )
    best_row = pass_rows[0]
    print(
        f"[DL GRID] Best run selected | run_id={best_row.get('run_id', '')} "
        f"model={best_row.get('model_name', '')} feature_set={best_row.get('feature_set', '')} "
        f"text_variant={best_row.get('text_variant', '')}",
        flush=True,
    )

    best_artifact_paths = save_best_run_artifacts(best_row, repo_root)
    print(f"[DL GRID] Best artifacts saved: {best_artifact_paths}", flush=True)

    leaderboard_rows = [
        build_leaderboard_row(run_row=row, primary_metric=primary_metric, selection_metric=primary_metric)
        for row in pass_rows
    ]
    leaderboard_path = repo_root / "experiments" / "dl" / "leaderboard.csv"
    write_leaderboard_csv(leaderboard_path, leaderboard_rows)

    output_best_config = Path(args.output_best_config)
    if not output_best_config.is_absolute():
        output_best_config = repo_root / output_best_config
    best_payload = build_best_config_payload(best_row, primary_metric=primary_metric, tie_breaker=tie_breaker)
    if isinstance(best_payload.get("best_run"), dict):
        best_payload["best_run"]["notes"] = {
            "source": "dl_hyperparam_search",
            "artifacts": best_artifact_paths,
        }
    best_payload["search_info"] = {
        "num_trials": int(args.num_trials),
        "num_trials_exploit": int(args.num_trials_exploit),
        "random_state": random_state,
        "search_mode": "grid",
    }
    write_best_config(output_best_config, best_payload)

    print(
        "[DL GRID] Completed | "
        f"pass={exploit_pass} fail={exploit_fail} "
        f"best_run_id={best_row.get('run_id', '')}",
        flush=True,
    )
    print(f"[DL GRID] best_config: {output_best_config.as_posix()}", flush=True)
    print(f"[DL GRID] leaderboard: {leaderboard_path.as_posix()}", flush=True)

    return exploit_pass, exploit_fail, output_best_config, leaderboard_path


# CLI entrypoint for hyperparameter search execution.
def main() -> None:
    args = parse_args()
    run_search(args)


if __name__ == "__main__":
    try:
        main()
    except (DlHyperparamSearchError, DlTrainError) as exc:
        raise SystemExit(f"[DL GRID ERROR] {exc}")