from __future__ import annotations

import csv
import json
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from joblib import dump
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

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
    from vectorizers import (  # type: ignore
        VectorizerConfigError,
        build_feature_pipeline,
        get_feature_set_names,
        get_text_variants,
        load_train_ml_config,
        resolve_config_path,
    )


RUNS_COLUMNS = [
    "run_id",
    "run_timestamp",
    "experiment_name",
    "config_version",
    "random_state",
    "model_name",
    "feature_set",
    "text_variant",
    "params_json",
    "val_f1_macro",
    "val_precision_macro",
    "val_recall_macro",
    "val_accuracy",
    "val_f1_fake",
    "test_f1_macro",
    "test_precision_macro",
    "test_recall_macro",
    "test_accuracy",
    "test_f1_fake",
    "status",
    "notes",
]


class TrainRunError(Exception):
    pass


# Resolve repository root path from current file location.
def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# Return local timestamp string for run tracking.
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# Create deterministic run ID with timestamp and short random suffix.
def generate_run_id(model_name: str, feature_set: str, text_variant: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    safe_model = model_name.replace(" ", "_")
    safe_feature = feature_set.replace(" ", "_")
    safe_variant = text_variant.replace(" ", "_")
    return f"{ts}_{safe_model}_{safe_feature}_{safe_variant}_{suffix}"


# Ensure runs CSV exists and has expected header before appending rows.
def ensure_runs_csv(runs_path: Path) -> None:
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    if not runs_path.exists():
        with runs_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RUNS_COLUMNS)
            writer.writeheader()
        return

    with runs_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    if header != RUNS_COLUMNS:
        raise TrainRunError(
            "experiments/ml/runs.csv header mismatch.\n"
            f"Expected: {RUNS_COLUMNS}\nFound:    {header}"
        )


# Append one run record row into experiments tracking file.
def append_run_record(runs_path: Path, run_record: dict[str, Any]) -> None:
    ensure_runs_csv(runs_path)
    with runs_path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_COLUMNS)
        writer.writerow({column: run_record.get(column, "") for column in RUNS_COLUMNS})


# Parse JSON params override string into dictionary format.
def parse_params_json(params_json: str | None) -> dict[str, Any] | None:
    if params_json is None:
        return None
    try:
        parsed = json.loads(params_json)
    except json.JSONDecodeError as exc:
        raise TrainRunError(f"Invalid --params-json value: {exc}") from exc
    if not isinstance(parsed, dict):
        raise TrainRunError("--params-json must decode to a JSON object.")
    return parsed


# Return optional run.default settings from train_ml config.
def get_run_default_config(config: dict[str, Any]) -> dict[str, Any]:
    run_section = config.get("run")
    if run_section is None:
        return {}
    if not isinstance(run_section, dict):
        raise TrainRunError("train_ml.yaml 'run' section must be a mapping.")

    default_cfg = run_section.get("default")
    if default_cfg is None:
        return {}
    if not isinstance(default_cfg, dict):
        raise TrainRunError("train_ml.yaml run.default must be a mapping.")

    return default_cfg


# Return first parameter combo available for one model name.
def get_first_params_for_model(
    model_name: str,
    model_combos: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    for combo_model_name, combo_params in model_combos:
        if combo_model_name == model_name:
            return dict(combo_params)
    raise TrainRunError(f"Model '{model_name}' has no available parameter combos.")


# Choose single run scope from config and optional CLI overrides.
def resolve_single_run_scope(
    config: dict[str, Any],
    requested_text_variant: str | None,
    requested_feature_set: str | None,
    requested_model_name: str | None,
    params_override: dict[str, Any] | None,
    max_combos_per_model: int,
) -> tuple[str, str, str, dict[str, Any]]:
    text_variants = get_text_variants(config)
    feature_sets = get_feature_set_names(config)
    run_default = get_run_default_config(config)
    default_text_variant = run_default.get("text_variant")
    default_feature_set = run_default.get("feature_set")
    default_model_name = run_default.get("model")
    default_params = run_default.get("params")

    selected_text_variant = requested_text_variant or default_text_variant or text_variants[0]
    selected_feature_set = requested_feature_set or default_feature_set or feature_sets[0]
    if selected_text_variant not in text_variants:
        raise TrainRunError(
            f"text_variant '{selected_text_variant}' is not available. Allowed: {text_variants}"
        )
    if selected_feature_set not in feature_sets:
        raise TrainRunError(
            f"feature_set '{selected_feature_set}' is not available. Allowed: {feature_sets}"
        )

    model_combos = get_model_param_combinations(config, max_combos_per_model=max_combos_per_model)
    combo_model_names = sorted({combo[0] for combo in model_combos})

    selected_model_name = requested_model_name or default_model_name
    if selected_model_name is None:
        selected_model_name = model_combos[0][0]
    if not isinstance(selected_model_name, str) or not selected_model_name.strip():
        raise TrainRunError("Selected model name must be a non-empty string.")
    selected_model_name = selected_model_name.strip()
    if selected_model_name not in combo_model_names:
        raise TrainRunError(
            f"Model '{selected_model_name}' is not available. Enabled models: {combo_model_names}"
        )

    if params_override is not None:
        return selected_text_variant, selected_feature_set, selected_model_name, params_override

    selected_params = get_first_params_for_model(selected_model_name, model_combos)
    if (
        requested_model_name is None
        and isinstance(default_model_name, str)
        and selected_model_name == default_model_name
        and isinstance(default_params, dict)
    ):
        selected_params.update(default_params)
    return selected_text_variant, selected_feature_set, selected_model_name, selected_params


# Compute evaluation metrics for one split using macro and fake-class targets.
def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "f1_macro": float(f1_macro),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
    }


# Compute diagnostics between configured max_features and observed vocabulary size.
def compute_feature_space_diagnostics(
    vectorizer_meta: dict[str, Any],
    observed_feature_count: int,
) -> dict[str, Any]:
    parts = vectorizer_meta.get("parts", [])
    expected_max_features = 0
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("max_features"), int):
                expected_max_features += int(part["max_features"])

    utilization_ratio = None
    if expected_max_features > 0:
        utilization_ratio = round(observed_feature_count / expected_max_features, 6)

    return {
        "expected_max_features": expected_max_features,
        "observed_feature_count": int(observed_feature_count),
        "utilization_ratio": utilization_ratio,
    }


# Save model, vectorizer, and metadata artifacts for one run.
def save_artifacts(
    run_id: str,
    model: Any,
    vectorizer: Any,
    metadata: dict[str, Any],
    save_as_best: bool,
) -> dict[str, str]:
    repo_root = get_repo_root()
    model_dir = repo_root / "models" / "ml"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{run_id}__model.joblib"
    vectorizer_path = model_dir / f"{run_id}__vectorizer.joblib"
    metadata_path = model_dir / f"{run_id}__metadata.json"

    dump(model, model_path)
    dump(vectorizer, vectorizer_path)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    artifact_paths = {
        "model_path": str(model_path.as_posix()),
        "vectorizer_path": str(vectorizer_path.as_posix()),
        "metadata_path": str(metadata_path.as_posix()),
    }

    if save_as_best:
        best_model_path = model_dir / "best_model.joblib"
        best_vectorizer_path = model_dir / "best_vectorizer.joblib"
        best_metadata_path = model_dir / "best_metadata.json"
        shutil.copy2(model_path, best_model_path)
        shutil.copy2(vectorizer_path, best_vectorizer_path)
        shutil.copy2(metadata_path, best_metadata_path)
        artifact_paths["best_model_path"] = str(best_model_path.as_posix())
        artifact_paths["best_vectorizer_path"] = str(best_vectorizer_path.as_posix())
        artifact_paths["best_metadata_path"] = str(best_metadata_path.as_posix())

    return artifact_paths


# Build a default run record structure with empty metric fields.
def init_run_record(
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


# Run one end-to-end ML training execution and persist artifacts plus run tracking.
def train_single_run(
    config_path: str | Path | None = None,
    text_variant: str | None = None,
    feature_set: str | None = None,
    model_name: str | None = None,
    params_json: str | None = None,
    allow_overlap: bool = False,
    max_combos_per_model: int = 0,
    save_as_best: bool = False,
) -> dict[str, Any]:
    resolved_config_path = resolve_config_path(config_path)
    config = load_train_ml_config(resolved_config_path)
    params_override = parse_params_json(params_json)

    experiment_cfg = config.get("experiment")
    if not isinstance(experiment_cfg, dict):
        raise TrainRunError("train_ml.yaml must define experiment section.")
    experiment_name = str(experiment_cfg.get("name", "ml_experiment")).strip() or "ml_experiment"
    random_state = experiment_cfg.get("random_state", "")
    if random_state != "" and not isinstance(random_state, int):
        raise TrainRunError("experiment.random_state must be integer.")
    config_version = config.get("version", "")
    if not isinstance(config_version, int):
        raise TrainRunError("train_ml.yaml version must be integer.")

    selected_text_variant, selected_feature_set, selected_model_name, selected_params = (
        resolve_single_run_scope(
            config=config,
            requested_text_variant=text_variant,
            requested_feature_set=feature_set,
            requested_model_name=model_name,
            params_override=params_override,
            max_combos_per_model=max_combos_per_model,
        )
    )

    run_id = generate_run_id(selected_model_name, selected_feature_set, selected_text_variant)
    run_timestamp = now_iso()
    params_string = json.dumps(selected_params, ensure_ascii=False, separators=(",", ":"))
    run_record = init_run_record(
        run_id=run_id,
        run_timestamp=run_timestamp,
        experiment_name=experiment_name,
        config_version=config_version,
        random_state=random_state,
        model_name=selected_model_name,
        feature_set=selected_feature_set,
        text_variant=selected_text_variant,
        params_json=params_string,
    )

    runs_path = get_repo_root() / "experiments" / "ml" / "runs.csv"

    try:
        print(
            "[ML TRAIN] Selected run scope | "
            f"text_variant={selected_text_variant} "
            f"feature_set={selected_feature_set} "
            f"model={selected_model_name} "
            f"params={selected_params}"
        )

        strict_overlap = not allow_overlap
        print("[ML TRAIN] Loading and validating train/val/test splits...")
        xy_splits = get_xy_splits(
            text_variant=selected_text_variant,
            config_path=resolved_config_path,
            strict_overlap=strict_overlap,
            print_summary=False,
        )
        x_train, y_train = xy_splits["train"]
        x_val, y_val = xy_splits["val"]
        x_test, y_test = xy_splits["test"]
        print(
            "[ML TRAIN] Dataset loaded | "
            f"train={len(x_train)} val={len(x_val)} test={len(x_test)}"
        )

        print("[ML TRAIN] Building feature pipeline...")
        vectorizer, vectorizer_meta = build_feature_pipeline(selected_feature_set, config)
        vec_fit_start = time.perf_counter()
        x_train_vec = vectorizer.fit_transform(x_train)
        vectorize_fit_seconds = round(time.perf_counter() - vec_fit_start, 6)
        print(
            "[ML TRAIN] Vectorizer fit done | "
            f"train_shape={x_train_vec.shape} fit={vectorize_fit_seconds}s"
        )

        vec_transform_start = time.perf_counter()
        x_val_vec = vectorizer.transform(x_val)
        x_test_vec = vectorizer.transform(x_test)
        vectorize_transform_seconds = round(time.perf_counter() - vec_transform_start, 6)
        print(
            "[ML TRAIN] Vectorizer transform done | "
            f"val_shape={x_val_vec.shape} test_shape={x_test_vec.shape} "
            f"transform={vectorize_transform_seconds}s"
        )

        print("[ML TRAIN] Initializing model...")
        model, final_params = build_model(selected_model_name, selected_params, random_state=random_state)
        print("[ML TRAIN] Training model...")
        model_fit_seconds = fit_model(model, x_train_vec, y_train)
        print(f"[ML TRAIN] Model trained in {model_fit_seconds}s")

        print("[ML TRAIN] Predicting on val/test...")
        y_val_pred, val_predict_seconds = predict_model(model, x_val_vec)
        y_test_pred, test_predict_seconds = predict_model(model, x_test_vec)
        print(
            "[ML TRAIN] Prediction done | "
            f"val={len(y_val_pred)} ({val_predict_seconds}s) "
            f"test={len(y_test_pred)} ({test_predict_seconds}s)"
        )

        val_metrics = compute_metrics(y_val, y_val_pred)
        test_metrics = compute_metrics(y_test, y_test_pred)
        feature_space_diagnostics = compute_feature_space_diagnostics(
            vectorizer_meta=vectorizer_meta,
            observed_feature_count=int(x_train_vec.shape[1]),
        )
        expected_max = feature_space_diagnostics["expected_max_features"]
        observed_count = feature_space_diagnostics["observed_feature_count"]
        if isinstance(expected_max, int) and expected_max > 0 and observed_count != expected_max:
            print(
                "[ML TRAIN][INFO] Feature count differs from configured max_features | "
                f"expected_max={expected_max} observed={observed_count} "
                f"utilization={feature_space_diagnostics['utilization_ratio']}"
            )
        print(
            "[ML TRAIN] Metrics | "
            f"val_f1_macro={round(val_metrics['f1_macro'], 6)} "
            f"test_f1_macro={round(test_metrics['f1_macro'], 6)}"
        )

        metadata = {
            "run_id": run_id,
            "run_timestamp": run_timestamp,
            "experiment_name": experiment_name,
            "config_path": str(resolved_config_path.as_posix()),
            "config_version": config_version,
            "selection": {
                "text_variant": selected_text_variant,
                "feature_set": selected_feature_set,
                "model_name": selected_model_name,
                "params": final_params,
                "random_state": random_state,
            },
            "shape": {
                "train_rows": int(x_train_vec.shape[0]),
                "val_rows": int(x_val_vec.shape[0]),
                "test_rows": int(x_test_vec.shape[0]),
                "feature_count": int(x_train_vec.shape[1]),
                "train_nnz": int(x_train_vec.nnz),
                "val_nnz": int(x_val_vec.nnz),
                "test_nnz": int(x_test_vec.nnz),
            },
            "timing_seconds": {
                "vectorize_fit": vectorize_fit_seconds,
                "vectorize_transform": vectorize_transform_seconds,
                "model_fit": model_fit_seconds,
                "val_predict": val_predict_seconds,
                "test_predict": test_predict_seconds,
            },
            "metrics": {
                "val": val_metrics,
                "test": test_metrics,
            },
            "vectorizer_meta": vectorizer_meta,
            "feature_space_diagnostics": feature_space_diagnostics,
        }
        artifact_paths = save_artifacts(
            run_id=run_id,
            model=model,
            vectorizer=vectorizer,
            metadata=metadata,
            save_as_best=save_as_best,
        )
        print("[ML TRAIN] Artifacts saved under models/ml/")

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
                "notes": json.dumps(artifact_paths, ensure_ascii=False, separators=(",", ":")),
            }
        )
    except Exception as exc:
        run_record["status"] = "FAIL"
        run_record["notes"] = str(exc)
    finally:
        print("[ML TRAIN] Writing run record to experiments/ml/runs.csv...")
        append_run_record(runs_path, run_record)

    if run_record["status"] != "PASS":
        raise TrainRunError(f"Training run failed and was logged to runs.csv: {run_record['notes']}")

    return run_record
