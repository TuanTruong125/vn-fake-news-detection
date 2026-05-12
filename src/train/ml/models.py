from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

try:
    from src.train.common.dataset_loader import DatasetLoaderError, get_xy_splits
    from src.train.common.split_utils import DatasetValidationError
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
    from vectorizers import (  # type: ignore
        VectorizerConfigError,
        build_feature_pipeline,
        get_feature_set_names,
        get_text_variants,
        load_train_ml_config,
        resolve_config_path,
    )


MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "linear_svm": LinearSVC,
    "multinomial_nb": MultinomialNB,
}

SUMMARY_COLUMNS = [
    "run_timestamp",
    "config_path",
    "text_variant",
    "feature_set",
    "model_name",
    "params_json",
    "status",
    "message",
    "train_rows",
    "val_rows",
    "test_rows",
    "train_features",
    "vectorize_fit_seconds",
    "vectorize_transform_seconds",
    "model_fit_seconds",
    "val_predict_seconds",
    "test_predict_seconds",
    "val_pred_rows",
    "test_pred_rows",
    "score_method",
]


# Config Error class for issues during model config parsing and validation.
class ModelConfigError(Exception):
    pass


# Error class for runtime issues during model building, fitting, or predicting.
class ModelRuntimeError(Exception):
    pass


# Resolve repository root path from current file location.
def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# Return local ISO timestamp for logging rows.
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# Parse CLI args for model-zoo smoke training.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML model-zoo smoke train/predict by config.")
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional path to train_ml.yaml. Default: configs/train_ml.yaml",
    )
    parser.add_argument(
        "--text-variant",
        type=str,
        default=None,
        help="Optional text variant to run. Default uses first variant in config.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default=None,
        help="Optional feature set to run. Default uses first feature_set in config.",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="If set, split overlap check becomes warning-only.",
    )
    parser.add_argument(
        "--max-combos-per-model",
        type=int,
        default=0,
        help="Limit number of parameter combinations per model (0 means all).",
    )
    return parser.parse_args()


# Validate root models config mapping and return it.
def load_models_section(config: dict[str, Any]) -> dict[str, Any]:
    models = config.get("models")
    if not isinstance(models, dict) or not models:
        raise ModelConfigError("train_ml.yaml must define non-empty 'models' section.")
    return models


# Validate enabled models against the registry and return enabled names.
def get_enabled_model_names(models_section: dict[str, Any]) -> list[str]:
    enabled_models: list[str] = []
    for model_name, model_conf in models_section.items():
        if model_name not in MODEL_REGISTRY:
            raise ModelConfigError(
                f"Unsupported model in config: {model_name}. "
                f"Allowed: {sorted(MODEL_REGISTRY.keys())}"
            )
        if not isinstance(model_conf, dict):
            raise ModelConfigError(f"models.{model_name} must be a mapping.")
        enabled = model_conf.get("enabled")
        if not isinstance(enabled, bool):
            raise ModelConfigError(f"models.{model_name}.enabled must be boolean.")
        if enabled:
            enabled_models.append(model_name)

    if not enabled_models:
        raise ModelConfigError("No model is enabled in train_ml.yaml.")
    return enabled_models


# Normalize one model param section to dictionary of lists for grid expansion.
def normalize_param_values(model_name: str, raw_params: Any) -> dict[str, list[Any]]:
    if not isinstance(raw_params, dict) or not raw_params:
        raise ModelConfigError(f"models.{model_name}.params must be a non-empty mapping.")

    normalized: dict[str, list[Any]] = {}
    for param_name, param_values in raw_params.items():
        values = param_values if isinstance(param_values, list) else [param_values]
        if not values:
            raise ModelConfigError(f"models.{model_name}.params.{param_name} must not be empty.")
        if any(value is None for value in values):
            raise ModelConfigError(f"models.{model_name}.params.{param_name} contains null value.")
        normalized[param_name] = values
    return normalized


# Expand dictionary-of-lists params into cartesian-product parameter combinations.
def expand_param_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(param_grid.keys())
    combinations = []
    for combo_values in itertools.product(*(param_grid[key] for key in keys)):
        combinations.append(dict(zip(keys, combo_values)))
    return combinations


# Build parameter combinations per enabled model from train_ml config.
def get_model_param_combinations(
    config: dict[str, Any],
    max_combos_per_model: int = 0,
) -> list[tuple[str, dict[str, Any]]]:
    models_section = load_models_section(config)
    enabled_models = get_enabled_model_names(models_section)

    model_combos: list[tuple[str, dict[str, Any]]] = []
    for model_name in enabled_models:
        params = normalize_param_values(model_name, models_section[model_name].get("params"))
        combos = expand_param_grid(params)
        if max_combos_per_model > 0:
            combos = combos[:max_combos_per_model]
        for combo in combos:
            model_combos.append((model_name, combo))

    if not model_combos:
        raise ModelConfigError("No model parameter combinations available to run.")
    return model_combos


# Inject random_state from experiment config when estimator supports this parameter.
def apply_random_state(
    model_name: str,
    params: dict[str, Any],
    random_state: int | None,
) -> dict[str, Any]:
    final_params = dict(params)
    if random_state is None:
        return final_params

    estimator_cls = MODEL_REGISTRY[model_name]
    supports_random_state = "random_state" in estimator_cls().get_params(deep=False)
    if supports_random_state and "random_state" not in final_params:
        final_params["random_state"] = random_state
    return final_params


# Build one sklearn estimator instance from model name and parameter mapping.
def build_model(
    model_name: str,
    params: dict[str, Any],
    random_state: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    if model_name not in MODEL_REGISTRY:
        raise ModelConfigError(f"Unknown model_name: {model_name}")
    estimator_cls = MODEL_REGISTRY[model_name]
    final_params = apply_random_state(model_name, params, random_state)
    try:
        model = estimator_cls(**final_params)
    except Exception as exc:  # pragma: no cover - explicit error path
        raise ModelConfigError(
            f"Failed to initialize model '{model_name}' with params={final_params}. Error: {exc}"
        ) from exc
    return model, final_params


# Fit one estimator and return elapsed seconds.
def fit_model(model: Any, x_train: Any, y_train: Any) -> float:
    start = time.perf_counter()
    model.fit(x_train, y_train)
    return round(time.perf_counter() - start, 6)


# Predict labels and return predictions with elapsed seconds.
def predict_model(model: Any, x_data: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    preds = model.predict(x_data)
    elapsed = round(time.perf_counter() - start, 6)
    return preds, elapsed


# Return score method used by model for probabilistic-like outputs.
def get_score_method(model: Any) -> str:
    if hasattr(model, "predict_proba"):
        return "predict_proba"
    if hasattr(model, "decision_function"):
        return "decision_function"
    return "none"


# Resolve one text variant and one feature set for smoke run defaults.
def resolve_smoke_scope(
    config: dict[str, Any],
    requested_text_variant: str | None,
    requested_feature_set: str | None,
) -> tuple[str, str]:
    text_variants = get_text_variants(config)
    feature_sets = get_feature_set_names(config)

    text_variant = requested_text_variant or text_variants[0]
    feature_set = requested_feature_set or feature_sets[0]

    if text_variant not in text_variants:
        raise ModelConfigError(f"text_variant '{text_variant}' is not available: {text_variants}")
    if feature_set not in feature_sets:
        raise ModelConfigError(f"feature_set '{feature_set}' is not available: {feature_sets}")

    return text_variant, feature_set


# Build one failure row for summary CSV logging.
def build_fail_row(
    config_path: Path,
    text_variant: str,
    feature_set: str,
    model_name: str,
    params: dict[str, Any],
    message: str,
) -> dict[str, Any]:
    return {
        "run_timestamp": now_iso(),
        "config_path": str(config_path.as_posix()),
        "text_variant": text_variant,
        "feature_set": feature_set,
        "model_name": model_name,
        "params_json": json.dumps(params, ensure_ascii=False, separators=(",", ":")),
        "status": "FAIL",
        "message": message,
        "train_rows": "",
        "val_rows": "",
        "test_rows": "",
        "train_features": "",
        "vectorize_fit_seconds": "",
        "vectorize_transform_seconds": "",
        "model_fit_seconds": "",
        "val_predict_seconds": "",
        "test_predict_seconds": "",
        "val_pred_rows": "",
        "test_pred_rows": "",
        "score_method": "",
    }


# Write smoke-run summary rows into logs folder CSV file.
def write_smoke_summary(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


# Run end-to-end smoke test for all enabled model/params on one feature representation.
def run_model_zoo_smoke(
    config_path: str | Path | None = None,
    text_variant: str | None = None,
    feature_set: str | None = None,
    allow_overlap: bool = False,
    max_combos_per_model: int = 0,
) -> tuple[list[dict[str, Any]], Path]:
    resolved_config_path = resolve_config_path(config_path)
    config = load_train_ml_config(resolved_config_path)
    selected_text_variant, selected_feature_set = resolve_smoke_scope(
        config=config,
        requested_text_variant=text_variant,
        requested_feature_set=feature_set,
    )

    experiment = config.get("experiment", {})
    random_state = experiment.get("random_state") if isinstance(experiment, dict) else None
    if random_state is not None and not isinstance(random_state, int):
        raise ModelConfigError("experiment.random_state must be integer when provided.")

    strict_overlap = not allow_overlap
    model_param_combos = get_model_param_combinations(config, max_combos_per_model=max_combos_per_model)

    # Build feature matrices once, then reuse for all model combinations.
    xy_splits = get_xy_splits(
        text_variant=selected_text_variant,
        config_path=resolved_config_path,
        strict_overlap=strict_overlap,
        print_summary=False,
    )
    x_train, y_train = xy_splits["train"]
    x_val, _ = xy_splits["val"]
    x_test, _ = xy_splits["test"]

    feature_pipeline, _ = build_feature_pipeline(selected_feature_set, config)
    vec_fit_start = time.perf_counter()
    x_train_vec = feature_pipeline.fit_transform(x_train)
    vectorize_fit_seconds = round(time.perf_counter() - vec_fit_start, 6)

    vec_transform_start = time.perf_counter()
    x_val_vec = feature_pipeline.transform(x_val)
    x_test_vec = feature_pipeline.transform(x_test)
    vectorize_transform_seconds = round(time.perf_counter() - vec_transform_start, 6)

    rows: list[dict[str, Any]] = []
    for model_name, param_combo in model_param_combos:
        print(f"[ML MODEL ZOO] Train model={model_name} params={param_combo}")
        try:
            model, final_params = build_model(model_name, param_combo, random_state=random_state)
            model_fit_seconds = fit_model(model, x_train_vec, y_train)
            val_pred, val_predict_seconds = predict_model(model, x_val_vec)
            test_pred, test_predict_seconds = predict_model(model, x_test_vec)

            if len(val_pred) != x_val_vec.shape[0]:
                raise ModelRuntimeError("Validation prediction row mismatch.")
            if len(test_pred) != x_test_vec.shape[0]:
                raise ModelRuntimeError("Test prediction row mismatch.")

            row = {
                "run_timestamp": now_iso(),
                "config_path": str(resolved_config_path.as_posix()),
                "text_variant": selected_text_variant,
                "feature_set": selected_feature_set,
                "model_name": model_name,
                "params_json": json.dumps(final_params, ensure_ascii=False, separators=(",", ":")),
                "status": "PASS",
                "message": "",
                "train_rows": int(x_train_vec.shape[0]),
                "val_rows": int(x_val_vec.shape[0]),
                "test_rows": int(x_test_vec.shape[0]),
                "train_features": int(x_train_vec.shape[1]),
                "vectorize_fit_seconds": vectorize_fit_seconds,
                "vectorize_transform_seconds": vectorize_transform_seconds,
                "model_fit_seconds": model_fit_seconds,
                "val_predict_seconds": val_predict_seconds,
                "test_predict_seconds": test_predict_seconds,
                "val_pred_rows": int(len(val_pred)),
                "test_pred_rows": int(len(test_pred)),
                "score_method": get_score_method(model),
            }
            rows.append(row)
            print(
                f"[ML MODEL ZOO] PASS model={model_name} "
                f"fit={model_fit_seconds}s val_pred={len(val_pred)} test_pred={len(test_pred)}"
            )
        except Exception as exc:
            fail_row = build_fail_row(
                config_path=resolved_config_path,
                text_variant=selected_text_variant,
                feature_set=selected_feature_set,
                model_name=model_name,
                params=param_combo,
                message=str(exc),
            )
            rows.append(fail_row)
            print(f"[ML MODEL ZOO] FAIL model={model_name} | {exc}")

    summary_path = get_repo_root() / "logs" / "ml_model_zoo_smoke_summary.csv"
    write_smoke_summary(summary_path, rows)
    return rows, summary_path


# Execute CLI smoke run and fail process if any model combination fails.
def main() -> None:
    args = parse_args()
    rows, summary_path = run_model_zoo_smoke(
        config_path=args.config_path,
        text_variant=args.text_variant,
        feature_set=args.feature_set,
        allow_overlap=args.allow_overlap,
        max_combos_per_model=args.max_combos_per_model,
    )
    pass_count = sum(1 for row in rows if row.get("status") == "PASS")
    fail_count = sum(1 for row in rows if row.get("status") == "FAIL")
    print(
        f"[ML MODEL ZOO] Done | pass={pass_count} fail={fail_count} | "
        f"summary={summary_path.as_posix()}"
    )
    if fail_count > 0:
        raise SystemExit("[ML MODEL ZOO ERROR] One or more model combos failed.")


# Expose model-zoo smoke run entrypoint for local and CI checks.
if __name__ == "__main__":
    try:
        main()
    except (
        ModelConfigError,
        ModelRuntimeError,
        DatasetLoaderError,
        DatasetValidationError,
        VectorizerConfigError,
    ) as exc:
        raise SystemExit(f"[ML MODEL ZOO ERROR] {exc}")
