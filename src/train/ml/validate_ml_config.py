from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ALLOWED_TEXT_VARIANTS = {"text_ml_seg", "text_ml_seg_lower"}
ALLOWED_SELECTION_MODES = {"all", "best"}
ALLOWED_METRICS = {"f1_macro", "precision_macro", "recall_macro", "accuracy", "f1_fake"}
REQUIRED_ROOT_SECTIONS = {"version", "data", "experiment", "vectorizers", "feature_sets", "models", "selection"}


class MlConfigValidationError(Exception):
    pass


# Load and parse one YAML file as a top-level mapping object.
def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MlConfigValidationError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise MlConfigValidationError(f"Config must be a mapping object: {path}")
    return data


# Read CSV header only to validate required columns without loading full data.
def read_csv_headers(csv_path: Path) -> set[str]:
    try:
        df = pd.read_csv(csv_path, nrows=0)
    except Exception as exc:  # pragma: no cover - explicit CLI error path
        raise MlConfigValidationError(f"Cannot read CSV header: {csv_path}. Error: {exc}") from exc
    return {str(col).strip() for col in df.columns}


# Return a required mapping section from config and validate its type.
def require_mapping(parent: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise MlConfigValidationError(f"{context}.{key} must be a mapping.")
    return value


# Return a required list section from config and validate its type.
def require_list(parent: dict[str, Any], key: str, context: str) -> list[Any]:
    value = parent.get(key)
    if not isinstance(value, list):
        raise MlConfigValidationError(f"{context}.{key} must be a list.")
    return value


# Validate a list of non-empty strings and return trimmed values.
def validate_string_list(values: Any, field_name: str) -> list[str]:
    if not isinstance(values, list) or not values:
        raise MlConfigValidationError(f"{field_name} must be a non-empty list.")
    out: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise MlConfigValidationError(f"{field_name} must contain only non-empty strings.")
        out.append(value.strip())
    return out


# Validate ngram_range as [min_n, max_n] with positive integers and min<=max.
def validate_ngram_range(value: Any, field_name: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise MlConfigValidationError(f"{field_name} must be a list with two integers.")
    min_n, max_n = value
    if not isinstance(min_n, int) or not isinstance(max_n, int):
        raise MlConfigValidationError(f"{field_name} values must be integers.")
    if min_n <= 0 or max_n <= 0 or min_n > max_n:
        raise MlConfigValidationError(f"{field_name} must satisfy 1 <= min_n <= max_n.")


# Validate min_df/max_df thresholds in either integer-count or float-ratio form.
def validate_df_threshold(value: Any, field_name: str, allow_zero: bool) -> None:
    if isinstance(value, int):
        if value < 0 or (not allow_zero and value == 0):
            raise MlConfigValidationError(f"{field_name} must be > 0 when integer.")
        return
    if isinstance(value, float):
        if value <= 0.0 or value > 1.0:
            raise MlConfigValidationError(f"{field_name} must be in (0,1] when float.")
        return
    raise MlConfigValidationError(f"{field_name} must be int or float.")


# Validate vectorizer settings and return declared vectorizer keys.
def validate_vectorizers_section(config: dict[str, Any]) -> set[str]:
    vectorizers = require_mapping(config, "vectorizers", "root")
    if not vectorizers:
        raise MlConfigValidationError("vectorizers must not be empty.")

    for vec_name, vec_conf in vectorizers.items():
        if not isinstance(vec_conf, dict):
            raise MlConfigValidationError(f"vectorizers.{vec_name} must be a mapping.")
        validate_ngram_range(vec_conf.get("ngram_range"), f"vectorizers.{vec_name}.ngram_range")
        validate_df_threshold(vec_conf.get("min_df"), f"vectorizers.{vec_name}.min_df", allow_zero=False)
        validate_df_threshold(vec_conf.get("max_df"), f"vectorizers.{vec_name}.max_df", allow_zero=False)

        max_features = vec_conf.get("max_features")
        if not isinstance(max_features, int) or max_features <= 0:
            raise MlConfigValidationError(f"vectorizers.{vec_name}.max_features must be a positive integer.")

        sublinear_tf = vec_conf.get("sublinear_tf")
        if not isinstance(sublinear_tf, bool):
            raise MlConfigValidationError(f"vectorizers.{vec_name}.sublinear_tf must be boolean.")

    return set(vectorizers.keys())


# Validate feature_set definitions and cross-check them against declared vectorizers.
def validate_feature_sets_section(config: dict[str, Any], vectorizer_names: set[str]) -> None:
    feature_sets = require_list(config, "feature_sets", "root")
    if not feature_sets:
        raise MlConfigValidationError("feature_sets must not be empty.")

    seen_names: set[str] = set()
    for idx, feature_set in enumerate(feature_sets):
        if not isinstance(feature_set, dict):
            raise MlConfigValidationError(f"feature_sets[{idx}] must be a mapping.")
        name = feature_set.get("name")
        if not isinstance(name, str) or not name.strip():
            raise MlConfigValidationError(f"feature_sets[{idx}].name must be a non-empty string.")
        name = name.strip()
        if name in seen_names:
            raise MlConfigValidationError(f"Duplicate feature_set name: {name}")
        seen_names.add(name)

        set_type = feature_set.get("type")
        if not isinstance(set_type, str) or not set_type.strip():
            raise MlConfigValidationError(f"feature_sets[{idx}].type must be a non-empty string.")
        set_type = set_type.strip()

        if set_type == "combined":
            parts = feature_set.get("parts")
            if not isinstance(parts, list) or len(parts) < 2:
                raise MlConfigValidationError(f"feature_sets[{idx}].parts must contain at least two vectorizers.")
            for part in parts:
                if not isinstance(part, str) or part.strip() not in vectorizer_names:
                    raise MlConfigValidationError(
                        f"feature_sets[{idx}].parts contains unknown vectorizer: {part}"
                    )
        elif set_type not in vectorizer_names:
            raise MlConfigValidationError(
                f"feature_sets[{idx}].type must be 'combined' or one of {sorted(vectorizer_names)}."
            )


# Validate ML model blocks and require at least one enabled model.
def validate_models_section(config: dict[str, Any]) -> None:
    models = require_mapping(config, "models", "root")
    if not models:
        raise MlConfigValidationError("models must not be empty.")

    enabled_count = 0
    for model_name, model_conf in models.items():
        if not isinstance(model_conf, dict):
            raise MlConfigValidationError(f"models.{model_name} must be a mapping.")
        enabled = model_conf.get("enabled")
        if not isinstance(enabled, bool):
            raise MlConfigValidationError(f"models.{model_name}.enabled must be boolean.")
        if enabled:
            enabled_count += 1

        params = model_conf.get("params")
        if not isinstance(params, dict) or not params:
            raise MlConfigValidationError(f"models.{model_name}.params must be a non-empty mapping.")
        for param_name, param_values in params.items():
            values = param_values if isinstance(param_values, list) else [param_values]
            if not values:
                raise MlConfigValidationError(f"models.{model_name}.params.{param_name} must not be empty.")
            for value in values:
                if value is None:
                    raise MlConfigValidationError(
                        f"models.{model_name}.params.{param_name} contains null value."
                    )

    if enabled_count == 0:
        raise MlConfigValidationError("At least one model must be enabled.")


# Validate experiment settings including metric names and random state.
def validate_experiment_section(config: dict[str, Any]) -> tuple[str, set[str]]:
    experiment = require_mapping(config, "experiment", "root")

    exp_name = experiment.get("name")
    if not isinstance(exp_name, str) or not exp_name.strip():
        raise MlConfigValidationError("experiment.name must be a non-empty string.")

    primary_metric = experiment.get("primary_metric")
    if not isinstance(primary_metric, str) or primary_metric not in ALLOWED_METRICS:
        raise MlConfigValidationError(f"experiment.primary_metric must be one of {sorted(ALLOWED_METRICS)}.")

    secondary_metrics = validate_string_list(experiment.get("secondary_metrics"), "experiment.secondary_metrics")
    invalid_secondary = [metric for metric in secondary_metrics if metric not in ALLOWED_METRICS]
    if invalid_secondary:
        raise MlConfigValidationError(
            f"experiment.secondary_metrics contains unsupported metrics: {invalid_secondary}"
        )

    random_state = experiment.get("random_state")
    if not isinstance(random_state, int):
        raise MlConfigValidationError("experiment.random_state must be integer.")

    max_trials = experiment.get("max_trials")
    if not isinstance(max_trials, int) or max_trials <= 0:
        raise MlConfigValidationError("experiment.max_trials must be a positive integer.")

    for bool_field in ["save_model", "save_vectorizer"]:
        if not isinstance(experiment.get(bool_field), bool):
            raise MlConfigValidationError(f"experiment.{bool_field} must be boolean.")

    all_metrics = set(secondary_metrics)
    all_metrics.add(primary_metric)
    return primary_metric, all_metrics


# Validate selection rules and cross-check chosen metrics.
def validate_selection_section(config: dict[str, Any], declared_metrics: set[str]) -> None:
    selection = require_mapping(config, "selection", "root")

    mode = selection.get("mode")
    if not isinstance(mode, str) or mode not in ALLOWED_SELECTION_MODES:
        raise MlConfigValidationError(f"selection.mode must be one of {sorted(ALLOWED_SELECTION_MODES)}.")

    choose_best_by = selection.get("choose_best_by")
    if not isinstance(choose_best_by, str) or choose_best_by not in ALLOWED_METRICS:
        raise MlConfigValidationError(f"selection.choose_best_by must be one of {sorted(ALLOWED_METRICS)}.")
    if choose_best_by not in declared_metrics:
        raise MlConfigValidationError(
            f"selection.choose_best_by='{choose_best_by}' is not declared in experiment metrics."
        )

    tie_breaker = selection.get("tie_breaker")
    if not isinstance(tie_breaker, str) or tie_breaker not in ALLOWED_METRICS:
        raise MlConfigValidationError(f"selection.tie_breaker must be one of {sorted(ALLOWED_METRICS)}.")


# Validate data section and optionally cross-check existing processed CSV headers.
def validate_data_section(config: dict[str, Any], repo_root: Path, warnings: list[str]) -> None:
    data = require_mapping(config, "data", "root")

    label_column = data.get("label_column")
    id_column = data.get("id_column")
    if not isinstance(label_column, str) or not label_column.strip():
        raise MlConfigValidationError("data.label_column must be a non-empty string.")
    if not isinstance(id_column, str) or not id_column.strip():
        raise MlConfigValidationError("data.id_column must be a non-empty string.")
    label_column = label_column.strip()
    id_column = id_column.strip()

    text_variants = validate_string_list(data.get("text_variants"), "data.text_variants")
    unknown_variants = [column for column in text_variants if column not in ALLOWED_TEXT_VARIANTS]
    if unknown_variants:
        raise MlConfigValidationError(
            f"data.text_variants contains unsupported columns: {unknown_variants}. "
            f"Allowed: {sorted(ALLOWED_TEXT_VARIANTS)}."
        )

    required_columns = set(text_variants + [label_column, id_column])
    for split_name in ["train_path", "val_path", "test_path"]:
        split_path_value = data.get(split_name)
        if not isinstance(split_path_value, str) or not split_path_value.strip():
            raise MlConfigValidationError(f"data.{split_name} must be a non-empty string.")
        split_path = repo_root / split_path_value

        # Check header constraints when file exists, and emit warning-only if not present yet.
        if not split_path.exists():
            warnings.append(
                f"data.{split_name}: file not found yet ({split_path_value}); skipped header cross-check."
            )
            continue

        headers = read_csv_headers(split_path)
        missing_columns = sorted(required_columns - headers)
        if missing_columns:
            raise MlConfigValidationError(
                f"data.{split_name}: missing required columns {missing_columns}."
            )


# Validate all root sections and enforce a strict feature bootstrap schema.
def validate_root_structure(config: dict[str, Any]) -> None:
    missing_sections = sorted(REQUIRED_ROOT_SECTIONS - set(config.keys()))
    if missing_sections:
        raise MlConfigValidationError(f"Missing required root sections: {missing_sections}")

    version = config.get("version")
    if not isinstance(version, int) or version <= 0:
        raise MlConfigValidationError("version must be a positive integer.")


# Execute end-to-end ML config validation with schema and cross-field checks.
def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "configs" / "train_ml.yaml"

    config = load_yaml(config_path)
    warnings: list[str] = []
    validate_root_structure(config)
    validate_data_section(config, repo_root, warnings)
    primary_metric, declared_metrics = validate_experiment_section(config)
    vectorizer_names = validate_vectorizers_section(config)
    validate_feature_sets_section(config, vectorizer_names)
    validate_models_section(config)
    validate_selection_section(config, declared_metrics | {primary_metric})

    print("[ML CONFIG] Validation passed.")
    for warning in warnings:
        print(f"[ML CONFIG][WARNING] {warning}")


# Provide CLI-friendly error output for local runs and CI checks.
if __name__ == "__main__":
    try:
        main()
    except MlConfigValidationError as exc:
        raise SystemExit(f"[ML CONFIG VALIDATION ERROR] {exc}")
