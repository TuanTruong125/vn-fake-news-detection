from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


EXPECTED_MASTER_COLUMNS = [
    "sample_id",
    "text_raw",
    "text_clean",
    "hash_text",
    "label_binary",
    "label_name",
    "source_file",
    "source_domain",
    "content_type",
    "published_at",
    "label_confidence",
    "text_length",
    "split",
]

ALLOWED_CONTENT_TYPES = {"news", "social"}
ALLOWED_BINARY_LABELS = {0, 1}
EXPECTED_DECLARED_SOURCES = 6


class ConfigValidationError(Exception):
    pass


# Load and parse one YAML file into a dictionary object.
def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigValidationError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigValidationError(f"Config file must contain a mapping object: {path}")
    return data


# Return True when an optional column setting is empty or null-like.
def _is_optional_column(column_name: Any) -> bool:
    if column_name is None:
        return True
    if isinstance(column_name, str):
        return column_name.strip().lower() in {"", "null"}
    return False


# Read only CSV headers to validate schema quickly.
def _read_csv_columns(csv_path: Path) -> set[str]:
    try:
        df = pd.read_csv(csv_path, nrows=0)
    except Exception as exc:  # pragma: no cover - explicit error path for CLI
        raise ConfigValidationError(f"Cannot read CSV header: {csv_path}. Error: {exc}") from exc
    # Normalize headers to avoid false mismatch caused by leading/trailing spaces.
    return {str(col).strip() for col in df.columns}


# Validate text_columns format before checking column existence in CSV.
def _validate_text_columns(source_id: str, text_columns: Any) -> list[str]:
    if not isinstance(text_columns, list) or not text_columns:
        raise ConfigValidationError(f"{source_id}: text_columns must be a non-empty list.")
    if not all(isinstance(col, str) and col.strip() for col in text_columns):
        raise ConfigValidationError(
            f"{source_id}: every item in text_columns must be a non-empty string."
        )
    # Keep a trimmed list so config formatting spaces do not break validation.
    return [col.strip() for col in text_columns]


# Validate data_sources and return declared/enabled source metadata for cross-checks.
def validate_data_sources(
    config: dict[str, Any], repo_root: Path
) -> tuple[list[str], list[str], dict[str, str]]:
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ConfigValidationError("configs/data_sources.yaml must define a non-empty 'sources' list.")

    declared_source_ids: list[str] = []
    enabled_source_ids: list[str] = []
    label_column_by_source: dict[str, str] = {}

    for source in sources:
        if not isinstance(source, dict):
            raise ConfigValidationError("Each source entry in data_sources.yaml must be a mapping.")

        required = ["source_id", "path", "text_columns", "label_column", "content_type", "label_confidence"]
        missing = [key for key in required if key not in source]
        if missing:
            raise ConfigValidationError(
                f"Source entry is missing required keys {missing}. Entry: {source}"
            )

        source_id = str(source["source_id"]).strip()
        if not source_id:
            raise ConfigValidationError("source_id cannot be empty.")
        if source_id in declared_source_ids:
            raise ConfigValidationError(f"Duplicate source_id found: {source_id}")
        declared_source_ids.append(source_id)

        enabled = source.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ConfigValidationError(f"{source_id}: enabled must be a boolean value.")
        # Skip deep validation for disabled sources by design.
        if not enabled:
            continue

        source_path = repo_root / str(source["path"])
        if not source_path.exists():
            raise ConfigValidationError(f"Configured source path does not exist: {source_path}")

        text_columns = _validate_text_columns(source_id, source["text_columns"])
        label_column = str(source["label_column"]).strip()
        if not label_column:
            raise ConfigValidationError(f"{source_id}: label_column must be a non-empty string.")

        headers = _read_csv_columns(source_path)
        if label_column not in headers:
            raise ConfigValidationError(
                f"{source_id}: label_column '{label_column}' not found in {source_path.name}."
            )

        # Support fallback text columns by requiring at least one column to exist.
        if not any(col in headers for col in text_columns):
            raise ConfigValidationError(
                f"{source_id}: none of text_columns {text_columns} found in {source_path.name}."
            )

        for optional_col in ["published_at_column", "url_column", "domain_column"]:
            configured_col = source.get(optional_col)
            if isinstance(configured_col, str):
                configured_col = configured_col.strip()
            if not _is_optional_column(configured_col) and configured_col not in headers:
                raise ConfigValidationError(
                    f"{source_id}: {optional_col} '{configured_col}' not found in {source_path.name}."
                )

        content_type = str(source["content_type"]).strip()
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise ConfigValidationError(
                f"{source_id}: content_type must be one of {sorted(ALLOWED_CONTENT_TYPES)}."
            )

        confidence = source["label_confidence"]
        if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
            raise ConfigValidationError(
                f"{source_id}: label_confidence must be a number in [0.0, 1.0]."
            )

        enabled_source_ids.append(source_id)
        label_column_by_source[source_id] = label_column

    # Prevent running the pipeline with an empty enabled source set.
    if not enabled_source_ids:
        raise ConfigValidationError("At least one source must be enabled in data_sources.yaml.")

    return declared_source_ids, enabled_source_ids, label_column_by_source


# Validate label_mapping for enabled sources and cross-check label_column consistency.
def validate_label_mapping(
    config: dict[str, Any],
    declared_source_ids: list[str],
    enabled_source_ids: list[str],
    label_column_by_source: dict[str, str],
) -> list[str]:
    sources = config.get("sources")
    if not isinstance(sources, dict):
        raise ConfigValidationError("configs/label_mapping.yaml must define a 'sources' mapping.")

    missing_source_mappings = [source_id for source_id in enabled_source_ids if source_id not in sources]
    if missing_source_mappings:
        raise ConfigValidationError(
            f"Missing label mapping for sources: {missing_source_mappings}"
        )

    extra_mappings = [source_id for source_id in sources if source_id not in declared_source_ids]
    if extra_mappings:
        raise ConfigValidationError(
            f"label_mapping.yaml contains unknown source_id(s): {extra_mappings}"
        )

    warnings: list[str] = []

    for source_id in enabled_source_ids:
        entry = sources[source_id]
        if not isinstance(entry, dict):
            raise ConfigValidationError(f"{source_id}: mapping entry must be a mapping object.")

        # Enforce cross-file consistency between data_sources and label_mapping.
        expected_label_col = label_column_by_source[source_id]
        actual_label_col = entry.get("label_column")
        if not isinstance(actual_label_col, str) or actual_label_col.strip() != expected_label_col:
            raise ConfigValidationError(
                f"{source_id}: label_column mismatch. data_sources='{expected_label_col}', "
                f"label_mapping='{actual_label_col}'."
            )

        mapping = entry.get("mapping")
        if not isinstance(mapping, dict) or not mapping:
            raise ConfigValidationError(f"{source_id}: mapping must be a non-empty key/value mapping.")
        # Keep mapping key types predictable for downstream normalization.
        if not all(isinstance(key, (str, int, bool)) for key in mapping.keys()):
            raise ConfigValidationError(
                f"{source_id}: mapping keys must be one of str/int/bool."
            )
        if not all(str(key).strip() for key in mapping.keys()):
            raise ConfigValidationError(f"{source_id}: mapping keys must be non-empty.")

        target_values = set(mapping.values())
        if not target_values.issubset(ALLOWED_BINARY_LABELS):
            raise ConfigValidationError(
                f"{source_id}: mapping values must be only 0 or 1. Found: {sorted(target_values)}"
            )
        # Missing one class is allowed at config level, so we emit a warning instead of failing.
        if ALLOWED_BINARY_LABELS - target_values:
            missing_labels = sorted(ALLOWED_BINARY_LABELS - target_values)
            warnings.append(
                f"{source_id}: mapping values={sorted(target_values)} "
                f"(missing {missing_labels}). Dataset may be single-class."
            )

    return warnings


# Validate master schema and enforce strict 13-column order for reproducibility.
def validate_schema(config: dict[str, Any]) -> None:
    master_schema = config.get("master_schema")
    if not isinstance(master_schema, dict):
        raise ConfigValidationError("configs/schema.yaml must define 'master_schema'.")

    columns = master_schema.get("columns")
    if not isinstance(columns, list):
        raise ConfigValidationError("schema.master_schema.columns must be a list.")

    names = [col.get("name") for col in columns if isinstance(col, dict)]
    # Keep strict column order to match the project report and downstream contracts.
    if names != EXPECTED_MASTER_COLUMNS:
        raise ConfigValidationError(
            "Master schema columns do not match expected 13-column order.\n"
            f"Expected: {EXPECTED_MASTER_COLUMNS}\nFound:    {names}"
        )

    constraints = config.get("constraints", {})
    if not isinstance(constraints, dict):
        raise ConfigValidationError("schema.constraints must be a mapping.")

    split_allowed = constraints.get("split_allowed")
    if not isinstance(split_allowed, list) or "external_test" not in split_allowed:
        raise ConfigValidationError("schema.constraints.split_allowed must include 'external_test'.")

    label_allowed = constraints.get("label_binary_allowed")
    if label_allowed != [0, 1]:
        raise ConfigValidationError("schema.constraints.label_binary_allowed must be exactly [0, 1].")


# Validate split ratios and split rules used in the internal pipeline.
def validate_split(config: dict[str, Any]) -> None:
    split = config.get("split")
    if not isinstance(split, dict):
        raise ConfigValidationError("configs/split.yaml must define a 'split' mapping.")

    ratios = split.get("ratios")
    if not isinstance(ratios, dict):
        raise ConfigValidationError("split.ratios must be a mapping with train/val/test keys.")

    expected_keys = {"train", "val", "test"}
    if set(ratios.keys()) != expected_keys:
        raise ConfigValidationError(
            f"split.ratios keys must be exactly {sorted(expected_keys)}."
        )

    ratio_total = 0.0
    for key, value in ratios.items():
        if not isinstance(value, (int, float)) or value <= 0:
            raise ConfigValidationError(f"split.ratios.{key} must be a positive number.")
        ratio_total += float(value)

    # Use math.isclose to avoid floating-point precision issues.
    if not math.isclose(ratio_total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ConfigValidationError(f"split.ratios must sum to 1.0. Found: {ratio_total}")

    random_state = split.get("random_state")
    if not isinstance(random_state, int):
        raise ConfigValidationError("split.random_state must be an integer.")

    if not isinstance(split.get("shuffle"), bool):
        raise ConfigValidationError("split.shuffle must be a boolean.")

    if split.get("stratify_by") != "label_binary":
        raise ConfigValidationError("split.stratify_by must be 'label_binary' for Phase 1.")

    if split.get("deduplicate_before_split") is not True:
        raise ConfigValidationError("split.deduplicate_before_split must be true.")

    # Force leakage checks to use deterministic text hash key.
    if split.get("leakage_key") != "hash_text":
        raise ConfigValidationError("split.leakage_key must be 'hash_text'.")


# Run end-to-end config validation for the feature/config-foundation gate.
def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "configs"

    # Load all config layers before performing cross-file validation.
    data_sources = load_yaml(config_dir / "data_sources.yaml")
    label_mapping = load_yaml(config_dir / "label_mapping.yaml")
    schema = load_yaml(config_dir / "schema.yaml")
    split = load_yaml(config_dir / "split.yaml")

    declared_source_ids, enabled_source_ids, label_column_by_source = validate_data_sources(
        data_sources, repo_root
    )
    # Keep this strict count for midterm scope control.
    if len(declared_source_ids) != EXPECTED_DECLARED_SOURCES:
        raise ConfigValidationError(
            f"Expected exactly {EXPECTED_DECLARED_SOURCES} declared internal sources in "
            f"data_sources.yaml, found {len(declared_source_ids)}."
        )

    mapping_warnings = validate_label_mapping(
        label_mapping,
        declared_source_ids,
        enabled_source_ids,
        label_column_by_source,
    )
    validate_schema(schema)
    validate_split(split)
    print(
        "Config validation passed. "
        f"Declared sources={len(declared_source_ids)}, enabled sources={len(enabled_source_ids)}."
    )
    for warning in mapping_warnings:
        print(f"[WARNING] {warning}")


# Provide a CLI-friendly error message for CI and local checks.
if __name__ == "__main__":
    try:
        main()
    except ConfigValidationError as exc:
        raise SystemExit(f"[CONFIG VALIDATION ERROR] {exc}")
