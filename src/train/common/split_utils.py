from __future__ import annotations

from typing import Any

import pandas as pd


# Error class for dataset validation issues during split loading and checking.
class DatasetValidationError(Exception):
    pass


# Validate that all required columns exist in a split dataframe.
def assert_required_columns(df: pd.DataFrame, required_columns: list[str], split_name: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise DatasetValidationError(
            f"{split_name}: missing required columns {missing}."
        )


# Validate labels are numeric binary values in the set {0,1} and return normalized labels.
def assert_binary_labels(df: pd.DataFrame, label_col: str, split_name: str) -> pd.Series:
    raw_labels = df[label_col]
    if raw_labels.isna().any():
        null_count = int(raw_labels.isna().sum())
        raise DatasetValidationError(
            f"{split_name}: label column '{label_col}' contains {null_count} null values."
        )

    labels_numeric = pd.to_numeric(raw_labels, errors="coerce")
    if labels_numeric.isna().any():
        bad_samples = raw_labels[labels_numeric.isna()].astype(str).head(5).tolist()
        raise DatasetValidationError(
            f"{split_name}: label column '{label_col}' has non-numeric values. "
            f"Examples: {bad_samples}"
        )

    unique_values = set(labels_numeric.unique().tolist())
    if not unique_values.issubset({0, 1}):
        bad_values = sorted(unique_values - {0, 1})
        raise DatasetValidationError(
            f"{split_name}: label column '{label_col}' has values outside {{0,1}}: {bad_values}"
        )

    return labels_numeric.astype(int)


# Validate text is not all-empty after strip normalization and return normalized text.
def assert_non_empty_text(df: pd.DataFrame, text_col: str, split_name: str) -> pd.Series:
    normalized_text = df[text_col].fillna("").astype(str).str.strip()
    if (normalized_text == "").all():
        raise DatasetValidationError(
            f"{split_name}: text column '{text_col}' is empty for all rows after strip."
        )
    return normalized_text


# Validate split dataframe has at least one row.
def assert_non_empty_split(df: pd.DataFrame, split_name: str) -> None:
    if df.empty:
        raise DatasetValidationError(f"{split_name}: split dataframe is empty.")


# Validate ID column has no null/empty values and return normalized IDs.
def assert_valid_ids(df: pd.DataFrame, id_col: str, split_name: str) -> pd.Series:
    normalized_ids = df[id_col].fillna("").astype(str).str.strip()
    empty_mask = normalized_ids == ""
    if empty_mask.any():
        empty_count = int(empty_mask.sum())
        raise DatasetValidationError(
            f"{split_name}: id column '{id_col}' has {empty_count} empty values."
        )
    return normalized_ids


# Validate no overlap of IDs across splits with strict or warning-only mode.
def assert_no_id_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str = "sample_id",
    strict: bool = True,
) -> dict[str, int]:
    train_ids = set(train_df[id_col].astype(str))
    val_ids = set(val_df[id_col].astype(str))
    test_ids = set(test_df[id_col].astype(str))

    overlap_train_val = len(train_ids & val_ids)
    overlap_train_test = len(train_ids & test_ids)
    overlap_val_test = len(val_ids & test_ids)

    overlaps = {
        "train_val": overlap_train_val,
        "train_test": overlap_train_test,
        "val_test": overlap_val_test,
    }

    total_overlap = overlap_train_val + overlap_train_test + overlap_val_test
    if total_overlap > 0 and strict:
        raise DatasetValidationError(
            "ID leakage detected across splits: "
            f"train_val={overlap_train_val}, "
            f"train_test={overlap_train_test}, "
            f"val_test={overlap_val_test}."
        )
    if total_overlap > 0 and not strict:
        print(
            "[DATASET LOADER][WARNING] ID overlap detected: "
            f"train_val={overlap_train_val}, "
            f"train_test={overlap_train_test}, "
            f"val_test={overlap_val_test}."
        )

    return overlaps


# Validate requested text variant is enabled by train_ml config.
def assert_text_variant_allowed(text_variant: str, allowed_variants: list[str]) -> None:
    if text_variant not in allowed_variants:
        raise DatasetValidationError(
            f"text_variant='{text_variant}' is not allowed. "
            f"Allowed variants: {allowed_variants}"
        )


# Build split summary dictionary and optionally print compact debug information.
def summarize_split(
    df: pd.DataFrame,
    split_name: str,
    label_col: str,
    text_col: str,
    print_summary: bool = True,
) -> dict[str, Any]:
    label_dist = df[label_col].value_counts().sort_index().to_dict()
    empty_text_count = int((df[text_col].fillna("").astype(str).str.strip() == "").sum())
    summary = {
        "split_name": split_name,
        "rows": int(len(df)),
        "label_dist": label_dist,
        "empty_text_count": empty_text_count,
    }

    if print_summary:
        print(
            f"[{split_name.upper()}] rows={summary['rows']} | "
            f"label_dist={summary['label_dist']} | "
            f"empty_text={summary['empty_text_count']}"
        )

    return summary
