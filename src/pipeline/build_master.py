from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "sample_id",
    "hash_text",
    "text_clean",
    "label_binary",
    "label_name",
    "source_file",
    "source_domain",
    "content_type",
    "label_confidence",
]
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
ISSUE_COLUMNS = ["run_timestamp", "severity", "issue_code", "count", "detail"]
ALLOWED_LABEL_BINARY = {0, 1}
ALLOWED_LABEL_NAME = {"real", "fake"}


# Error class for build-master stage issues.
class BuildMasterError(Exception):
    pass


# Append one standardized issue row to the issue list.
def add_issue(
    issues: list[dict[str, Any]],
    run_timestamp: str,
    severity: str,
    issue_code: str,
    count: int,
    detail: str,
) -> None:
    issues.append(
        {
            "run_timestamp": run_timestamp,
            "severity": severity,
            "issue_code": issue_code,
            "count": int(count),
            "detail": detail,
        }
    )


# Determine stage status based on issue severities.
def determine_status(issues: list[dict[str, Any]]) -> str:
    severities = {item["severity"] for item in issues}
    if "ERROR" in severities:
        return "FAIL"
    if "WARNING" in severities:
        return "WARNING"
    return "PASS"


# Validate required columns and return missing-column list.
def missing_required_columns(df: pd.DataFrame) -> list[str]:
    return sorted(set(REQUIRED_COLUMNS) - set(df.columns))


# Build master columns with fallback strategy for text_raw and published_at.
def build_master_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "text_raw" in out.columns:
        text_raw_series = out["text_raw"].fillna("").astype(str)
    elif "text_raw_source" in out.columns:
        text_raw_series = out["text_raw_source"].fillna("").astype(str)
    else:
        text_raw_series = out["text_clean"].fillna("").astype(str)
    out["text_raw"] = text_raw_series

    if "published_at_raw" in out.columns:
        out["published_at"] = out["published_at_raw"].fillna("").astype(str)
    else:
        out["published_at"] = ""

    text_clean_series = out["text_clean"].fillna("").astype(str)
    out["text_clean"] = text_clean_series
    if "text_length" not in out.columns:
        out["text_length"] = text_clean_series.str.len()
    else:
        out["text_length"] = pd.to_numeric(out["text_length"], errors="coerce").fillna(0).astype(int)

    out["split"] = ""

    out["label_binary"] = pd.to_numeric(out["label_binary"], errors="coerce")
    out["label_name"] = out["label_name"].fillna("").astype(str).str.strip().str.lower()

    master = out[EXPECTED_MASTER_COLUMNS].copy()
    return master


# Run strict master validations and append errors/warnings.
def validate_master_frame(
    master_df: pd.DataFrame,
    source_df: pd.DataFrame,
    run_timestamp: str,
    issues: list[dict[str, Any]],
) -> None:
    if list(master_df.columns) != EXPECTED_MASTER_COLUMNS:
        add_issue(
            issues,
            run_timestamp,
            "ERROR",
            "invalid_column_order",
            1,
            f"expected={EXPECTED_MASTER_COLUMNS}; actual={list(master_df.columns)}",
        )

    sample_hash_mismatch = int((master_df["sample_id"] != master_df["hash_text"]).sum())
    if sample_hash_mismatch > 0:
        add_issue(
            issues,
            run_timestamp,
            "ERROR",
            "sample_hash_mismatch",
            sample_hash_mismatch,
            "sample_id must equal hash_text for all rows",
        )

    invalid_binary_mask = ~master_df["label_binary"].isin(list(ALLOWED_LABEL_BINARY))
    invalid_binary_count = int(invalid_binary_mask.sum())
    if invalid_binary_count > 0:
        add_issue(
            issues,
            run_timestamp,
            "ERROR",
            "invalid_label_binary",
            invalid_binary_count,
            "label_binary must be in {0,1}",
        )

    invalid_name_mask = ~master_df["label_name"].isin(ALLOWED_LABEL_NAME)
    invalid_name_count = int(invalid_name_mask.sum())
    if invalid_name_count > 0:
        add_issue(
            issues,
            run_timestamp,
            "ERROR",
            "invalid_label_name",
            invalid_name_count,
            "label_name must be in {real,fake}",
        )

    empty_text_clean_mask = master_df["text_clean"].str.len().le(0)
    empty_text_clean_count = int(empty_text_clean_mask.sum())
    if empty_text_clean_count > 0:
        add_issue(
            issues,
            run_timestamp,
            "ERROR",
            "empty_text_clean",
            empty_text_clean_count,
            "text_clean must be non-empty after quality_filter and deduplicate",
        )

    duplicated_hash_count = int(master_df["hash_text"].duplicated(keep=False).sum())
    if duplicated_hash_count > 0:
        add_issue(
            issues,
            run_timestamp,
            "ERROR",
            "hash_not_unique",
            duplicated_hash_count,
            "hash_text must be globally unique in master dataset",
        )

    split_non_empty_count = int((master_df["split"].astype(str).str.strip() != "").sum())
    if split_non_empty_count > 0:
        add_issue(
            issues,
            run_timestamp,
            "ERROR",
            "split_not_empty_before_split_stage",
            split_non_empty_count,
            "split column must be empty in build-master stage",
        )

    if "published_at_raw" not in source_df.columns:
        add_issue(
            issues,
            run_timestamp,
            "WARNING",
            "missing_optional_column_published_at_raw",
            1,
            "published_at_raw missing; published_at filled with empty string",
        )


# Write summary and issue logs for the build-master stage.
def write_logs(
    repo_root: Path,
    summary_row: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_path = logs_dir / "build_master_summary.csv"
    issues_path = logs_dir / "build_master_issues.csv"

    pd.DataFrame([summary_row]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(issues, columns=ISSUE_COLUMNS).to_csv(issues_path, index=False, encoding="utf-8-sig")


# Execute build-master stage and emit master dataset plus logs.
def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    run_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    issues: list[dict[str, Any]] = []

    input_path = repo_root / "data" / "staging" / "deduplicate_master_internal.csv"
    output_path = repo_root / "data" / "processed" / "master_dataset_v1.csv"

    if not input_path.exists():
        raise BuildMasterError(f"Missing input file: {input_path}")

    source_df = pd.read_csv(input_path)
    rows_input = int(len(source_df))

    missing_cols = missing_required_columns(source_df)
    if missing_cols:
        add_issue(
            issues,
            run_timestamp,
            "ERROR",
            "missing_required_columns",
            len(missing_cols),
            f"missing={missing_cols}",
        )
        status = determine_status(issues)
        summary_row = {
            "run_timestamp": run_timestamp,
            "rows_input": rows_input,
            "rows_output": 0,
            "unique_hash_count": 0,
            "label_0_count": 0,
            "label_1_count": 0,
            "split_empty_count": 0,
            "status": status,
        }
        write_logs(repo_root, summary_row, issues)
        raise SystemExit("[BUILD_MASTER ERROR] Missing required columns. Check logs/build_master_issues.csv.")

    master_df = build_master_frame(source_df)
    validate_master_frame(master_df, source_df, run_timestamp, issues)

    status = determine_status(issues)
    rows_output = int(len(master_df))
    unique_hash_count = int(master_df["hash_text"].nunique(dropna=False))
    label_0_count = int((master_df["label_binary"] == 0).sum())
    label_1_count = int((master_df["label_binary"] == 1).sum())
    split_empty_count = int((master_df["split"].astype(str).str.strip() == "").sum())

    summary_row = {
        "run_timestamp": run_timestamp,
        "rows_input": rows_input,
        "rows_output": rows_output,
        "unique_hash_count": unique_hash_count,
        "label_0_count": label_0_count,
        "label_1_count": label_1_count,
        "split_empty_count": split_empty_count,
        "status": status,
    }
    write_logs(repo_root, summary_row, issues)

    if status == "FAIL":
        raise SystemExit("[BUILD_MASTER ERROR] Validation failed. Check logs/build_master_issues.csv.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(
        f"[BUILD_MASTER] status={status} rows_output={rows_output} "
        f"output={output_path.as_posix()}"
    )


# Expose a CLI-friendly entrypoint for local runs and CI checks.
if __name__ == "__main__":
    main()
