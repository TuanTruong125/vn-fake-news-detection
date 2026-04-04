from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED_COLUMNS = ["source_id", "row_index_raw", "text_clean", "label_binary", "label_name"]
ISSUE_COLUMNS = ["run_timestamp", "source_id", "severity", "issue_code", "count", "detail"]
REMOVED_ROW_COLUMNS = [
    "run_timestamp",
    "source_id",
    "row_index_raw",
    "remove_reason",
    "text_length",
    "label_binary",
    "text_preview",
]
TEXT_TOO_SHORT_THRESHOLD = 20
HIGH_REMOVED_RATIO_WARNING_THRESHOLD = 0.30
TEXT_PREVIEW_MAX_LEN = 100


class QualityFilterError(Exception):
    pass


# Load one YAML file and return a dictionary object.
def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise QualityFilterError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise QualityFilterError(f"Config file must contain a mapping object: {path}")
    return data


# Append one standardized issue row to the issue list.
def add_issue(
    issues: list[dict[str, Any]],
    run_timestamp: str,
    source_id: str,
    severity: str,
    issue_code: str,
    count: int,
    detail: str,
) -> None:
    issues.append(
        {
            "run_timestamp": run_timestamp,
            "source_id": source_id,
            "severity": severity,
            "issue_code": issue_code,
            "count": int(count),
            "detail": detail,
        }
    )


# Determine per-source status based on issue severities.
def determine_status(source_issues: list[dict[str, Any]]) -> str:
    severities = {item["severity"] for item in source_issues}
    if "ERROR" in severities:
        return "FAIL"
    if "WARNING" in severities:
        return "WARNING"
    return "PASS"


# Build a compact one-line text preview for removed row logs.
def to_text_preview(value: Any) -> str:
    if pd.isna(value):
        return "<EMPTY>"
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    if text == "":
        return "<EMPTY>"
    if len(text) <= TEXT_PREVIEW_MAX_LEN:
        return text
    return text[:TEXT_PREVIEW_MAX_LEN] + "..."


# Validate required columns and return missing-column list.
def missing_required_columns(df: pd.DataFrame) -> list[str]:
    return sorted(set(REQUIRED_COLUMNS) - set(df.columns))


# Build row-level removed records with compact previews.
def build_removed_rows_log(
    removed_df: pd.DataFrame,
    run_timestamp: str,
    source_id: str,
) -> pd.DataFrame:
    if removed_df.empty:
        return pd.DataFrame(columns=REMOVED_ROW_COLUMNS)

    # Keep removed row index while assigning scalar metadata columns.
    out = pd.DataFrame(index=removed_df.index.copy())
    out["run_timestamp"] = run_timestamp
    out["source_id"] = source_id
    out["row_index_raw"] = removed_df["row_index_raw"]
    out["remove_reason"] = removed_df["remove_reason"]
    out["text_length"] = removed_df["text_length"]
    out["label_binary"] = removed_df["label_binary"]
    out["text_preview"] = removed_df["text_clean"].map(to_text_preview)
    return out[REMOVED_ROW_COLUMNS]


# Process one mapped-label source file and apply quality filters.
def filter_one_source(
    source_cfg: dict[str, Any],
    run_timestamp: str,
    repo_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    source_id = str(source_cfg["source_id"]).strip()
    input_path = repo_root / "data" / "staging" / f"map_labels_{source_id}.csv"
    output_path = repo_root / "data" / "staging" / f"quality_filter_{source_id}.csv"

    source_issues: list[dict[str, Any]] = []
    summary = {
        "run_timestamp": run_timestamp,
        "source_id": source_id,
        "rows_input": 0,
        "rows_kept": 0,
        "rows_removed": 0,
        "removed_text_empty": 0,
        "removed_text_too_short": 0,
        "removed_invalid_record": 0,
        "status": "PASS",
    }

    # Stop per-source checks early when mapped-label file is missing.
    if not input_path.exists():
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "ERROR",
            "missing_file",
            1,
            f"expected_path={input_path.as_posix()}",
        )
        summary["status"] = "FAIL"
        return summary, source_issues, pd.DataFrame(), pd.DataFrame(columns=REMOVED_ROW_COLUMNS)

    df = pd.read_csv(input_path)
    summary["rows_input"] = int(len(df))

    missing_cols = missing_required_columns(df)
    if missing_cols:
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "ERROR",
            "missing_required_columns",
            len(missing_cols),
            f"missing={missing_cols}",
        )
        summary["status"] = "FAIL"
        return summary, source_issues, pd.DataFrame(), pd.DataFrame(columns=REMOVED_ROW_COLUMNS)

    # Normalize text and compute text_length before applying any filter rule.
    out_df = df.copy()
    text_series = out_df["text_clean"].fillna("").astype(str).str.strip()
    out_df["text_clean"] = text_series
    out_df["text_length"] = text_series.str.len()
    out_df["remove_reason"] = pd.Series([None] * len(out_df), index=out_df.index, dtype="object")

    # Rule 1: remove rows whose cleaned text is empty.
    empty_mask = out_df["text_clean"] == ""
    out_df.loc[empty_mask, "remove_reason"] = "text_empty"

    # Rule 2: remove rows whose cleaned text length is too short.
    too_short_mask = (out_df["remove_reason"].isna()) & (out_df["text_length"] <= TEXT_TOO_SHORT_THRESHOLD)
    out_df.loc[too_short_mask, "remove_reason"] = "text_too_short"

    # Rule 3: remove rows with clearly invalid labels as a guardrail.
    invalid_binary_mask = ~out_df["label_binary"].isin([0, 1])
    invalid_name_mask = ~out_df["label_name"].astype(str).isin(["real", "fake"])
    invalid_record_mask = (out_df["remove_reason"].isna()) & (invalid_binary_mask | invalid_name_mask)
    out_df.loc[invalid_record_mask, "remove_reason"] = "invalid_record"

    removed_df = out_df[out_df["remove_reason"].notna()].copy()
    kept_df = out_df[out_df["remove_reason"].isna()].copy()

    summary["rows_kept"] = int(len(kept_df))
    summary["rows_removed"] = int(len(removed_df))
    summary["removed_text_empty"] = int((removed_df["remove_reason"] == "text_empty").sum())
    summary["removed_text_too_short"] = int((removed_df["remove_reason"] == "text_too_short").sum())
    summary["removed_invalid_record"] = int((removed_df["remove_reason"] == "invalid_record").sum())

    # Enforce row accounting invariant for trustable quality filter statistics.
    if summary["rows_input"] != summary["rows_kept"] + summary["rows_removed"]:
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "ERROR",
            "row_count_mismatch",
            abs(summary["rows_input"] - (summary["rows_kept"] + summary["rows_removed"])),
            (
                f"rows_input={summary['rows_input']}, rows_kept={summary['rows_kept']}, "
                f"rows_removed={summary['rows_removed']}"
            ),
        )

    removed_ratio = (
        float(summary["rows_removed"] / summary["rows_input"]) if summary["rows_input"] > 0 else 0.0
    )
    if removed_ratio > HIGH_REMOVED_RATIO_WARNING_THRESHOLD:
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "WARNING",
            "high_removed_ratio",
            summary["rows_removed"],
            f"ratio={removed_ratio:.6f}, threshold={HIGH_REMOVED_RATIO_WARNING_THRESHOLD:.2f}",
        )

    summary["status"] = determine_status(source_issues)
    if summary["status"] != "FAIL":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        kept_df = kept_df.drop(columns=["remove_reason"])
        kept_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    removed_rows_log_df = build_removed_rows_log(removed_df, run_timestamp, source_id)
    return summary, source_issues, kept_df, removed_rows_log_df


# Write stage logs for summary/issues/removed rows.
def write_logs(
    repo_root: Path,
    summary_rows: list[dict[str, Any]],
    issue_rows: list[dict[str, Any]],
    removed_rows_df: pd.DataFrame,
) -> None:
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_path = logs_dir / "quality_filter_summary.csv"
    issues_path = logs_dir / "quality_filter_issues.csv"
    removed_path = logs_dir / "quality_filter_removed_rows.csv"

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(issue_rows, columns=ISSUE_COLUMNS).to_csv(issues_path, index=False, encoding="utf-8-sig")
    removed_rows_df.to_csv(removed_path, index=False, encoding="utf-8-sig")


# Execute quality filter for all enabled sources and emit logs.
def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_sources_cfg = load_yaml(repo_root / "configs" / "data_sources.yaml")

    sources = data_sources_cfg.get("sources")
    if not isinstance(sources, list) or not sources:
        raise QualityFilterError("configs/data_sources.yaml must define a non-empty 'sources' list.")
    enabled_sources = [s for s in sources if isinstance(s, dict) and s.get("enabled", True)]
    if not enabled_sources:
        raise QualityFilterError("No enabled source found in configs/data_sources.yaml.")

    run_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    summary_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    removed_rows_parts: list[pd.DataFrame] = []

    for source_cfg in enabled_sources:
        summary, source_issues, _, removed_rows_log_df = filter_one_source(
            source_cfg=source_cfg,
            run_timestamp=run_timestamp,
            repo_root=repo_root,
        )
        summary_rows.append(summary)
        issue_rows.extend(source_issues)
        removed_rows_parts.append(removed_rows_log_df)
        print(
            f"[QUALITY_FILTER] source_id={summary['source_id']} status={summary['status']} "
            f"rows_input={summary['rows_input']} rows_kept={summary['rows_kept']} rows_removed={summary['rows_removed']}"
        )

    if removed_rows_parts:
        removed_rows_df = pd.concat(removed_rows_parts, ignore_index=True)
    else:
        removed_rows_df = pd.DataFrame(columns=REMOVED_ROW_COLUMNS)
    write_logs(
        repo_root=repo_root,
        summary_rows=summary_rows,
        issue_rows=issue_rows,
        removed_rows_df=removed_rows_df,
    )

    failed_sources = sum(1 for row in summary_rows if row["status"] == "FAIL")
    warning_sources = sum(1 for row in summary_rows if row["status"] == "WARNING")
    if failed_sources > 0:
        overall_status = "FAIL"
    elif warning_sources > 0:
        overall_status = "WARNING"
    else:
        overall_status = "PASS"

    print(
        f"[QUALITY_FILTER] Completed {len(summary_rows)} source(s). "
        f"overall_status={overall_status}. Logs written to logs/quality_filter_*"
    )

    if overall_status == "FAIL":
        raise SystemExit("[QUALITY_FILTER ERROR] Quality filtering failed. Check logs/quality_filter_issues.csv.")


# Expose a CLI-friendly entrypoint for local runs and CI checks.
if __name__ == "__main__":
    main()
