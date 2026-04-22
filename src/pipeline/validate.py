from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


STAGING_COLUMNS = [
    "source_id",
    "source_file",
    "source_path",
    "output_path",
    "row_index_raw",
    "text_source_column",
    "text_fallback_used",
    "text_raw_source",
    "label_raw",
    "published_at_raw",
    "url_raw",
    "source_domain_raw",
    "source_domain",
    "content_type",
    "label_confidence",
]
ISSUE_ROW_COLUMNS = [
    "run_timestamp",
    "source_id",
    "issue_code",
    "row_index_raw",
    "label_raw",
    "text_preview",
]
MAX_ISSUE_ROWS_PER_TYPE = 200
TEXT_PREVIEW_MAX_LEN = 220

PLACEHOLDER_ONLY_PATTERN = re.compile(
    r"^(?:<\s*url\s*>|url|https?://\S+|www\.\S+)$", re.IGNORECASE
)


class ValidateError(Exception):
    pass


# Load one YAML file and return a dictionary object.
def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValidateError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValidateError(f"Config file must contain a mapping object: {path}")
    return data


# Convert an absolute path into a project-relative POSIX path string.
def to_repo_relative_posix(abs_path: Path, repo_root: Path) -> str:
    return str(abs_path.resolve().relative_to(repo_root.resolve()).as_posix())


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


# Build a compact one-line text preview for row-level issue logs.
def to_text_preview(value: Any) -> str:
    if pd.isna(value):
        return "<EMPTY>"
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    if text == "":
        return "<EMPTY>"
    if len(text) <= TEXT_PREVIEW_MAX_LEN:
        return text
    return text[:TEXT_PREVIEW_MAX_LEN] + "..."


# Append row-level issue traces with a fixed schema and capped row volume.
def add_issue_rows(
    issue_rows: list[dict[str, Any]],
    run_timestamp: str,
    source_id: str,
    issue_code: str,
    df_rows: pd.DataFrame,
) -> None:
    limited_rows = df_rows.head(MAX_ISSUE_ROWS_PER_TYPE)
    for _, row in limited_rows.iterrows():
        label_raw_value = row.get("label_raw", "")
        if pd.isna(label_raw_value) or str(label_raw_value).strip() == "":
            label_raw_value = "<EMPTY>"
        issue_rows.append(
            {
                "run_timestamp": run_timestamp,
                "source_id": source_id,
                "issue_code": issue_code,
                "row_index_raw": row.get("row_index_raw", ""),
                "label_raw": str(label_raw_value),
                "text_preview": to_text_preview(row.get("text_raw_source", "")),
            }
        )


# Get allowed content types from schema constraints.
def get_allowed_content_types(schema_cfg: dict[str, Any]) -> set[str]:
    constraints = schema_cfg.get("constraints")
    if not isinstance(constraints, dict):
        raise ValidateError("configs/schema.yaml must define 'constraints'.")
    allowed = constraints.get("content_type_allowed")
    if not isinstance(allowed, list) or not allowed:
        raise ValidateError("schema.constraints.content_type_allowed must be a non-empty list.")
    return {str(x).strip() for x in allowed if str(x).strip()}


# Build per-source default summary row before applying checks.
def init_summary_row(
    run_timestamp: str,
    source_id: str,
    source_path: str,
    staging_path: str,
) -> dict[str, Any]:
    return {
        "run_timestamp": run_timestamp,
        "source_id": source_id,
        "source_path": source_path,
        "staging_path": staging_path,
        "status": "PASS",
        "rows": 0,
        "error_count": 0,
        "warning_count": 0,
        "null_text_count": 0,
        "null_label_count": 0,
        "placeholder_only_count": 0,
        "placeholder_only_ratio": 0.0,
    }


# Determine source status from issue severities.
def determine_status(source_issues: list[dict[str, Any]]) -> str:
    severities = {item["severity"] for item in source_issues}
    if "ERROR" in severities:
        return "FAIL"
    if "WARNING" in severities:
        return "WARNING"
    return "PASS"


# Validate one source staging file and return summary plus issue rows.
def validate_one_source(
    source_cfg: dict[str, Any],
    label_mapping_cfg: dict[str, Any],
    allowed_content_types: set[str],
    repo_root: Path,
    run_timestamp: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_id = str(source_cfg["source_id"]).strip()
    source_path_abs = (repo_root / str(source_cfg["path"])).resolve()
    staging_path_abs = (repo_root / "data" / "staging" / f"ingest_{source_id}.csv").resolve()
    source_path_rel = to_repo_relative_posix(source_path_abs, repo_root)
    staging_path_rel = to_repo_relative_posix(staging_path_abs, repo_root)

    summary = init_summary_row(
        run_timestamp=run_timestamp,
        source_id=source_id,
        source_path=source_path_rel,
        staging_path=staging_path_rel,
    )
    source_issues: list[dict[str, Any]] = []
    label_value_rows: list[dict[str, Any]] = []
    issue_row_rows: list[dict[str, Any]] = []

    # Stop per-source checks early when staging file is missing.
    if not staging_path_abs.exists():
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "ERROR",
            "missing_file",
            1,
            f"expected_path={staging_path_rel}",
        )
        summary["status"] = "FAIL"
        summary["error_count"] = 1
        return summary, source_issues, label_value_rows, issue_row_rows

    df = pd.read_csv(staging_path_abs)
    summary["rows"] = int(len(df))

    if summary["rows"] == 0:
        add_issue(source_issues, run_timestamp, source_id, "ERROR", "empty_dataset", 1, "rows=0")

    missing_required_cols = sorted(set(STAGING_COLUMNS) - set(df.columns))
    if missing_required_cols:
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "ERROR",
            "missing_required_columns",
            len(missing_required_cols),
            f"missing={missing_required_cols}",
        )
        summary["status"] = determine_status(source_issues)
        summary["error_count"] = sum(1 for x in source_issues if x["severity"] == "ERROR")
        summary["warning_count"] = sum(1 for x in source_issues if x["severity"] == "WARNING")
        return summary, source_issues, label_value_rows, issue_row_rows

    # Validate source_id consistency inside staging output.
    source_id_series = df["source_id"].fillna("").astype(str).str.strip()
    unique_source_ids = sorted(source_id_series.unique().tolist())
    if len(unique_source_ids) != 1 or unique_source_ids[0] != source_id:
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "ERROR",
            "invalid_source_id",
            len(unique_source_ids),
            f"expected={source_id}; actual={unique_source_ids}",
        )

    # Validate source_file consistency with config path basename.
    expected_source_file = source_path_abs.name
    source_file_series = df["source_file"].fillna("").astype(str).str.strip()
    unique_source_files = sorted(source_file_series.unique().tolist())
    if len(unique_source_files) != 1 or unique_source_files[0] != expected_source_file:
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "ERROR",
            "invalid_source_file",
            len(unique_source_files),
            f"expected={expected_source_file}; actual={unique_source_files}",
        )

    # Validate text empty rules and placeholder-only rows.
    text_series = df["text_raw_source"].fillna("").astype(str).str.strip()
    text_empty_mask = text_series == ""
    null_text_count = int(text_empty_mask.sum())
    summary["null_text_count"] = null_text_count

    if len(text_series) > 0 and null_text_count == len(text_series):
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "ERROR",
            "text_all_empty",
            null_text_count,
            "all text_raw_source rows are empty",
        )
    elif null_text_count > 0:
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "WARNING",
            "text_partial_empty",
            null_text_count,
            "some text_raw_source rows are empty",
        )
        add_issue_rows(
            issue_row_rows,
            run_timestamp,
            source_id,
            "text_partial_empty",
            df[text_empty_mask],
        )

    placeholder_mask = text_series.str.match(PLACEHOLDER_ONLY_PATTERN, na=False)
    placeholder_count = int(placeholder_mask.sum())
    summary["placeholder_only_count"] = placeholder_count
    summary["placeholder_only_ratio"] = (
        float(placeholder_count / len(text_series)) if len(text_series) > 0 else 0.0
    )
    if placeholder_count > 0:
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "WARNING",
            "text_placeholder_only",
            placeholder_count,
            f"ratio={summary['placeholder_only_ratio']:.6f}",
        )
        add_issue_rows(
            issue_row_rows,
            run_timestamp,
            source_id,
            "text_placeholder_only",
            df[placeholder_mask],
        )

    # Validate label nulls and label distribution counts.
    label_series = df["label_raw"].fillna("").astype(str).str.strip()
    label_null_mask = label_series == ""
    null_label_count = int(label_null_mask.sum())
    summary["null_label_count"] = null_label_count

    if null_label_count > 0:
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "WARNING",
            "label_null",
            null_label_count,
            "empty label_raw rows found; rows will be dropped in map_labels",
        )

    label_counts = label_series.value_counts(dropna=False).sort_values(ascending=False)
    for label_value, count in label_counts.items():
        label_value_rows.append(
            {
                "run_timestamp": run_timestamp,
                "source_id": source_id,
                "label_raw": "<EMPTY>" if label_value == "" else str(label_value),
                "count": int(count),
            }
        )

    # Validate mapping presence and source-specific label validity.
    mapping_sources = label_mapping_cfg.get("sources", {})
    mapping_entry = mapping_sources.get(source_id) if isinstance(mapping_sources, dict) else None
    if not isinstance(mapping_entry, dict) or not isinstance(mapping_entry.get("mapping"), dict):
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "ERROR",
            "missing_label_mapping",
            1,
            f"source_id={source_id} has no valid mapping in label_mapping.yaml",
        )
    else:
        allowed_raw_labels = {
            str(k).strip() for k in mapping_entry["mapping"].keys() if str(k).strip()
        }
        non_empty_labels = label_series[~label_null_mask]
        invalid_label_mask = (~label_null_mask) & (~label_series.isin(allowed_raw_labels))
        invalid_labels = label_series[invalid_label_mask]
        invalid_count = int(len(invalid_labels))
        if invalid_count > 0:
            top_invalid = invalid_labels.value_counts().head(5).index.tolist()
            add_issue(
                source_issues,
                run_timestamp,
                source_id,
                "ERROR",
                "label_out_of_mapping",
                invalid_count,
                f"top_invalid={top_invalid}",
            )
            add_issue_rows(
                issue_row_rows,
                run_timestamp,
                source_id,
                "label_out_of_mapping",
                df[invalid_label_mask],
            )

    # Validate content_type consistency and allowed values from schema.
    content_type_series = df["content_type"].fillna("").astype(str).str.strip()
    unique_content_types = sorted(content_type_series.unique().tolist())
    if len(unique_content_types) != 1:
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "ERROR",
            "invalid_content_type",
            len(unique_content_types),
            f"content_types={unique_content_types}",
        )
    else:
        content_type_value = unique_content_types[0]
        if content_type_value not in allowed_content_types:
            add_issue(
                source_issues,
                run_timestamp,
                source_id,
                "ERROR",
                "invalid_content_type",
                1,
                f"value={content_type_value}; allowed={sorted(allowed_content_types)}",
            )

    # Validate label_confidence parse and value range with vectorized checks.
    confidence_raw = df["label_confidence"]
    confidence_num = pd.to_numeric(confidence_raw, errors="coerce")
    parse_invalid_mask = confidence_num.isna()
    parse_invalid_count = int(parse_invalid_mask.sum())
    if parse_invalid_count > 0:
        invalid_samples = (
            confidence_raw[parse_invalid_mask].astype(str).str.strip().value_counts().head(5).index.tolist()
        )
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "ERROR",
            "invalid_label_confidence_parse",
            parse_invalid_count,
            f"samples={invalid_samples}",
        )

    range_invalid_mask = (~parse_invalid_mask) & ((confidence_num < 0) | (confidence_num > 1))
    range_invalid_count = int(range_invalid_mask.sum())
    if range_invalid_count > 0:
        range_samples = confidence_num[range_invalid_mask].head(5).tolist()
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "ERROR",
            "invalid_label_confidence_range",
            range_invalid_count,
            f"samples={range_samples}",
        )

    # Warn on duplicate text/label pairs inside one source staging file.
    duplicate_count = int(df.duplicated(subset=["text_raw_source", "label_raw"]).sum())
    if duplicate_count > 0:
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "WARNING",
            "duplicate_rows",
            duplicate_count,
            "duplicated on subset=[text_raw_source,label_raw]",
        )
        duplicate_mask = df.duplicated(subset=["text_raw_source", "label_raw"], keep=False)
        add_issue_rows(
            issue_row_rows,
            run_timestamp,
            source_id,
            "duplicate_rows",
            df[duplicate_mask],
        )

    summary["status"] = determine_status(source_issues)
    summary["error_count"] = sum(1 for x in source_issues if x["severity"] == "ERROR")
    summary["warning_count"] = sum(1 for x in source_issues if x["severity"] == "WARNING")
    return summary, source_issues, label_value_rows, issue_row_rows


# Write validate outputs to CSV/JSON files under the logs directory.
def write_outputs(
    repo_root: Path,
    summary_rows: list[dict[str, Any]],
    issue_rows: list[dict[str, Any]],
    label_value_rows: list[dict[str, Any]],
    issue_row_rows: list[dict[str, Any]],
    report_obj: dict[str, Any],
) -> None:
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_path = logs_dir / "validate_summary.csv"
    issues_path = logs_dir / "validate_issues.csv"
    label_counts_path = logs_dir / "validate_label_value_counts.csv"
    issue_rows_path = logs_dir / "validate_issue_rows.csv"
    report_path = logs_dir / "validate_report.json"

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(issue_rows).to_csv(issues_path, index=False, encoding="utf-8-sig")

    label_df = pd.DataFrame(label_value_rows)
    if not label_df.empty:
        label_df = label_df.sort_values(by=["source_id", "count"], ascending=[True, False])
    label_df.to_csv(label_counts_path, index=False, encoding="utf-8-sig")

    issue_row_df = pd.DataFrame(issue_row_rows, columns=ISSUE_ROW_COLUMNS)
    issue_row_df.to_csv(issue_rows_path, index=False, encoding="utf-8-sig")

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report_obj, f, ensure_ascii=False, indent=2)


# Run validation across all enabled sources and emit reports.
def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_sources_cfg = load_yaml(repo_root / "configs" / "data_sources.yaml")
    label_mapping_cfg = load_yaml(repo_root / "configs" / "label_mapping.yaml")
    schema_cfg = load_yaml(repo_root / "configs" / "schema.yaml")

    allowed_content_types = get_allowed_content_types(schema_cfg)
    run_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    sources = data_sources_cfg.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValidateError("configs/data_sources.yaml must define a non-empty 'sources' list.")
    enabled_sources = [s for s in sources if isinstance(s, dict) and s.get("enabled", True)]
    if not enabled_sources:
        raise ValidateError("No enabled source found in configs/data_sources.yaml.")

    summary_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    label_value_rows: list[dict[str, Any]] = []
    issue_row_rows: list[dict[str, Any]] = []

    for source_cfg in enabled_sources:
        summary, source_issues, source_label_values, source_issue_rows = validate_one_source(
            source_cfg=source_cfg,
            label_mapping_cfg=label_mapping_cfg,
            allowed_content_types=allowed_content_types,
            repo_root=repo_root,
            run_timestamp=run_timestamp,
        )
        summary_rows.append(summary)
        issue_rows.extend(source_issues)
        label_value_rows.extend(source_label_values)
        issue_row_rows.extend(source_issue_rows)
        print(
            f"[VALIDATE] source_id={summary['source_id']} status={summary['status']} "
            f"errors={summary['error_count']} warnings={summary['warning_count']}"
        )

    failed_sources = sum(1 for row in summary_rows if row["status"] == "FAIL")
    warning_sources = sum(1 for row in summary_rows if row["status"] == "WARNING")
    if failed_sources > 0:
        overall_status = "FAIL"
    elif warning_sources > 0:
        overall_status = "WARNING"
    else:
        overall_status = "PASS"

    report_obj = {
        "run_timestamp": run_timestamp,
        "overall_status": overall_status,
        "total_sources": len(summary_rows),
        "failed_sources": failed_sources,
        "warning_sources": warning_sources,
        "sources": [
            {
                "source_id": row["source_id"],
                "status": row["status"],
                "error_count": row["error_count"],
                "warning_count": row["warning_count"],
            }
            for row in summary_rows
        ],
    }

    write_outputs(
        repo_root=repo_root,
        summary_rows=summary_rows,
        issue_rows=issue_rows,
        label_value_rows=label_value_rows,
        issue_row_rows=issue_row_rows,
        report_obj=report_obj,
    )

    print(
        f"[VALIDATE] Completed {len(summary_rows)} source(s). "
        f"overall_status={overall_status}. Logs written to logs/validate_*"
    )

    if overall_status == "FAIL":
        raise SystemExit("[VALIDATE ERROR] Validation failed. Check logs/validate_issues.csv.")


# Expose a CLI-friendly entrypoint for local runs and CI checks.
if __name__ == "__main__":
    main()
