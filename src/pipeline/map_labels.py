from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED_COLUMNS = ["source_id", "label_raw"]
ISSUE_COLUMNS = ["run_timestamp", "source_id", "severity", "issue_code", "count", "detail"]


# Error class for label mapping stage issues.
class MapLabelsError(Exception):
    pass


# Load one YAML file and return a dictionary object.
def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MapLabelsError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise MapLabelsError(f"Config file must contain a mapping object: {path}")
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


# Build a report table that documents raw-to-binary mapping per source.
def build_label_mapping_table(mapping_cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    global_map = mapping_cfg.get("global", {})
    label_binary_to_name = global_map.get("label_binary_to_name", {})
    sources_map = mapping_cfg.get("sources", {})

    if not isinstance(sources_map, dict):
        raise MapLabelsError("configs/label_mapping.yaml must define 'sources' mapping.")

    for source_id, entry in sources_map.items():
        if not isinstance(entry, dict):
            continue
        source_column = str(entry.get("label_column", "")).strip()
        mapping = entry.get("mapping", {})
        if not isinstance(mapping, dict):
            continue
        for raw_label, binary_label in mapping.items():
            label_name = label_binary_to_name.get(int(binary_label), "")
            rows.append(
                {
                    "source_id": str(source_id),
                    "mapping_source_column": source_column,
                    "raw_label_value": str(raw_label),
                    "label_binary": int(binary_label),
                    "label_name": str(label_name),
                }
            )

    table_df = pd.DataFrame(rows)
    if not table_df.empty:
        table_df = table_df.sort_values(
            by=["source_id", "label_binary", "raw_label_value"],
            ascending=[True, True, True],
        )
    return table_df


# Validate required columns and return missing-column list.
def missing_required_columns(df: pd.DataFrame) -> list[str]:
    return sorted(set(REQUIRED_COLUMNS) - set(df.columns))


# Process one normalized source file and return mapping summary.
def map_one_source(
    source_cfg: dict[str, Any],
    mapping_cfg: dict[str, Any],
    run_timestamp: str,
    repo_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_id = str(source_cfg["source_id"]).strip()
    normalize_path = repo_root / "data" / "staging" / f"normalize_{source_id}.csv"
    output_path = repo_root / "data" / "staging" / f"map_labels_{source_id}.csv"

    source_issues: list[dict[str, Any]] = []
    summary = {
        "run_timestamp": run_timestamp,
        "source_id": source_id,
        "rows_input": 0,
        "rows_output": 0,
        "mapped_count": 0,
        "unmapped_count": 0,
        "label_0_count": 0,
        "label_1_count": 0,
        "status": "PASS",
    }

    # Skip this source when normalized staging file is missing.
    if not normalize_path.exists():
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "ERROR",
            "missing_file",
            1,
            f"expected_path={normalize_path.as_posix()}",
        )
        summary["status"] = "FAIL"
        return summary, source_issues

    df = pd.read_csv(normalize_path)
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
        return summary, source_issues

    # Validate source-specific mapping availability in config.
    sources_map = mapping_cfg.get("sources", {})
    mapping_entry = sources_map.get(source_id) if isinstance(sources_map, dict) else None
    if not isinstance(mapping_entry, dict) or not isinstance(mapping_entry.get("mapping"), dict):
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "ERROR",
            "missing_label_mapping",
            1,
            f"source_id={source_id} has no valid mapping entry",
        )
        summary["status"] = "FAIL"
        return summary, source_issues

    # Normalize raw labels to string before lookup to avoid type mismatches.
    label_raw_norm = df["label_raw"].fillna("").astype(str).str.strip()
    label_null_mask = label_raw_norm == ""
    label_null_count = int(label_null_mask.sum())
    if label_null_count > 0:
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "WARNING",
            "label_null",
            label_null_count,
            "empty label_raw rows are dropped before mapping",
        )

    raw_mapping = {str(k).strip(): int(v) for k, v in mapping_entry["mapping"].items()}
    mapped_series = label_raw_norm.map(raw_mapping)
    invalid_mask = (~label_null_mask) & (mapped_series.isna())
    invalid_count = int(invalid_mask.sum())
    if invalid_count > 0:
        top_invalid = label_raw_norm[invalid_mask].value_counts().head(5).index.tolist()
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "ERROR",
            "label_out_of_mapping",
            invalid_count,
            f"top_invalid={top_invalid}",
        )

    # Keep guardrail check that mapped labels must be exactly {0,1}.
    valid_mapped = mapped_series.dropna().astype(int)
    invalid_binary_mask = ~valid_mapped.isin([0, 1])
    invalid_binary_count = int(invalid_binary_mask.sum())
    if invalid_binary_count > 0:
        invalid_binary_values = valid_mapped[invalid_binary_mask].value_counts().index.tolist()
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "ERROR",
            "invalid_label_binary_after_map",
            invalid_binary_count,
            f"invalid_values={invalid_binary_values}",
        )

    summary["mapped_count"] = int(valid_mapped.shape[0])
    summary["unmapped_count"] = int(label_null_count + invalid_count)
    summary["label_0_count"] = int((valid_mapped == 0).sum())
    summary["label_1_count"] = int((valid_mapped == 1).sum())

    # Warn when a source contributes only one class after successful mapping.
    if summary["mapped_count"] > 0 and (summary["label_0_count"] == 0 or summary["label_1_count"] == 0):
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "WARNING",
            "single_class_source",
            1,
            f"label_0_count={summary['label_0_count']}, label_1_count={summary['label_1_count']}",
        )

    summary["status"] = determine_status(source_issues)
    if summary["status"] == "FAIL":
        summary["rows_output"] = 0
        return summary, source_issues

    # Drop empty raw labels and keep only valid mapped rows.
    keep_mask = (~label_null_mask) & (~mapped_series.isna())

    # Apply mapped labels and canonical label names for downstream steps.
    out_df = df.loc[keep_mask].copy()
    out_df["label_binary"] = mapped_series.loc[keep_mask].astype(int)
    out_df["label_name"] = out_df["label_binary"].map({0: "real", 1: "fake"})
    summary["rows_output"] = int(len(out_df))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return summary, source_issues


# Write summary and issue logs for the mapping stage.
def write_logs(
    repo_root: Path,
    summary_rows: list[dict[str, Any]],
    issue_rows: list[dict[str, Any]],
) -> None:
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_path = logs_dir / "map_labels_summary.csv"
    issues_path = logs_dir / "map_labels_issues.csv"

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(issue_rows, columns=ISSUE_COLUMNS).to_csv(issues_path, index=False, encoding="utf-8-sig")


# Write the mapping table report for documentation and submission artifacts.
def write_label_mapping_table(repo_root: Path, mapping_table_df: pd.DataFrame) -> None:
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "label_mapping_table.csv"
    mapping_table_df.to_csv(output_path, index=False, encoding="utf-8-sig")


# Execute source-wise label mapping and emit stage outputs.
def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_sources_cfg = load_yaml(repo_root / "configs" / "data_sources.yaml")
    mapping_cfg = load_yaml(repo_root / "configs" / "label_mapping.yaml")

    sources = data_sources_cfg.get("sources")
    if not isinstance(sources, list) or not sources:
        raise MapLabelsError("configs/data_sources.yaml must define a non-empty 'sources' list.")
    enabled_sources = [s for s in sources if isinstance(s, dict) and s.get("enabled", True)]
    if not enabled_sources:
        raise MapLabelsError("No enabled source found in configs/data_sources.yaml.")

    run_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    summary_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []

    for source_cfg in enabled_sources:
        summary, source_issues = map_one_source(
            source_cfg=source_cfg,
            mapping_cfg=mapping_cfg,
            run_timestamp=run_timestamp,
            repo_root=repo_root,
        )
        summary_rows.append(summary)
        issue_rows.extend(source_issues)
        print(
            f"[MAP_LABELS] source_id={summary['source_id']} status={summary['status']} "
            f"mapped={summary['mapped_count']} unmapped={summary['unmapped_count']}"
        )

    write_logs(repo_root=repo_root, summary_rows=summary_rows, issue_rows=issue_rows)
    mapping_table_df = build_label_mapping_table(mapping_cfg)
    write_label_mapping_table(repo_root=repo_root, mapping_table_df=mapping_table_df)

    failed_sources = sum(1 for row in summary_rows if row["status"] == "FAIL")
    warning_sources = sum(1 for row in summary_rows if row["status"] == "WARNING")
    if failed_sources > 0:
        overall_status = "FAIL"
    elif warning_sources > 0:
        overall_status = "WARNING"
    else:
        overall_status = "PASS"

    print(
        f"[MAP_LABELS] Completed {len(summary_rows)} source(s). overall_status={overall_status}. "
        "Logs written to logs/map_labels_* and reports/label_mapping_table.csv"
    )

    if overall_status == "FAIL":
        raise SystemExit("[MAP_LABELS ERROR] Label mapping failed. Check logs/map_labels_issues.csv.")


# Expose a CLI-friendly entrypoint for local runs and CI checks.
if __name__ == "__main__":
    main()
