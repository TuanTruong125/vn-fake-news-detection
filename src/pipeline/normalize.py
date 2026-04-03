from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED_COLUMNS = ["source_id", "row_index_raw", "text_raw_source"]
EMPTY_AFTER_CLEAN_WARNING_THRESHOLD = 0.20
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
URL_PLACEHOLDER_PATTERN = re.compile(r"<\s*url\s*>", flags=re.IGNORECASE)
ISSUE_COLUMNS = ["run_timestamp", "source_id", "severity", "issue_code", "count", "detail"]


class NormalizeError(Exception):
    pass


# Load one YAML file and return a dictionary object.
def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise NormalizeError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise NormalizeError(f"Config file must contain a mapping object: {path}")
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


# Normalize whitespace with explicit tab/space/newline rules.
def normalize_whitespace(text: str) -> str:
    # Convert Windows-style line endings and tabs before collapsing spaces.
    out = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    # Remove spaces around newlines and collapse repeated spaces.
    out = re.sub(r"[ ]+\n", "\n", out)
    out = re.sub(r"\n[ ]+", "\n", out)
    out = re.sub(r"[ ]{2,}", " ", out)
    # Keep at most one empty line by collapsing repeated newlines.
    out = re.sub(r"\n{2,}", "\n", out)
    return out.strip()


# Run the full normalization pipeline for one text value and return flags.
def normalize_text(text: str) -> dict[str, Any]:
    # Step 1: replace literal "/n" with actual newline.
    step_1 = text.replace("/n", "\n")
    flag_replaced_slash_n = step_1 != text

    # Step 2: normalize Unicode to NFC for stable Vietnamese rendering.
    step_2 = unicodedata.normalize("NFC", step_1)
    flag_changed_nfc = step_2 != step_1

    # Step 3: remove URL placeholder token before generic URL regex.
    step_3 = URL_PLACEHOLDER_PATTERN.sub(" ", step_2)

    # Step 4: remove URL patterns such as http(s) links and www links.
    step_4 = URL_PATTERN.sub(" ", step_3)
    flag_removed_url = step_4 != step_2

    # Step 5: normalize spaces/newlines and trim final boundaries.
    step_5 = normalize_whitespace(step_4)
    flag_whitespace_normalized = step_5 != step_4

    return {
        "text_clean": step_5,
        "flag_replaced_slash_n": bool(flag_replaced_slash_n),
        "flag_changed_nfc": bool(flag_changed_nfc),
        "flag_removed_url": bool(flag_removed_url),
        "flag_whitespace_normalized": bool(flag_whitespace_normalized),
        "flag_empty_after_clean": bool(step_5 == ""),
    }


# Validate required columns and return a missing-column list.
def missing_required_columns(df: pd.DataFrame) -> list[str]:
    return sorted(set(REQUIRED_COLUMNS) - set(df.columns))


# Process one source ingest file and return normalized output metadata.
def normalize_one_source(
    source_cfg: dict[str, Any],
    repo_root: Path,
    run_timestamp: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame | None]:
    source_id = str(source_cfg["source_id"]).strip()
    ingest_path = repo_root / "data" / "staging" / f"ingest_{source_id}.csv"
    normalize_path = repo_root / "data" / "staging" / f"normalize_{source_id}.csv"

    source_issues: list[dict[str, Any]] = []
    summary = {
        "run_timestamp": run_timestamp,
        "source_id": source_id,
        "rows_input": 0,
        "rows_output": 0,
        "changed_count": 0,
        "slash_n_replaced_count": 0,
        "nfc_changed_count": 0,
        "url_removed_count": 0,
        "whitespace_normalized_count": 0,
        "empty_after_clean_count": 0,
        "empty_after_clean_ratio": 0.0,
        "status": "PASS",
    }

    # Skip this source when the expected ingest file is missing.
    if not ingest_path.exists():
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "ERROR",
            "missing_file",
            1,
            f"expected_path={ingest_path.as_posix()}",
        )
        summary["status"] = "FAIL"
        return summary, source_issues, None

    df = pd.read_csv(ingest_path)
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
        return summary, source_issues, None

    text_raw_series = df["text_raw_source"].fillna("").astype(str)
    normalized_records = [normalize_text(text) for text in text_raw_series.tolist()]
    normalized_df = pd.DataFrame(normalized_records)

    out_df = df.copy()
    out_df["text_raw"] = text_raw_series
    out_df["text_clean"] = normalized_df["text_clean"]
    out_df["flag_replaced_slash_n"] = normalized_df["flag_replaced_slash_n"]
    out_df["flag_changed_nfc"] = normalized_df["flag_changed_nfc"]
    out_df["flag_removed_url"] = normalized_df["flag_removed_url"]
    out_df["flag_whitespace_normalized"] = normalized_df["flag_whitespace_normalized"]
    out_df["flag_empty_after_clean"] = normalized_df["flag_empty_after_clean"]

    summary["rows_output"] = int(len(out_df))

    # Treat unexpected row count changes as a pipeline bug.
    if summary["rows_output"] != summary["rows_input"]:
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "ERROR",
            "row_count_mismatch_after_normalize",
            abs(summary["rows_output"] - summary["rows_input"]),
            f"rows_input={summary['rows_input']}, rows_output={summary['rows_output']}",
        )

    text_clean_null_count = int(out_df["text_clean"].isna().sum())
    if text_clean_null_count > 0:
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "ERROR",
            "text_clean_null",
            text_clean_null_count,
            "unexpected null text_clean values detected",
        )

    summary["changed_count"] = int((out_df["text_raw"] != out_df["text_clean"]).sum())
    summary["slash_n_replaced_count"] = int(out_df["flag_replaced_slash_n"].sum())
    summary["nfc_changed_count"] = int(out_df["flag_changed_nfc"].sum())
    summary["url_removed_count"] = int(out_df["flag_removed_url"].sum())
    summary["whitespace_normalized_count"] = int(out_df["flag_whitespace_normalized"].sum())
    summary["empty_after_clean_count"] = int(out_df["flag_empty_after_clean"].sum())
    summary["empty_after_clean_ratio"] = (
        float(summary["empty_after_clean_count"] / summary["rows_output"])
        if summary["rows_output"] > 0
        else 0.0
    )

    if summary["empty_after_clean_ratio"] > EMPTY_AFTER_CLEAN_WARNING_THRESHOLD:
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "WARNING",
            "high_empty_after_clean_ratio",
            summary["empty_after_clean_count"],
            (
                f"ratio={summary['empty_after_clean_ratio']:.6f}, "
                f"threshold={EMPTY_AFTER_CLEAN_WARNING_THRESHOLD:.2f}"
            ),
        )

    summary["status"] = determine_status(source_issues)
    if summary["status"] != "FAIL":
        normalize_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(normalize_path, index=False, encoding="utf-8-sig")

    return summary, source_issues, out_df


# Write normalization summary and issue logs to disk.
def write_logs(
    repo_root: Path,
    summary_rows: list[dict[str, Any]],
    issue_rows: list[dict[str, Any]],
) -> None:
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_path = logs_dir / "normalize_summary.csv"
    issues_path = logs_dir / "normalize_issues.csv"

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(issue_rows, columns=ISSUE_COLUMNS).to_csv(issues_path, index=False, encoding="utf-8-sig")


# Execute normalization for all enabled sources and emit logs.
def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_sources_cfg = load_yaml(repo_root / "configs" / "data_sources.yaml")

    sources = data_sources_cfg.get("sources")
    if not isinstance(sources, list) or not sources:
        raise NormalizeError("configs/data_sources.yaml must define a non-empty 'sources' list.")

    enabled_sources = [s for s in sources if isinstance(s, dict) and s.get("enabled", True)]
    if not enabled_sources:
        raise NormalizeError("No enabled source found in configs/data_sources.yaml.")

    run_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    summary_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []

    for source_cfg in enabled_sources:
        summary, source_issues, _ = normalize_one_source(
            source_cfg=source_cfg,
            repo_root=repo_root,
            run_timestamp=run_timestamp,
        )
        summary_rows.append(summary)
        issue_rows.extend(source_issues)
        print(
            f"[NORMALIZE] source_id={summary['source_id']} status={summary['status']} "
            f"rows={summary['rows_input']} changed={summary['changed_count']}"
        )

    write_logs(repo_root=repo_root, summary_rows=summary_rows, issue_rows=issue_rows)

    failed_sources = sum(1 for row in summary_rows if row["status"] == "FAIL")
    warning_sources = sum(1 for row in summary_rows if row["status"] == "WARNING")
    if failed_sources > 0:
        overall_status = "FAIL"
    elif warning_sources > 0:
        overall_status = "WARNING"
    else:
        overall_status = "PASS"

    print(
        f"[NORMALIZE] Completed {len(summary_rows)} source(s). "
        f"overall_status={overall_status}. Logs written to logs/normalize_*"
    )

    if overall_status == "FAIL":
        raise SystemExit("[NORMALIZE ERROR] Normalization failed. Check logs/normalize_issues.csv.")


# Expose a CLI-friendly entrypoint for local runs and CI checks.
if __name__ == "__main__":
    main()
