from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    from src.utils.hashing import md5_text
except ModuleNotFoundError:  # pragma: no cover - fallback for direct script execution
    import sys

    # Add repo root to sys.path so `src.*` imports work in script mode.
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.utils.hashing import md5_text


REQUIRED_COLUMNS = ["source_id", "row_index_raw", "text_clean"]
ISSUE_COLUMNS = ["run_timestamp", "source_id", "severity", "issue_code", "count", "detail"]
REMOVED_COLUMNS = [
    "run_timestamp",
    "source_id",
    "row_index_raw",
    "hash_text",
    "remove_reason",
    "kept_sample_id",
    "text_preview",
]
TEXT_PREVIEW_MAX_LEN = 100


# Error class for deduplication stage issues.
class DeduplicateError(Exception):
    pass


# Load one YAML file and return a dictionary object.
def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DeduplicateError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise DeduplicateError(f"Config file must contain a mapping object: {path}")
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


# Build a compact one-line preview for removed duplicate logs.
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


# Read one quality-filter file and attach source order for deterministic sorting.
def read_source_frame(
    source_cfg: dict[str, Any],
    source_order: int,
    run_timestamp: str,
    repo_root: Path,
    issue_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    source_id = str(source_cfg["source_id"]).strip()
    input_path = repo_root / "data" / "staging" / f"quality_filter_{source_id}.csv"

    if not input_path.exists():
        add_issue(
            issue_rows,
            run_timestamp,
            source_id,
            "ERROR",
            "missing_file",
            1,
            f"expected_path={input_path.as_posix()}",
        )
        return pd.DataFrame()

    df = pd.read_csv(input_path)
    missing_cols = missing_required_columns(df)
    if missing_cols:
        add_issue(
            issue_rows,
            run_timestamp,
            source_id,
            "ERROR",
            "missing_required_columns",
            len(missing_cols),
            f"missing={missing_cols}",
        )
        return pd.DataFrame()

    out = df.copy()
    out["source_order"] = source_order
    out["source_id"] = out["source_id"].fillna("").astype(str).str.strip()
    return out


# Build per-source summary rows from input/output counts and issue rows.
def build_summary_rows(
    run_timestamp: str,
    enabled_source_ids: list[str],
    input_counts: dict[str, int],
    output_counts: dict[str, int],
    issue_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_id in enabled_source_ids:
        source_issues = [x for x in issue_rows if x["source_id"] == source_id]
        rows_input = int(input_counts.get(source_id, 0))
        rows_output = int(output_counts.get(source_id, 0))
        duplicates_removed = max(rows_input - rows_output, 0)
        duplicates_removed_ratio = (duplicates_removed / rows_input) if rows_input > 0 else 0.0
        rows.append(
            {
                "run_timestamp": run_timestamp,
                "source_id": source_id,
                "rows_input": rows_input,
                "rows_output": rows_output,
                "duplicates_removed": duplicates_removed,
                "duplicates_removed_ratio": duplicates_removed_ratio,
                "status": determine_status(source_issues),
            }
        )

    total_rows_input = int(sum(input_counts.get(sid, 0) for sid in enabled_source_ids))
    total_rows_output = int(sum(output_counts.get(sid, 0) for sid in enabled_source_ids))
    total_removed = max(total_rows_input - total_rows_output, 0)
    total_removed_ratio = (total_removed / total_rows_input) if total_rows_input > 0 else 0.0
    overall_status = determine_status(issue_rows)
    rows.append(
        {
            "run_timestamp": run_timestamp,
            "source_id": "__ALL__",
            "rows_input": total_rows_input,
            "rows_output": total_rows_output,
            "duplicates_removed": total_removed,
            "duplicates_removed_ratio": total_removed_ratio,
            "status": overall_status,
        }
    )
    return rows


# Write logs for deduplicate stage.
def write_logs(
    repo_root: Path,
    summary_rows: list[dict[str, Any]],
    issue_rows: list[dict[str, Any]],
    removed_rows_df: pd.DataFrame,
) -> None:
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_path = logs_dir / "deduplicate_summary.csv"
    issues_path = logs_dir / "deduplicate_issues.csv"
    removed_path = logs_dir / "deduplicate_removed_rows.csv"

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(issue_rows, columns=ISSUE_COLUMNS).to_csv(issues_path, index=False, encoding="utf-8-sig")
    removed_rows_df.to_csv(removed_path, index=False, encoding="utf-8-sig")


# Execute global deduplication across enabled sources and emit outputs.
def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_sources_cfg = load_yaml(repo_root / "configs" / "data_sources.yaml")

    sources = data_sources_cfg.get("sources")
    if not isinstance(sources, list) or not sources:
        raise DeduplicateError("configs/data_sources.yaml must define a non-empty 'sources' list.")
    enabled_sources = [s for s in sources if isinstance(s, dict) and s.get("enabled", True)]
    if not enabled_sources:
        raise DeduplicateError("No enabled source found in configs/data_sources.yaml.")

    run_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    issue_rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    enabled_source_ids = [str(s["source_id"]).strip() for s in enabled_sources]

    # Read source files in config order so keep=first stays deterministic.
    for idx, source_cfg in enumerate(enabled_sources):
        frame = read_source_frame(
            source_cfg=source_cfg,
            source_order=idx,
            run_timestamp=run_timestamp,
            repo_root=repo_root,
            issue_rows=issue_rows,
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise SystemExit("[DEDUPLICATE ERROR] No valid source data available for deduplication.")

    full_df = pd.concat(frames, ignore_index=True)
    input_counts = full_df["source_id"].value_counts().to_dict()

    # Ensure deterministic order before hash-based keep=first deduplication.
    full_df = full_df.sort_values(by=["source_order", "row_index_raw"], ascending=[True, True]).reset_index(drop=True)

    text_clean_series = full_df["text_clean"].fillna("").astype(str)
    full_df["text_clean"] = text_clean_series
    full_df["hash_text"] = text_clean_series.map(md5_text)
    full_df["sample_id"] = full_df["hash_text"]

    null_hash_count = int(full_df["hash_text"].isna().sum())
    null_sample_count = int(full_df["sample_id"].isna().sum())
    if null_hash_count > 0:
        add_issue(
            issue_rows,
            run_timestamp,
            "__ALL__",
            "ERROR",
            "hash_text_null",
            null_hash_count,
            "hash_text contains null values",
        )
    if null_sample_count > 0:
        add_issue(
            issue_rows,
            run_timestamp,
            "__ALL__",
            "ERROR",
            "sample_id_null",
            null_sample_count,
            "sample_id contains null values",
        )

    # Keep first row per hash_text and mark remaining rows as duplicates.
    duplicate_mask = full_df.duplicated(subset=["hash_text"], keep="first")
    kept_df = full_df[~duplicate_mask].copy()
    removed_df = full_df[duplicate_mask].copy()

    # Map each removed hash to the kept sample_id for traceability.
    kept_sample_by_hash = kept_df.set_index("hash_text")["sample_id"].to_dict()
    removed_df["kept_sample_id"] = removed_df["hash_text"].map(kept_sample_by_hash)
    removed_df["remove_reason"] = "duplicate_hash"

    # Guardrail: hash_text must be globally unique after deduplication.
    post_dedup_duplicates = int(kept_df.duplicated(subset=["hash_text"], keep=False).sum())
    if post_dedup_duplicates > 0:
        add_issue(
            issue_rows,
            run_timestamp,
            "__ALL__",
            "ERROR",
            "hash_not_unique_after_dedup",
            post_dedup_duplicates,
            "hash_text still has duplicates after deduplication",
        )

    # Write per-source deduplicated outputs and global master output.
    output_counts: dict[str, int] = {}
    for source_id in enabled_source_ids:
        source_out = kept_df[kept_df["source_id"] == source_id].copy()
        output_counts[source_id] = int(len(source_out))
        out_path = repo_root / "data" / "staging" / f"deduplicate_{source_id}.csv"
        source_out = source_out.drop(columns=["source_order"], errors="ignore")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        source_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    master_path = repo_root / "data" / "staging" / "deduplicate_master_internal.csv"
    kept_master = kept_df.drop(columns=["source_order"], errors="ignore")
    kept_master.to_csv(master_path, index=False, encoding="utf-8-sig")

    removed_rows_df = pd.DataFrame(columns=REMOVED_COLUMNS)
    if not removed_df.empty:
        removed_rows_df = pd.DataFrame(
            {
                "run_timestamp": run_timestamp,
                "source_id": removed_df["source_id"],
                "row_index_raw": removed_df["row_index_raw"],
                "hash_text": removed_df["hash_text"],
                "remove_reason": removed_df["remove_reason"],
                "kept_sample_id": removed_df["kept_sample_id"],
                "text_preview": removed_df["text_clean"].map(to_text_preview),
            }
        )[REMOVED_COLUMNS]

    summary_rows = build_summary_rows(
        run_timestamp=run_timestamp,
        enabled_source_ids=enabled_source_ids,
        input_counts=input_counts,
        output_counts=output_counts,
        issue_rows=issue_rows,
    )
    write_logs(
        repo_root=repo_root,
        summary_rows=summary_rows,
        issue_rows=issue_rows,
        removed_rows_df=removed_rows_df,
    )

    overall_status = next(x["status"] for x in summary_rows if x["source_id"] == "__ALL__")
    print(
        f"[DEDUPLICATE] Completed {len(enabled_source_ids)} source(s). "
        f"overall_status={overall_status}. Logs written to logs/deduplicate_*"
    )

    if overall_status == "FAIL":
        raise SystemExit("[DEDUPLICATE ERROR] Deduplication failed. Check logs/deduplicate_issues.csv.")


# Expose a CLI-friendly entrypoint for local runs and CI checks.
if __name__ == "__main__":
    main()
