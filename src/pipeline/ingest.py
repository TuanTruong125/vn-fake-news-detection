from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ENCODING_CANDIDATES = ["utf-8", "utf-8-sig", "cp1258", "latin-1"]
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


class IngestError(Exception):
    pass


# Load ingest config YAML as a dictionary.
def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise IngestError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise IngestError(f"Config file must contain a mapping object: {path}")
    return data


# Return True when a configured optional column is null-like.
def is_optional_column(column_name: Any) -> bool:
    if column_name is None:
        return True
    if isinstance(column_name, str):
        return column_name.strip().lower() in {"", "null"}
    return False


# Convert an absolute path to a project-relative POSIX path string.
def to_repo_relative_posix(abs_path: Path, repo_root: Path) -> str:
    return str(abs_path.resolve().relative_to(repo_root.resolve()).as_posix())


# Select the first available text column and report fallback usage.
def select_text_column(headers: set[str], text_columns: list[str]) -> dict[str, Any]:
    cleaned_columns = [str(col).strip() for col in text_columns]
    for idx, col in enumerate(cleaned_columns):
        if col in headers:
            return {"column": col, "is_fallback": idx > 0}
    raise IngestError(f"None of configured text_columns exist in source headers: {cleaned_columns}")


# Read CSV with encoding fallbacks and capture parser warnings.
def read_source_csv(csv_path: Path) -> tuple[pd.DataFrame, str, list[str]]:
    last_error: Exception | None = None

    for encoding in ENCODING_CANDIDATES:
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                df = pd.read_csv(
                    csv_path,
                    encoding=encoding,
                    engine="python",
                    on_bad_lines="skip",
                )

            parser_warnings = [str(w.message) for w in caught_warnings]
            return df, encoding, parser_warnings
        except Exception as exc:
            last_error = exc
            continue

    raise IngestError(f"Cannot read source CSV '{csv_path}' with known encodings. Error: {last_error}")


# Get an optional raw column as cleaned string values or empty strings.
def optional_series(
    df: pd.DataFrame, configured_col: Any, warning_messages: list[str], warn_key: str
) -> pd.Series:
    if is_optional_column(configured_col):
        return pd.Series([""] * len(df), index=df.index)

    col_name = str(configured_col).strip()
    if col_name not in df.columns:
        warning_messages.append(f"optional_column_missing:{warn_key}:{col_name}")
        return pd.Series([""] * len(df), index=df.index)
    series = df[col_name].fillna("").astype(str).str.strip()
    if (series == "").all():
        warning_messages.append(f"{warn_key}_all_empty")
    return series


# Build source_domain_raw and source_domain with explicit fallback rules.
def resolve_source_domain(df: pd.DataFrame, source_cfg: dict[str, Any], warning_messages: list[str]) -> tuple[pd.Series, pd.Series]:
    default_domain = str(source_cfg.get("source_domain_default", "unknown")).strip() or "unknown"
    domain_column = source_cfg.get("domain_column")

    if is_optional_column(domain_column):
        raw_domain = pd.Series([""] * len(df), index=df.index)
        resolved_domain = pd.Series([default_domain] * len(df), index=df.index)
        return raw_domain, resolved_domain

    domain_col_name = str(domain_column).strip()
    if domain_col_name not in df.columns:
        warning_messages.append(f"optional_column_missing:domain_column:{domain_col_name}")
        raw_domain = pd.Series([""] * len(df), index=df.index)
        resolved_domain = pd.Series([default_domain] * len(df), index=df.index)
        return raw_domain, resolved_domain

    raw_domain = df[domain_col_name].fillna("").astype(str).str.strip()
    resolved_domain = raw_domain.mask(raw_domain == "", default_domain)
    if (raw_domain == "").any():
        warning_messages.append("domain_fallback_default")
    return raw_domain, resolved_domain


# Ensure all staging columns exist before strict reordering.
def enforce_staging_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in STAGING_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[STAGING_COLUMNS]


# Build a standardized ingest dataframe and source-level summary.
def build_ingest_frame(
    df_raw: pd.DataFrame,
    source_cfg: dict[str, Any],
    repo_root: Path,
    run_timestamp: str,
    encoding_used: str,
    parser_warnings: list[str],
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    warning_messages: list[str] = []

    source_id = str(source_cfg["source_id"]).strip()
    source_path_abs = (repo_root / str(source_cfg["path"])).resolve()
    output_path_abs = (repo_root / "data" / "staging" / f"ingest_{source_id}.csv").resolve()

    df = df_raw.copy()
    df.columns = [str(col).strip() for col in df.columns]
    headers = set(df.columns)

    selected = select_text_column(headers, source_cfg["text_columns"])
    selected_text_column = selected["column"]
    text_fallback_used = bool(selected["is_fallback"])
    if text_fallback_used:
        warning_messages.append("fallback_text_column")

    label_column = str(source_cfg["label_column"]).strip()
    if label_column not in headers:
        raise IngestError(f"{source_id}: label_column '{label_column}' not found in source.")

    # Reset index so row_index_raw is deterministic across runs.
    df = df.reset_index(drop=True)
    text_raw_source = df[selected_text_column].fillna("").astype(str).str.strip()
    label_raw = df[label_column].fillna("").astype(str).str.strip()

    if (text_raw_source == "").all():
        warning_messages.append("text_raw_all_empty")
    if parser_warnings:
        warning_messages.append("parser_skip_bad_lines")

    published_at_raw = optional_series(df, source_cfg.get("published_at_column"), warning_messages, "published_at_column")
    url_raw = optional_series(df, source_cfg.get("url_column"), warning_messages, "url_column")
    source_domain_raw, source_domain = resolve_source_domain(df, source_cfg, warning_messages)

    content_type = str(source_cfg.get("content_type", "")).strip()
    label_confidence = source_cfg.get("label_confidence")
    if not content_type:
        raise IngestError(f"{source_id}: content_type cannot be empty.")
    if label_confidence is None:
        raise IngestError(f"{source_id}: label_confidence cannot be null.")
    try:
        label_confidence_val = float(label_confidence)
    except Exception as exc:
        raise IngestError(f"{source_id}: label_confidence must be a valid number.") from exc

    # Keep warning list unique while preserving insertion order for stable logs.
    warning_messages = list(dict.fromkeys(warning_messages))

    staging_df = pd.DataFrame(
        {
            "source_id": source_id,
            "source_file": source_path_abs.name,
            "source_path": to_repo_relative_posix(source_path_abs, repo_root),
            "output_path": to_repo_relative_posix(output_path_abs, repo_root),
            "row_index_raw": df.index,
            "text_source_column": selected_text_column,
            "text_fallback_used": text_fallback_used,
            "text_raw_source": text_raw_source,
            "label_raw": label_raw,
            "published_at_raw": published_at_raw,
            "url_raw": url_raw,
            "source_domain_raw": source_domain_raw,
            "source_domain": source_domain,
            "content_type": content_type,
            "label_confidence": label_confidence_val,
        }
    )
    staging_df = enforce_staging_columns(staging_df)

    null_text_count = int((staging_df["text_raw_source"] == "").sum())
    null_label_count = int((staging_df["label_raw"] == "").sum())

    summary_row = {
        "run_timestamp": run_timestamp,
        "pipeline_stage": "ingest",
        "source_id": source_id,
        "source_file": source_path_abs.name,
        "source_path": to_repo_relative_posix(source_path_abs, repo_root),
        "output_path": to_repo_relative_posix(output_path_abs, repo_root),
        "encoding_used": encoding_used,
        "rows_input_after_read": int(len(df_raw)),
        "rows_output": int(len(staging_df)),
        "selected_text_column": selected_text_column,
        "text_fallback_used": text_fallback_used,
        "null_text_count": null_text_count,
        "null_label_count": null_label_count,
        "content_type": content_type,
        "label_confidence": label_confidence_val,
        "warnings_count": len(warning_messages),
        "warning_messages": " | ".join(warning_messages),
    }

    source_warning_lines: list[str] = []
    for msg in warning_messages:
        source_warning_lines.append(f"{run_timestamp}\t{source_id}\t{msg}")
    for parser_msg in parser_warnings:
        source_warning_lines.append(f"{run_timestamp}\t{source_id}\tparser_warning_detail:{parser_msg}")

    return staging_df, summary_row, source_warning_lines


# Write one staging ingest file in UTF-8 BOM for Excel-friendly inspection.
def write_staging_file(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


# Write ingest summary CSV and warning log for the full run.
def write_logs(
    summary_rows: list[dict[str, Any]],
    warning_lines: list[str],
    repo_root: Path,
) -> None:
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_path = logs_dir / "ingest_summary.csv"
    warning_path = logs_dir / "ingest_warnings.log"

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    with warning_path.open("w", encoding="utf-8") as f:
        if warning_lines:
            f.write("\n".join(warning_lines) + "\n")
        else:
            f.write("No warnings.\n")


# Run ingest for all enabled sources and produce staging outputs.
def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = load_yaml(repo_root / "configs" / "data_sources.yaml")

    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise IngestError("configs/data_sources.yaml must define a non-empty 'sources' list.")

    run_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    summary_rows: list[dict[str, Any]] = []
    warning_lines: list[str] = []

    for source_cfg in sources:
        if not isinstance(source_cfg, dict):
            raise IngestError("Each source entry in data_sources.yaml must be a mapping.")
        if not source_cfg.get("enabled", True):
            continue

        source_path_abs = (repo_root / str(source_cfg["path"])).resolve()
        source_id = str(source_cfg["source_id"]).strip()
        output_path_abs = (repo_root / "data" / "staging" / f"ingest_{source_id}.csv").resolve()

        df_raw, encoding_used, parser_warnings = read_source_csv(source_path_abs)
        staging_df, summary_row, source_warning_lines = build_ingest_frame(
            df_raw=df_raw,
            source_cfg=source_cfg,
            repo_root=repo_root,
            run_timestamp=run_timestamp,
            encoding_used=encoding_used,
            parser_warnings=parser_warnings,
        )

        write_staging_file(staging_df, output_path_abs)
        summary_rows.append(summary_row)
        warning_lines.extend(source_warning_lines)

        print(
            f"[INGEST] source_id={source_id} rows={len(staging_df)} "
            f"output={to_repo_relative_posix(output_path_abs, repo_root)}"
        )

    if not summary_rows:
        raise IngestError("No enabled source found. Nothing to ingest.")

    write_logs(summary_rows, warning_lines, repo_root)
    print(
        f"[INGEST] Completed {len(summary_rows)} source(s). "
        "Summary log: logs/ingest_summary.csv, warnings log: logs/ingest_warnings.log"
    )


# Expose a CLI-friendly entrypoint for local runs and CI checks.
if __name__ == "__main__":
    try:
        main()
    except IngestError as exc:
        raise SystemExit(f"[INGEST ERROR] {exc}")
