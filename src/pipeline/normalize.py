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
SUSPECTED_GLUED_WARNING_THRESHOLD = 0.02
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
URL_PLACEHOLDER_PATTERN = re.compile(r"<\s*url\s*>", flags=re.IGNORECASE)
ABBREVIATION_DOT_PATTERN = re.compile(r"\b(?:[A-ZĐ]{1,6}\.){1,}[A-ZĐ]{1,10}\b")
DOTTED_NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)+\b")
ISSUE_COLUMNS = ["run_timestamp", "source_id", "severity", "issue_code", "count", "detail"]
PUNCT_SIDE_EFFECT_WARNING_MIN_BASE = 20
PUNCT_SIDE_EFFECT_WARNING_DROP_RATIO = 0.25


# Error class for normalization stage issues.
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


# Normalize punctuation spacing without breaking numeric time patterns.
def normalize_punctuation_spacing(text: str) -> str:
    out = text

    # Protect ellipsis tokens to avoid accidental split like ". . .".
    out = out.replace("...", "__ELLIPSIS__")

    # Normalize spacing around comma/semicolon/question/exclamation marks.
    out = re.sub(r"\s+([,;!?])", r"\1", out)
    out = re.sub(r"([,;!?])(?=[^\s\n])", r"\1 ", out)

    # Normalize colon spacing only when not part of numeric tokens like 13:30.
    out = re.sub(r"(?<!\d)\s+:(?!\d)", ":", out)
    out = re.sub(r"(?<!\d):(?![\s\n\d])", ": ", out)

    # Normalize period spacing conservatively for likely sentence boundaries only.
    out = re.sub(r"(?<=[a-zà-ỹ0-9\"'\)\]])\.(?=[A-ZÀ-Ỹ])", ". ", out)

    # Restore ellipsis tokens.
    out = out.replace("__ELLIPSIS__", "...")
    return out


# Count suspicious glued-word tokens for warning-only quality monitoring.
def count_suspected_glued_tokens(text: str) -> int:
    count = 0
    for token in text.split():
        bare = token.strip(".,;:!?\"'()[]{}")
        if len(bare) < 8:
            continue
        if "_" in bare or any(ch.isdigit() for ch in bare):
            continue
        if not re.fullmatch(r"[A-Za-zÀ-ỹ]+", bare):
            continue
        
        # Focus only on tokens carrying Vietnamese diacritics.
        decomposed = unicodedata.normalize("NFD", bare.lower())
        has_diacritic = any(unicodedata.category(ch) == "Mn" for ch in decomposed)
        if not has_diacritic:
            continue
        
        # Convert to base Latin letters and count vowel groups as syllable proxy.
        base = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
        base = base.replace("đ", "d")
        vowel_groups = re.findall(r"[aeiouy]+", base)

        # Mark as suspicious when one token likely contains multiple glued syllables.
        if len(vowel_groups) >= 2:
            count += 1
    return count


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

    # Count dot-based tokens before punctuation spacing to detect side effects.
    abbr_before_punct = len(ABBREVIATION_DOT_PATTERN.findall(step_4))
    numdot_before_punct = len(DOTTED_NUMBER_PATTERN.findall(step_4))

    # Step 5: normalize punctuation spacing for readability and consistency.
    step_5 = normalize_punctuation_spacing(step_4)
    flag_punctuation_spacing_normalized = step_5 != step_4

    # Step 6: normalize spaces/newlines and trim final boundaries.
    step_6 = normalize_whitespace(step_5)
    flag_whitespace_normalized = step_6 != step_5
    suspected_glued_token_count = count_suspected_glued_tokens(step_6)
    abbr_after_punct = len(ABBREVIATION_DOT_PATTERN.findall(step_6))
    numdot_after_punct = len(DOTTED_NUMBER_PATTERN.findall(step_6))

    return {
        "text_clean": step_6,
        "flag_replaced_slash_n": bool(flag_replaced_slash_n),
        "flag_changed_nfc": bool(flag_changed_nfc),
        "flag_removed_url": bool(flag_removed_url),
        "flag_punctuation_spacing_normalized": bool(flag_punctuation_spacing_normalized),
        "flag_whitespace_normalized": bool(flag_whitespace_normalized),
        "flag_empty_after_clean": bool(step_6 == ""),
        "suspected_glued_token_count": int(suspected_glued_token_count),
        "flag_suspected_glued_token": bool(suspected_glued_token_count > 0),
        "abbr_before_punct": int(abbr_before_punct),
        "abbr_after_punct": int(abbr_after_punct),
        "numdot_before_punct": int(numdot_before_punct),
        "numdot_after_punct": int(numdot_after_punct),
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
        "punctuation_spacing_normalized_count": 0,
        "whitespace_normalized_count": 0,
        "empty_after_clean_count": 0,
        "empty_after_clean_ratio": 0.0,
        "suspected_glued_rows": 0,
        "suspected_glued_ratio": 0.0,
        "abbr_before_punct_total": 0,
        "abbr_after_punct_total": 0,
        "numdot_before_punct_total": 0,
        "numdot_after_punct_total": 0,
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
    out_df["flag_punctuation_spacing_normalized"] = normalized_df["flag_punctuation_spacing_normalized"]
    out_df["flag_whitespace_normalized"] = normalized_df["flag_whitespace_normalized"]
    out_df["flag_empty_after_clean"] = normalized_df["flag_empty_after_clean"]
    out_df["suspected_glued_token_count"] = normalized_df["suspected_glued_token_count"]
    out_df["flag_suspected_glued_token"] = normalized_df["flag_suspected_glued_token"]
    out_df["abbr_before_punct"] = normalized_df["abbr_before_punct"]
    out_df["abbr_after_punct"] = normalized_df["abbr_after_punct"]
    out_df["numdot_before_punct"] = normalized_df["numdot_before_punct"]
    out_df["numdot_after_punct"] = normalized_df["numdot_after_punct"]

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
    summary["punctuation_spacing_normalized_count"] = int(
        out_df["flag_punctuation_spacing_normalized"].sum()
    )
    summary["whitespace_normalized_count"] = int(out_df["flag_whitespace_normalized"].sum())
    summary["empty_after_clean_count"] = int(out_df["flag_empty_after_clean"].sum())
    summary["empty_after_clean_ratio"] = (
        float(summary["empty_after_clean_count"] / summary["rows_output"])
        if summary["rows_output"] > 0
        else 0.0
    )
    summary["suspected_glued_rows"] = int(out_df["flag_suspected_glued_token"].sum())
    summary["suspected_glued_ratio"] = (
        float(summary["suspected_glued_rows"] / summary["rows_output"])
        if summary["rows_output"] > 0
        else 0.0
    )
    summary["abbr_before_punct_total"] = int(out_df["abbr_before_punct"].sum())
    summary["abbr_after_punct_total"] = int(out_df["abbr_after_punct"].sum())
    summary["numdot_before_punct_total"] = int(out_df["numdot_before_punct"].sum())
    summary["numdot_after_punct_total"] = int(out_df["numdot_after_punct"].sum())

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
    if summary["suspected_glued_ratio"] > SUSPECTED_GLUED_WARNING_THRESHOLD:
        add_issue(
            source_issues,
            run_timestamp,
            source_id,
            "WARNING",
            "high_suspected_glued_ratio",
            summary["suspected_glued_rows"],
            (
                f"ratio={summary['suspected_glued_ratio']:.6f}, "
                f"threshold={SUSPECTED_GLUED_WARNING_THRESHOLD:.2f}"
            ),
        )
    if summary["abbr_before_punct_total"] >= PUNCT_SIDE_EFFECT_WARNING_MIN_BASE:
        abbr_drop_ratio = 1.0 - (
            summary["abbr_after_punct_total"] / summary["abbr_before_punct_total"]
        )
        if abbr_drop_ratio > PUNCT_SIDE_EFFECT_WARNING_DROP_RATIO:
            add_issue(
                source_issues,
                run_timestamp,
                source_id,
                "WARNING",
                "punctuation_side_effect_abbreviation_drop",
                int(summary["abbr_before_punct_total"] - summary["abbr_after_punct_total"]),
                (
                    f"drop_ratio={abbr_drop_ratio:.6f}, before={summary['abbr_before_punct_total']}, "
                    f"after={summary['abbr_after_punct_total']}"
                ),
            )
    if summary["numdot_before_punct_total"] >= PUNCT_SIDE_EFFECT_WARNING_MIN_BASE:
        numdot_drop_ratio = 1.0 - (
            summary["numdot_after_punct_total"] / summary["numdot_before_punct_total"]
        )
        if numdot_drop_ratio > PUNCT_SIDE_EFFECT_WARNING_DROP_RATIO:
            add_issue(
                source_issues,
                run_timestamp,
                source_id,
                "WARNING",
                "punctuation_side_effect_numdot_drop",
                int(summary["numdot_before_punct_total"] - summary["numdot_after_punct_total"]),
                (
                    f"drop_ratio={numdot_drop_ratio:.6f}, before={summary['numdot_before_punct_total']}, "
                    f"after={summary['numdot_after_punct_total']}"
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
