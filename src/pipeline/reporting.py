from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ISSUE_COLUMNS = ["run_timestamp", "severity", "issue_code", "count", "detail"]


class ReportingError(Exception):
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


# Determine report status from issue severities.
def determine_status(issues: list[dict[str, Any]]) -> str:
    severities = {x["severity"] for x in issues}
    if "ERROR" in severities:
        return "FAIL"
    if "WARNING" in severities:
        return "WARNING"
    return "PASS"


# Load one CSV file and fail with a clear message if missing.
def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ReportingError(f"Missing required input file: {path}")
    return pd.read_csv(path)


# Keep only the latest run block when a summary log has multiple run timestamps.
def select_latest_run(df: pd.DataFrame) -> pd.DataFrame:
    if "run_timestamp" not in df.columns or df.empty:
        return df
    ts = pd.to_datetime(df["run_timestamp"], errors="coerce")
    if ts.isna().all():
        return df
    latest = ts.max()
    return df[ts == latest].copy()


# Convert one value from dataframe row to json-safe primitive type.
def json_value(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


# Keep markdown table cells compact and safe for preview rendering.
def to_md_example(value: Any, max_words: int = 10, max_chars: int = 80) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    # Escape markdown table separator to avoid broken columns.
    text = text.replace("|", r"\|")
    if not text:
        return ""
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]) + " ..."
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + " ..."
    return text


# Build a markdown data dictionary table for the 13 master columns.
def build_data_dictionary_md(master_df: pd.DataFrame) -> str:
    example_row = master_df.iloc[0].to_dict() if not master_df.empty else {}
    specs = [
        ("sample_id", "string", "Deterministic sample id, equal to hash_text."),
        ("text_raw", "string", "Raw text kept for audit and trace."),
        ("text_clean", "string", "Normalized text used for modeling."),
        ("hash_text", "string", "MD5 hash of text_clean for dedup/leakage checks."),
        ("label_binary", "int", "Model label (0=real, 1=fake)."),
        ("label_name", "string", "Human-readable label (real/fake)."),
        ("source_file", "string", "Original source file name."),
        ("source_domain", "string", "Content source domain/platform."),
        ("content_type", "string", "Content type (news/social)."),
        ("published_at", "string", "Published timestamp if available."),
        ("label_confidence", "float", "Source-level label confidence."),
        ("text_length", "int", "Length of text_clean."),
        ("split", "string", "Data split assignment (train/val/test)."),
    ]
    lines = [
        "# Data Dictionary",
        "",
        "| Column | Type | Description | Example |",
        "|---|---|---|---|",
    ]
    for col, dtype, desc in specs:
        example = example_row.get(col, "")
        lines.append(f"| {col} | {dtype} | {desc} | {to_md_example(example)} |")
    return "\n".join(lines) + "\n"


# Build a text-based pipeline overview with the agreed stage flow.
def build_pipeline_overview_md() -> str:
    lines = [
        "# Pipeline Overview",
        "",
        "raw -> ingest -> validate -> normalize -> map_labels -> quality_filter -> deduplicate -> build_master -> split -> reporting",
        "",
        "## Core Rules",
        "- Label convention: 0=real, 1=fake.",
        "- Raw files are immutable.",
        "- Deduplicate before split.",
        "- Leakage checks use hash_text intersection across train/val/test.",
    ]
    return "\n".join(lines) + "\n"


# Plot label distribution for one dataset split/master.
def plot_label_distribution(df: pd.DataFrame, title: str, output_path: Path) -> None:
    counts = df["label_name"].value_counts().reindex(["real", "fake"]).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"])
    ax.set_title(title)
    ax.set_xlabel("label_name")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# Plot stacked label counts by source_file or content_type.
def plot_stacked_label_distribution(
    df: pd.DataFrame,
    group_col: str,
    title: str,
    output_path: Path,
) -> None:
    pivot = (
        df.groupby([group_col, "label_name"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["real", "fake"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=["#4C78A8", "#F58518"])
    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# Plot text_length distribution by split as a compact boxplot.
def plot_text_length_boxplot(master_df: pd.DataFrame, output_path: Path) -> None:
    data = []
    labels = []
    for split_name in ["train", "val", "test"]:
        part = master_df[master_df["split"] == split_name]["text_length"]
        data.append(part.values)
        labels.append(split_name)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_title("Text Length Distribution by Split")
    ax.set_xlabel("split")
    ax.set_ylabel("text_length")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# Plot stage row flow before/after using stage summary values.
def plot_pipeline_row_flow(stage_rows: list[dict[str, Any]], output_path: Path) -> None:
    stages = [row["stage"] for row in stage_rows]
    before_vals = [row["rows_before"] for row in stage_rows]
    after_vals = [row["rows_after"] for row in stage_rows]

    x = range(len(stages))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, before_vals, marker="o", label="before")
    ax.plot(x, after_vals, marker="o", label="after")
    ax.set_xticks(list(x))
    ax.set_xticklabels(stages, rotation=30, ha="right")
    ax.set_ylabel("rows")
    ax.set_title("Pipeline Row Flow (Before vs After)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# Build a compact stage before/after summary from stage logs.
def build_stage_rows(
    ingest_df: pd.DataFrame,
    normalize_df: pd.DataFrame,
    map_df: pd.DataFrame,
    qf_df: pd.DataFrame,
    dedup_df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    dedup_all = dedup_df[dedup_df["source_id"] == "__ALL__"].iloc[0]
    split_row = split_df.iloc[0]
    return [
        {
            "stage": "ingest",
            "rows_before": int(ingest_df["rows_input_after_read"].sum()),
            "rows_after": int(ingest_df["rows_output"].sum()),
        },
        {
            "stage": "normalize",
            "rows_before": int(normalize_df["rows_input"].sum()),
            "rows_after": int(normalize_df["rows_output"].sum()),
        },
        {
            "stage": "map_labels",
            "rows_before": int(map_df["rows_input"].sum()),
            "rows_after": int(map_df["rows_output"].sum()),
        },
        {
            "stage": "quality_filter",
            "rows_before": int(qf_df["rows_input"].sum()),
            "rows_after": int(qf_df["rows_kept"].sum()),
        },
        {
            "stage": "deduplicate",
            "rows_before": int(dedup_all["rows_input"]),
            "rows_after": int(dedup_all["rows_output"]),
        },
        {
            "stage": "split",
            "rows_before": int(split_row["total_rows"]),
            "rows_after": int(split_row["train_rows"] + split_row["val_rows"] + split_row["test_rows"]),
        },
    ]


# Execute reporting stage and generate report artifacts.
def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    reports_dir = repo_root / "reports"
    figures_dir = reports_dir / "figures"
    logs_dir = repo_root / "logs"
    processed_dir = repo_root / "data" / "processed"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    issues: list[dict[str, Any]] = []

    master_df = read_csv_required(processed_dir / "master_dataset_v1.csv")
    train_df = read_csv_required(processed_dir / "train.csv")
    val_df = read_csv_required(processed_dir / "val.csv")
    test_df = read_csv_required(processed_dir / "test.csv")

    ingest_df = select_latest_run(read_csv_required(logs_dir / "ingest_summary.csv"))
    normalize_df = select_latest_run(read_csv_required(logs_dir / "normalize_summary.csv"))
    map_df = select_latest_run(read_csv_required(logs_dir / "map_labels_summary.csv"))
    qf_df = select_latest_run(read_csv_required(logs_dir / "quality_filter_summary.csv"))
    dedup_df = select_latest_run(read_csv_required(logs_dir / "deduplicate_summary.csv"))
    split_df = select_latest_run(read_csv_required(logs_dir / "split_summary.csv"))

    # Cross-check stage-level row consistency using both logs and processed outputs.
    if "__ALL__" not in set(dedup_df["source_id"].astype(str)):
        add_issue(issues, run_timestamp, "ERROR", "missing_deduplicate_all_row", 1, "source_id=__ALL__ not found")
    else:
        dedup_all_rows_out = int(dedup_df[dedup_df["source_id"] == "__ALL__"].iloc[0]["rows_output"])
        if dedup_all_rows_out != len(master_df):
            add_issue(
                issues,
                run_timestamp,
                "ERROR",
                "dedup_master_row_mismatch",
                abs(dedup_all_rows_out - len(master_df)),
                f"deduplicate_rows_output={dedup_all_rows_out}, master_rows={len(master_df)}",
            )

    if split_df.empty:
        add_issue(issues, run_timestamp, "ERROR", "missing_split_summary", 1, "split_summary.csv is empty")
    else:
        split_row = split_df.iloc[0]
        if int(split_row["total_rows"]) != len(master_df):
            add_issue(
                issues,
                run_timestamp,
                "ERROR",
                "split_master_row_mismatch",
                abs(int(split_row["total_rows"]) - len(master_df)),
                f"split_total_rows={int(split_row['total_rows'])}, master_rows={len(master_df)}",
            )

    if len(train_df) + len(val_df) + len(test_df) != len(master_df):
        add_issue(
            issues,
            run_timestamp,
            "ERROR",
            "processed_split_row_mismatch",
            abs((len(train_df) + len(val_df) + len(test_df)) - len(master_df)),
            "len(train)+len(val)+len(test) must equal len(master)",
        )

    # Recompute leakage directly from processed data for trustable reporting.
    train_hash = set(train_df["hash_text"].astype(str))
    val_hash = set(val_df["hash_text"].astype(str))
    test_hash = set(test_df["hash_text"].astype(str))
    leakage_train_val = len(train_hash.intersection(val_hash))
    leakage_train_test = len(train_hash.intersection(test_hash))
    leakage_val_test = len(val_hash.intersection(test_hash))
    leakage_total = leakage_train_val + leakage_train_test + leakage_val_test
    if leakage_total > 0:
        add_issue(
            issues,
            run_timestamp,
            "ERROR",
            "leakage_detected_recomputed",
            leakage_total,
            (
                f"train_val={leakage_train_val}, "
                f"train_test={leakage_train_test}, "
                f"val_test={leakage_val_test}"
            ),
        )

    # Build distribution tables used in TXT/JSON reports.
    by_source_file = (
        master_df.groupby(["source_file", "label_name"]).size().unstack(fill_value=0).sort_index()
    )
    by_content_type = (
        master_df.groupby(["content_type", "label_name"]).size().unstack(fill_value=0).sort_index()
    )
    source_content_table = (
        master_df.groupby(["source_file", "content_type"]).size().reset_index(name="count")
    )

    text_length_stats = {
        "min": float(master_df["text_length"].min()),
        "p25": float(master_df["text_length"].quantile(0.25)),
        "mean": float(master_df["text_length"].mean()),
        "median": float(master_df["text_length"].median()),
        "p75": float(master_df["text_length"].quantile(0.75)),
        "max": float(master_df["text_length"].max()),
    }

    stage_rows = build_stage_rows(
        ingest_df=ingest_df,
        normalize_df=normalize_df,
        map_df=map_df,
        qf_df=qf_df,
        dedup_df=dedup_df,
        split_df=split_df,
    )

    # Generate required and supporting figures for the midterm report package.
    figure_paths = {
        "label_distribution_master": "reports/figures/label_distribution_master.png",
        "label_distribution_train": "reports/figures/label_distribution_train.png",
        "label_distribution_val": "reports/figures/label_distribution_val.png",
        "label_distribution_test": "reports/figures/label_distribution_test.png",
        "label_distribution_by_source_file": "reports/figures/label_distribution_by_source_file.png",
        "label_distribution_by_content_type": "reports/figures/label_distribution_by_content_type.png",
        "text_length_boxplot_by_split": "reports/figures/text_length_boxplot_by_split.png",
        "pipeline_row_flow_before_after": "reports/figures/pipeline_row_flow_before_after.png",
    }
    plot_label_distribution(master_df, "Label Distribution - Master", repo_root / figure_paths["label_distribution_master"])
    plot_label_distribution(train_df, "Label Distribution - Train", repo_root / figure_paths["label_distribution_train"])
    plot_label_distribution(val_df, "Label Distribution - Val", repo_root / figure_paths["label_distribution_val"])
    plot_label_distribution(test_df, "Label Distribution - Test", repo_root / figure_paths["label_distribution_test"])
    plot_stacked_label_distribution(
        master_df,
        "source_file",
        "Label Distribution by Source File",
        repo_root / figure_paths["label_distribution_by_source_file"],
    )
    plot_stacked_label_distribution(
        master_df,
        "content_type",
        "Label Distribution by Content Type",
        repo_root / figure_paths["label_distribution_by_content_type"],
    )
    plot_text_length_boxplot(master_df, repo_root / figure_paths["text_length_boxplot_by_split"])
    plot_pipeline_row_flow(stage_rows, repo_root / figure_paths["pipeline_row_flow_before_after"])

    overall_status = determine_status(issues)

    # Build machine-readable JSON report with a stable top-level schema.
    data_report_json = {
        "run_metadata": {
            "run_timestamp": run_timestamp,
            "overall_status": overall_status,
            "project": "vn-fake-news-detection",
            "stage": "reporting",
        },
        "stages": {
            row["stage"]: {
                "rows_before": row["rows_before"],
                "rows_after": row["rows_after"],
                "removed": row["rows_before"] - row["rows_after"],
                "removed_ratio": (
                    (row["rows_before"] - row["rows_after"]) / row["rows_before"]
                    if row["rows_before"] > 0
                    else 0.0
                ),
            }
            for row in stage_rows
        },
        "consistency_checks": {
            "deduplicate_rows_output_equals_master_rows": "__ALL__" in set(dedup_df["source_id"].astype(str))
            and int(dedup_df[dedup_df["source_id"] == "__ALL__"].iloc[0]["rows_output"]) == len(master_df),
            "split_total_rows_equals_master_rows": (not split_df.empty) and int(split_df.iloc[0]["total_rows"]) == len(master_df),
            "processed_split_rows_sum_equals_master_rows": (len(train_df) + len(val_df) + len(test_df)) == len(master_df),
            "leakage_recomputed": {
                "train_val": leakage_train_val,
                "train_test": leakage_train_test,
                "val_test": leakage_val_test,
                "total": leakage_total,
            },
        },
        "master_stats": {
            "rows": len(master_df),
            "columns": list(master_df.columns),
            "label_0_count": int((master_df["label_binary"] == 0).sum()),
            "label_1_count": int((master_df["label_binary"] == 1).sum()),
            "content_type_counts": {
                str(k): int(v) for k, v in master_df["content_type"].value_counts().to_dict().items()
            },
            "source_file_count": int(master_df["source_file"].nunique()),
        },
        "split_stats": {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_ratio": len(train_df) / len(master_df) if len(master_df) > 0 else 0.0,
            "val_ratio": len(val_df) / len(master_df) if len(master_df) > 0 else 0.0,
            "test_ratio": len(test_df) / len(master_df) if len(master_df) > 0 else 0.0,
        },
        "distributions": {
            "by_source_file": {
                idx: {col: int(val) for col, val in row.items()}
                for idx, row in by_source_file.fillna(0).astype(int).to_dict(orient="index").items()
            },
            "by_content_type": {
                idx: {col: int(val) for col, val in row.items()}
                for idx, row in by_content_type.fillna(0).astype(int).to_dict(orient="index").items()
            },
            "source_file_x_content_type": [
                {k: json_value(v) for k, v in rec.items()}
                for rec in source_content_table.to_dict(orient="records")
            ],
        },
        "text_length_stats": text_length_stats,
        "figures": figure_paths,
        "issues": issues,
    }

    # Write machine-readable JSON report.
    with (reports_dir / "data_report.json").open("w", encoding="utf-8") as f:
        json.dump(data_report_json, f, ensure_ascii=False, indent=2)

    # Write human-readable TXT report from the same computed metrics.
    lines = [
        "DATA REPORT (MIDTERM)",
        f"Run timestamp: {run_timestamp}",
        f"Overall status: {overall_status}",
        "",
        "1) Stage Row Flow (Before vs After)",
    ]
    for row in stage_rows:
        removed = row["rows_before"] - row["rows_after"]
        removed_ratio = (removed / row["rows_before"]) if row["rows_before"] > 0 else 0.0
        lines.append(
            f"- {row['stage']}: input={row['rows_before']}, output={row['rows_after']}, "
            f"removed={removed} ({removed_ratio:.2%})"
        )
    lines.extend(
        [
            "",
            "2) Master Stats",
            f"- Rows: {len(master_df)}",
            f"- Label 0 (real): {(master_df['label_binary'] == 0).sum()}",
            f"- Label 1 (fake): {(master_df['label_binary'] == 1).sum()}",
            f"- Source files: {master_df['source_file'].nunique()}",
            f"- Content types: {master_df['content_type'].value_counts().to_dict()}",
            "",
            "3) Split Stats",
            f"- Train: {len(train_df)} ({len(train_df)/len(master_df):.2%})",
            f"- Val:   {len(val_df)} ({len(val_df)/len(master_df):.2%})",
            f"- Test:  {len(test_df)} ({len(test_df)/len(master_df):.2%})",
            "",
            "4) Leakage (recomputed from processed data)",
            f"- train ∩ val:  {leakage_train_val}",
            f"- train ∩ test: {leakage_train_test}",
            f"- val ∩ test:   {leakage_val_test}",
            f"- total overlap: {leakage_total}",
            "",
            "5) Text Length Stats (master)",
            (
                f"- min={text_length_stats['min']:.0f}, p25={text_length_stats['p25']:.2f}, "
                f"mean={text_length_stats['mean']:.2f}, median={text_length_stats['median']:.2f}, "
                f"p75={text_length_stats['p75']:.2f}, max={text_length_stats['max']:.0f}"
            ),
            "",
            "6) Issues",
        ]
    )
    if issues:
        for issue in issues:
            lines.append(
                f"- [{issue['severity']}] {issue['issue_code']} (count={issue['count']}): {issue['detail']}"
            )
    else:
        lines.append("- None")
    lines.append("")
    with (reports_dir / "data_report.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Write markdown artifacts required by the deliverable checklist.
    with (reports_dir / "data_dictionary.md").open("w", encoding="utf-8") as f:
        f.write(build_data_dictionary_md(master_df))
    with (reports_dir / "pipeline_overview.md").open("w", encoding="utf-8") as f:
        f.write(build_pipeline_overview_md())

    # Keep a dedicated reporting issue log for traceability.
    pd.DataFrame(issues, columns=ISSUE_COLUMNS).to_csv(
        logs_dir / "reporting_issues.csv", index=False, encoding="utf-8-sig"
    )

    print(
        "[REPORTING] Generated reports/data_report.txt, reports/data_report.json, "
        "reports/data_dictionary.md, reports/pipeline_overview.md, and figures under reports/figures."
    )

    if overall_status == "FAIL":
        raise SystemExit("[REPORTING ERROR] Consistency checks failed. See logs/reporting_issues.csv.")


# Expose a CLI-friendly entrypoint for local runs and CI checks.
if __name__ == "__main__":
    main()
