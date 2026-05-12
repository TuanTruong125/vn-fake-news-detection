from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ISSUE_COLUMNS = ["run_timestamp", "severity", "issue_code", "count", "detail"]


# Error class for reporting stage issues.
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
        "raw -> ingest -> validate -> normalize -> map_labels -> quality_filter -> deduplicate -> build_master -> split -> prepare_ml_text -> reporting",
        "",
        "## Core Rules",
        "- Label convention: 0=real, 1=fake.",
        "- Raw files are immutable.",
        "- Deduplicate before split.",
        "- Leakage checks use hash_text intersection across train/val/test.",
    ]
    return "\n".join(lines) + "\n"


# Annotate bar charts with value labels on top of each bar.
def annotate_bar_values(ax: Any, value_fmt: str = "{:.0f}", as_percent: bool = False) -> None:
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
    for patch in ax.patches:
        height = float(patch.get_height())
        if height <= 0:
            continue
        x = patch.get_x() + patch.get_width() / 2.0
        y = height + ymax * 0.01
        text = value_fmt.format(height * 100.0) if as_percent else value_fmt.format(height)
        ax.text(x, y, text, ha="center", va="bottom", fontsize=8)


# Annotate line charts with value labels near each point.
def annotate_line_points(ax: Any, x_vals: list[int], y_vals: list[float], value_fmt: str = "{:.0f}") -> None:
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
    for x, y in zip(x_vals, y_vals):
        ax.text(x, y + ymax * 0.012, value_fmt.format(y), ha="center", va="bottom", fontsize=8)


# Plot label distribution for one dataset split/master.
def plot_label_distribution(df: pd.DataFrame, title: str, output_path: Path) -> None:
    counts = df["label_name"].value_counts().reindex(["real", "fake"]).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"])
    ax.set_title(title)
    ax.set_xlabel("label_name")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)
    annotate_bar_values(ax, value_fmt="{:.0f}")
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
    
    for container in ax.containers:
        labels = [f"{int(v)}" if float(v) > 0 else "" for v in container.datavalues]
        ax.bar_label(container, labels=labels, label_type="center", fontsize=8, color="black")
    
    totals = pivot.sum(axis=1).tolist()
    for idx, total in enumerate(totals):
        ax.text(idx, total + max(1.0, max(totals) * 0.01), f"{int(total)}", ha="center", va="bottom", fontsize=8)
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
    box_obj = ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_title("Text Length Distribution by Split")
    ax.set_xlabel("split")
    ax.set_ylabel("text_length")
    ax.grid(axis="y", alpha=0.25)
    
    for idx, median_line in enumerate(box_obj["medians"], start=1):
        median_val = float(median_line.get_ydata()[0])
        ax.text(idx, median_val, f"{median_val:.0f}", ha="center", va="bottom", fontsize=8)
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
    annotate_line_points(ax, list(x), [float(v) for v in before_vals], value_fmt="{:.0f}")
    annotate_line_points(ax, list(x), [float(v) for v in after_vals], value_fmt="{:.0f}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# Plot suspected glued-text ratio by source to compare relative noise levels.
def plot_normalize_suspected_ratio(normalize_df: pd.DataFrame, output_path: Path) -> None:
    if "source_id" not in normalize_df.columns:
        return
    if "suspected_glued_ratio" in normalize_df.columns:
        ratio_df = normalize_df[["source_id", "suspected_glued_ratio"]].copy()
    elif all(col in normalize_df.columns for col in ["suspected_glued_rows", "rows_output"]):
        ratio_df = normalize_df[["source_id", "suspected_glued_rows", "rows_output"]].copy()
        ratio_df["suspected_glued_ratio"] = ratio_df["suspected_glued_rows"] / ratio_df["rows_output"].clip(lower=1)
        ratio_df = ratio_df[["source_id", "suspected_glued_ratio"]]
    else:
        return
    ratio_df = ratio_df[ratio_df["source_id"].astype(str) != "__ALL__"].set_index("source_id")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ratio_percent = ratio_df["suspected_glued_ratio"] * 100.0
    ratio_percent.plot(kind="bar", ax=ax, color="#E45756")
    ax.set_title("Suspected Glued-Text Ratio by Source")
    ax.set_xlabel("source_id")
    ax.set_ylabel("ratio (%)")
    ax.grid(axis="y", alpha=0.25)
    annotate_bar_values(ax, value_fmt="{:.2f}")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# Plot normalize changed ratio by source to show text modification coverage.
def plot_normalize_changed_ratio(normalize_df: pd.DataFrame, output_path: Path) -> None:
    required_cols = ["source_id", "changed_count", "rows_output"]
    if any(col not in normalize_df.columns for col in required_cols):
        return
    plot_df = normalize_df[required_cols].copy()
    plot_df = plot_df[plot_df["source_id"].astype(str) != "__ALL__"]
    plot_df["changed_ratio"] = plot_df["changed_count"] / plot_df["rows_output"].clip(lower=1)
    plot_df = plot_df.set_index("source_id")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    (plot_df["changed_ratio"] * 100.0).plot(kind="bar", ax=ax, color="#4C78A8")
    ax.set_title("Normalize Changed Ratio by Source")
    ax.set_xlabel("source_id")
    ax.set_ylabel("ratio (%)")
    ax.grid(axis="y", alpha=0.25)
    annotate_bar_values(ax, value_fmt="{:.2f}")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# Plot empty-after-clean ratio by source to verify empty text is controlled.
def plot_normalize_empty_ratio(normalize_df: pd.DataFrame, output_path: Path) -> None:
    if "source_id" not in normalize_df.columns:
        return
    if "empty_after_clean_ratio" in normalize_df.columns:
        ratio_df = normalize_df[["source_id", "empty_after_clean_ratio"]].copy()
    elif all(col in normalize_df.columns for col in ["empty_after_clean_count", "rows_output"]):
        ratio_df = normalize_df[["source_id", "empty_after_clean_count", "rows_output"]].copy()
        ratio_df["empty_after_clean_ratio"] = (
            ratio_df["empty_after_clean_count"] / ratio_df["rows_output"].clip(lower=1)
        )
        ratio_df = ratio_df[["source_id", "empty_after_clean_ratio"]]
    else:
        return
    ratio_df = ratio_df[ratio_df["source_id"].astype(str) != "__ALL__"].set_index("source_id")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    (ratio_df["empty_after_clean_ratio"] * 100.0).plot(kind="bar", ax=ax, color="#72B7B2")
    ax.set_title("Empty-After-Clean Ratio by Source")
    ax.set_xlabel("source_id")
    ax.set_ylabel("ratio (%)")
    ax.grid(axis="y", alpha=0.25)
    annotate_bar_values(ax, value_fmt="{:.2f}")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# Pick one available ML text column in preferred order.
def select_ml_text_column(df: pd.DataFrame) -> str | None:
    for col in ["text_ml_seg_lower", "text_ml", "text_ml_seg"]:
        if col in df.columns:
            return col
    return None


# Build n-gram counter from one tokenized text series.
def build_ngram_counter(text_series: pd.Series, n: int) -> Counter:
    counter: Counter = Counter()
    for text in text_series.fillna("").astype(str):
        tokens = [tok for tok in text.split() if tok]
        if len(tokens) < n:
            continue
        grams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        counter.update(grams)
    return counter


# Plot top n-grams with one subplot per label and numeric annotations.
def plot_top_ngrams_by_label(
    train_ml_df: pd.DataFrame,
    text_col: str,
    n: int,
    top_k: int,
    title: str,
    output_path: Path,
) -> None:
    labels = [("real", 0), ("fake", 1)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (label_name, label_bin) in zip(axes, labels):
        subset = train_ml_df[train_ml_df["label_binary"] == label_bin]
        counter = build_ngram_counter(subset[text_col], n=n)
        top_items = counter.most_common(top_k)
        if not top_items:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(f"{label_name}")
            ax.axis("off")
            continue
        grams = [x[0] for x in top_items][::-1]
        counts = [x[1] for x in top_items][::-1]
        bars = ax.barh(range(len(grams)), counts, color="#4C78A8" if label_bin == 0 else "#F58518")
        ax.set_yticks(range(len(grams)))
        ax.set_yticklabels(grams, fontsize=8)
        ax.set_title(f"{label_name}")
        ax.grid(axis="x", alpha=0.25)
        
        for bar, value in zip(bars, counts):
            ax.text(bar.get_width() + max(1, max(counts) * 0.01), bar.get_y() + bar.get_height() / 2, f"{int(value)}", va="center", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# Plot before/after ML text length to show preprocessing impact.
def plot_doc_length_before_after_ml(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_ml_df: pd.DataFrame,
    val_ml_df: pd.DataFrame,
    test_ml_df: pd.DataFrame,
    output_path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    split_pairs = [
        ("train", train_df, train_ml_df),
        ("val", val_df, val_ml_df),
        ("test", test_df, test_ml_df),
    ]
    for split_name, raw_df, ml_df in split_pairs:
        token_col = "text_ml_token_count" if "text_ml_token_count" in ml_df.columns else None
        if token_col is None:
            continue
        rows.append(
            {
                "split": split_name,
                "avg_text_length_chars": float(raw_df["text_length"].mean()),
                "avg_text_length_tokens": float(ml_df[token_col].mean()),
            }
        )

    if not rows:
        return

    stats_df = pd.DataFrame(rows)
    x = range(len(stats_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars_chars = ax.bar([i - width / 2 for i in x], stats_df["avg_text_length_chars"], width=width, label="chars", color="#72B7B2")
    bars_tokens = ax.bar([i + width / 2 for i in x], stats_df["avg_text_length_tokens"], width=width, label="tokens", color="#E45756")
    ax.set_xticks(list(x))
    ax.set_xticklabels(stats_df["split"].tolist())
    ax.set_title("Average Document Length Before/After ML Prep")
    ax.set_xlabel("split")
    ax.set_ylabel("average length")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for bar in list(bars_chars) + list(bars_tokens):
        h = float(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2, h + max(1.0, float(stats_df["avg_text_length_chars"].max()) * 0.01), f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# Render word cloud when library is available, or fallback to token-frequency bars.
def plot_wordcloud_or_fallback(
    text_series: pd.Series,
    title: str,
    output_path: Path,
    top_k_fallback: int = 25,
) -> None:
    text_blob = " ".join(text_series.fillna("").astype(str).tolist()).strip()
    if not text_blob:
        return

    try:
        from wordcloud import WordCloud  # type: ignore

        wc = WordCloud(
            width=1400,
            height=800,
            background_color="white",
            collocations=False,
        ).generate(text_blob)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=140)
        plt.close(fig)
    except Exception:
        counter = build_ngram_counter(text_series, n=1)
        top_items = counter.most_common(top_k_fallback)
        if not top_items:
            return
        tokens = [x[0] for x in top_items][::-1]
        counts = [x[1] for x in top_items][::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(tokens)), counts, color="#4C78A8")
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_title(f"{title} (fallback)")
        ax.set_xlabel("count")
        ax.grid(axis="x", alpha=0.25)
        for bar, value in zip(bars, counts):
            ax.text(bar.get_width() + max(1, max(counts) * 0.01), bar.get_y() + bar.get_height() / 2, f"{int(value)}", va="center", fontsize=8)
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
    ml_prep_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    dedup_all = dedup_df[dedup_df["source_id"] == "__ALL__"].iloc[0]
    split_row = split_df.iloc[0]
    stage_rows = [
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

    # Include ML text preparation stage when logs are available.
    if ml_prep_df is not None and not ml_prep_df.empty:
        ml_all = ml_prep_df[ml_prep_df["input_file"].astype(str) == "__ALL__"]
        if not ml_all.empty:
            ml_row = ml_all.iloc[0]
            stage_rows.append(
                {
                    "stage": "prepare_ml_text",
                    "rows_before": int(ml_row["rows_input"]),
                    "rows_after": int(ml_row["rows_output"]),
                }
            )
    return stage_rows


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
    train_ml_path = processed_dir / "train_ml.csv"
    val_ml_path = processed_dir / "val_ml.csv"
    test_ml_path = processed_dir / "test_ml.csv"
    train_ml_df = read_csv_required(train_ml_path) if train_ml_path.exists() else pd.DataFrame()
    val_ml_df = read_csv_required(val_ml_path) if val_ml_path.exists() else pd.DataFrame()
    test_ml_df = read_csv_required(test_ml_path) if test_ml_path.exists() else pd.DataFrame()
    if train_ml_df.empty or val_ml_df.empty or test_ml_df.empty:
        add_issue(
            issues,
            run_timestamp,
            "WARNING",
            "missing_processed_ml_split_files",
            1,
            "Expected data/processed/train_ml.csv, val_ml.csv, test_ml.csv",
        )

    ingest_df = select_latest_run(read_csv_required(logs_dir / "ingest_summary.csv"))
    normalize_df = select_latest_run(read_csv_required(logs_dir / "normalize_summary.csv"))
    map_df = select_latest_run(read_csv_required(logs_dir / "map_labels_summary.csv"))
    qf_df = select_latest_run(read_csv_required(logs_dir / "quality_filter_summary.csv"))
    dedup_df = select_latest_run(read_csv_required(logs_dir / "deduplicate_summary.csv"))
    split_df = select_latest_run(read_csv_required(logs_dir / "split_summary.csv"))
    ml_prep_summary_path = logs_dir / "prepare_ml_text_summary.csv"
    if ml_prep_summary_path.exists():
        ml_prep_df = select_latest_run(read_csv_required(ml_prep_summary_path))
    else:
        ml_prep_df = pd.DataFrame()
        add_issue(
            issues,
            run_timestamp,
            "WARNING",
            "missing_prepare_ml_text_summary",
            1,
            "prepare_ml_text_summary.csv not found; stage metrics are omitted",
        )

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

    # Cross-check ML-prepared split rows against processed master size.
    if not ml_prep_df.empty:
        ml_all = ml_prep_df[ml_prep_df["input_file"].astype(str) == "__ALL__"]
        if ml_all.empty:
            add_issue(
                issues,
                run_timestamp,
                "WARNING",
                "missing_prepare_ml_text_all_row",
                1,
                "prepare_ml_text_summary.csv has no __ALL__ row",
            )
        else:
            ml_all_row = ml_all.iloc[0]
            if int(ml_all_row["rows_output"]) != len(master_df):
                add_issue(
                    issues,
                    run_timestamp,
                    "ERROR",
                    "prepare_ml_text_master_row_mismatch",
                    abs(int(ml_all_row["rows_output"]) - len(master_df)),
                    f"prepare_ml_text_rows_output={int(ml_all_row['rows_output'])}, master_rows={len(master_df)}",
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
    normalize_noise_signals = {
        "punctuation_spacing_normalized_count": int(
            normalize_df["punctuation_spacing_normalized_count"].sum()
        )
        if "punctuation_spacing_normalized_count" in normalize_df.columns
        else 0,
        "suspected_glued_rows": int(normalize_df["suspected_glued_rows"].sum())
        if "suspected_glued_rows" in normalize_df.columns
        else 0,
        "rows_output_total": int(normalize_df["rows_output"].sum())
        if "rows_output" in normalize_df.columns
        else 0,
    }
    normalize_noise_signals["suspected_glued_ratio"] = (
        normalize_noise_signals["suspected_glued_rows"] / normalize_noise_signals["rows_output_total"]
        if normalize_noise_signals["rows_output_total"] > 0
        else 0.0
    )
    ml_text_prep_signals = {
        "segment_backend_actual": "",
        "rows_output_total": 0,
        "changed_count_seg": 0,
        "changed_count_seg_lower": 0,
        "presegmented_passthrough_count": 0,
        "changed_count": 0,
        "changed_ratio": 0.0,
        "empty_text_ml_count": 0,
        "empty_text_ml_ratio": 0.0,
        "avg_tokens_before_filter": 0.0,
        "avg_tokens_seg": 0.0,
        "avg_tokens_seg_lower": 0.0,
        "avg_tokens_after_filter": 0.0,
    }
    if not ml_prep_df.empty:
        ml_all = ml_prep_df[ml_prep_df["input_file"].astype(str) == "__ALL__"]
        if not ml_all.empty:
            ml_all_row = ml_all.iloc[0]
            rows_output_total = int(ml_all_row["rows_output"])
            changed_count = int(ml_all_row["changed_count"])
            ml_text_prep_signals = {
                "segment_backend_actual": str(ml_all_row.get("segment_backend_actual", "")),
                "rows_output_total": rows_output_total,
                "changed_count_seg": int(ml_all_row.get("changed_count_seg", changed_count)),
                "changed_count_seg_lower": int(ml_all_row.get("changed_count_seg_lower", changed_count)),
                "presegmented_passthrough_count": int(ml_all_row.get("presegmented_passthrough_count", 0)),
                "changed_count": changed_count,
                "changed_ratio": (changed_count / rows_output_total) if rows_output_total > 0 else 0.0,
                "empty_text_ml_count": int(ml_all_row["empty_text_ml_count"]),
                "empty_text_ml_ratio": float(ml_all_row["empty_text_ml_ratio"]),
                "avg_tokens_before_filter": float(ml_all_row["avg_tokens_before_filter"]),
                "avg_tokens_seg": float(ml_all_row.get("avg_tokens_seg", ml_all_row["avg_tokens_after_filter"])),
                "avg_tokens_seg_lower": float(ml_all_row.get("avg_tokens_seg_lower", ml_all_row["avg_tokens_after_filter"])),
                "avg_tokens_after_filter": float(ml_all_row["avg_tokens_after_filter"]),
            }

    stage_rows = build_stage_rows(
        ingest_df=ingest_df,
        normalize_df=normalize_df,
        map_df=map_df,
        qf_df=qf_df,
        dedup_df=dedup_df,
        split_df=split_df,
        ml_prep_df=ml_prep_df,
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
        "normalize_suspected_ratio_by_source": "reports/figures/normalize_suspected_ratio_by_source.png",
        "normalize_changed_ratio_by_source": "reports/figures/normalize_changed_ratio_by_source.png",
        "empty_after_clean_ratio_by_source": "reports/figures/empty_after_clean_ratio_by_source.png",
        "doc_length_before_vs_after_ml": "reports/figures/doc_length_before_vs_after_ml.png",
        "wordcloud_real": "reports/figures/wordcloud_real.png",
        "wordcloud_fake": "reports/figures/wordcloud_fake.png",
        "top_unigram_by_label": "reports/figures/top_unigram_by_label.png",
        "top_bigram_by_label": "reports/figures/top_bigram_by_label.png",
        "top_trigram_by_label": "reports/figures/top_trigram_by_label.png",
        "top_fourgram_by_label": "reports/figures/top_fourgram_by_label.png",
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
    plot_normalize_suspected_ratio(
        normalize_df,
        repo_root / figure_paths["normalize_suspected_ratio_by_source"],
    )
    plot_normalize_changed_ratio(
        normalize_df,
        repo_root / figure_paths["normalize_changed_ratio_by_source"],
    )
    plot_normalize_empty_ratio(
        normalize_df,
        repo_root / figure_paths["empty_after_clean_ratio_by_source"],
    )
    if not train_ml_df.empty and not val_ml_df.empty and not test_ml_df.empty:
        plot_doc_length_before_after_ml(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_ml_df=train_ml_df,
            val_ml_df=val_ml_df,
            test_ml_df=test_ml_df,
            output_path=repo_root / figure_paths["doc_length_before_vs_after_ml"],
        )
        ml_text_col = select_ml_text_column(train_ml_df)
        if ml_text_col is None:
            add_issue(
                issues,
                run_timestamp,
                "WARNING",
                "missing_ml_text_column_for_vocab_figures",
                1,
                "No text_ml_seg_lower/text_ml/text_ml_seg column found in train_ml.csv",
            )
        else:
            train_real_ml = train_ml_df[train_ml_df["label_binary"] == 0]
            train_fake_ml = train_ml_df[train_ml_df["label_binary"] == 1]
            plot_wordcloud_or_fallback(
                text_series=train_real_ml[ml_text_col],
                title="WordCloud - Real (train)",
                output_path=repo_root / figure_paths["wordcloud_real"],
            )
            plot_wordcloud_or_fallback(
                text_series=train_fake_ml[ml_text_col],
                title="WordCloud - Fake (train)",
                output_path=repo_root / figure_paths["wordcloud_fake"],
            )
            plot_top_ngrams_by_label(
                train_ml_df=train_ml_df,
                text_col=ml_text_col,
                n=1,
                top_k=20,
                title="Top Unigram by Label (train)",
                output_path=repo_root / figure_paths["top_unigram_by_label"],
            )
            plot_top_ngrams_by_label(
                train_ml_df=train_ml_df,
                text_col=ml_text_col,
                n=2,
                top_k=20,
                title="Top Bigram by Label (train)",
                output_path=repo_root / figure_paths["top_bigram_by_label"],
            )
            plot_top_ngrams_by_label(
                train_ml_df=train_ml_df,
                text_col=ml_text_col,
                n=3,
                top_k=20,
                title="Top Trigram by Label (train)",
                output_path=repo_root / figure_paths["top_trigram_by_label"],
            )
            plot_top_ngrams_by_label(
                train_ml_df=train_ml_df,
                text_col=ml_text_col,
                n=4,
                top_k=20,
                title="Top Four-gram by Label (train)",
                output_path=repo_root / figure_paths["top_fourgram_by_label"],
            )

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
        "normalize_noise_signals": normalize_noise_signals,
        "ml_text_prep_signals": ml_text_prep_signals,
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
            "6) Normalize Noise Signals",
            (
                "- punctuation_spacing_normalized_count: "
                f"{normalize_noise_signals['punctuation_spacing_normalized_count']}"
            ),
            (
                f"- suspected_glued_rows: {normalize_noise_signals['suspected_glued_rows']} "
                f"({normalize_noise_signals['suspected_glued_ratio']:.2%})"
            ),
            "",
            "7) ML Text Prep Signals",
            f"- segment_backend_actual: {ml_text_prep_signals['segment_backend_actual']}",
            f"- changed_count_seg: {ml_text_prep_signals['changed_count_seg']}",
            f"- changed_count_seg_lower: {ml_text_prep_signals['changed_count_seg_lower']}",
            f"- presegmented_passthrough_count: {ml_text_prep_signals['presegmented_passthrough_count']}",
            f"- changed_count: {ml_text_prep_signals['changed_count']} ({ml_text_prep_signals['changed_ratio']:.2%})",
            (
                f"- empty_text_ml_count: {ml_text_prep_signals['empty_text_ml_count']} "
                f"({ml_text_prep_signals['empty_text_ml_ratio']:.2%})"
            ),
            (
                f"- avg_tokens_before_filter: {ml_text_prep_signals['avg_tokens_before_filter']:.2f}, "
                f"avg_tokens_seg: {ml_text_prep_signals['avg_tokens_seg']:.2f}, "
                f"avg_tokens_seg_lower: {ml_text_prep_signals['avg_tokens_seg_lower']:.2f}, "
                f"avg_tokens_after_filter: {ml_text_prep_signals['avg_tokens_after_filter']:.2f}"
            ),
            "",
            "8) Issues",
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
