from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.train.dl.train_phobert import DlTrainError, get_repo_root, load_yaml
except ModuleNotFoundError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from train_phobert import DlTrainError, get_repo_root, load_yaml  # type: ignore


DEFAULT_REPORT_PATH = "reports/error_analysis_dl.md"
DEFAULT_SPLITS = "val,test"
DEFAULT_TOP_N = 20
DEFAULT_SOURCE_FIG = "reports/figures/error_dl_by_source.png"
DEFAULT_CONTENT_FIG = "reports/figures/error_dl_by_content_type.png"
DEFAULT_LENGTH_FIG = "reports/figures/error_dl_by_length.png"

REQUIRED_CORE_COLUMNS = ["sample_id", "label_binary"]
REQUIRED_CONTEXT_COLUMNS = ["content_type", "text_length"]


# Error class for issues during DL error analysis.
class DlErrorAnalysisError(Exception):
    pass


# Data class to hold all relevant information for a split-level error analysis.
@dataclass
class SplitAnalysis:
    split_name: str
    frame: pd.DataFrame
    predictions: pd.DataFrame
    merged: pd.DataFrame
    confidence_method: str
    metrics: dict[str, float]
    fp_fn_summary: pd.DataFrame
    top_fp: pd.DataFrame
    top_fn: pd.DataFrame


# Parse CLI arguments for DL error analysis.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DL error analysis and generate markdown + figures.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run_id override. Default: best run from experiments/dl/best_config.json.",
    )
    parser.add_argument(
        "--splits",
        "--split",
        dest="splits",
        type=str,
        default=DEFAULT_SPLITS,
        help="Comma-separated splits to analyze. Default: val,test.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of top FP/FN samples to include in the report.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=DEFAULT_REPORT_PATH,
        help="Markdown report output path relative to repo root or absolute path.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/train_dl.yaml",
        help="Path to train_dl.yaml.",
    )
    return parser.parse_args()


# Resolve repository root path from current file location.
def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# Parse ISO timestamp safely.
def parse_run_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return datetime.min


# Normalize a comma-separated split list.
def parse_split_names(raw_value: str) -> list[str]:
    parts = [part.strip().lower() for part in str(raw_value).split(",")]
    split_names = [part for part in parts if part]
    if not split_names:
        raise DlErrorAnalysisError("At least one split must be provided.")
    allowed = {"val", "test"}
    invalid = [part for part in split_names if part not in allowed]
    if invalid:
        raise DlErrorAnalysisError(f"Unsupported split(s): {invalid}. Allowed values: val,test")
    ordered: list[str] = []
    for split_name in split_names:
        if split_name not in ordered:
            ordered.append(split_name)
    return ordered


# Read a JSON artifact and enforce top-level mapping.
def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DlErrorAnalysisError(f"Missing JSON artifact: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise DlErrorAnalysisError(f"JSON artifact must be an object: {path}")
    return payload


# Load DL best_config.json with fallback to latest PASS run row.
def load_best_config(repo_root: Path) -> tuple[dict[str, Any] | None, Path]:
    best_config_path = repo_root / "experiments" / "dl" / "best_config.json"
    if not best_config_path.exists():
        return None, best_config_path
    payload = load_json(best_config_path)
    return payload, best_config_path


# Load runs.csv rows for run metadata lookup.
def load_runs_rows(runs_path: Path) -> list[dict[str, Any]]:
    if not runs_path.exists():
        raise DlErrorAnalysisError(f"Missing runs file: {runs_path}")
    with runs_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise DlErrorAnalysisError("experiments/dl/runs.csv has no run records.")
    return rows


# Resolve target run either by explicit run_id or best_config fallback.
def resolve_target_run(
    run_rows: list[dict[str, Any]],
    best_config: dict[str, Any] | None,
    requested_run_id: str | None,
) -> dict[str, Any]:
    if requested_run_id:
        requested = requested_run_id.strip()
        for row in run_rows:
            if str(row.get("run_id", "")).strip() == requested:
                return row
        raise DlErrorAnalysisError(f"run_id '{requested}' not found in experiments/dl/runs.csv.")

    if isinstance(best_config, dict):
        best_run = best_config.get("best_run")
        if isinstance(best_run, dict):
            best_run_id = str(best_run.get("run_id", "")).strip()
            if best_run_id:
                for row in run_rows:
                    if str(row.get("run_id", "")).strip() == best_run_id:
                        return row

    pass_rows = [row for row in run_rows if str(row.get("status", "")).upper() == "PASS"]
    if not pass_rows:
        raise DlErrorAnalysisError("No PASS run found in experiments/dl/runs.csv.")
    pass_rows.sort(key=lambda row: parse_run_timestamp(str(row.get("run_timestamp", ""))), reverse=True)
    return pass_rows[0]


# Resolve the physical run directory using run notes or fallback naming convention.
def resolve_run_dir(target_run: dict[str, Any], repo_root: Path) -> Path:
    run_id = str(target_run.get("run_id", "")).strip()
    if not run_id:
        raise DlErrorAnalysisError("Target run row is missing run_id.")

    notes_raw = str(target_run.get("notes", "")).strip()
    if notes_raw:
        try:
            notes_obj = json.loads(notes_raw)
            if isinstance(notes_obj, dict):
                run_dir_value = str(notes_obj.get("run_dir", "")).strip()
                if run_dir_value:
                    run_dir = Path(run_dir_value)
                    if run_dir.exists():
                        return run_dir
        except json.JSONDecodeError:
            pass

    fallback_run_dir = repo_root / "models" / "dl" / run_id
    if not fallback_run_dir.exists():
        raise DlErrorAnalysisError(f"Run directory not found: {fallback_run_dir}")
    return fallback_run_dir


# Load split dataframe using train_dl.yaml configuration.
def load_split_frame(config: dict[str, Any], split_name: str, repo_root: Path) -> pd.DataFrame:
    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        raise DlErrorAnalysisError("train_dl.yaml missing data section.")

    key = f"{split_name}_path"
    split_rel_path = str(data_cfg.get(key, "")).strip()
    if not split_rel_path:
        raise DlErrorAnalysisError(f"train_dl.yaml data.{key} must be non-empty.")

    split_path = repo_root / split_rel_path
    if not split_path.exists():
        raise DlErrorAnalysisError(f"Split file not found: {split_path}")

    frame = pd.read_csv(split_path)
    return frame


# Resolve the best available text column for snippet extraction.
def resolve_text_column(frame: pd.DataFrame, config: dict[str, Any]) -> str:
    data_cfg = config.get("data")
    candidate = ""
    if isinstance(data_cfg, dict):
        candidate = str(data_cfg.get("text_column", "")).strip()
    preferred_candidates = [candidate, "text_clean", "text_raw"]
    for col in preferred_candidates:
        if col and col in frame.columns:
            return col
    raise DlErrorAnalysisError("Cannot resolve snippet text column. Expected text_column, text_clean or text_raw.")


# Resolve the source grouping column used for source-file analysis.
def resolve_source_column(frame: pd.DataFrame) -> str:
    if "source_file" in frame.columns:
        return "source_file"
    raise DlErrorAnalysisError("Missing source_file column for source analysis.")


# Validate columns required for analysis.
def validate_analysis_columns(frame: pd.DataFrame, split_name: str, text_col: str, source_col: str) -> None:
    missing = [col for col in REQUIRED_CORE_COLUMNS if col not in frame.columns]
    missing.extend([col for col in REQUIRED_CONTEXT_COLUMNS if col not in frame.columns])
    if source_col not in frame.columns:
        missing.append(source_col)
    if text_col not in frame.columns:
        missing.append(text_col)
    if missing:
        raise DlErrorAnalysisError(f"{split_name}: missing required columns: {sorted(set(missing))}")


# Load prediction rows for the target run.
def load_predictions(run_dir: Path) -> pd.DataFrame:
    pred_path = run_dir / "predictions.csv"
    if not pred_path.exists():
        raise DlErrorAnalysisError(f"Missing prediction artifact: {pred_path}")
    df = pd.read_csv(pred_path)
    required = ["sample_id", "split", "y_true", "y_pred", "prob_fake", "is_error"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DlErrorAnalysisError(f"predictions.csv missing required columns: {missing}")
    return df


# Load metrics summary for the target run.
def load_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise DlErrorAnalysisError(f"Missing metrics artifact: {metrics_path}")
    return load_json(metrics_path)


# Compute standard binary metrics from a pair of arrays.
def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "f1_macro": float(f1_macro),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
    }


# Derive confidence score from DL probability outputs.
def add_confidence_scores(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    out = frame.copy()
    if "prob_fake" in out.columns:
        prob_fake = pd.to_numeric(out["prob_fake"], errors="coerce")
        out["confidence_score"] = pd.concat([prob_fake, 1.0 - prob_fake], axis=1).max(axis=1)
        return out, "prob_fake_max"
    out["confidence_score"] = pd.NA
    return out, "unavailable"


# Return human-readable definition for each confidence method used in reports.
def describe_confidence_method(confidence_method: str) -> str:
    mapping = {
        "prob_fake_max": "predicted-class probability confidence = max(P(fake), 1 - P(fake)).",
        "unavailable": "confidence unavailable in stored predictions.",
    }
    return mapping.get(confidence_method, "confidence method description unavailable.")


# Build split-level analysis dataframe by joining split metadata and predictions.
def build_split_analysis(
    split_name: str,
    config: dict[str, Any],
    repo_root: Path,
    predictions: pd.DataFrame,
    top_n: int,
) -> SplitAnalysis:
    frame = load_split_frame(config, split_name, repo_root)
    text_col = resolve_text_column(frame, config)
    source_col = resolve_source_column(frame)
    validate_analysis_columns(frame, split_name, text_col, source_col)

    split_predictions = predictions[predictions["split"].astype(str).str.lower() == split_name].copy()
    if split_predictions.empty:
        raise DlErrorAnalysisError(f"No prediction rows found for split='{split_name}'.")

    merged = frame.merge(
        split_predictions,
        on="sample_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_pred"),
    )
    if len(merged) != len(frame):
        raise DlErrorAnalysisError(
            f"{split_name}: merged rows mismatch. split_rows={len(frame)} predictions={len(split_predictions)} merged={len(merged)}"
        )

    y_true = pd.to_numeric(merged["label_binary"], errors="coerce")
    y_pred = pd.to_numeric(merged["y_pred"], errors="coerce")
    if y_true.isna().any() or y_pred.isna().any():
        raise DlErrorAnalysisError(f"{split_name}: y_true/y_pred contains invalid values.")

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    if not set(y_true.unique()).issubset({0, 1}) or not set(y_pred.unique()).issubset({0, 1}):
        raise DlErrorAnalysisError(f"{split_name}: labels outside {{0,1}} detected in predictions or ground truth.")

    if "y_true_pred" in merged.columns:
        compare = pd.to_numeric(merged["y_true_pred"], errors="coerce").fillna(-1).astype(int)
        if not compare.equals(y_true):
            raise DlErrorAnalysisError(f"{split_name}: label mismatch between split CSV and predictions.csv.")

    merged["y_true"] = y_true
    merged["y_pred"] = y_pred
    merged["is_error"] = merged["y_true"] != merged["y_pred"]
    merged["error_type"] = "CORRECT"
    merged.loc[(merged["y_true"] == 0) & (merged["y_pred"] == 1), "error_type"] = "FP"
    merged.loc[(merged["y_true"] == 1) & (merged["y_pred"] == 0), "error_type"] = "FN"
    merged["text_length"] = pd.to_numeric(merged["text_length"], errors="coerce").fillna(0).astype(int)
    merged["source_file"] = merged[source_col].fillna("UNKNOWN").astype(str)
    merged["content_type"] = merged["content_type"].fillna("UNKNOWN").astype(str)
    merged, confidence_method = add_confidence_scores(merged)
    merged["snippet_text"] = merged[text_col].fillna("").astype(str)

    metrics = compute_metrics(merged["y_true"], merged["y_pred"])
    fp_fn_summary = summarize_fp_fn(merged)
    top_fp = select_top_errors(merged, "FP", top_n=top_n)
    top_fn = select_top_errors(merged, "FN", top_n=top_n)

    return SplitAnalysis(
        split_name=split_name,
        frame=frame,
        predictions=split_predictions,
        merged=merged,
        confidence_method=confidence_method,
        metrics=metrics,
        fp_fn_summary=fp_fn_summary,
        top_fp=top_fp,
        top_fn=top_fn,
    )


# Summarize FP/FN counts for a split.
def summarize_fp_fn(df: pd.DataFrame) -> pd.DataFrame:
    total_samples = int(len(df))
    errors = int(df["is_error"].sum())
    fp_count = int(((df["error_type"] == "FP")).sum())
    fn_count = int(((df["error_type"] == "FN")).sum())
    summary = pd.DataFrame(
        [
            {"metric": "samples", "value": total_samples},
            {"metric": "errors", "value": errors},
            {"metric": "fp_count", "value": fp_count},
            {"metric": "fn_count", "value": fn_count},
            {"metric": "error_rate_pct", "value": round((errors / total_samples * 100.0) if total_samples else 0.0, 4)},
        ]
    )
    return summary


# Add length bins for grouped analysis.
def add_length_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bins = [-1, 200, 500, 1000, 2000, 4000, 10_000_000]
    labels = ["0-200", "201-500", "501-1000", "1001-2000", "2001-4000", ">4000"]
    out["length_bin"] = pd.cut(out["text_length"], bins=bins, labels=labels)
    return out


# Aggregate errors by a categorical column.
def aggregate_error(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            samples=("sample_id", "count"),
            errors=("is_error", "sum"),
            fp_count=("error_type", lambda s: int((s == "FP").sum())),
            fn_count=("error_type", lambda s: int((s == "FN").sum())),
        )
        .reset_index()
    )
    grouped[group_col] = grouped[group_col].fillna("UNKNOWN").astype(str)
    grouped["error_rate"] = grouped["errors"] / grouped["samples"].replace(0, 1)
    grouped = grouped.sort_values(["error_rate", "errors", "samples"], ascending=[False, False, False])
    return grouped


# Select top informative mistakes for one error class.
def select_top_errors(df: pd.DataFrame, error_type: str, top_n: int) -> pd.DataFrame:
    subset = df[df["error_type"] == error_type].copy()
    if subset.empty:
        return subset

    confidence_available = "confidence_score" in subset.columns and subset["confidence_score"].notna().any()
    if confidence_available:
        subset = subset.sort_values(["confidence_score", "text_length"], ascending=[False, False])
    else:
        subset = subset.sort_values(["text_length"], ascending=[False])

    top = subset.head(top_n).copy()
    top["snippet"] = top["snippet_text"].apply(build_snippet)
    top["error_rate_hint"] = top["confidence_score"]
    return top[
        [
            "sample_id",
            "split",
            "source_file",
            "content_type",
            "text_length",
            "y_true",
            "y_pred",
            "confidence_score",
            "snippet",
        ]
    ]


# Build a compact text snippet for markdown tables.
def build_snippet(text: Any, max_len: int = 220) -> str:
    normalized = str(text).replace("\n", " ").strip()
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3] + "..."


# Format a dataframe as markdown.
def dataframe_to_markdown(df: pd.DataFrame, float_cols: list[str] | None = None) -> str:
    if df.empty:
        return "_No rows._"

    float_cols = float_cols or []
    formatted = df.copy()
    for col in formatted.columns:
        if str(formatted[col].dtype) == "category":
            formatted[col] = formatted[col].astype(str)
    for col in float_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(lambda v: f"{float(v):.6f}" if pd.notna(v) else "")

    headers = [str(col) for col in formatted.columns.tolist()]
    rows = [[str(value) for value in row] for row in formatted.fillna("").values.tolist()]
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = ["| " + " | ".join(headers) + " |", separator]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# Save a bar chart figure for error-rate analysis.
def save_error_bar_figure(
    df_agg: pd.DataFrame,
    category_col: str,
    out_path: Path,
    title: str,
    top_k: int | None = None,
) -> None:
    chart_df = df_agg.copy()
    if top_k is not None and len(chart_df) > top_k:
        chart_df = chart_df.head(top_k)

    labels = chart_df[category_col].astype(str).tolist()
    rates_pct = (chart_df["error_rate"] * 100).tolist()
    colors = ["#d62728"] + ["#4e79a7"] * (max(len(labels) - 1, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    bars = ax.bar(labels, rates_pct, color=colors)
    ax.set_title(title)
    ax.set_ylabel("Error Rate (%)")
    ax.set_xlabel(category_col)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, rates_pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# Build insight bullets from the aggregated error views.
def build_insights(
    total_samples: int,
    total_errors: int,
    overall_error_rate: float,
    by_source: pd.DataFrame,
    by_content_type: pd.DataFrame,
    by_length: pd.DataFrame,
    wrong_confidence: pd.Series,
) -> list[str]:
    insights: list[str] = []
    insights.append(f"Overall error rate across analyzed splits is {overall_error_rate * 100:.2f}% ({total_errors}/{total_samples}).")

    if not by_source.empty:
        top_source = by_source.iloc[0]
        insights.append(
            f"Highest-error source file: `{top_source.iloc[0]}` with error_rate={top_source['error_rate'] * 100:.2f}% "
            f"(errors={int(top_source['errors'])}/{int(top_source['samples'])})."
        )

    if not by_content_type.empty:
        top_ct = by_content_type.iloc[0]
        if len(by_content_type) >= 2:
            second_ct = by_content_type.iloc[1]
            diff = (float(top_ct["error_rate"]) - float(second_ct["error_rate"])) * 100
            insights.append(
                f"Most difficult content type: `{top_ct.iloc[0]}` is harder than `{second_ct.iloc[0]}` by {diff:.2f} percentage points."
            )
        else:
            insights.append(
                f"Only one content type observed in analysis: `{top_ct.iloc[0]}` with error_rate={top_ct['error_rate'] * 100:.2f}% ."
            )

    if not by_length.empty:
        top_len = by_length.iloc[0]
        insights.append(
            f"Most difficult length bin: `{top_len.iloc[0]}` with error_rate={top_len['error_rate'] * 100:.2f}%."
        )

    if wrong_confidence.notna().any():
        insights.append(
            f"Average confidence on incorrect samples is {wrong_confidence.mean():.6f} (median={wrong_confidence.median():.6f})."
        )
    else:
        insights.append("Confidence scores are unavailable in the stored predictions, so wrong-sample confidence could not be ranked.")

    return insights


# Build markdown report for the DL error analysis deliverable.
def build_report(
    run_info: dict[str, Any],
    selected_splits: list[str],
    metrics_payload: dict[str, Any],
    split_analyses: list[SplitAnalysis],
    by_source: pd.DataFrame,
    by_content_type: pd.DataFrame,
    by_length: pd.DataFrame,
    overall_fp_fn: pd.DataFrame,
    top_fp: pd.DataFrame,
    top_fn: pd.DataFrame,
    insights: list[str],
    source_fig: Path,
    content_fig: Path,
    length_fig: Path,
    source_col: str,
) -> str:
    confidence_methods = sorted({analysis.confidence_method for analysis in split_analyses})
    confidence_notes = "; ".join(
        f"{method}: {describe_confidence_method(method)}" for method in confidence_methods
    )

    lines: list[str] = []
    lines.append("# Error Analysis Report (DL)")
    lines.append("")
    lines.append("## Run Info")
    lines.append(f"- run_id: `{run_info['run_id']}`")
    lines.append(f"- run_timestamp: `{run_info['run_timestamp']}`")
    lines.append(f"- model_name: `{run_info['model_name']}`")
    lines.append(f"- feature_set: `{run_info['feature_set']}`")
    lines.append(f"- text_variant: `{run_info['text_variant']}`")
    lines.append(f"- run_dir: `{run_info['run_dir']}`")
    lines.append(f"- source_file_column: `{source_col}`")
    lines.append(f"- analyzed_splits: `{', '.join(selected_splits)}`")
    lines.append(f"- confidence_definition: {confidence_notes}")
    lines.append("")

    lines.append("## Official Run Metrics")
    lines.append("| Split | F1 Macro | Precision Macro | Recall Macro | Accuracy | F1 Fake |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for split_name in ["val", "test"]:
        metrics = metrics_payload.get("metrics", {}).get(split_name, {}) if isinstance(metrics_payload.get("metrics"), dict) else {}
        if isinstance(metrics, dict):
            lines.append(
                f"| {split_name} | {float(metrics.get('f1_macro', 0.0)):.6f} | {float(metrics.get('precision_macro', 0.0)):.6f} | "
                f"{float(metrics.get('recall_macro', 0.0)):.6f} | {float(metrics.get('accuracy', 0.0)):.6f} | {float(metrics.get('f1_fake', 0.0)):.6f} |"
            )
    lines.append("")

    lines.append("## FP/FN Summary")
    lines.append(dataframe_to_markdown(overall_fp_fn))
    lines.append("")

    lines.append("## Split Metrics")
    summary_rows = []
    for analysis in split_analyses:
        merged = analysis.merged
        total_samples = int(len(merged))
        total_errors = int(merged["is_error"].sum())
        fp_count = int((merged["error_type"] == "FP").sum())
        fn_count = int((merged["error_type"] == "FN").sum())
        summary_rows.append(
            {
                "split": analysis.split_name,
                "confidence_method": analysis.confidence_method,
                "samples": total_samples,
                "errors": total_errors,
                "fp_count": fp_count,
                "fn_count": fn_count,
                "error_rate": round(total_errors / total_samples if total_samples else 0.0, 6),
                **analysis.metrics,
            }
        )
    lines.append(dataframe_to_markdown(pd.DataFrame(summary_rows), float_cols=["error_rate", "f1_macro", "precision_macro", "recall_macro", "accuracy", "f1_fake"]))
    lines.append("")

    lines.append("## Error By Source")
    lines.append(dataframe_to_markdown(by_source, float_cols=["error_rate"]))
    lines.append("")
    lines.append(f"![Error By Source]({source_fig.as_posix()})")
    lines.append("")

    lines.append("## Error By Content Type")
    lines.append(dataframe_to_markdown(by_content_type, float_cols=["error_rate"]))
    lines.append("")
    lines.append(f"![Error By Content Type]({content_fig.as_posix()})")
    lines.append("")

    lines.append("## Error By Length")
    lines.append(dataframe_to_markdown(by_length, float_cols=["error_rate"]))
    lines.append("")
    lines.append(f"![Error By Length]({length_fig.as_posix()})")
    lines.append("")

    lines.append("## Top False Positives")
    lines.append(dataframe_to_markdown(top_fp, float_cols=["confidence_score"]))
    lines.append("")

    lines.append("## Top False Negatives")
    lines.append(dataframe_to_markdown(top_fn, float_cols=["confidence_score"]))
    lines.append("")

    lines.append("## Key Insights")
    for insight in insights:
        lines.append(f"- {insight}")
    lines.append("")

    lines.append("## Suggested Improvements")
    lines.append("- Add or reweight data from the highest-error source/domain group.")
    lines.append("- If FN dominates, optimize recall-oriented fine-tuning or adjust class balance / threshold calibration.")
    lines.append("- If FP dominates, review calibration and hard negatives from the most error-prone content type.")
    lines.append("- For the hardest length bin, consider longer max_length, chunking, or length-aware data augmentation.")
    lines.append("")
    return "\n".join(lines)


# Run end-to-end DL error analysis.
def run_error_analysis(
    run_id: str | None,
    splits_raw: str,
    top_n: int,
    output_report: str | Path,
    config_path: str | Path,
) -> tuple[dict[str, Any], Path, list[Path]]:
    repo_root = get_repo_root()
    selected_splits = parse_split_names(splits_raw)

    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = repo_root / resolved_config_path
    config = load_yaml(resolved_config_path)

    best_config, best_config_path = load_best_config(repo_root)
    run_rows = load_runs_rows(repo_root / "experiments" / "dl" / "runs.csv")
    target_run = resolve_target_run(run_rows, best_config, run_id)
    target_run_id = str(target_run.get("run_id", "")).strip()

    run_dir = resolve_run_dir(target_run, repo_root)
    predictions = load_predictions(run_dir)
    metrics_payload = load_metrics(run_dir)

    split_analyses: list[SplitAnalysis] = []
    for split_name in selected_splits:
        split_analysis = build_split_analysis(split_name, config, repo_root, predictions, top_n)
        split_analyses.append(split_analysis)

    combined = pd.concat([analysis.merged.assign(split=analysis.split_name) for analysis in split_analyses], ignore_index=True)
    combined = add_length_bins(combined)

    source_col = resolve_source_column(split_analyses[0].frame)
    by_source = aggregate_error(combined, source_col)
    by_content_type = aggregate_error(combined, "content_type")
    by_length = aggregate_error(combined, "length_bin")

    overall_fp = int((combined["error_type"] == "FP").sum())
    overall_fn = int((combined["error_type"] == "FN").sum())
    total_samples = int(len(combined))
    total_errors = int(combined["is_error"].sum())
    overall_error_rate = total_errors / total_samples if total_samples else 0.0
    wrong_confidence = pd.to_numeric(combined.loc[combined["is_error"], "confidence_score"], errors="coerce")
    insights = build_insights(total_samples, total_errors, overall_error_rate, by_source, by_content_type, by_length, wrong_confidence)

    overall_fp_fn = pd.DataFrame(
        [
            {"error_type": "FP", "count": overall_fp, "share_of_errors": round(overall_fp / total_errors if total_errors else 0.0, 6)},
            {"error_type": "FN", "count": overall_fn, "share_of_errors": round(overall_fn / total_errors if total_errors else 0.0, 6)},
        ]
    )

    top_fp = pd.concat([analysis.top_fp.assign(split=analysis.split_name) for analysis in split_analyses if not analysis.top_fp.empty], ignore_index=True) if any(not analysis.top_fp.empty for analysis in split_analyses) else pd.DataFrame()
    top_fn = pd.concat([analysis.top_fn.assign(split=analysis.split_name) for analysis in split_analyses if not analysis.top_fn.empty], ignore_index=True) if any(not analysis.top_fn.empty for analysis in split_analyses) else pd.DataFrame()
    if not top_fp.empty:
        top_fp = top_fp.head(top_n)
    if not top_fn.empty:
        top_fn = top_fn.head(top_n)

    figures_dir = repo_root / "reports" / "figures"
    source_fig = figures_dir / "error_dl_by_source.png"
    content_fig = figures_dir / "error_dl_by_content_type.png"
    length_fig = figures_dir / "error_dl_by_length.png"

    save_error_bar_figure(by_source, by_source.columns[0], source_fig, "DL Error Rate by Source", top_k=20)
    save_error_bar_figure(by_content_type, by_content_type.columns[0], content_fig, "DL Error Rate by Content Type", top_k=None)
    save_error_bar_figure(by_length, by_length.columns[0], length_fig, "DL Error Rate by Text Length Bin", top_k=None)

    run_info = {
        "run_id": target_run_id,
        "run_timestamp": str(target_run.get("run_timestamp", "")),
        "model_name": str(target_run.get("model_name", "")),
        "feature_set": str(target_run.get("feature_set", "")),
        "text_variant": str(target_run.get("text_variant", "")),
        "run_dir": str(run_dir.as_posix()),
    }

    report_md = build_report(
        run_info=run_info,
        selected_splits=selected_splits,
        metrics_payload=metrics_payload,
        split_analyses=split_analyses,
        by_source=by_source,
        by_content_type=by_content_type,
        by_length=by_length,
        overall_fp_fn=overall_fp_fn,
        top_fp=top_fp,
        top_fn=top_fn,
        insights=insights,
        source_fig=source_fig.relative_to(repo_root / "reports"),
        content_fig=content_fig.relative_to(repo_root / "reports"),
        length_fig=length_fig.relative_to(repo_root / "reports"),
        source_col=source_col,
    )

    report_path = Path(output_report)
    if not report_path.is_absolute():
        report_path = repo_root / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")

    summary = {
        "run_id": target_run_id,
        "run_timestamp": run_info["run_timestamp"],
        "run_dir": run_info["run_dir"],
        "selected_splits": selected_splits,
        "total_samples": total_samples,
        "total_errors": total_errors,
        "overall_error_rate": overall_error_rate,
        "overall_fp": overall_fp,
        "overall_fn": overall_fn,
        "best_config_path": str(best_config_path.as_posix()),
    }
    return summary, report_path, [source_fig, content_fig, length_fig]


# Build concise CLI output.
def main() -> None:
    args = parse_args()
    print(f"[DL ERROR ANALYSIS] Starting analysis with args: {args}")
    summary, report_path, figure_paths = run_error_analysis(
        run_id=args.run_id,
        splits_raw=args.splits,
        top_n=int(args.top_n),
        output_report=args.output_report,
        config_path=args.config_path,
    )
    print(
        "[DL ERROR ANALYSIS] PASS | "
        f"run_id={summary['run_id']} "
        f"splits={','.join(summary['selected_splits'])} "
        f"errors={summary['total_errors']}/{summary['total_samples']}"
    )
    print(f"[DL ERROR ANALYSIS] Report: {report_path.as_posix()}")
    print(
        "[DL ERROR ANALYSIS] Figures: "
        + ", ".join(path.as_posix() for path in figure_paths)
    )


if __name__ == "__main__":
    try:
        main()
    except DlErrorAnalysisError as exc:
        raise SystemExit(f"[DL ERROR ANALYSIS ERROR] {exc}")
