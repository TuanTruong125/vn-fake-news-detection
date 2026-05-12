from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from joblib import load

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.train.ml.vectorizers import load_train_ml_config, resolve_config_path
except ModuleNotFoundError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from vectorizers import load_train_ml_config, resolve_config_path  # type: ignore


REQUIRED_COLUMNS = [
    "sample_id",
    "label_binary",
    "source_file",
    "content_type",
    "text_length",
    "text_clean",
]


# Error class for error analysis stage issues.
class ErrorAnalysisError(Exception):
    pass


# Resolve repository root path from current file location.
def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# Parse CLI arguments for error analysis run.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML error analysis and generate markdown + figures.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run_id override. If provided, skip best_config.json and resolve artifacts by run_id.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Data split to analyze. Default: test.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top FP/FN rows to include in report.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="reports/error_analysis_ml.md",
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional path to train_ml.yaml. Default configs/train_ml.yaml",
    )
    return parser.parse_args()


# Load best config JSON artifact.
def load_best_config(best_config_path: Path) -> dict[str, Any]:
    if not best_config_path.exists():
        raise ErrorAnalysisError(f"Missing best_config.json: {best_config_path}")
    with best_config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ErrorAnalysisError("best_config.json must be a JSON object.")
    return payload


# Resolve run_id to analyze using CLI override or best_config fallback.
def resolve_run_id(best_config: dict[str, Any], requested_run_id: str | None) -> str:
    if requested_run_id:
        return requested_run_id.strip()

    best_run = best_config.get("best_run")
    if not isinstance(best_run, dict):
        raise ErrorAnalysisError("best_config.json missing best_run object.")
    run_id = str(best_run.get("run_id", "")).strip()
    if not run_id:
        raise ErrorAnalysisError("best_config.json best_run.run_id is empty.")
    return run_id


# Load runs.csv rows for run metadata lookup.
def load_runs_rows(runs_path: Path) -> list[dict[str, Any]]:
    if not runs_path.exists():
        raise ErrorAnalysisError(f"Missing runs file: {runs_path}")
    with runs_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


# Resolve text_variant from run metadata with deterministic fallback order.
def resolve_text_variant(
    run_id: str,
    best_config: dict[str, Any] | None,
    config: dict[str, Any],
    repo_root: Path,
) -> str:
    runs_path = repo_root / "experiments" / "ml" / "runs.csv"
    rows = load_runs_rows(runs_path)
    for row in rows:
        if str(row.get("run_id", "")).strip() == run_id:
            candidate = str(row.get("text_variant", "")).strip()
            if candidate:
                return candidate

    if isinstance(best_config, dict):
        best_run = best_config.get("best_run")
        if isinstance(best_run, dict):
            candidate = str(best_run.get("text_variant", "")).strip()
            if candidate:
                return candidate

    candidate = str(config.get("run", {}).get("default", {}).get("text_variant", "")).strip()
    if candidate:
        return candidate
    raise ErrorAnalysisError("Cannot resolve text_variant from run metadata or config defaults.")


# Resolve artifact paths from best_config or default run_id naming convention.
def resolve_artifact_paths(
    best_config: dict[str, Any] | None,
    run_id: str,
    repo_root: Path,
) -> tuple[Path, Path, Path | None]:
    model_path = repo_root / "models" / "ml" / f"{run_id}__model.joblib"
    vectorizer_path = repo_root / "models" / "ml" / f"{run_id}__vectorizer.joblib"
    metadata_path = repo_root / "models" / "ml" / f"{run_id}__metadata.json"

    if isinstance(best_config, dict):
        best_run = best_config.get("best_run")
        if isinstance(best_run, dict):
            notes = best_run.get("notes")
            if isinstance(notes, dict):
                artifacts = notes.get("artifacts")
                if isinstance(artifacts, dict):
                    model_from_notes = artifacts.get("model_path")
                    vectorizer_from_notes = artifacts.get("vectorizer_path")
                    metadata_from_notes = artifacts.get("metadata_path")
                    if isinstance(model_from_notes, str) and model_from_notes.strip():
                        model_path = Path(model_from_notes)
                    if isinstance(vectorizer_from_notes, str) and vectorizer_from_notes.strip():
                        vectorizer_path = Path(vectorizer_from_notes)
                    if isinstance(metadata_from_notes, str) and metadata_from_notes.strip():
                        metadata_path = Path(metadata_from_notes)

    # Mandatory artifacts for one run.
    if not model_path.exists():
        raise ErrorAnalysisError(f"Model artifact not found: {model_path}")
    if not vectorizer_path.exists():
        raise ErrorAnalysisError(f"Vectorizer artifact not found: {vectorizer_path}")
    if not metadata_path.exists():
        metadata_path = None
    return model_path, vectorizer_path, metadata_path


# Resolve split CSV path from train_ml config data section.
def resolve_split_path(config: dict[str, Any], split_name: str, repo_root: Path) -> Path:
    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        raise ErrorAnalysisError("train_ml.yaml missing data section.")
    key = f"{split_name}_path"
    split_path = data_cfg.get(key)
    if not isinstance(split_path, str) or not split_path.strip():
        raise ErrorAnalysisError(f"train_ml.yaml data.{key} must be non-empty string.")
    final_path = repo_root / split_path
    if not final_path.exists():
        raise ErrorAnalysisError(f"Split file not found: {final_path}")
    return final_path


# Validate required metadata columns for error analysis.
def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ErrorAnalysisError(f"Split dataframe missing required columns: {missing}")


# Compute model confidence scores for predictions with method fallback.
def compute_confidence_scores(model: Any, x_vec: Any, y_pred: pd.Series) -> tuple[pd.Series, str]:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_vec)
        confidence = probs.max(axis=1)
        return pd.Series(confidence, index=y_pred.index), "predict_proba_max"

    if hasattr(model, "decision_function"):
        decision = model.decision_function(x_vec)
        if hasattr(decision, "ndim") and decision.ndim > 1:
            confidence = pd.Series(pd.DataFrame(decision).abs().max(axis=1).values, index=y_pred.index)
        else:
            confidence = pd.Series(pd.Series(decision).abs().values, index=y_pred.index)
        return confidence, "decision_function_abs"

    return pd.Series([float("nan")] * len(y_pred), index=y_pred.index), "unavailable"


# Return human-readable definition for each confidence method used in reports.
def describe_confidence_method(confidence_method: str) -> str:
    mapping = {
        "predict_proba_max": "predicted-class probability confidence = max(P(real), P(fake)).",
        "decision_function_abs": "margin confidence = |decision score| (not a probability).",
        "unavailable": "confidence unavailable for this estimator.",
    }
    return mapping.get(confidence_method, "confidence method description unavailable.")


# Build row-level prediction dataframe enriched with error metadata.
def build_prediction_dataframe(
    df_raw: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    confidence: pd.Series,
) -> pd.DataFrame:
    out = df_raw.copy()
    out["y_true"] = y_true.astype(int)
    out["y_pred"] = y_pred.astype(int)
    out["is_error"] = out["y_true"] != out["y_pred"]
    out["confidence_score"] = confidence
    out["error_type"] = "CORRECT"
    out.loc[(out["y_true"] == 0) & (out["y_pred"] == 1), "error_type"] = "FP"
    out.loc[(out["y_true"] == 1) & (out["y_pred"] == 0), "error_type"] = "FN"
    return out


# Aggregate errors by one grouping key with fp/fn counts and error rate.
def aggregate_error(df_pred: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = (
        df_pred.groupby(group_col, dropna=False)
        .agg(
            samples=("sample_id", "count"),
            errors=("is_error", "sum"),
            fp_count=("error_type", lambda s: int((s == "FP").sum())),
            fn_count=("error_type", lambda s: int((s == "FN").sum())),
        )
        .reset_index()
    )
    grouped["error_rate"] = grouped["errors"] / grouped["samples"].replace(0, 1)
    grouped = grouped.sort_values(["error_rate", "errors", "samples"], ascending=[False, False, False])
    return grouped


# Add deterministic length bins for grouped error analysis.
def add_length_bins(df_pred: pd.DataFrame) -> pd.DataFrame:
    out = df_pred.copy()
    out["text_length"] = pd.to_numeric(out["text_length"], errors="coerce").fillna(0).astype(int)
    bins = [-1, 200, 500, 1000, 2000, 4000, 10_000_000]
    labels = ["0-200", "201-500", "501-1000", "1001-2000", "2001-4000", ">4000"]
    out["length_bin"] = pd.cut(out["text_length"], bins=bins, labels=labels)
    return out


# Build concise snippet for markdown tables.
def build_snippet(text: Any, max_len: int = 220) -> str:
    normalized = str(text).replace("\n", " ").strip()
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3] + "..."


# Select top FP/FN rows sorted by confidence when available.
def select_top_errors(df_pred: pd.DataFrame, error_type: str, top_n: int) -> pd.DataFrame:
    subset = df_pred[df_pred["error_type"] == error_type].copy()
    if subset.empty:
        return subset

    has_conf = subset["confidence_score"].notna().any()
    if has_conf:
        subset = subset.sort_values("confidence_score", ascending=False)
    else:
        subset = subset.sort_values("text_length", ascending=False)

    subset = subset.head(top_n).copy()
    snippet_col = "text_clean" if "text_clean" in subset.columns else "text_raw"
    subset["snippet"] = subset[snippet_col].apply(build_snippet)
    return subset[
        [
            "sample_id",
            "source_file",
            "content_type",
            "text_length",
            "y_true",
            "y_pred",
            "confidence_score",
            "snippet",
        ]
    ]


# Save bar chart for error rate with descending sort and highlighted top category.
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
    colors = ["#d62728"] + ["#4e79a7"] * (len(labels) - 1)

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


# Convert dataframe to markdown table text.
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
    lines = [
        "| " + " | ".join(headers) + " |",
        separator,
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# Build auto insights from aggregated error tables and wrong-confidence rows.
def build_insights(
    overall_error_rate: float,
    by_source: pd.DataFrame,
    by_content_type: pd.DataFrame,
    by_length: pd.DataFrame,
    wrong_confidence: pd.Series,
) -> list[str]:
    insights: list[str] = []
    insights.append(f"Overall error rate on the analysis split is {overall_error_rate * 100:.2f}%.")

    if not by_source.empty:
        top_source = by_source.iloc[0]
        insights.append(
            f"Highest source of error: `{top_source['source_file']}` with error_rate={top_source['error_rate'] * 100:.2f}% "
            f"(errors={int(top_source['errors'])}/{int(top_source['samples'])})."
        )

    if len(by_content_type) >= 2:
        top_ct = by_content_type.iloc[0]
        second_ct = by_content_type.iloc[1]
        diff = (float(top_ct["error_rate"]) - float(second_ct["error_rate"])) * 100
        insights.append(
            f"The most difficult group: `{top_ct['content_type']}` is harder than `{second_ct['content_type']}` by {diff:.2f} percentage points."
        )
    elif len(by_content_type) == 1:
        only_ct = by_content_type.iloc[0]
        insights.append(
            f"Only 1 content_type in the split: `{only_ct['content_type']}` with error_rate={only_ct['error_rate'] * 100:.2f}%."
        )

    if not by_length.empty:
        top_len = by_length.iloc[0]
        insights.append(
            f"The most difficult length Bin: `{top_len['length_bin']}` with error_rate={top_len['error_rate'] * 100:.2f}%."
        )

    if wrong_confidence.notna().any():
        insights.append(
            f"Average confidence level on incorrect samples: {wrong_confidence.mean():.6f} "
            f"(median={wrong_confidence.median():.6f})."
        )
    else:
        insights.append("The model does not directly support confidence scores and does not rank errors according to confidence level.")

    return insights


# Render full markdown report for error analysis deliverable.
def build_error_analysis_markdown(
    run_id: str,
    split_name: str,
    confidence_method: str,
    total_samples: int,
    total_errors: int,
    overall_error_rate: float,
    by_source: pd.DataFrame,
    by_content_type: pd.DataFrame,
    by_length: pd.DataFrame,
    top_fp: pd.DataFrame,
    top_fn: pd.DataFrame,
    insights: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# Error Analysis Report (ML)")
    lines.append("")
    lines.append("## Run Info")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- split: `{split_name}`")
    lines.append(f"- confidence_method: `{confidence_method}`")
    lines.append(f"- confidence_definition: {describe_confidence_method(confidence_method)}")
    lines.append("")
    lines.append("## Overall")
    lines.append(f"- total_samples: `{total_samples}`")
    lines.append(f"- total_errors: `{total_errors}`")
    lines.append(f"- overall_error_rate: `{overall_error_rate * 100:.2f}%`")
    lines.append("")
    lines.append("## Error By Source")
    lines.append(dataframe_to_markdown(by_source, float_cols=["error_rate"]))
    lines.append("")
    lines.append("![Error By Source](figures/error_ml_by_source.png)")
    lines.append("")
    lines.append("## Error By Content Type")
    lines.append(dataframe_to_markdown(by_content_type, float_cols=["error_rate"]))
    lines.append("")
    lines.append("![Error By Content Type](figures/error_ml_by_content_type.png)")
    lines.append("")
    lines.append("## Error By Length")
    lines.append(dataframe_to_markdown(by_length, float_cols=["error_rate"]))
    lines.append("")
    lines.append("![Error By Length](figures/error_ml_by_length.png)")
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
    lines.append("- Increase the data/quality for the source group with the highest error_rate.")
    lines.append("- Compare threshold tuning or calibration if FNs are highly confident.")
    lines.append("- With the high-error length group, try adding feature_set (word_char) or tuning n-gram.")
    lines.append("")
    return "\n".join(lines)


# Execute end-to-end error analysis and generate report + figures.
def run_error_analysis(
    run_id: str | None = None,
    split_name: str = "test",
    top_n: int = 20,
    output_report: str | Path = "reports/error_analysis_ml.md",
    config_path: str | Path | None = None,
) -> tuple[str, Path]:
    repo_root = get_repo_root()
    best_config: dict[str, Any] | None = None
    requested_run_id = (run_id or "").strip()
    if requested_run_id:
        selected_run_id = requested_run_id
    else:
        best_config = load_best_config(repo_root / "experiments" / "ml" / "best_config.json")
        selected_run_id = resolve_run_id(best_config, requested_run_id)

    model_path, vectorizer_path, _ = resolve_artifact_paths(best_config, selected_run_id, repo_root)
    model = load(model_path)
    vectorizer = load(vectorizer_path)

    config = load_train_ml_config(resolve_config_path(config_path))
    split_path = resolve_split_path(config, split_name, repo_root)
    df_split = pd.read_csv(split_path)
    validate_required_columns(df_split)

    text_variant = resolve_text_variant(selected_run_id, best_config, config, repo_root)
    if not text_variant or text_variant not in df_split.columns:
        raise ErrorAnalysisError(
            f"Cannot resolve text_variant for analysis. Resolved='{text_variant}', available columns check failed."
        )

    x_text = df_split[text_variant].fillna("").astype(str)
    y_true = pd.to_numeric(df_split["label_binary"], errors="coerce").fillna(-1).astype(int)
    if not set(y_true.unique()).issubset({0, 1}):
        raise ErrorAnalysisError("label_binary contains values outside {0,1}.")

    print(
        f"[ML ERROR ANALYSIS] Predicting split={split_name} | run_id={selected_run_id} | "
        f"rows={len(df_split)} text_variant={text_variant}"
    )
    x_vec = vectorizer.transform(x_text)
    y_pred = pd.Series(model.predict(x_vec), index=df_split.index).astype(int)
    confidence_scores, confidence_method = compute_confidence_scores(model, x_vec, y_pred)

    df_pred = build_prediction_dataframe(df_split, y_true, y_pred, confidence_scores)
    df_pred = add_length_bins(df_pred)

    by_source = aggregate_error(df_pred, "source_file")
    by_content_type = aggregate_error(df_pred, "content_type")
    by_length = aggregate_error(df_pred, "length_bin")

    top_fp = select_top_errors(df_pred, "FP", top_n=top_n)
    top_fn = select_top_errors(df_pred, "FN", top_n=top_n)

    figures_dir = repo_root / "reports" / "figures"
    save_error_bar_figure(
        by_source,
        category_col="source_file",
        out_path=figures_dir / "error_ml_by_source.png",
        title="ML Error Rate by Source",
        top_k=20,
    )
    save_error_bar_figure(
        by_content_type,
        category_col="content_type",
        out_path=figures_dir / "error_ml_by_content_type.png",
        title="ML Error Rate by Content Type",
        top_k=None,
    )
    save_error_bar_figure(
        by_length,
        category_col="length_bin",
        out_path=figures_dir / "error_ml_by_length.png",
        title="ML Error Rate by Text Length Bin",
        top_k=None,
    )

    total_samples = int(len(df_pred))
    total_errors = int(df_pred["is_error"].sum())
    overall_error_rate = total_errors / total_samples if total_samples else 0.0
    wrong_conf = df_pred.loc[df_pred["is_error"], "confidence_score"]
    insights = build_insights(overall_error_rate, by_source, by_content_type, by_length, wrong_conf)

    report_md = build_error_analysis_markdown(
        run_id=selected_run_id,
        split_name=split_name,
        confidence_method=confidence_method,
        total_samples=total_samples,
        total_errors=total_errors,
        overall_error_rate=overall_error_rate,
        by_source=by_source,
        by_content_type=by_content_type,
        by_length=by_length,
        top_fp=top_fp,
        top_fn=top_fn,
        insights=insights,
    )

    report_path = Path(output_report)
    if not report_path.is_absolute():
        report_path = repo_root / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")
    return selected_run_id, report_path


# Run CLI entrypoint and print generated artifact locations.
def main() -> None:
    args = parse_args()
    print(f"[ML ERROR ANALYSIS] Starting analysis with args: {args}")
    run_id, report_path = run_error_analysis(
        run_id=args.run_id,
        split_name=args.split,
        top_n=args.top_n,
        output_report=args.output_report,
        config_path=args.config_path,
    )
    print(f"[ML ERROR ANALYSIS] PASS | run_id={run_id}")
    print(f"[ML ERROR ANALYSIS] Report: {report_path.as_posix()}")
    print(f"[ML ERROR ANALYSIS] Figures: reports/figures/error_ml_by_source.png, error_ml_by_content_type.png, error_ml_by_length.png")


# Expose strict CLI-friendly error handling.
if __name__ == "__main__":
    try:
        main()
    except ErrorAnalysisError as exc:
        raise SystemExit(f"[ML ERROR ANALYSIS ERROR] {exc}")
