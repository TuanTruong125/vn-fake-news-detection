from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.train.ml.vectorizers import load_train_ml_config, resolve_config_path
except ModuleNotFoundError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from vectorizers import load_train_ml_config, resolve_config_path  # type: ignore


class CalibrationError(Exception):
    pass


OBJECTIVES = ["f1_macro", "f1_fake"]
TIE_BREAKERS = ["f1_fake", "precision_fake"]


# Resolve repository root path from current file location.
def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# Return local timestamp string for logging and metadata fields.
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# Parse CLI args for threshold calibration workflow.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate ML decision threshold and update run metadata.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run_id. If provided, calibration uses this run directly.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="f1_macro",
        choices=OBJECTIVES,
        help="Primary objective to optimize on calibration split.",
    )
    parser.add_argument(
        "--tie-breaker",
        type=str,
        default="f1_fake",
        choices=TIE_BREAKERS,
        help="Tie-break metric when primary objective is equal.",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=0.05,
        help="Minimum threshold for sweep grid (inclusive).",
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=0.95,
        help="Maximum threshold for sweep grid (inclusive).",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
        help="Threshold sweep step size.",
    )
    parser.add_argument(
        "--split-calibration",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Split used to optimize threshold. Default: val.",
    )
    parser.add_argument(
        "--split-eval",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Split used to verify post-calibration performance. Default: test.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="reports/threshold_calibration_ml.md",
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--output-figure",
        type=str,
        default=None,
        help="Optional figure path. Default: reports/figures/threshold_curve_ml_<run_id>.png",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional path to train_ml.yaml. Default: configs/train_ml.yaml",
    )
    return parser.parse_args()


# Validate threshold sweep bounds.
def validate_threshold_args(min_threshold: float, max_threshold: float, step: float) -> None:
    if step <= 0:
        raise CalibrationError("--step must be > 0.")
    if min_threshold < 0 or max_threshold > 1:
        raise CalibrationError("--min-threshold and --max-threshold must be within [0,1].")
    if min_threshold >= max_threshold:
        raise CalibrationError("--min-threshold must be < --max-threshold.")


# Load best_config JSON artifact for fallback run resolution.
def load_best_config(best_config_path: Path) -> dict[str, Any]:
    if not best_config_path.exists():
        raise CalibrationError(f"Missing best_config.json: {best_config_path}")
    with best_config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise CalibrationError("best_config.json must be a JSON object.")
    return payload


# Resolve run_id from CLI input or best_config fallback.
def resolve_run_id(run_id: str | None, repo_root: Path) -> str:
    requested_run_id = (run_id or "").strip()
    if requested_run_id:
        return requested_run_id

    best_config = load_best_config(repo_root / "experiments" / "ml" / "best_config.json")
    best_run = best_config.get("best_run")
    if not isinstance(best_run, dict):
        raise CalibrationError("best_config.json missing best_run object.")

    fallback_run_id = str(best_run.get("run_id", "")).strip()
    if not fallback_run_id:
        raise CalibrationError("best_config.json best_run.run_id is empty.")
    return fallback_run_id


# Resolve mandatory model/vectorizer/metadata artifacts by run_id naming convention.
def resolve_artifact_paths(run_id: str, repo_root: Path) -> tuple[Path, Path, Path]:
    model_path = repo_root / "models" / "ml" / f"{run_id}__model.joblib"
    vectorizer_path = repo_root / "models" / "ml" / f"{run_id}__vectorizer.joblib"
    metadata_path = repo_root / "models" / "ml" / f"{run_id}__metadata.json"

    if not model_path.exists():
        raise CalibrationError(f"Model artifact not found: {model_path}")
    if not vectorizer_path.exists():
        raise CalibrationError(f"Vectorizer artifact not found: {vectorizer_path}")
    if not metadata_path.exists():
        raise CalibrationError(f"Metadata artifact not found: {metadata_path}")

    return model_path, vectorizer_path, metadata_path


# Load one metadata JSON file and validate object shape.
def load_metadata(metadata_path: Path) -> dict[str, Any]:
    with metadata_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise CalibrationError(f"Metadata must be JSON object: {metadata_path}")
    return payload


# Resolve split CSV path from train_ml config data section.
def resolve_split_path(config: dict[str, Any], split_name: str, repo_root: Path) -> Path:
    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        raise CalibrationError("train_ml.yaml missing data section.")

    key = f"{split_name}_path"
    split_path = data_cfg.get(key)
    if not isinstance(split_path, str) or not split_path.strip():
        raise CalibrationError(f"train_ml.yaml data.{key} must be non-empty string.")

    final_path = repo_root / split_path
    if not final_path.exists():
        raise CalibrationError(f"Split file not found: {final_path}")
    return final_path


# Resolve text variant with metadata-first strategy.
def resolve_text_variant(
    run_id: str,
    metadata: dict[str, Any],
    config: dict[str, Any],
    repo_root: Path,
) -> str:
    selection = metadata.get("selection")
    if isinstance(selection, dict):
        candidate = str(selection.get("text_variant", "")).strip()
        if candidate:
            return candidate

    runs_path = repo_root / "experiments" / "ml" / "runs.csv"
    if runs_path.exists():
        with runs_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("run_id", "")).strip() == run_id:
                    candidate = str(row.get("text_variant", "")).strip()
                    if candidate:
                        return candidate

    candidate = str(config.get("run", {}).get("default", {}).get("text_variant", "")).strip()
    if candidate:
        return candidate
    raise CalibrationError("Cannot resolve text_variant from metadata, runs.csv, or config defaults.")


# Validate split dataframe contains required columns and labels are binary.
def validate_split_frame(df: pd.DataFrame, split_name: str, text_variant: str) -> None:
    required = ["label_binary", text_variant]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise CalibrationError(f"{split_name}: missing required columns: {missing}")

    y_values = pd.to_numeric(df["label_binary"], errors="coerce").dropna().astype(int)
    if y_values.empty:
        raise CalibrationError(f"{split_name}: label_binary contains no valid values.")
    if not set(y_values.unique()).issubset({0, 1}):
        raise CalibrationError(f"{split_name}: label_binary contains values outside {{0,1}}.")


# Create sweep thresholds inclusive of both bounds.
def build_threshold_grid(min_threshold: float, max_threshold: float, step: float) -> np.ndarray:
    raw_values = np.arange(min_threshold, max_threshold + (step / 2.0), step)
    rounded = np.round(raw_values, 6)
    clipped = np.clip(rounded, 0.0, 1.0)
    unique_values = np.unique(clipped)
    if len(unique_values) == 0:
        raise CalibrationError("Threshold grid is empty after processing bounds.")
    return unique_values


# Return model scores for class-1 decision with deterministic fallback rules.
def compute_scores(model: Any, x_vec: Any) -> tuple[np.ndarray, str]:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_vec)
        if probs.ndim != 2 or probs.shape[1] < 2:
            raise CalibrationError("predict_proba output shape is invalid for binary classification.")

        classes = list(getattr(model, "classes_", []))
        if 1 not in classes:
            raise CalibrationError(f"Model classes do not contain class 1: {classes}")
        class_one_index = classes.index(1)
        scores = probs[:, class_one_index]
        return np.asarray(scores, dtype=float), "predict_proba"

    if hasattr(model, "decision_function"):
        decision = model.decision_function(x_vec)
        decision_arr = np.asarray(decision)

        if decision_arr.ndim == 1:
            raw_scores = decision_arr
        elif decision_arr.ndim == 2:
            classes = list(getattr(model, "classes_", []))
            if 1 in classes and len(classes) == decision_arr.shape[1]:
                raw_scores = decision_arr[:, classes.index(1)]
            else:
                raise CalibrationError("decision_function returns 2D output but class-1 index cannot be resolved.")
        else:
            raise CalibrationError("decision_function output rank is unsupported.")

        safe_scores = np.clip(raw_scores.astype(float), -60.0, 60.0)
        calibrated = 1.0 / (1.0 + np.exp(-safe_scores))
        return calibrated, "decision_function_sigmoid"

    raise CalibrationError("Model does not support predict_proba or decision_function. Cannot calibrate.")


# Compute evaluation metrics for one threshold.
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
        "precision_fake": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


# Sweep thresholds and collect metrics table.
def evaluate_threshold_grid(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        y_pred = (scores >= float(threshold)).astype(int)
        metrics = compute_metrics(y_true, y_pred)
        rows.append({"threshold": float(threshold), **metrics})

    return pd.DataFrame(rows)


# Pick best threshold by objective and tie-break metrics.
def select_best_threshold(
    sweep_df: pd.DataFrame,
    objective: str,
    tie_breaker: str,
) -> dict[str, Any]:
    if sweep_df.empty:
        raise CalibrationError("Sweep metrics dataframe is empty.")

    for metric_name in [objective, tie_breaker]:
        if metric_name not in sweep_df.columns:
            raise CalibrationError(f"Metric '{metric_name}' missing in sweep dataframe.")

    ranked = sweep_df.copy()
    ranked["tie_distance_05"] = (ranked["threshold"] - 0.5).abs()
    ranked = ranked.sort_values(
        [objective, tie_breaker, "tie_distance_05", "threshold"],
        ascending=[False, False, True, True],
    )
    best_row = ranked.iloc[0].to_dict()
    best_row.pop("tie_distance_05", None)
    return best_row


# Evaluate one split at one threshold.
def evaluate_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (scores >= float(threshold)).astype(int)
    return compute_metrics(y_true, y_pred)


# Resolve baseline threshold for before/after comparison.
def resolve_before_threshold(metadata: dict[str, Any]) -> float:
    calibration = metadata.get("threshold_calibration")
    if isinstance(calibration, dict):
        candidate = calibration.get("recommended_threshold")
        if isinstance(candidate, (int, float)) and not math.isnan(float(candidate)):
            return float(candidate)
    return 0.5


# Compute compact score distribution summary for debugging.
def build_score_distribution(scores: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
    }


# Convert one dict of metrics to markdown table text.
def metrics_pair_to_markdown(
    before_threshold: float,
    before_metrics: dict[str, float],
    after_threshold: float,
    after_metrics: dict[str, float],
) -> str:
    lines = []
    lines.append("| setup | threshold | f1_macro | f1_fake | precision_fake | recall_fake | accuracy |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    lines.append(
        "| before | "
        f"{before_threshold:.6f} | "
        f"{before_metrics['f1_macro']:.6f} | {before_metrics['f1_fake']:.6f} | "
        f"{before_metrics['precision_fake']:.6f} | {before_metrics['recall_fake']:.6f} | {before_metrics['accuracy']:.6f} |"
    )
    lines.append(
        "| after | "
        f"{after_threshold:.6f} | "
        f"{after_metrics['f1_macro']:.6f} | {after_metrics['f1_fake']:.6f} | "
        f"{after_metrics['precision_fake']:.6f} | {after_metrics['recall_fake']:.6f} | {after_metrics['accuracy']:.6f} |"
    )
    return "\n".join(lines)


# Convert top threshold rows to markdown table.
def top_thresholds_to_markdown(sweep_df: pd.DataFrame, objective: str, tie_breaker: str, top_n: int = 10) -> str:
    ranked = sweep_df.copy()
    ranked["distance_05"] = (ranked["threshold"] - 0.5).abs()
    ranked = ranked.sort_values(
        [objective, tie_breaker, "distance_05", "threshold"],
        ascending=[False, False, True, True],
    ).head(top_n)

    headers = ["threshold", "f1_macro", "f1_fake", "precision_fake", "recall_fake", "accuracy"]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in ranked.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{float(row['threshold']):.6f}",
                    f"{float(row['f1_macro']):.6f}",
                    f"{float(row['f1_fake']):.6f}",
                    f"{float(row['precision_fake']):.6f}",
                    f"{float(row['recall_fake']):.6f}",
                    f"{float(row['accuracy']):.6f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


# Save threshold sweep curve figure for report diagnostics.
def save_threshold_curve(
    sweep_df: pd.DataFrame,
    figure_path: Path,
    recommended_threshold: float,
    objective: str,
    tie_breaker: str,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    chart_df = sweep_df.sort_values("threshold")
    x = chart_df["threshold"].astype(float)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(x, chart_df["f1_macro"], label="f1_macro", linewidth=2.0)
    ax.plot(x, chart_df["f1_fake"], label="f1_fake", linewidth=2.0)
    ax.plot(x, chart_df["precision_fake"], label="precision_fake", linewidth=1.5, linestyle="--")
    ax.plot(x, chart_df["recall_fake"], label="recall_fake", linewidth=1.5, linestyle="--")

    ax.axvline(recommended_threshold, color="#d62728", linestyle=":", linewidth=2.0)
    ax.text(
        recommended_threshold,
        ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0,
        f"best={recommended_threshold:.3f}",
        color="#d62728",
        ha="left",
        va="top",
        fontsize=9,
    )

    ax.set_title(f"Threshold Sweep ({objective}, tie-break={tie_breaker})")
    ax.set_xlabel("threshold")
    ax.set_ylabel("metric value")
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# Build markdown report for calibration output.
def build_report_markdown(
    run_id: str,
    text_variant: str,
    score_method: str,
    objective: str,
    tie_breaker: str,
    min_threshold: float,
    max_threshold: float,
    step: float,
    split_calibration: str,
    split_eval: str,
    before_threshold: float,
    recommended_threshold: float,
    val_before: dict[str, float],
    val_after: dict[str, float],
    test_before: dict[str, float],
    test_after: dict[str, float],
    sweep_df: pd.DataFrame,
    figure_path: Path,
    report_path: Path,
) -> str:
    figure_rel_path = Path(".")
    try:
        figure_rel_path = Path(figure_path.relative_to(report_path.parent))
    except Exception:
        figure_rel_path = figure_path

    lines: list[str] = []
    lines.append("# ML Threshold Calibration Report")
    lines.append("")
    lines.append("## Run Info")
    lines.append(f"- run_id: {run_id}")
    lines.append(f"- text_variant: {text_variant}")
    lines.append(f"- score_method: {score_method}")
    lines.append(f"- objective: {objective}")
    lines.append(f"- tie_breaker: {tie_breaker}")
    lines.append(f"- sweep: min={min_threshold:.6f}, max={max_threshold:.6f}, step={step:.6f}")
    lines.append(f"- calibration_split: {split_calibration}")
    lines.append(f"- evaluation_split: {split_eval}")
    lines.append("")
    lines.append("## Threshold")
    lines.append(f"- default_threshold: 0.500000")
    lines.append(f"- before_threshold: {before_threshold:.6f}")
    lines.append(f"- recommended_threshold: {recommended_threshold:.6f}")
    lines.append("")
    lines.append("## Calibration Split Metrics (Before vs After)")
    lines.append(metrics_pair_to_markdown(before_threshold, val_before, recommended_threshold, val_after))
    lines.append("")
    lines.append("## Evaluation Split Metrics (Before vs After)")
    lines.append(metrics_pair_to_markdown(before_threshold, test_before, recommended_threshold, test_after))
    lines.append("")
    lines.append("## Top Threshold Candidates")
    lines.append(top_thresholds_to_markdown(sweep_df, objective=objective, tie_breaker=tie_breaker, top_n=10))
    lines.append("")
    lines.append("## Threshold Curve")
    lines.append(f"![threshold_curve]({figure_rel_path.as_posix()})")
    lines.append("")
    return "\n".join(lines)


# Convert metrics dict to metadata-safe payload with rounded values.
def to_metadata_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "f1_macro": float(metrics["f1_macro"]),
        "f1_fake": float(metrics["f1_fake"]),
        "precision_fake": float(metrics["precision_fake"]),
        "recall_fake": float(metrics["recall_fake"]),
        "accuracy": float(metrics["accuracy"]),
    }


# Persist calibration block into run metadata JSON.
def write_calibration_metadata(
    metadata_path: Path,
    metadata: dict[str, Any],
    split_calibration: str,
    score_method: str,
    before_threshold: float,
    recommended_threshold: float,
    objective: str,
    tie_breaker: str,
    val_before: dict[str, float],
    val_after: dict[str, float],
    test_before: dict[str, float],
    test_after: dict[str, float],
    score_distribution: dict[str, float],
    min_threshold: float,
    max_threshold: float,
    step: float,
) -> None:
    metadata["threshold_calibration"] = {
        "enabled": True,
        "calibrated_on_split": split_calibration,
        "score_method": score_method,
        "default_threshold": 0.5,
        "before_threshold": float(before_threshold),
        "recommended_threshold": float(recommended_threshold),
        "objective": objective,
        "tie_breaker": tie_breaker,
        "val_before": to_metadata_metrics(val_before),
        "val_after": to_metadata_metrics(val_after),
        "test_before": to_metadata_metrics(test_before),
        "test_after": to_metadata_metrics(test_after),
        "score_distribution": {
            "min": float(score_distribution["min"]),
            "max": float(score_distribution["max"]),
            "mean": float(score_distribution["mean"]),
        },
        "sweep": {
            "min_threshold": float(min_threshold),
            "max_threshold": float(max_threshold),
            "step": float(step),
        },
        "calibrated_at": now_iso(),
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


# Execute end-to-end threshold calibration and produce all outputs.
def run_calibration(args: argparse.Namespace) -> tuple[str, float, Path, Path]:
    validate_threshold_args(args.min_threshold, args.max_threshold, args.step)

    print("[ML CALIBRATE] Starting threshold calibration...")

    repo_root = get_repo_root()
    run_id = resolve_run_id(args.run_id, repo_root)

    model_path, vectorizer_path, metadata_path = resolve_artifact_paths(run_id, repo_root)
    model = load(model_path)
    vectorizer = load(vectorizer_path)
    metadata = load_metadata(metadata_path)

    config = load_train_ml_config(resolve_config_path(args.config_path))
    text_variant = resolve_text_variant(run_id, metadata, config, repo_root)

    calibration_split_path = resolve_split_path(config, args.split_calibration, repo_root)
    eval_split_path = resolve_split_path(config, args.split_eval, repo_root)

    df_calibration = pd.read_csv(calibration_split_path)
    df_eval = pd.read_csv(eval_split_path)
    validate_split_frame(df_calibration, args.split_calibration, text_variant)
    validate_split_frame(df_eval, args.split_eval, text_variant)

    x_calibration = df_calibration[text_variant].fillna("").astype(str)
    y_calibration = pd.to_numeric(df_calibration["label_binary"], errors="coerce").fillna(-1).astype(int).to_numpy()

    x_eval = df_eval[text_variant].fillna("").astype(str)
    y_eval = pd.to_numeric(df_eval["label_binary"], errors="coerce").fillna(-1).astype(int).to_numpy()

    x_calibration_vec = vectorizer.transform(x_calibration)
    x_eval_vec = vectorizer.transform(x_eval)

    calibration_scores, score_method = compute_scores(model, x_calibration_vec)
    eval_scores, eval_score_method = compute_scores(model, x_eval_vec)
    if eval_score_method != score_method:
        raise CalibrationError(
            f"Score method mismatch between splits: calibration={score_method}, eval={eval_score_method}"
        )

    thresholds = build_threshold_grid(args.min_threshold, args.max_threshold, args.step)
    sweep_df = evaluate_threshold_grid(y_calibration, calibration_scores, thresholds)
    best_row = select_best_threshold(sweep_df, objective=args.objective, tie_breaker=args.tie_breaker)
    recommended_threshold = float(best_row["threshold"])

    before_threshold = resolve_before_threshold(metadata)

    val_before = evaluate_at_threshold(y_calibration, calibration_scores, before_threshold)
    val_after = evaluate_at_threshold(y_calibration, calibration_scores, recommended_threshold)
    test_before = evaluate_at_threshold(y_eval, eval_scores, before_threshold)
    test_after = evaluate_at_threshold(y_eval, eval_scores, recommended_threshold)

    score_distribution = build_score_distribution(calibration_scores)

    report_path = Path(args.output_report)
    if not report_path.is_absolute():
        report_path = repo_root / report_path

    if args.output_figure:
        figure_path = Path(args.output_figure)
        if not figure_path.is_absolute():
            figure_path = repo_root / figure_path
    else:
        figure_path = repo_root / "reports" / "figures" / f"threshold_curve_ml_{run_id}.png"

    save_threshold_curve(
        sweep_df=sweep_df,
        figure_path=figure_path,
        recommended_threshold=recommended_threshold,
        objective=args.objective,
        tie_breaker=args.tie_breaker,
    )

    write_calibration_metadata(
        metadata_path=metadata_path,
        metadata=metadata,
        split_calibration=args.split_calibration,
        score_method=score_method,
        before_threshold=before_threshold,
        recommended_threshold=recommended_threshold,
        objective=args.objective,
        tie_breaker=args.tie_breaker,
        val_before=val_before,
        val_after=val_after,
        test_before=test_before,
        test_after=test_after,
        score_distribution=score_distribution,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        step=args.step,
    )

    report_md = build_report_markdown(
        run_id=run_id,
        text_variant=text_variant,
        score_method=score_method,
        objective=args.objective,
        tie_breaker=args.tie_breaker,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        step=args.step,
        split_calibration=args.split_calibration,
        split_eval=args.split_eval,
        before_threshold=before_threshold,
        recommended_threshold=recommended_threshold,
        val_before=val_before,
        val_after=val_after,
        test_before=test_before,
        test_after=test_after,
        sweep_df=sweep_df,
        figure_path=figure_path,
        report_path=report_path,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")

    print("[ML CALIBRATE] PASS")
    print(f"[ML CALIBRATE] run_id={run_id}")
    print(f"[ML CALIBRATE] score_method={score_method}")
    print(
        "[ML CALIBRATE] thresholds "
        f"default=0.500000 before={before_threshold:.6f} recommended={recommended_threshold:.6f}"
    )
    print(
        "[ML CALIBRATE] val "
        f"before(f1_macro={val_before['f1_macro']:.6f}, f1_fake={val_before['f1_fake']:.6f}, "
        f"precision_fake={val_before['precision_fake']:.6f}, recall_fake={val_before['recall_fake']:.6f}, "
        f"accuracy={val_before['accuracy']:.6f}) "
        f"after(f1_macro={val_after['f1_macro']:.6f}, f1_fake={val_after['f1_fake']:.6f}, "
        f"precision_fake={val_after['precision_fake']:.6f}, recall_fake={val_after['recall_fake']:.6f}, "
        f"accuracy={val_after['accuracy']:.6f})"
    )
    print(
        "[ML CALIBRATE] test "
        f"before(f1_macro={test_before['f1_macro']:.6f}, f1_fake={test_before['f1_fake']:.6f}, "
        f"precision_fake={test_before['precision_fake']:.6f}, recall_fake={test_before['recall_fake']:.6f}, "
        f"accuracy={test_before['accuracy']:.6f}) "
        f"after(f1_macro={test_after['f1_macro']:.6f}, f1_fake={test_after['f1_fake']:.6f}, "
        f"precision_fake={test_after['precision_fake']:.6f}, recall_fake={test_after['recall_fake']:.6f}, "
        f"accuracy={test_after['accuracy']:.6f})"
    )
    print(f"[ML CALIBRATE] metadata={metadata_path.as_posix()}")
    print(f"[ML CALIBRATE] report={report_path.as_posix()}")
    print(f"[ML CALIBRATE] figure={figure_path.as_posix()}")

    return run_id, recommended_threshold, report_path, figure_path


# CLI entrypoint for threshold calibration.
def main() -> None:
    args = parse_args()
    run_calibration(args)


if __name__ == "__main__":
    try:
        main()
    except CalibrationError as exc:
        raise SystemExit(f"[ML CALIBRATE ERROR] {exc}")
