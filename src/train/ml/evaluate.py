from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.train.common.dataset_loader import DatasetLoaderError, get_xy_splits
    from src.train.common.split_utils import DatasetValidationError
    from src.train.ml.vectorizers import VectorizerConfigError, resolve_config_path
except ModuleNotFoundError:
    CURRENT_DIR = Path(__file__).resolve().parent
    COMMON_DIR = CURRENT_DIR.parent / "common"
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    if str(COMMON_DIR) not in sys.path:
        sys.path.insert(0, str(COMMON_DIR))

    from dataset_loader import DatasetLoaderError, get_xy_splits  # type: ignore
    from split_utils import DatasetValidationError  # type: ignore
    from vectorizers import VectorizerConfigError, resolve_config_path  # type: ignore


class EvaluationError(Exception):
    pass


# Resolve repository root path from current file location.
def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# Parse CLI arguments for ML evaluation and report generation.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one trained ML run and generate report artifacts.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run_id from experiments/ml/runs.csv. Default: latest PASS run.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional path to train_ml.yaml. Default: configs/train_ml.yaml",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="reports/model_report_ml.md",
        help="Report output path relative to repo root or absolute path.",
    )
    return parser.parse_args()


# Load all run rows from experiments tracking CSV.
def load_runs_csv(runs_path: Path) -> list[dict[str, Any]]:
    if not runs_path.exists():
        raise EvaluationError(f"Missing runs tracking file: {runs_path}")

    with runs_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise EvaluationError("experiments/ml/runs.csv has no run records.")
    return rows


# Parse run timestamp safely for latest-run selection.
def parse_run_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return datetime.min


# Resolve target run either by explicit run_id or latest PASS row.
def resolve_target_run(run_rows: list[dict[str, Any]], requested_run_id: str | None) -> dict[str, Any]:
    if requested_run_id:
        for row in run_rows:
            if str(row.get("run_id", "")).strip() == requested_run_id:
                return row
        raise EvaluationError(f"run_id '{requested_run_id}' not found in runs.csv.")

    pass_rows = [row for row in run_rows if str(row.get("status", "")).upper() == "PASS"]
    if not pass_rows:
        raise EvaluationError("No PASS run found in runs.csv to evaluate.")
    pass_rows.sort(key=lambda row: parse_run_timestamp(str(row.get("run_timestamp", ""))), reverse=True)
    return pass_rows[0]


# Resolve model and vectorizer artifact paths from run notes or run_id fallback pattern.
def resolve_artifact_paths(target_run: dict[str, Any], repo_root: Path) -> tuple[Path, Path, Path]:
    run_id = str(target_run.get("run_id", "")).strip()
    if not run_id:
        raise EvaluationError("Target run row is missing run_id.")

    model_path = None
    vectorizer_path = None
    metadata_path = None

    notes_raw = str(target_run.get("notes", "")).strip()
    if notes_raw:
        try:
            notes_obj = json.loads(notes_raw)
            if isinstance(notes_obj, dict):
                model_path = notes_obj.get("model_path")
                vectorizer_path = notes_obj.get("vectorizer_path")
                metadata_path = notes_obj.get("metadata_path")
        except json.JSONDecodeError:
            pass

    if not model_path:
        model_path = str((repo_root / "models" / "ml" / f"{run_id}__model.joblib").as_posix())
    if not vectorizer_path:
        vectorizer_path = str((repo_root / "models" / "ml" / f"{run_id}__vectorizer.joblib").as_posix())
    if not metadata_path:
        metadata_path = str((repo_root / "models" / "ml" / f"{run_id}__metadata.json").as_posix())

    model_file = Path(model_path)
    vectorizer_file = Path(vectorizer_path)
    metadata_file = Path(metadata_path)

    if not model_file.exists():
        raise EvaluationError(f"Model artifact not found: {model_file}")
    if not vectorizer_file.exists():
        raise EvaluationError(f"Vectorizer artifact not found: {vectorizer_file}")
    if not metadata_file.exists():
        raise EvaluationError(f"Metadata artifact not found: {metadata_file}")

    return model_file, vectorizer_file, metadata_file


# Compute aggregate evaluation metrics for one split.
def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
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


# Generate confusion matrix image with count annotations per cell.
def save_confusion_matrix_figure(cm: Any, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    class_labels = ["real(0)", "fake(1)"]
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=class_labels,
        yticklabels=class_labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            color = "white" if value > threshold else "black"
            ax.text(j, i, f"{value}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# Convert classification report dictionary to markdown table lines.
def classification_report_to_markdown(report_dict: dict[str, Any], split_name: str) -> list[str]:
    keys = ["0", "1", "macro avg", "weighted avg"]
    lines = [
        f"### Classification Report ({split_name})",
        "| Label | Precision | Recall | F1-score | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in keys:
        if key not in report_dict:
            continue
        row = report_dict[key]
        label_name = "real(0)" if key == "0" else "fake(1)" if key == "1" else key
        lines.append(
            f"| {label_name} | {row.get('precision', 0):.6f} | {row.get('recall', 0):.6f} | "
            f"{row.get('f1-score', 0):.6f} | {int(row.get('support', 0))} |"
        )
    lines.append("")
    return lines


# Build markdown report body with run metadata, metrics, and figure references.
def build_markdown_report(
    target_run: dict[str, Any],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    val_report: dict[str, Any],
    test_report: dict[str, Any],
    val_fig_path: Path,
    test_fig_path: Path,
) -> str:
    run_id = str(target_run.get("run_id", ""))
    run_timestamp = str(target_run.get("run_timestamp", ""))
    model_name = str(target_run.get("model_name", ""))
    feature_set = str(target_run.get("feature_set", ""))
    text_variant = str(target_run.get("text_variant", ""))
    params_json = str(target_run.get("params_json", ""))

    lines: list[str] = []
    lines.append("# ML Model Evaluation Report")
    lines.append("")
    lines.append("## Run Information")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- run_timestamp: `{run_timestamp}`")
    lines.append(f"- model_name: `{model_name}`")
    lines.append(f"- feature_set: `{feature_set}`")
    lines.append(f"- text_variant: `{text_variant}`")
    lines.append(f"- params_json: `{params_json}`")
    lines.append("")
    lines.append("## Aggregate Metrics")
    lines.append("| Split | F1 Macro | Precision Macro | Recall Macro | Accuracy | F1 Fake |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| val | {val_metrics['f1_macro']:.6f} | {val_metrics['precision_macro']:.6f} | "
        f"{val_metrics['recall_macro']:.6f} | {val_metrics['accuracy']:.6f} | {val_metrics['f1_fake']:.6f} |"
    )
    lines.append(
        f"| test | {test_metrics['f1_macro']:.6f} | {test_metrics['precision_macro']:.6f} | "
        f"{test_metrics['recall_macro']:.6f} | {test_metrics['accuracy']:.6f} | {test_metrics['f1_fake']:.6f} |"
    )
    lines.append("")
    lines.append("## Confusion Matrix")
    lines.append(f"### Validation Split")
    lines.append(f"![Confusion Matrix Val]({val_fig_path.as_posix()})")
    lines.append("")
    lines.append(f"### Test Split")
    lines.append(f"![Confusion Matrix Test]({test_fig_path.as_posix()})")
    lines.append("")
    lines.extend(classification_report_to_markdown(val_report, "val"))
    lines.extend(classification_report_to_markdown(test_report, "test"))
    lines.append("## Notes")
    lines.append("- Class mapping: `0=real`, `1=fake`.")
    lines.append("- Metrics are recomputed from stored model/vectorizer artifacts on current val/test splits.")
    lines.append("")
    return "\n".join(lines)


# Run end-to-end evaluation and generate report markdown plus confusion matrix figures.
def run_evaluation(
    run_id: str | None = None,
    config_path: str | Path | None = None,
    output_report: str | Path = "reports/model_report_ml.md",
) -> tuple[dict[str, Any], Path, Path, Path]:
    repo_root = get_repo_root()
    resolved_config_path = resolve_config_path(config_path)
    runs_path = repo_root / "experiments" / "ml" / "runs.csv"

    run_rows = load_runs_csv(runs_path)
    target_run = resolve_target_run(run_rows, run_id)
    target_run_id = str(target_run.get("run_id", "")).strip()
    text_variant = str(target_run.get("text_variant", "")).strip()
    if not text_variant:
        raise EvaluationError("Target run row missing text_variant.")

    model_file, vectorizer_file, _ = resolve_artifact_paths(target_run, repo_root)
    model = load(model_file)
    vectorizer = load(vectorizer_file)

    xy_splits = get_xy_splits(
        text_variant=text_variant,
        config_path=resolved_config_path,
        strict_overlap=True,
        print_summary=False,
    )
    x_val, y_val = xy_splits["val"]
    x_test, y_test = xy_splits["test"]

    x_val_vec = vectorizer.transform(x_val)
    x_test_vec = vectorizer.transform(x_test)
    y_val_pred = model.predict(x_val_vec)
    y_test_pred = model.predict(x_test_vec)

    val_metrics = compute_metrics(y_val, y_val_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)
    val_report = classification_report(y_val, y_val_pred, output_dict=True, zero_division=0)
    test_report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)

    val_cm = confusion_matrix(y_val, y_val_pred, labels=[0, 1])
    test_cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])

    figures_dir = repo_root / "reports" / "figures"
    val_cm_file = figures_dir / f"confusion_matrix_ml_val_{target_run_id}.png"
    test_cm_file = figures_dir / f"confusion_matrix_ml_test_{target_run_id}.png"
    save_confusion_matrix_figure(val_cm, f"Confusion Matrix (VAL) - {target_run_id}", val_cm_file)
    save_confusion_matrix_figure(test_cm, f"Confusion Matrix (TEST) - {target_run_id}", test_cm_file)

    report_path = Path(output_report)
    if not report_path.is_absolute():
        report_path = repo_root / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    relative_val_figure = val_cm_file.relative_to(report_path.parent)
    relative_test_figure = test_cm_file.relative_to(report_path.parent)

    report_md = build_markdown_report(
        target_run=target_run,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        val_report=val_report,
        test_report=test_report,
        val_fig_path=relative_val_figure,
        test_fig_path=relative_test_figure,
    )
    report_path.write_text(report_md, encoding="utf-8")

    summary = {
        "run_id": target_run_id,
        "run_timestamp": target_run.get("run_timestamp", ""),
        "model_name": target_run.get("model_name", ""),
        "feature_set": target_run.get("feature_set", ""),
        "text_variant": text_variant,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    return summary, report_path, val_cm_file, test_cm_file


# Execute CLI entrypoint and print concise artifact locations.
def main() -> None:
    args = parse_args()
    print("[ML EVAL] Starting single-run evaluation...")
    summary, report_path, val_cm_path, test_cm_path = run_evaluation(
        run_id=args.run_id,
        config_path=args.config_path,
        output_report=args.output_report,
    )
    print(
        "[ML EVAL] PASS | "
        f"run_id={summary['run_id']} "
        f"model={summary['model_name']} "
        f"feature_set={summary['feature_set']} "
        f"text_variant={summary['text_variant']}"
    )
    print(
        "[ML EVAL] Metrics | "
        f"val_f1_macro={summary['val_metrics']['f1_macro']:.6f} "
        f"test_f1_macro={summary['test_metrics']['f1_macro']:.6f}"
    )
    print(f"[ML EVAL] Report: {report_path.as_posix()}")
    print(f"[ML EVAL] Figure (val): {val_cm_path.as_posix()}")
    print(f"[ML EVAL] Figure (test): {test_cm_path.as_posix()}")


# Expose evaluation entrypoint with strict error mapping.
if __name__ == "__main__":
    try:
        main()
    except (
        EvaluationError,
        DatasetLoaderError,
        DatasetValidationError,
        VectorizerConfigError,
    ) as exc:
        raise SystemExit(f"[ML EVAL ERROR] {exc}")
