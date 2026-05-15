from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src.train.dl.train_phobert import (
        DlTrainError,
        TextDataset,
        evaluate_model,
        get_repo_root,
        load_yaml,
        tokenize_split,
    )
except ModuleNotFoundError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from train_phobert import DlTrainError, TextDataset, evaluate_model, get_repo_root, load_yaml, tokenize_split  # type: ignore

try:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ModuleNotFoundError as exc:
    raise SystemExit(
        "[DL EVAL ERROR] Missing DL dependencies. Please install requirements including torch and transformers."
    ) from exc


# Error class for evaluation stage issues.
class EvaluationError(Exception):
    pass


# Parse CLI args for DL evaluation and report generation.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one trained DL run and generate report artifacts.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run_id from experiments/dl/runs.csv. Default: latest PASS run.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/train_dl.yaml",
        help="Path to train_dl.yaml.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="reports/model_report_dl.md",
        help="Report output path relative to repo root or absolute path.",
    )
    parser.add_argument(
        "--output-pred-csv",
        type=str,
        default=None,
        help="Optional path to save concatenated val/test predictions CSV.",
    )
    return parser.parse_args()


# Load all run rows from DL runs tracking CSV.
def load_runs_csv(runs_path: Path) -> list[dict[str, Any]]:
    if not runs_path.exists():
        raise EvaluationError(f"Missing runs tracking file: {runs_path}")

    with runs_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise EvaluationError("experiments/dl/runs.csv has no run records.")
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


# Resolve DL run directory from run notes or fallback run_id path.
def resolve_run_dir(target_run: dict[str, Any], repo_root: Path) -> Path:
    run_id = str(target_run.get("run_id", "")).strip()
    if not run_id:
        raise EvaluationError("Target run row is missing run_id.")

    run_dir_value = None
    notes_raw = str(target_run.get("notes", "")).strip()
    if notes_raw:
        try:
            notes_obj = json.loads(notes_raw)
            if isinstance(notes_obj, dict):
                run_dir_value = notes_obj.get("run_dir")
        except json.JSONDecodeError:
            pass

    if run_dir_value:
        run_dir = Path(str(run_dir_value))
    else:
        run_dir = repo_root / "models" / "dl" / run_id

    if not run_dir.exists():
        raise EvaluationError(f"Run directory not found: {run_dir}")
    return run_dir


# Validate and resolve compute device from config.
def resolve_device(runtime_cfg: dict[str, Any]) -> torch.device:
    device_name = str(runtime_cfg.get("device", "auto")).strip().lower()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise EvaluationError("runtime.device=cuda but CUDA not available")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    raise EvaluationError("runtime.device must be one of: auto, cpu, cuda")


# Load split file according to train_dl.yaml config.
def load_split_frame(config: dict[str, Any], split_name: str, repo_root: Path) -> pd.DataFrame:
    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        raise EvaluationError("train_dl.yaml missing data section")

    if split_name == "train":
        split_path = repo_root / str(data_cfg.get("train_path", ""))
    elif split_name == "val":
        split_path = repo_root / str(data_cfg.get("val_path", ""))
    else:
        split_path = repo_root / str(data_cfg.get("test_path", ""))

    if not split_path.exists():
        raise EvaluationError(f"Split file not found: {split_path}")

    frame = pd.read_csv(split_path)
    text_col = str(data_cfg.get("text_column", "")).strip()
    label_col = str(data_cfg.get("label_column", "")).strip()
    id_col = str(data_cfg.get("id_column", "")).strip()

    required = [text_col, label_col, id_col]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise EvaluationError(f"{split_name}: missing required columns: {missing}")

    return frame


# Evaluate one split and return metric payload and prediction rows.
def evaluate_split(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    split_name: str,
    repo_root: Path,
    max_length: int,
    eval_batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    frame = load_split_frame(config, split_name, repo_root)
    data_cfg = config["data"]
    text_col = str(data_cfg.get("text_column"))
    label_col = str(data_cfg.get("label_column"))
    id_col = str(data_cfg.get("id_column"))

    texts = frame[text_col].fillna("").astype(str).tolist()
    labels = pd.to_numeric(frame[label_col], errors="coerce").fillna(-1).astype(int).to_numpy()
    sample_ids = frame[id_col].fillna("").astype(str).tolist()

    encodings = tokenize_split(tokenizer, texts, max_length=max_length)
    dataset = TextDataset(encodings, labels)
    dataloader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
    metrics, pred_rows = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        sample_ids=sample_ids,
        split_name=split_name,
    )
    return metrics, pred_rows


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


# Load best ML metrics payload from experiments/ml/best_config.json if available.
def load_ml_best_metrics(repo_root: Path) -> tuple[dict[str, Any] | None, Path]:
    best_config_path = repo_root / "experiments" / "ml" / "best_config.json"
    if not best_config_path.exists():
        return None, best_config_path

    with best_config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None, best_config_path

    best_run = payload.get("best_run")
    if not isinstance(best_run, dict):
        return None, best_config_path

    metrics = best_run.get("metrics")
    if not isinstance(metrics, dict):
        return None, best_config_path
    return metrics, best_config_path


# Build markdown report body with conditional DL-vs-ML comparison.
def build_markdown_report(
    target_run: dict[str, Any],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    val_report: dict[str, Any],
    test_report: dict[str, Any],
    val_fig_path: Path,
    test_fig_path: Path,
    ml_best_metrics: dict[str, Any] | None,
) -> str:
    run_id = str(target_run.get("run_id", ""))
    run_timestamp = str(target_run.get("run_timestamp", ""))
    model_name = str(target_run.get("model_name", ""))
    feature_set = str(target_run.get("feature_set", ""))
    text_variant = str(target_run.get("text_variant", ""))
    params_json = str(target_run.get("params_json", ""))

    lines: list[str] = []
    lines.append("# DL Model Evaluation Report")
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

    if ml_best_metrics is not None:
        ml_val = ml_best_metrics.get("val") if isinstance(ml_best_metrics.get("val"), dict) else {}
        ml_test = ml_best_metrics.get("test") if isinstance(ml_best_metrics.get("test"), dict) else {}
        lines.append("## DL vs ML Best Comparison")
        lines.append("| Split | DL f1_macro | ML Best f1_macro | Delta | DL f1_fake | ML Best f1_fake | Delta | DL accuracy | ML Best accuracy | Delta |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        val_ml_f1 = float(ml_val.get("f1_macro", 0.0))
        val_ml_fake = float(ml_val.get("f1_fake", 0.0))
        val_ml_acc = float(ml_val.get("accuracy", 0.0))
        lines.append(
            f"| val | {val_metrics['f1_macro']:.6f} | {val_ml_f1:.6f} | {val_metrics['f1_macro'] - val_ml_f1:+.6f} | "
            f"{val_metrics['f1_fake']:.6f} | {val_ml_fake:.6f} | {val_metrics['f1_fake'] - val_ml_fake:+.6f} | "
            f"{val_metrics['accuracy']:.6f} | {val_ml_acc:.6f} | {val_metrics['accuracy'] - val_ml_acc:+.6f} |"
        )
        test_ml_f1 = float(ml_test.get("f1_macro", 0.0))
        test_ml_fake = float(ml_test.get("f1_fake", 0.0))
        test_ml_acc = float(ml_test.get("accuracy", 0.0))
        lines.append(
            f"| test | {test_metrics['f1_macro']:.6f} | {test_ml_f1:.6f} | {test_metrics['f1_macro'] - test_ml_f1:+.6f} | "
            f"{test_metrics['f1_fake']:.6f} | {test_ml_fake:.6f} | {test_metrics['f1_fake'] - test_ml_fake:+.6f} | "
            f"{test_metrics['accuracy']:.6f} | {test_ml_acc:.6f} | {test_metrics['accuracy'] - test_ml_acc:+.6f} |"
        )
        lines.append("")
    else:
        lines.append("## DL vs ML Best Comparison")
        lines.append("- ML best_config not found. Comparison table was skipped.")
        lines.append("")

    lines.append("## Confusion Matrix")
    lines.append("### Validation Split")
    lines.append(f"![Confusion Matrix Val]({val_fig_path.as_posix()})")
    lines.append("")
    lines.append("### Test Split")
    lines.append(f"![Confusion Matrix Test]({test_fig_path.as_posix()})")
    lines.append("")
    lines.extend(classification_report_to_markdown(val_report, "val"))
    lines.extend(classification_report_to_markdown(test_report, "test"))
    lines.append("## Notes")
    lines.append("- Class mapping: `0=real`, `1=fake`.")
    lines.append("- Metrics are recomputed from stored tokenizer/model checkpoint on current val/test splits.")
    lines.append("")
    return "\n".join(lines)


# Run end-to-end DL evaluation and generate report markdown plus confusion matrix figures.
def run_evaluation(
    run_id: str | None = None,
    config_path: str | Path = "configs/train_dl.yaml",
    output_report: str | Path = "reports/model_report_dl.md",
    output_pred_csv: str | Path | None = None,
) -> tuple[dict[str, Any], Path, Path, Path]:
    repo_root = get_repo_root()
    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = repo_root / resolved_config_path

    config = load_yaml(resolved_config_path)
    model_cfg = config.get("model")
    runtime_cfg = config.get("runtime")
    training_cfg = config.get("training")
    if not isinstance(model_cfg, dict) or not isinstance(runtime_cfg, dict) or not isinstance(training_cfg, dict):
        raise EvaluationError("train_dl.yaml missing model/runtime/training section")

    max_length = int(model_cfg.get("max_length", 256))
    eval_batch_size = int(training_cfg.get("eval_batch_size", 16))
    device = resolve_device(runtime_cfg)

    runs_path = repo_root / "experiments" / "dl" / "runs.csv"
    run_rows = load_runs_csv(runs_path)
    target_run = resolve_target_run(run_rows, run_id)
    target_run_id = str(target_run.get("run_id", "")).strip()

    run_dir = resolve_run_dir(target_run, repo_root)
    checkpoint_dir = run_dir / "best_checkpoint"
    if not checkpoint_dir.exists():
        raise EvaluationError(f"Missing checkpoint directory: {checkpoint_dir}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.to(device)

    val_metrics, val_pred_rows = evaluate_split(
        model=model,
        tokenizer=tokenizer,
        config=config,
        split_name="val",
        repo_root=repo_root,
        max_length=max_length,
        eval_batch_size=eval_batch_size,
        device=device,
    )
    test_metrics, test_pred_rows = evaluate_split(
        model=model,
        tokenizer=tokenizer,
        config=config,
        split_name="test",
        repo_root=repo_root,
        max_length=max_length,
        eval_batch_size=eval_batch_size,
        device=device,
    )

    if output_pred_csv is not None:
        output_pred_path = Path(output_pred_csv)
        if not output_pred_path.is_absolute():
            output_pred_path = repo_root / output_pred_path
        output_pred_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([val_pred_rows, test_pred_rows], axis=0, ignore_index=True).to_csv(
            output_pred_path,
            index=False,
            encoding="utf-8-sig",
        )

    val_y_true = val_pred_rows["y_true"].astype(int).to_numpy()
    val_y_pred = val_pred_rows["y_pred"].astype(int).to_numpy()
    test_y_true = test_pred_rows["y_true"].astype(int).to_numpy()
    test_y_pred = test_pred_rows["y_pred"].astype(int).to_numpy()

    val_report = classification_report(val_y_true, val_y_pred, output_dict=True, zero_division=0)
    test_report = classification_report(test_y_true, test_y_pred, output_dict=True, zero_division=0)
    val_cm = confusion_matrix(val_y_true, val_y_pred, labels=[0, 1])
    test_cm = confusion_matrix(test_y_true, test_y_pred, labels=[0, 1])

    figures_dir = repo_root / "reports" / "figures"
    val_cm_file = figures_dir / f"confusion_matrix_dl_val_{target_run_id}.png"
    test_cm_file = figures_dir / f"confusion_matrix_dl_test_{target_run_id}.png"
    save_confusion_matrix_figure(val_cm, f"Confusion Matrix (VAL) - {target_run_id}", val_cm_file)
    save_confusion_matrix_figure(test_cm, f"Confusion Matrix (TEST) - {target_run_id}", test_cm_file)

    ml_best_metrics, ml_best_path = load_ml_best_metrics(repo_root)
    if ml_best_metrics is not None:
        print(f"[DL EVAL] ML best_config FOUND at {ml_best_path.as_posix()} -> comparison table will be included.")
    else:
        print(f"[DL EVAL] ML best_config NOT FOUND at {ml_best_path.as_posix()} -> comparison table will be skipped.")

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
        ml_best_metrics=ml_best_metrics,
    )
    report_path.write_text(report_md, encoding="utf-8")

    summary = {
        "run_id": target_run_id,
        "run_timestamp": target_run.get("run_timestamp", ""),
        "model_name": target_run.get("model_name", ""),
        "feature_set": target_run.get("feature_set", ""),
        "text_variant": target_run.get("text_variant", ""),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    return summary, report_path, val_cm_file, test_cm_file


# Execute CLI entrypoint and print concise artifact locations.
def main() -> None:
    args = parse_args()
    print("[DL EVAL] Starting single-run evaluation...")
    summary, report_path, val_cm_path, test_cm_path = run_evaluation(
        run_id=args.run_id,
        config_path=args.config_path,
        output_report=args.output_report,
        output_pred_csv=args.output_pred_csv,
    )
    print(
        "[DL EVAL] PASS | "
        f"run_id={summary['run_id']} "
        f"model={summary['model_name']} "
        f"feature_set={summary['feature_set']} "
        f"text_variant={summary['text_variant']}"
    )
    print(
        "[DL EVAL] Metrics | "
        f"val_f1_macro={summary['val_metrics']['f1_macro']:.6f} "
        f"test_f1_macro={summary['test_metrics']['f1_macro']:.6f}"
    )
    print(f"[DL EVAL] Report: {report_path.as_posix()}")
    print(f"[DL EVAL] Figure (val): {val_cm_path.as_posix()}")
    print(f"[DL EVAL] Figure (test): {test_cm_path.as_posix()}")


if __name__ == "__main__":
    try:
        main()
    except (DlTrainError, EvaluationError) as exc:
        raise SystemExit(f"[DL EVAL ERROR] {exc}")
