from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    from src.train.dl.train_phobert import DlTrainError, get_repo_root, train_phobert
except ModuleNotFoundError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from train_phobert import DlTrainError, get_repo_root, train_phobert  # type: ignore


# Parse CLI args for DL run orchestration.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PhoBERT training and generate DL-vs-ML comparison report.")
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/train_dl.yaml",
        help="Path to train_dl.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional artifact root override (e.g., models/dl)",
    )
    return parser.parse_args()


# Load YAML config helper.
def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DlTrainError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise DlTrainError(f"Config must be a mapping object: {path}")
    return payload


# Build markdown table from metric dictionaries.
def build_metrics_table(dl_metrics: dict[str, Any], ml_metrics: dict[str, Any] | None) -> str:
    lines: list[str] = []
    lines.append("| Model | Split | f1_macro | f1_fake | precision_macro | recall_macro | accuracy |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")

    for split_name in ["train", "val", "test"]:
        values = dl_metrics["metrics"][split_name]
        lines.append(
            "| PhoBERT | "
            f"{split_name} | {values['f1_macro']:.6f} | {values['f1_fake']:.6f} | "
            f"{values['precision_macro']:.6f} | {values['recall_macro']:.6f} | {values['accuracy']:.6f} |"
        )

    if ml_metrics is not None:
        for split_name in ["val", "test"]:
            values = ml_metrics.get(split_name)
            if not isinstance(values, dict):
                continue
            lines.append(
                "| ML Best | "
                f"{split_name} | {float(values.get('f1_macro', 0.0)):.6f} | {float(values.get('f1_fake', 0.0)):.6f} | "
                f"{float(values.get('precision_macro', 0.0)):.6f} | {float(values.get('recall_macro', 0.0)):.6f} | "
                f"{float(values.get('accuracy', 0.0)):.6f} |"
            )

    return "\n".join(lines)


# Build ML-vs-DL delta table for val/test.
def build_delta_table(dl_metrics: dict[str, Any], ml_metrics: dict[str, Any] | None) -> str:
    if ml_metrics is None:
        return "_ML baseline not available from best_config.json._"

    lines: list[str] = []
    lines.append("| Split | Delta f1_macro (DL-ML) | Delta f1_fake (DL-ML) | Delta accuracy (DL-ML) |")
    lines.append("| --- | --- | --- | --- |")

    for split_name in ["val", "test"]:
        dl_values = dl_metrics["metrics"][split_name]
        ml_values = ml_metrics.get(split_name)
        if not isinstance(ml_values, dict):
            continue
        delta_f1_macro = float(dl_values["f1_macro"]) - float(ml_values.get("f1_macro", 0.0))
        delta_f1_fake = float(dl_values["f1_fake"]) - float(ml_values.get("f1_fake", 0.0))
        delta_accuracy = float(dl_values["accuracy"]) - float(ml_values.get("accuracy", 0.0))
        lines.append(
            f"| {split_name} | {delta_f1_macro:+.6f} | {delta_f1_fake:+.6f} | {delta_accuracy:+.6f} |"
        )

    return "\n".join(lines)


# Load ML baseline metrics from best_config.json if available.
def load_ml_baseline_metrics(config: dict[str, Any], repo_root: Path) -> dict[str, Any] | None:
    comparison_cfg = config.get("comparison")
    if not isinstance(comparison_cfg, dict):
        return None

    best_config_rel = str(comparison_cfg.get("ml_best_config_path", "")).strip()
    if not best_config_rel:
        return None

    best_config_path = repo_root / best_config_rel
    if not best_config_path.exists():
        return None

    with best_config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None

    best_run = payload.get("best_run")
    if not isinstance(best_run, dict):
        return None

    metrics = best_run.get("metrics")
    if not isinstance(metrics, dict):
        return None

    return metrics


# Render markdown report for DL run and ML comparison.
def build_report_content(
    train_dl_config: dict[str, Any],
    dl_metrics_payload: dict[str, Any],
    ml_metrics: dict[str, Any] | None,
) -> str:
    lines: list[str] = []
    lines.append("# Model Report (DL: PhoBERT)")
    lines.append("")
    lines.append("## Run Info")
    lines.append(f"- run_id: {dl_metrics_payload['run_id']}")
    lines.append(f"- run_timestamp: {dl_metrics_payload['run_timestamp']}")
    lines.append(f"- experiment_name: {dl_metrics_payload['experiment_name']}")
    lines.append(f"- pretrained_name: {dl_metrics_payload['pretrained_name']}")
    lines.append(f"- max_length: {dl_metrics_payload['max_length']}")
    lines.append(f"- best_epoch: {dl_metrics_payload['best_epoch']}")
    lines.append(f"- device: {dl_metrics_payload['device']}")
    lines.append(f"- elapsed_seconds: {dl_metrics_payload['elapsed_seconds']}")
    lines.append("")

    lines.append("## DL Metrics")
    lines.append(build_metrics_table(dl_metrics_payload, ml_metrics))
    lines.append("")

    lines.append("## ML vs DL Delta")
    lines.append(build_delta_table(dl_metrics_payload, ml_metrics))
    lines.append("")

    lines.append("## Notes")
    lines.append("- DL input column: text_clean from data/processed/train.csv, val.csv, test.csv")
    lines.append("- Label convention: 0=real, 1=fake")
    lines.append("- Checkpoint selection: best validation f1_macro with tie-break by f1_fake")
    lines.append("")

    return "\n".join(lines)


# Write markdown report to configured output path.
def write_report(train_dl_config: dict[str, Any], report_content: str, repo_root: Path) -> Path:
    comparison_cfg = train_dl_config.get("comparison")
    report_rel = "reports/model_report_dl.md"
    if isinstance(comparison_cfg, dict):
        report_rel = str(comparison_cfg.get("report_path", report_rel))

    report_path = repo_root / report_rel
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding="utf-8")
    return report_path


# Execute DL training and emit final report with ML comparison.
def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()

    resolved_config_path = Path(args.config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = repo_root / resolved_config_path

    train_dl_config = load_yaml(resolved_config_path)

    print("[DL TRAIN] Starting PhoBERT training run...")
    artifacts = train_phobert(
        config_path=resolved_config_path,
        output_root_override=args.output_root,
    )

    ml_metrics = load_ml_baseline_metrics(train_dl_config, repo_root)
    report_content = build_report_content(
        train_dl_config=train_dl_config,
        dl_metrics_payload=artifacts.metrics,
        ml_metrics=ml_metrics,
    )
    report_path = write_report(train_dl_config, report_content, repo_root)

    print(
        "[DL TRAIN] PASS | "
        f"run_id={artifacts.run_id} "
        f"val_f1_macro={artifacts.metrics['metrics']['val']['f1_macro']:.6f} "
        f"test_f1_macro={artifacts.metrics['metrics']['test']['f1_macro']:.6f}"
    )
    print(f"[DL TRAIN] Artifacts: {artifacts.run_dir.as_posix()}")
    print(f"[DL TRAIN] Report: {report_path.as_posix()}")


if __name__ == "__main__":
    try:
        main()
    except DlTrainError as exc:
        raise SystemExit(f"[DL TRAIN ERROR] {exc}")
