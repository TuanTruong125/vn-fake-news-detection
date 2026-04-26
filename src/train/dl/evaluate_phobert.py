from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

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


# Parse CLI args for standalone PhoBERT evaluation.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PhoBERT checkpoint on a selected split.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to DL run directory containing best_checkpoint.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split to evaluate.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/train_dl.yaml",
        help="Path to train_dl.yaml.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to save split predictions CSV.",
    )
    return parser.parse_args()


# Load split file according to train_dl.yaml config.
def load_split_frame(config: dict[str, Any], split_name: str, repo_root: Path) -> pd.DataFrame:
    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        raise DlTrainError("train_dl.yaml missing data section")

    if split_name == "train":
        split_path = repo_root / str(data_cfg.get("train_path", ""))
    elif split_name == "val":
        split_path = repo_root / str(data_cfg.get("val_path", ""))
    else:
        split_path = repo_root / str(data_cfg.get("test_path", ""))

    if not split_path.exists():
        raise DlTrainError(f"Split file not found: {split_path}")

    frame = pd.read_csv(split_path)
    text_col = str(data_cfg.get("text_column", "")).strip()
    label_col = str(data_cfg.get("label_column", "")).strip()
    id_col = str(data_cfg.get("id_column", "")).strip()

    required = [text_col, label_col, id_col]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise DlTrainError(f"{split_name}: missing required columns: {missing}")

    return frame


# Run standalone evaluation from saved checkpoint and print metrics.
def evaluate_phobert_run(
    run_dir: str | Path,
    split_name: str,
    config_path: str | Path = "configs/train_dl.yaml",
    output_csv: str | Path | None = None,
) -> dict[str, float]:
    repo_root = get_repo_root()

    resolved_run_dir = Path(run_dir)
    if not resolved_run_dir.is_absolute():
        resolved_run_dir = repo_root / resolved_run_dir

    checkpoint_dir = resolved_run_dir / "best_checkpoint"
    if not checkpoint_dir.exists():
        raise DlTrainError(f"Missing checkpoint directory: {checkpoint_dir}")

    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = repo_root / resolved_config_path

    config = load_yaml(resolved_config_path)
    model_cfg = config.get("model")
    runtime_cfg = config.get("runtime")
    training_cfg = config.get("training")
    if not isinstance(model_cfg, dict) or not isinstance(runtime_cfg, dict) or not isinstance(training_cfg, dict):
        raise DlTrainError("train_dl.yaml missing model/runtime/training section")

    max_length = int(model_cfg.get("max_length", 256))
    eval_batch_size = int(training_cfg.get("eval_batch_size", 16))

    device_name = str(runtime_cfg.get("device", "auto")).strip().lower()
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name == "cuda":
        if not torch.cuda.is_available():
            raise DlTrainError("runtime.device=cuda but CUDA not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    frame = load_split_frame(config, split_name, repo_root)
    data_cfg = config["data"]
    text_col = str(data_cfg.get("text_column"))
    label_col = str(data_cfg.get("label_column"))
    id_col = str(data_cfg.get("id_column"))

    texts = frame[text_col].fillna("").astype(str).tolist()
    labels = pd.to_numeric(frame[label_col], errors="coerce").fillna(-1).astype(int).to_numpy()
    sample_ids = frame[id_col].fillna("").astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.to(device)

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

    if output_csv is not None:
        output_path = Path(output_csv)
        if not output_path.is_absolute():
            output_path = repo_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pred_rows.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(
        "[DL EVAL] "
        f"split={split_name} "
        f"f1_macro={metrics['f1_macro']:.6f} "
        f"f1_fake={metrics['f1_fake']:.6f} "
        f"accuracy={metrics['accuracy']:.6f}"
    )

    return metrics


# CLI entrypoint for standalone evaluation.
def main() -> None:
    args = parse_args()
    evaluate_phobert_run(
        run_dir=args.run_dir,
        split_name=args.split,
        config_path=args.config_path,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    try:
        main()
    except DlTrainError as exc:
        raise SystemExit(f"[DL EVAL ERROR] {exc}")
