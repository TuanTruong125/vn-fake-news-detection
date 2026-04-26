from __future__ import annotations

import csv
import json
import math
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

try:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "[DL TRAIN ERROR] Missing DL dependencies. Please install requirements including torch and transformers."
    ) from exc


class DlTrainError(Exception):
    pass


@dataclass
class SplitData:
    sample_ids: list[str]
    texts: list[str]
    labels: np.ndarray


@dataclass
class TrainArtifacts:
    run_id: str
    run_dir: Path
    best_checkpoint_dir: Path
    metrics: dict[str, Any]
    metadata_path: Path


class TextDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], labels: np.ndarray) -> None:
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


RUNS_COLUMNS = [
    "run_id",
    "run_timestamp",
    "experiment_name",
    "config_version",
    "random_state",
    "model_name",
    "feature_set",
    "text_variant",
    "params_json",
    "val_f1_macro",
    "val_precision_macro",
    "val_recall_macro",
    "val_accuracy",
    "val_f1_fake",
    "test_f1_macro",
    "test_precision_macro",
    "test_recall_macro",
    "test_accuracy",
    "test_f1_fake",
    "status",
    "notes",
]


# Resolve repository root from current file location.
def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# Return local ISO timestamp.
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# Build reproducible run id for DL runs.
def build_run_id(experiment_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    safe_name = experiment_name.replace(" ", "_")
    return f"{ts}_{safe_name}_{suffix}"


# Read YAML config and validate object type.
def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DlTrainError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise DlTrainError(f"Config must be a mapping object: {path}")
    return payload


# Ensure DL runs tracking CSV exists and matches expected header schema.
def ensure_runs_csv(runs_path: Path) -> None:
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    if not runs_path.exists():
        with runs_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RUNS_COLUMNS)
            writer.writeheader()
        return

    with runs_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    if header != RUNS_COLUMNS:
        raise DlTrainError(
            "experiments/dl/runs.csv header mismatch.\n"
            f"Expected: {RUNS_COLUMNS}\nFound:    {header}"
        )


# Append one DL run record row into experiments/dl/runs.csv.
def append_run_record(runs_path: Path, run_record: dict[str, Any]) -> None:
    ensure_runs_csv(runs_path)
    with runs_path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_COLUMNS)
        writer.writerow({column: run_record.get(column, "") for column in RUNS_COLUMNS})


# Build default FAIL-first run record, then update to PASS on successful training.
def init_run_record(
    run_id: str,
    run_timestamp: str,
    experiment_name: str,
    config_version: int | str,
    random_state: int,
    model_name: str,
    params_json: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "experiment_name": experiment_name,
        "config_version": config_version,
        "random_state": random_state,
        "model_name": model_name,
        "feature_set": "transformer",
        "text_variant": "text_clean",
        "params_json": params_json,
        "val_f1_macro": "",
        "val_precision_macro": "",
        "val_recall_macro": "",
        "val_accuracy": "",
        "val_f1_fake": "",
        "test_f1_macro": "",
        "test_precision_macro": "",
        "test_recall_macro": "",
        "test_accuracy": "",
        "test_f1_fake": "",
        "status": "FAIL",
        "notes": "",
    }


# Set deterministic seed for reproducibility.
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Validate and resolve compute device from config.
def resolve_device(device_cfg: str) -> torch.device:
    candidate = device_cfg.strip().lower()
    if candidate == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if candidate == "cuda":
        if not torch.cuda.is_available():
            raise DlTrainError("runtime.device=cuda but CUDA is not available.")
        return torch.device("cuda")
    if candidate == "cpu":
        return torch.device("cpu")
    raise DlTrainError("runtime.device must be one of: auto, cpu, cuda")


# Validate dataframe required columns and convert to split payload.
def build_split_data(df: pd.DataFrame, text_col: str, label_col: str, id_col: str, split_name: str) -> SplitData:
    required_columns = [text_col, label_col, id_col]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise DlTrainError(f"{split_name}: missing required columns: {missing}")

    text_series = df[text_col].fillna("").astype(str)
    label_series = pd.to_numeric(df[label_col], errors="coerce").fillna(-1).astype(int)
    id_series = df[id_col].fillna("").astype(str)

    if not set(label_series.unique()).issubset({0, 1}):
        raise DlTrainError(f"{split_name}: {label_col} contains values outside {{0,1}}")

    return SplitData(
        sample_ids=id_series.tolist(),
        texts=text_series.tolist(),
        labels=label_series.to_numpy(),
    )


# Load configured train/val/test split files using text_clean.
def load_splits(config: dict[str, Any], repo_root: Path) -> dict[str, SplitData]:
    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        raise DlTrainError("train_dl.yaml missing data section")

    text_col = str(data_cfg.get("text_column", "")).strip()
    label_col = str(data_cfg.get("label_column", "")).strip()
    id_col = str(data_cfg.get("id_column", "")).strip()
    if not text_col or not label_col or not id_col:
        raise DlTrainError("data.text_column, data.label_column, data.id_column must be non-empty")

    split_paths = {
        "train": repo_root / str(data_cfg.get("train_path", "")),
        "val": repo_root / str(data_cfg.get("val_path", "")),
        "test": repo_root / str(data_cfg.get("test_path", "")),
    }
    for split_name, split_path in split_paths.items():
        if not split_path.exists():
            raise DlTrainError(f"{split_name}: split file not found: {split_path}")

    split_frames = {split_name: pd.read_csv(split_path) for split_name, split_path in split_paths.items()}

    return {
        split_name: build_split_data(
            df=frame,
            text_col=text_col,
            label_col=label_col,
            id_col=id_col,
            split_name=split_name,
        )
        for split_name, frame in split_frames.items()
    }


# Tokenize one split and return model-ready tensor dictionary.
def tokenize_split(tokenizer: Any, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


# Compute binary metrics aligned with ML naming conventions.
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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


# Evaluate model on one dataloader and return metrics and optional prediction rows.
def evaluate_model(
    model: Any,
    dataloader: DataLoader,
    device: torch.device,
    sample_ids: list[str],
    split_name: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob_fake: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())
            y_prob_fake.extend(probs[:, 1].detach().cpu().numpy().tolist())

    metrics = compute_metrics(np.array(y_true, dtype=int), np.array(y_pred, dtype=int))

    rows = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "split": split_name,
            "y_true": y_true,
            "y_pred": y_pred,
            "prob_fake": y_prob_fake,
        }
    )
    rows["is_error"] = rows["y_true"] != rows["y_pred"]

    return metrics, rows


# Train PhoBERT classifier and persist best run artifacts.
def train_phobert(config_path: str | Path | None = None, output_root_override: str | Path | None = None) -> TrainArtifacts:
    repo_root = get_repo_root()
    resolved_config_path = repo_root / "configs" / "train_dl.yaml" if config_path is None else Path(config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = repo_root / resolved_config_path

    config = load_yaml(resolved_config_path)

    experiment_cfg = config.get("experiment", {})
    if not isinstance(experiment_cfg, dict):
        raise DlTrainError("train_dl.yaml missing experiment section")
    experiment_name = str(experiment_cfg.get("name", "phobert_extension")).strip() or "phobert_extension"
    seed = int(experiment_cfg.get("random_state", 42))
    primary_metric = str(experiment_cfg.get("primary_metric", "f1_macro")).strip() or "f1_macro"
    tie_breaker = str(experiment_cfg.get("tie_breaker", "f1_fake")).strip() or "f1_fake"

    model_cfg = config.get("model")
    training_cfg = config.get("training")
    runtime_cfg = config.get("runtime")
    if not isinstance(model_cfg, dict) or not isinstance(training_cfg, dict) or not isinstance(runtime_cfg, dict):
        raise DlTrainError("train_dl.yaml must define model, training, runtime sections")

    pretrained_name = str(model_cfg.get("pretrained_name", "")).strip()
    if not pretrained_name:
        raise DlTrainError("model.pretrained_name must be non-empty")

    config_version = config.get("version", "")

    max_length = int(model_cfg.get("max_length", 256))
    num_labels = int(model_cfg.get("num_labels", 2))
    if num_labels != 2:
        raise DlTrainError("PhoBERT extension currently supports binary labels only (num_labels=2)")

    num_epochs = int(training_cfg.get("num_epochs", 3))
    train_batch_size = int(training_cfg.get("train_batch_size", 8))
    eval_batch_size = int(training_cfg.get("eval_batch_size", 16))
    learning_rate = float(training_cfg.get("learning_rate", 2e-5))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(training_cfg.get("warmup_ratio", 0.1))
    grad_acc_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    early_stopping_patience = int(training_cfg.get("early_stopping_patience", 2))

    run_id = build_run_id(experiment_name)
    run_timestamp = now_iso()
    params_json = json.dumps(
        {
            "learning_rate": learning_rate,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "max_length": max_length,
            "num_epochs": num_epochs,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "gradient_accumulation_steps": grad_acc_steps,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    run_record = init_run_record(
        run_id=run_id,
        run_timestamp=run_timestamp,
        experiment_name=experiment_name,
        config_version=config_version,
        random_state=seed,
        model_name=pretrained_name,
        params_json=params_json,
    )
    runs_path = repo_root / "experiments" / "dl" / "runs.csv"

    device = resolve_device(str(runtime_cfg.get("device", "auto")))
    fp16 = bool(runtime_cfg.get("fp16", False)) and device.type == "cuda"
    num_workers = int(runtime_cfg.get("num_workers", 0))
    pin_memory = bool(runtime_cfg.get("pin_memory", False)) and device.type == "cuda"

    output_root = repo_root / str(runtime_cfg.get("output_root", "models/dl"))
    if output_root_override is not None:
        output_root = Path(output_root_override)
        if not output_root.is_absolute():
            output_root = repo_root / output_root

    try:
        seed_everything(seed)

        splits = load_splits(config, repo_root)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_name, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_name, num_labels=num_labels)
        model.to(device)

        train_encodings = tokenize_split(tokenizer, splits["train"].texts, max_length=max_length)
        val_encodings = tokenize_split(tokenizer, splits["val"].texts, max_length=max_length)
        test_encodings = tokenize_split(tokenizer, splits["test"].texts, max_length=max_length)

        train_dataset = TextDataset(train_encodings, splits["train"].labels)
        val_dataset = TextDataset(val_encodings, splits["val"].labels)
        test_dataset = TextDataset(test_encodings, splits["test"].labels)

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        train_steps_per_epoch = max(1, int(np.ceil(len(train_loader) / max(1, grad_acc_steps))))
        total_train_steps = train_steps_per_epoch * num_epochs
        warmup_steps = int(total_train_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps,
        )

        run_dir = output_root / run_id
        checkpoint_dir = run_dir / "best_checkpoint"
        run_dir.mkdir(parents=True, exist_ok=True)

        scaler = torch.cuda.amp.GradScaler(enabled=fp16)

        best_val_metrics: dict[str, float] | None = None
        best_epoch = 0
        no_improve_epochs = 0

        epoch_history: list[dict[str, Any]] = []
        training_start = time.perf_counter()

        for epoch in range(1, num_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss_sum = 0.0
            seen_batches = 0

            for step, batch in enumerate(train_loader, start=1):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.cuda.amp.autocast(enabled=fp16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / max(1, grad_acc_steps)

                scaler.scale(loss).backward()

                if step % max(1, grad_acc_steps) == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                epoch_loss_sum += float(loss.detach().cpu().item() * max(1, grad_acc_steps))
                seen_batches += 1

            if seen_batches == 0:
                raise DlTrainError("No training batches were produced. Check dataset and batch size.")

            train_loss = epoch_loss_sum / seen_batches

            val_metrics, _ = evaluate_model(
                model=model,
                dataloader=val_loader,
                device=device,
                sample_ids=splits["val"].sample_ids,
                split_name="val",
            )

            print(
                f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.6f} | "
                f"Val f1_macro: {val_metrics['f1_macro']:.6f} | Val f1_fake: {val_metrics['f1_fake']:.6f}",
                flush=True,
            )

            improved = False
            if best_val_metrics is None:
                improved = True
            else:
                best_primary = float(best_val_metrics.get(primary_metric, float("-inf")))
                current_primary = float(val_metrics.get(primary_metric, float("-inf")))
                if current_primary > best_primary:
                    improved = True
                elif math.isclose(current_primary, best_primary):
                    best_tie = float(best_val_metrics.get(tie_breaker, float("-inf")))
                    current_tie = float(val_metrics.get(tie_breaker, float("-inf")))
                    if current_tie > best_tie:
                        improved = True

            if improved:
                best_val_metrics = dict(val_metrics)
                best_epoch = epoch
                no_improve_epochs = 0
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
            else:
                no_improve_epochs += 1

            epoch_history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_metrics": val_metrics,
                    "improved": improved,
                }
            )

            if no_improve_epochs >= early_stopping_patience:
                print(
                    f"[DL TRAIN] Early stopping triggered at epoch={epoch} (patience={early_stopping_patience}).",
                    flush=True,
                )
                break

        if best_val_metrics is None:
            raise DlTrainError("Training completed but no best checkpoint was selected.")

        best_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        best_model.to(device)

        train_metrics, train_pred_rows = evaluate_model(
            model=best_model,
            dataloader=DataLoader(
                train_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
            device=device,
            sample_ids=splits["train"].sample_ids,
            split_name="train",
        )
        val_metrics, val_pred_rows = evaluate_model(
            model=best_model,
            dataloader=val_loader,
            device=device,
            sample_ids=splits["val"].sample_ids,
            split_name="val",
        )
        test_metrics, test_pred_rows = evaluate_model(
            model=best_model,
            dataloader=test_loader,
            device=device,
            sample_ids=splits["test"].sample_ids,
            split_name="test",
        )

        elapsed_seconds = round(time.perf_counter() - training_start, 3)

        metrics_payload = {
            "run_id": run_id,
            "run_timestamp": run_timestamp,
            "experiment_name": experiment_name,
            "config_path": str(resolved_config_path.as_posix()),
            "seed": seed,
            "device": str(device),
            "fp16": bool(fp16),
            "pretrained_name": pretrained_name,
            "max_length": max_length,
            "best_epoch": int(best_epoch),
            "elapsed_seconds": elapsed_seconds,
            "metrics": {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
            },
            "epoch_history": epoch_history,
        }

        metrics_path = run_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

        metadata_path = run_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "run_timestamp": run_timestamp,
                    "source": "train_phobert",
                    "config_snapshot": config,
                    "artifact_paths": {
                        "best_checkpoint_dir": str(checkpoint_dir.as_posix()),
                        "metrics_path": str(metrics_path.as_posix()),
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        pred_rows = pd.concat([train_pred_rows, val_pred_rows, test_pred_rows], axis=0, ignore_index=True)
        pred_rows.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")

        run_record.update(
            {
                "val_f1_macro": round(val_metrics["f1_macro"], 6),
                "val_precision_macro": round(val_metrics["precision_macro"], 6),
                "val_recall_macro": round(val_metrics["recall_macro"], 6),
                "val_accuracy": round(val_metrics["accuracy"], 6),
                "val_f1_fake": round(val_metrics["f1_fake"], 6),
                "test_f1_macro": round(test_metrics["f1_macro"], 6),
                "test_precision_macro": round(test_metrics["precision_macro"], 6),
                "test_recall_macro": round(test_metrics["recall_macro"], 6),
                "test_accuracy": round(test_metrics["accuracy"], 6),
                "test_f1_fake": round(test_metrics["f1_fake"], 6),
                "status": "PASS",
                "notes": json.dumps(
                    {
                        "best_epoch": int(best_epoch),
                        "elapsed_seconds": elapsed_seconds,
                        "device": str(device),
                        "run_dir": str(run_dir.as_posix()),
                        "metrics_path": str(metrics_path.as_posix()),
                        "metadata_path": str(metadata_path.as_posix()),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )

        return TrainArtifacts(
            run_id=run_id,
            run_dir=run_dir,
            best_checkpoint_dir=checkpoint_dir,
            metrics=metrics_payload,
            metadata_path=metadata_path,
        )
    except Exception as exc:
        run_record["status"] = "FAIL"
        run_record["notes"] = str(exc)
        raise
    finally:
        print("[DL TRAIN] Writing run record to experiments/dl/runs.csv...")
        append_run_record(runs_path, run_record)
