from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

dl_eval = pytest.importorskip("src.train.dl.evaluate_phobert")


# Build one compact fake tokenizer used across DL evaluation-helper tests.
# Tokenizer stub returning a fixed padded batch for evaluation tests.
class FakeTokenizer:
    def __call__(self, texts, truncation, padding, max_length, return_tensors):
        batch_size = len(texts)
        input_ids = torch.tensor([[1, 2, 3, 0][:max_length] for _ in range(batch_size)], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 0][:max_length] for _ in range(batch_size)], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


# Build one compact fake model used for evaluate_split tests.
# Model stub returning deterministic logits for evaluation tests.
class FakeModel:
    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        batch_size = int(input_ids.shape[0])
        logits = torch.tensor([[0.0, 1.0]] * batch_size, dtype=torch.float32, device=input_ids.device)
        return SimpleNamespace(logits=logits)


# Validate device resolution accepts CPU and rejects unsupported names.
def test_resolve_device_validation() -> None:
    assert dl_eval.resolve_device({"device": "cpu"}).type == "cpu"

    with pytest.raises(dl_eval.EvaluationError):
        dl_eval.resolve_device({"device": "invalid"})


# Validate split loading reads the configured CSV and enforces required columns.
def test_load_split_frame(tmp_path: Path) -> None:
    repo_root = tmp_path
    split_path = repo_root / "data" / "processed" / "val.csv"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"sample_id": ["s1"], "label_binary": [0], "text_clean": ["hello"]}
    ).to_csv(split_path, index=False, encoding="utf-8-sig")

    cfg = {"data": {"val_path": "data/processed/val.csv", "text_column": "text_clean", "label_column": "label_binary", "id_column": "sample_id"}}
    frame = dl_eval.load_split_frame(cfg, "val", repo_root)

    assert frame.loc[0, "sample_id"] == "s1"


# Validate target run resolution prefers explicit IDs and otherwise picks the latest PASS run.
def test_resolve_target_run_selection() -> None:
    rows = [
        {"run_id": "old", "run_timestamp": "2026-05-01T00:00:00+07:00", "status": "PASS"},
        {"run_id": "new", "run_timestamp": "2026-05-03T00:00:00+07:00", "status": "PASS"},
        {"run_id": "fail", "run_timestamp": "2026-05-04T00:00:00+07:00", "status": "FAIL"},
    ]

    assert dl_eval.resolve_target_run(rows, "old")["run_id"] == "old"
    assert dl_eval.resolve_target_run(rows, None)["run_id"] == "new"


# Validate run directory resolution supports both notes-based paths and fallback paths.
def test_resolve_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "models" / "dl" / "run-1"
    run_dir.mkdir(parents=True, exist_ok=True)

    target_run = {"run_id": "run-1", "notes": '{"run_dir": "' + str(run_dir).replace("\\", "/") + '"}'}
    assert dl_eval.resolve_run_dir(target_run, tmp_path) == run_dir


# Validate evaluation on one split produces metrics, predictions, and fake probabilities.
def test_evaluate_split_end_to_end(tmp_path: Path) -> None:
    repo_root = tmp_path
    split_path = repo_root / "data" / "processed" / "val.csv"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "label_binary": [0, 1],
            "text_clean": ["hello", "world"],
        }
    ).to_csv(split_path, index=False, encoding="utf-8-sig")

    cfg = {
        "data": {
            "val_path": "data/processed/val.csv",
            "text_column": "text_clean",
            "label_column": "label_binary",
            "id_column": "sample_id",
        }
    }

    metrics, pred_rows = dl_eval.evaluate_split(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        config=cfg,
        split_name="val",
        repo_root=repo_root,
        max_length=4,
        eval_batch_size=1,
        device=torch.device("cpu"),
    )

    assert set(metrics) == {"f1_macro", "precision_macro", "recall_macro", "accuracy", "f1_fake"}
    assert pred_rows["split"].tolist() == ["val", "val"]
    assert pred_rows["prob_fake"].tolist() == [pytest.approx(0.7310586), pytest.approx(0.7310586)]


# Validate markdown report assembly includes run details and comparison blocks.
def test_build_markdown_report_and_ml_loading(tmp_path: Path) -> None:
    repo_root = tmp_path
    best_config_path = repo_root / "experiments" / "ml" / "best_config.json"
    best_config_path.parent.mkdir(parents=True, exist_ok=True)
    best_config_path.write_text(
        '{"best_run": {"metrics": {"val": {"f1_macro": 0.9, "f1_fake": 0.8, "accuracy": 0.85}, "test": {"f1_macro": 0.88, "f1_fake": 0.77, "accuracy": 0.84}}}}',
        encoding="utf-8",
    )

    ml_metrics, resolved_path = dl_eval.load_ml_best_metrics(repo_root)
    assert resolved_path == best_config_path
    assert ml_metrics is not None

    report = dl_eval.build_markdown_report(
        target_run={
            "run_id": "run-1",
            "run_timestamp": "2026-05-02T00:00:00+07:00",
            "model_name": "phobert",
            "feature_set": "transformer",
            "text_variant": "text_clean",
            "params_json": "{}",
        },
        val_metrics={"f1_macro": 0.9, "precision_macro": 0.91, "recall_macro": 0.89, "accuracy": 0.88, "f1_fake": 0.87},
        test_metrics={"f1_macro": 0.88, "precision_macro": 0.89, "recall_macro": 0.87, "accuracy": 0.86, "f1_fake": 0.85},
        val_report={"0": {"precision": 0.9, "recall": 0.8, "f1-score": 0.85, "support": 10}},
        test_report={"0": {"precision": 0.9, "recall": 0.8, "f1-score": 0.85, "support": 10}},
        val_fig_path=Path("reports/figures/val.png"),
        test_fig_path=Path("reports/figures/test.png"),
        ml_best_metrics=ml_metrics,
    )

    assert "DL Model Evaluation Report" in report
    assert "DL vs ML Best Comparison" in report
    assert "Confusion Matrix" in report
