from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

dl_train = pytest.importorskip("src.train.dl.train_phobert")


# Build one compact fake tokenizer used across DL training-helper tests.
# Tokenizer stub returning deterministic tensors for training helpers.
class FakeTokenizer:
    def __call__(self, texts, truncation, padding, max_length, return_tensors):
        batch_size = len(texts)
        input_ids = torch.tensor([[1, 2, 3, 0][:max_length] for _ in range(batch_size)], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 0][:max_length] for _ in range(batch_size)], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


# Build one compact fake model used for evaluate_model tests.
# Model stub returning fixed logits for evaluation metrics coverage.
class FakeModel:
    def eval(self) -> None:
        self.was_eval = True

    def __call__(self, input_ids, attention_mask):
        batch_size = int(input_ids.shape[0])
        logits = torch.tensor([[0.0, 1.0]] * batch_size, dtype=torch.float32, device=input_ids.device)
        return SimpleNamespace(logits=logits)


import torch


# Validate seed setting keeps the RNGs reproducible.
def test_seed_everything_reproducible() -> None:
    dl_train.seed_everything(123)
    first = (random_value := np.random.rand(), torch.rand(1).item())
    dl_train.seed_everything(123)
    second = (np.random.rand(), torch.rand(1).item())

    assert first == second


# Validate device resolution accepts CPU and rejects invalid values.
def test_resolve_device_validation() -> None:
    assert dl_train.resolve_device("cpu").type == "cpu"

    with pytest.raises(dl_train.DlTrainError):
        dl_train.resolve_device("invalid")


# Validate split building enforces required columns and binary labels.
def test_build_split_data_validation() -> None:
    frame = pd.DataFrame({"text_clean": ["a", "b"], "label_binary": [0, 1], "sample_id": ["s1", "s2"]})
    split = dl_train.build_split_data(frame, "text_clean", "label_binary", "sample_id", "train")

    assert split.sample_ids == ["s1", "s2"]
    assert split.texts == ["a", "b"]
    assert split.labels.tolist() == [0, 1]

    with pytest.raises(dl_train.DlTrainError):
        dl_train.build_split_data(frame.drop(columns=["sample_id"]), "text_clean", "label_binary", "sample_id", "train")


# Validate tokenization caches one encoded payload per split and max_length.
def test_tokenize_split_cache() -> None:
    tokenizer = FakeTokenizer()
    cache: dict[tuple[int, str], dict[str, torch.Tensor]] = {}

    encoded_first = dl_train.tokenize_split(tokenizer, ["a", "b"], max_length=4, encoded_cache=cache, split_name="train")
    encoded_second = dl_train.tokenize_split(tokenizer, ["changed"], max_length=4, encoded_cache=cache, split_name="train")

    assert encoded_first is encoded_second
    assert set(encoded_first) == {"input_ids", "attention_mask"}


# Validate label and encoding bound checks raise only on invalid inputs.
def test_label_and_encoding_bounds() -> None:
    labels = np.array([0, 1], dtype=int)
    sample_ids = ["s1", "s2"]
    dl_train.assert_label_bounds(labels, 2, "train", sample_ids)

    with pytest.raises(dl_train.DlTrainError):
        dl_train.assert_label_bounds(np.array([0, 2], dtype=int), 2, "train", sample_ids)

    encodings = {"input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long)}
    dl_train.assert_encoding_lengths(encodings, labels, "train")
    dl_train.assert_encoding_bounds(encodings, vocab_size=10, split_name="train")

    with pytest.raises(dl_train.DlTrainError):
        dl_train.assert_encoding_lengths({"input_ids": torch.tensor([[1, 2]])}, labels, "train")

    with pytest.raises(dl_train.DlTrainError):
        dl_train.assert_encoding_bounds({"input_ids": torch.tensor([[1, 11]], dtype=torch.long)}, vocab_size=10, split_name="train")


# Validate binary metrics and model evaluation both produce the expected output schema.
def test_compute_metrics_and_evaluate_model() -> None:
    metrics = dl_train.compute_metrics(np.array([0, 1, 1]), np.array([0, 1, 0]))
    assert set(metrics) == {"f1_macro", "precision_macro", "recall_macro", "accuracy", "f1_fake"}

    encodings = {
        "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
    }
    dataset = dl_train.TextDataset(encodings, np.array([0, 1], dtype=int))
    dataloader = dl_train.DataLoader(dataset, batch_size=1, shuffle=False)

    model = FakeModel()
    out_metrics, pred_rows = dl_train.evaluate_model(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
        sample_ids=["s1", "s2"],
        split_name="val",
    )

    assert set(out_metrics) == {"f1_macro", "precision_macro", "recall_macro", "accuracy", "f1_fake"}
    assert list(pred_rows["split"].unique()) == ["val"]
    assert list(pred_rows["prob_fake"].round(6)) == [0.731059, 0.731059]


# Validate autocast helper returns a context manager even when mixed precision is disabled.
def test_autocast_context_disabled() -> None:
    with dl_train.autocast_context(False):
        assert True
