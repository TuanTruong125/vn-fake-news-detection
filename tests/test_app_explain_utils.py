from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.explain_dl import build_dl_explanation
from src.app.explain_ml import explain_linear_prediction


# Build a compact linear model stub used for ML explanation tests.
# Linear-model stub with fixed weights for attribution tests.
class FakeLinearModel:
    coef_ = np.asarray([[2.0, -1.0]])
    classes_ = np.asarray([0, 1])
    intercept_ = np.asarray([0.25])


# Build a compact vectorizer stub used for ML explanation tests.
# Vectorizer stub exposing a fixed feature-name list.
class FakeVectorizer:
    def get_feature_names_out(self):
        return np.asarray(["tok_a", "tok_b"])


# Build a compact tokenizer stub used for DL occlusion tests.
# Tokenizer stub exposing the special token ids needed by occlusion.
class FakeTokenizer:
    all_special_ids = [0]
    mask_token_id = 99
    pad_token_id = None
    unk_token_id = None

    def convert_ids_to_tokens(self, ids):
        mapping = {1: "hello", 2: "world", 99: "[MASK]"}
        return [mapping[i] for i in ids]


# Validate linear ML explanation returns per-feature contributions and decomposition.
def test_explain_linear_prediction_basic() -> None:
    model = FakeLinearModel()
    vectorizer = FakeVectorizer()
    x_vec = csr_matrix([[3.0, 4.0]])

    explanation = explain_linear_prediction(
        model=model,
        vectorizer=vectorizer,
        x_vec=x_vec,
        top_k_per_direction=5,
        include_decomposition=True,
        raw_decision_score=5.25,
    )

    assert explanation["explanation_available"] is True
    assert explanation["top_features_towards_fake"][0]["feature"] == "tok_a"
    assert explanation["explanation_decomposition"]["raw_decision_score"] == 5.25


# Validate DL occlusion explanation returns token-level features and an approximate decomposition.
def test_build_dl_explanation_basic() -> None:
    tokenizer = FakeTokenizer()
    encoded = {
        "input_ids": torch.tensor([[1, 2, 1, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
    }

    def score_fn(masked_ids, attention_mask):
        token_ids = masked_ids[0].tolist()
        if token_ids[0] == 99:
            return 0.2, None
        return 0.8, None

    explanation = build_dl_explanation(
        tokenizer=tokenizer,
        encoded=encoded,
        base_score=0.8,
        raw_decision_score=0.4,
        top_k_per_direction=5,
        return_explanation=True,
        explanation_enabled=True,
        include_decomposition=True,
        score_fn=score_fn,
    )

    assert explanation["explanation_available"] is True
    assert explanation["top_features_towards_fake"][0]["feature"] == "hello"
    assert explanation["explanation_decomposition"]["estimated_decision_score"] == 0.8
