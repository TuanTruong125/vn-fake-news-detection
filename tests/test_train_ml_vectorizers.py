from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.ml.vectorizers import (
    VectorizerConfigError,
    build_feature_pipeline,
    get_feature_set_names,
    get_text_variants,
    normalize_ngram_range,
    vectorizer_meta_to_json,
)


# Build a compact train_ml-like config payload for vectorizer tests.
def build_vectorizer_config() -> dict:
    return {
        "data": {
            "text_variants": ["text_ml_seg", "text_ml_seg_lower"],
        },
        "vectorizers": {
            "word_tfidf": {
                "ngram_range": [1, 2],
                "min_df": 3,
                "max_df": 0.95,
                "max_features": 1000,
                "sublinear_tf": True,
            },
            "char_tfidf": {
                "ngram_range": [3, 5],
                "min_df": 3,
                "max_df": 0.98,
                "max_features": 1200,
                "sublinear_tf": True,
            },
        },
        "feature_sets": [
            {"name": "word", "type": "word_tfidf"},
            {"name": "word_char", "type": "combined", "parts": ["word_tfidf", "char_tfidf"]},
        ],
    }


# Validate text variants extraction reads normalized non-empty columns.
def test_get_text_variants_pass() -> None:
    cfg = build_vectorizer_config()
    assert get_text_variants(cfg) == ["text_ml_seg", "text_ml_seg_lower"]


# Validate feature-set name extraction preserves configured order.
def test_get_feature_set_names_pass() -> None:
    cfg = build_vectorizer_config()
    assert get_feature_set_names(cfg) == ["word", "word_char"]


# Validate char ngram upper bound is clipped to 4 when config requests >4.
def test_normalize_ngram_range_char_clip() -> None:
    min_n, max_n, clipped = normalize_ngram_range([3, 6], "char", "char_tfidf")
    assert (min_n, max_n, clipped) == (3, 4, True)


# Validate single feature-set pipeline builds one TfidfVectorizer instance.
def test_build_feature_pipeline_single_vectorizer() -> None:
    cfg = build_vectorizer_config()
    pipeline, metadata = build_feature_pipeline("word", cfg)

    assert isinstance(pipeline, TfidfVectorizer)
    assert metadata["feature_set_type"] == "single"
    assert metadata["parts"][0]["vectorizer_key"] == "word_tfidf"


# Validate combined feature-set pipeline builds FeatureUnion with two parts.
def test_build_feature_pipeline_combined() -> None:
    cfg = build_vectorizer_config()
    pipeline, metadata = build_feature_pipeline("word_char", cfg)

    assert isinstance(pipeline, FeatureUnion)
    assert metadata["feature_set_type"] == "combined"
    assert len(metadata["parts"]) == 2
    assert metadata["parts"][1]["char_ngram_clipped_to_4"] is True


# Validate vectorizer metadata serialization returns valid compact JSON text.
def test_vectorizer_meta_to_json_roundtrip() -> None:
    payload = {
        "feature_set_name": "word_char",
        "feature_set_type": "combined",
        "parts": [{"vectorizer_key": "word_tfidf"}],
    }

    raw = vectorizer_meta_to_json(payload)
    assert " " not in raw
    assert json.loads(raw)["feature_set_name"] == "word_char"


# Validate feature pipeline builder rejects unknown feature-set names.
def test_build_feature_pipeline_unknown_feature_set() -> None:
    cfg = build_vectorizer_config()
    with pytest.raises(VectorizerConfigError):
        build_feature_pipeline("unknown_set", cfg)
