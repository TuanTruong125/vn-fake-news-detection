from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.ml.validate_ml_config import (
    MlConfigValidationError,
    validate_data_section,
    validate_models_section,
    validate_ngram_range,
    validate_optional_run_section,
    validate_root_structure,
)


# Build a minimal valid models section used across ML config tests.
def build_models_section() -> dict:
    return {
        "models": {
            "linear_svm": {
                "enabled": True,
                "params": {
                    "C": [1.0],
                    "class_weight": ["balanced"],
                    "random_state": [42],
                },
            },
            "logistic_regression": {
                "enabled": False,
                "params": {
                    "C": [1.0],
                    "solver": ["liblinear"],
                    "max_iter": [1000],
                    "class_weight": ["balanced"],
                    "random_state": [42],
                },
            },
            "multinomial_nb": {
                "enabled": False,
                "params": {
                    "alpha": [1.0],
                    "fit_prior": [True],
                },
            },
        }
    }


# Validate ngram range accepts valid input and rejects invalid ordering.
def test_validate_ngram_range_pass_and_fail() -> None:
    validate_ngram_range([1, 2], "vectorizers.word_tfidf.ngram_range")

    with pytest.raises(MlConfigValidationError):
        validate_ngram_range([3, 1], "vectorizers.word_tfidf.ngram_range")


# Validate root structure rejects missing required top-level sections.
def test_validate_root_structure_missing_sections() -> None:
    with pytest.raises(MlConfigValidationError):
        validate_root_structure({"version": 1, "data": {}})


# Validate models section fails when no model is enabled.
def test_validate_models_section_requires_enabled_model() -> None:
    cfg = build_models_section()
    cfg["models"]["linear_svm"]["enabled"] = False

    with pytest.raises(MlConfigValidationError):
        validate_models_section(cfg)


# Validate optional run section cross-checks default model against models section.
def test_validate_optional_run_section_fails_unknown_default_model() -> None:
    cfg = {
        "models": build_models_section()["models"],
        "feature_sets": [{"name": "word_char", "type": "combined", "parts": ["word_tfidf", "char_tfidf"]}],
        "data": {"text_variants": ["text_ml_seg", "text_ml_seg_lower"]},
        "run": {
            "default": {
                "model": "unknown_model",
                "feature_set": "word_char",
                "text_variant": "text_ml_seg",
            }
        },
    }

    with pytest.raises(MlConfigValidationError):
        validate_optional_run_section(cfg)


# Validate data section header cross-check succeeds with complete CSV columns.
def test_validate_data_section_header_cross_check_pass(tmp_path: Path) -> None:
    repo_root = tmp_path
    data_dir = repo_root / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)

    columns = ["sample_id", "label_binary", "text_ml_seg", "text_ml_seg_lower"]
    for split_name in ["train_ml.csv", "val_ml.csv", "test_ml.csv"]:
        pd.DataFrame(columns=columns).to_csv(data_dir / split_name, index=False, encoding="utf-8-sig")

    cfg = {
        "data": {
            "train_path": "data/processed/train_ml.csv",
            "val_path": "data/processed/val_ml.csv",
            "test_path": "data/processed/test_ml.csv",
            "label_column": "label_binary",
            "id_column": "sample_id",
            "text_variants": ["text_ml_seg", "text_ml_seg_lower"],
        }
    }
    warnings: list[str] = []

    validate_data_section(cfg, repo_root, warnings)

    assert warnings == []


# Validate data section fails when required text variant column is absent.
def test_validate_data_section_missing_required_column(tmp_path: Path) -> None:
    repo_root = tmp_path
    data_dir = repo_root / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Missing text_ml_seg_lower on purpose.
    pd.DataFrame(columns=["sample_id", "label_binary", "text_ml_seg"]).to_csv(
        data_dir / "train_ml.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["sample_id", "label_binary", "text_ml_seg"]).to_csv(
        data_dir / "val_ml.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(columns=["sample_id", "label_binary", "text_ml_seg"]).to_csv(
        data_dir / "test_ml.csv", index=False, encoding="utf-8-sig"
    )

    cfg = {
        "data": {
            "train_path": "data/processed/train_ml.csv",
            "val_path": "data/processed/val_ml.csv",
            "test_path": "data/processed/test_ml.csv",
            "label_column": "label_binary",
            "id_column": "sample_id",
            "text_variants": ["text_ml_seg", "text_ml_seg_lower"],
        }
    }

    with pytest.raises(MlConfigValidationError):
        validate_data_section(cfg, repo_root, [])
