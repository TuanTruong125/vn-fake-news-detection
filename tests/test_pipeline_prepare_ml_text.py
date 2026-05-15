from __future__ import annotations

from pathlib import Path
from collections import Counter

import pandas as pd
import pytest

from src.pipeline import prepare_ml_text


# Validate configuration parsing and helper transforms used by prepare_ml_text.
def test_prepare_ml_text_helpers_basic(tmp_repo: Path) -> None:
    cfg = prepare_ml_text.parse_ml_config(
        {
            "ml_text": {
                "enabled": True,
                "input_files": ["data/processed/train.csv"],
                "text_input_column": "text_clean",
                "output_suffix": "_ml",
                "primary_variant": "lower",
                "segment_backend": "regex",
            }
        },
        tmp_repo,
    )
    assert cfg["enabled"] is True
    assert prepare_ml_text.ml_output_path(Path("a.csv"), "_ml").name == "a_ml.csv"
    assert prepare_ml_text.is_likely_presegmented("tin_nong abc_def ghi_jkl") is True
    assert prepare_ml_text.light_normalize_text("12,500 covid 19") == "12500 covid_19"
    assert prepare_ml_text.pre_clean_text("#TinNoiBo https://x.com 😊", "social", True, True, True, False).startswith("HASHTAG_")
    text_ml, before_n, after_n, tokens = prepare_ml_text.tokenize_and_filter(
        "tin_nong covid_19 123",
        lowercase=True,
        keep_numbers=True,
        min_token_length=1,
        remove_stopwords=False,
        stopwords=set(),
    )
    assert before_n >= after_n >= 1
    assert isinstance(tokens, list)
    assert text_ml


# Validate one-file processing in regex mode writes the derived ML-text file.
def test_process_one_file_regex(tmp_repo: Path) -> None:
    input_rel = "data/processed/train.csv"
    input_path = tmp_repo / input_rel
    input_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"text_clean": ["Tin nóng covid-19", "Thêm 12,500 đồng"], "content_type": ["news", "social"]}
    ).to_csv(input_path, index=False, encoding="utf-8-sig")

    cfg = {
        "enabled": True,
        "input_files": [input_rel],
        "text_input_column": "text_clean",
        "output_suffix": "_ml",
        "replace_underscore_with_space": True,
        "remove_url_tokens": True,
        "keep_numbers": True,
        "min_token_length": 1,
        "remove_stopwords": False,
        "stopwords_path": None,
        "log_top_tokens": 5,
        "segment_backend": "regex",
        "segment_model_dir": None,
        "primary_variant": "lower",
        "preserve_social_emoji": True,
        "remove_news_emoji": True,
        "respect_presegmented_input": True,
    }
    summary, issues, seg_counter, lower_counter = prepare_ml_text.process_one_file(
        input_rel_path=input_rel,
        cfg=cfg,
        repo_root=tmp_repo,
        run_timestamp="2026-04-04T00:00:00+07:00",
        stopwords=set(),
        segment_backend_actual="regex",
        segment_fn=lambda text: text,
        stage_warnings=[],
    )

    assert summary["status"] in {"PASS", "WARNING"}
    assert issues == [] or isinstance(issues, list)
    assert isinstance(seg_counter, Counter)
    assert isinstance(lower_counter, Counter)
    assert (tmp_repo / "data" / "processed" / "train_ml.csv").exists()

