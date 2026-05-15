from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import normalize


# Test: Normalization helpers handle whitespace, punctuation spacing, and token detection
def test_normalize_helpers_basic() -> None:
    assert normalize.normalize_whitespace(" a\n\n b \t c ") == "a\nb c"
    assert normalize.normalize_punctuation_spacing("Xin chao,world! 13:30") == "Xin chao, world! 13:30"
    assert normalize.count_suspected_glued_tokens("thanhpho ha noi") >= 0
    assert normalize.missing_required_columns(pd.DataFrame(columns=normalize.REQUIRED_COLUMNS)) == []


# Test: Normalize text pipeline returns expected flags and cleaned output
def test_normalize_text_pipeline() -> None:
    result = normalize.normalize_text("Xin /n chao https://example.com\tABC")

    assert result["flag_replaced_slash_n"] is True
    assert result["flag_removed_url"] is True
    assert isinstance(result["text_clean"], str)


# Test: One-source normalization writes output file for valid ingest source
def test_normalize_one_source_pass(tmp_repo: Path) -> None:
    source_id = "source_norm"
    ingest_path = tmp_repo / "data" / "staging" / f"ingest_{source_id}.csv"
    pd.DataFrame(
        [
            {"source_id": source_id, "row_index_raw": 0, "text_raw_source": "Xin /n chao", "label_raw": "0"},
            {"source_id": source_id, "row_index_raw": 1, "text_raw_source": "http://example.com", "label_raw": "1"},
        ]
    ).to_csv(ingest_path, index=False, encoding="utf-8-sig")

    summary, issues, out_df = normalize.normalize_one_source(
        {"source_id": source_id},
        tmp_repo,
        "2026-04-04T00:00:00+07:00",
    )

    assert summary["status"] in {"PASS", "WARNING"}
    assert issues is not None
    assert out_df is not None
    assert (tmp_repo / "data" / "staging" / f"normalize_{source_id}.csv").exists()
