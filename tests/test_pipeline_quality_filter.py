from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import quality_filter


# Validate text preview helpers and removed-row logging.
def test_quality_filter_helpers_basic() -> None:
    assert quality_filter.to_text_preview(None) == "<EMPTY>"
    assert quality_filter.to_text_preview("abc") == "abc"

    removed_df = pd.DataFrame(
        {
            "row_index_raw": [0],
            "remove_reason": ["text_empty"],
            "text_length": [0],
            "label_binary": [0],
            "text_clean": [""],
        }
    )
    log_df = quality_filter.build_removed_rows_log(removed_df, "2026-04-04T00:00:00+07:00", "src")
    assert list(log_df.columns) == quality_filter.REMOVED_ROW_COLUMNS


# Validate one-source filtering keeps valid rows and records removed rows.
def test_filter_one_source_pass(tmp_repo: Path) -> None:
    source_id = "src_qf"
    input_path = tmp_repo / "data" / "staging" / f"map_labels_{source_id}.csv"
    pd.DataFrame(
        [
            {"source_id": source_id, "row_index_raw": 0, "text_clean": "valid text enough for quality filtering", "label_binary": 0, "label_name": "real"},
            {"source_id": source_id, "row_index_raw": 1, "text_clean": "another valid text enough for filtering", "label_binary": 1, "label_name": "fake"},
            {"source_id": source_id, "row_index_raw": 2, "text_clean": "third valid text enough for filtering", "label_binary": 0, "label_name": "real"},
            {"source_id": source_id, "row_index_raw": 3, "text_clean": "x", "label_binary": 1, "label_name": "fake"},
        ]
    ).to_csv(input_path, index=False, encoding="utf-8-sig")

    summary, issues, kept_df, removed_log_df = quality_filter.filter_one_source(
        {"source_id": source_id},
        "2026-04-04T00:00:00+07:00",
        tmp_repo,
    )

    assert summary["rows_input"] == 4
    assert summary["rows_removed"] == 1
    assert kept_df.shape[0] == 3
    assert removed_log_df.shape[0] == 1
    assert issues == []


# Validate main fails when there are no enabled sources.
def test_quality_filter_main_fails_when_no_enabled_sources(tmp_repo: Path, write_yaml, patch_module_repo_root) -> None:
    write_yaml(tmp_repo / "configs" / "data_sources.yaml", {"sources": []})
    patch_module_repo_root(quality_filter, "src/pipeline/quality_filter.py")

    with pytest.raises(quality_filter.QualityFilterError):
        quality_filter.main()
