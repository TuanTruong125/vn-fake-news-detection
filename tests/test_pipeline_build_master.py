from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import build_master


# Validate master-frame construction and strict validation rules.
def test_build_master_frame_and_validate() -> None:
    source_df = pd.DataFrame(
        {
            "sample_id": ["id_1", "id_2"],
            "hash_text": ["id_1", "id_2"],
            "text_clean": ["text a", "text b"],
            "text_raw_source": ["raw a", "raw b"],
            "label_binary": [0, 1],
            "label_name": ["real", "fake"],
            "source_file": ["s.csv", "s.csv"],
            "source_domain": ["example.com", "example.com"],
            "content_type": ["news", "social"],
            "label_confidence": [1.0, 1.0],
            "published_at_raw": ["2026-01-01", "2026-01-02"],
        }
    )
    master_df = build_master.build_master_frame(source_df)
    issues: list[dict[str, object]] = []
    build_master.validate_master_frame(master_df, source_df, "2026-04-04T00:00:00+07:00", issues)

    assert list(master_df.columns) == build_master.EXPECTED_MASTER_COLUMNS
    assert master_df["split"].eq("").all()
    assert issues == []


# Validate main writes a summary and fails when the source file is missing.
def test_build_master_main_fails_when_input_missing(tmp_repo: Path, patch_module_repo_root) -> None:
    patch_module_repo_root(build_master, "src/pipeline/build_master.py")

    with pytest.raises(build_master.BuildMasterError):
        build_master.main()

