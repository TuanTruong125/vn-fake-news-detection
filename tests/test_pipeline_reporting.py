from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipeline import reporting


# Validate reporting helpers used by the final pipeline stage.
def test_reporting_helpers_basic() -> None:
    df = pd.DataFrame({"run_timestamp": ["2026-04-01T00:00:00+07:00", "2026-04-02T00:00:00+07:00"], "value": [1, 2]})
    latest = reporting.select_latest_run(df)
    assert latest.iloc[0]["value"] == 2

    assert reporting.determine_status([]) == "PASS"
    assert "# Pipeline Overview" in reporting.build_pipeline_overview_md()


# Validate data dictionary generation uses the expected 13-column contract.
def test_build_data_dictionary_md() -> None:
    master_df = pd.DataFrame(columns=[
        "sample_id", "text_raw", "text_clean", "hash_text", "label_binary", "label_name", "source_file", "source_domain", "content_type", "published_at", "label_confidence", "text_length", "split",
    ])
    md = reporting.build_data_dictionary_md(master_df)
    assert "# Data Dictionary" in md
    assert "sample_id" in md

