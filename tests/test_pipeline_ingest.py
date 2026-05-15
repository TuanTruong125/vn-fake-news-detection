from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import ingest


# Build one source CSV row for ingest-stage tests.
def build_source_row(text: str, label: str, domain: str = "example.com") -> dict[str, object]:
    return {"content": text, "label": label, "domain": domain, "published_at": "2026-01-01"}


# Validate helper behavior for optional columns, fallback selection, and path conversion.
def test_ingest_helpers_basic(tmp_path: Path) -> None:
    assert ingest.is_optional_column(None) is True
    assert ingest.is_optional_column("null") is True
    assert ingest.is_optional_column("content") is False

    selected = ingest.select_text_column({"title", "content"}, ["title", "content"])
    assert selected == {"column": "title", "is_fallback": False}

    rel = ingest.to_repo_relative_posix(tmp_path / "data" / "raw" / "internal" / "x.csv", tmp_path)
    assert rel == "data/raw/internal/x.csv"


# Validate ingest frame building creates staging columns and summary metadata.
def test_build_ingest_frame_and_write(tmp_path: Path) -> None:
    repo_root = tmp_path
    df_raw = pd.DataFrame([build_source_row("Tin 1", "0"), build_source_row("Tin 2", "1", domain="")])
    source_cfg = {
        "source_id": "src_1",
        "path": "data/raw/internal/src_1.csv",
        "text_columns": ["content"],
        "label_column": "label",
        "content_type": "news",
        "label_confidence": 0.9,
        "published_at_column": "published_at",
        "url_column": None,
        "domain_column": "domain",
        "source_domain_default": "unknown",
    }

    staging_df, summary_row, warning_lines = ingest.build_ingest_frame(
        df_raw=df_raw,
        source_cfg=source_cfg,
        repo_root=repo_root,
        run_timestamp="2026-04-04T00:00:00+07:00",
        encoding_used="utf-8",
        parser_warnings=["bad_line"],
    )

    assert list(staging_df.columns) == ingest.STAGING_COLUMNS
    assert summary_row["rows_input_after_read"] == 2
    assert summary_row["warnings_count"] >= 1
    assert warning_lines

    out_path = repo_root / "data" / "staging" / "ingest_src_1.csv"
    ingest.write_staging_file(staging_df, out_path)
    assert out_path.exists()


# Validate main emits a failure when no enabled sources are configured.
def test_ingest_main_fails_with_no_enabled_sources(tmp_repo: Path, write_yaml, patch_module_repo_root) -> None:
    write_yaml(tmp_repo / "configs" / "data_sources.yaml", {"sources": []})
    patch_module_repo_root(ingest, "src/pipeline/ingest.py")

    with pytest.raises(ingest.IngestError):
        ingest.main()
