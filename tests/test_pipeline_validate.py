from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.pipeline.validate import STAGING_COLUMNS, validate_one_source


# Build one staging row with all required columns for validate stage.
def build_staging_row(
    source_id: str,
    source_file: str,
    text_raw_source: str,
    label_raw: str,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_file": source_file,
        "source_path": f"data/raw/internal/{source_file}",
        "output_path": f"data/staging/ingest_{source_id}.csv",
        "row_index_raw": 0,
        "text_source_column": "content",
        "text_fallback_used": False,
        "text_raw_source": text_raw_source,
        "label_raw": label_raw,
        "published_at_raw": "",
        "url_raw": "",
        "source_domain_raw": "",
        "source_domain": "example.com",
        "content_type": "news",
        "label_confidence": 1.0,
    }


# Write one ingest staging file for a given source.
def write_ingest_file(repo_root: Path, source_id: str, rows: list[dict[str, Any]]) -> None:
    path = repo_root / "data" / "staging" / f"ingest_{source_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


# Validate pass case with valid schema, source consistency, and labels.
def test_validate_one_source_pass(tmp_repo: Path) -> None:
    source_id = "source_a"
    source_file = "source_a.csv"
    source_cfg = {"source_id": source_id, "path": f"data/raw/internal/{source_file}"}
    (tmp_repo / source_cfg["path"]).write_text("placeholder\n", encoding="utf-8")

    rows = [
        build_staging_row(source_id, source_file, "tin that 1", "0"),
        {**build_staging_row(source_id, source_file, "tin gia 2", "1"), "row_index_raw": 1},
    ]
    write_ingest_file(tmp_repo, source_id, rows)

    label_mapping_cfg = {"sources": {source_id: {"mapping": {"0": 0, "1": 1}}}}
    summary, issues, label_counts, issue_rows = validate_one_source(
        source_cfg=source_cfg,
        label_mapping_cfg=label_mapping_cfg,
        allowed_content_types={"news", "social"},
        repo_root=tmp_repo,
        run_timestamp="2026-04-04T00:00:00+07:00",
    )

    assert summary["status"] == "PASS"
    assert summary["rows"] == 2
    assert issues == []
    assert len(label_counts) == 2
    assert issue_rows == []


# Validate fail case when one required column is missing.
def test_validate_missing_required_columns(tmp_repo: Path) -> None:
    source_id = "source_b"
    source_file = "source_b.csv"
    source_cfg = {"source_id": source_id, "path": f"data/raw/internal/{source_file}"}
    (tmp_repo / source_cfg["path"]).write_text("placeholder\n", encoding="utf-8")

    row = build_staging_row(source_id, source_file, "text", "0")
    slim_row = {k: v for k, v in row.items() if k != "label_raw"}
    write_ingest_file(tmp_repo, source_id, [slim_row])

    label_mapping_cfg = {"sources": {source_id: {"mapping": {"0": 0, "1": 1}}}}
    summary, issues, _, _ = validate_one_source(
        source_cfg=source_cfg,
        label_mapping_cfg=label_mapping_cfg,
        allowed_content_types={"news"},
        repo_root=tmp_repo,
        run_timestamp="2026-04-04T00:00:00+07:00",
    )

    missing = sorted(set(STAGING_COLUMNS) - set([k for k in slim_row.keys()]))
    assert summary["status"] == "FAIL"
    assert any(x["issue_code"] == "missing_required_columns" for x in issues)
    assert missing == ["label_raw"]


# Validate fail case when label values are outside source mapping.
def test_validate_label_out_of_mapping(tmp_repo: Path) -> None:
    source_id = "source_c"
    source_file = "source_c.csv"
    source_cfg = {"source_id": source_id, "path": f"data/raw/internal/{source_file}"}
    (tmp_repo / source_cfg["path"]).write_text("placeholder\n", encoding="utf-8")

    rows = [build_staging_row(source_id, source_file, "valid text", "UNKNOWN")]
    write_ingest_file(tmp_repo, source_id, rows)

    label_mapping_cfg = {"sources": {source_id: {"mapping": {"0": 0, "1": 1}}}}
    summary, issues, _, issue_rows = validate_one_source(
        source_cfg=source_cfg,
        label_mapping_cfg=label_mapping_cfg,
        allowed_content_types={"news"},
        repo_root=tmp_repo,
        run_timestamp="2026-04-04T00:00:00+07:00",
    )

    assert summary["status"] == "FAIL"
    assert any(x["issue_code"] == "label_out_of_mapping" for x in issues)
    assert len(issue_rows) == 1


# Validate fail case when source_id inside file differs from config source_id.
def test_validate_invalid_source_id(tmp_repo: Path) -> None:
    source_id = "source_d"
    source_file = "source_d.csv"
    source_cfg = {"source_id": source_id, "path": f"data/raw/internal/{source_file}"}
    (tmp_repo / source_cfg["path"]).write_text("placeholder\n", encoding="utf-8")

    wrong = build_staging_row(source_id, source_file, "text", "0")
    wrong["source_id"] = "unexpected_source"
    write_ingest_file(tmp_repo, source_id, [wrong])

    label_mapping_cfg = {"sources": {source_id: {"mapping": {"0": 0}}}}
    summary, issues, _, _ = validate_one_source(
        source_cfg=source_cfg,
        label_mapping_cfg=label_mapping_cfg,
        allowed_content_types={"news"},
        repo_root=tmp_repo,
        run_timestamp="2026-04-04T00:00:00+07:00",
    )

    assert summary["status"] == "FAIL"
    assert any(x["issue_code"] == "invalid_source_id" for x in issues)


# Validate fail case when all text rows are empty after trimming.
def test_validate_text_all_empty(tmp_repo: Path) -> None:
    source_id = "source_e"
    source_file = "source_e.csv"
    source_cfg = {"source_id": source_id, "path": f"data/raw/internal/{source_file}"}
    (tmp_repo / source_cfg["path"]).write_text("placeholder\n", encoding="utf-8")

    rows = [
        build_staging_row(source_id, source_file, "", "0"),
        {**build_staging_row(source_id, source_file, "   ", "1"), "row_index_raw": 1},
    ]
    write_ingest_file(tmp_repo, source_id, rows)

    label_mapping_cfg = {"sources": {source_id: {"mapping": {"0": 0, "1": 1}}}}
    summary, issues, _, issue_rows = validate_one_source(
        source_cfg=source_cfg,
        label_mapping_cfg=label_mapping_cfg,
        allowed_content_types={"news"},
        repo_root=tmp_repo,
        run_timestamp="2026-04-04T00:00:00+07:00",
    )

    assert summary["status"] == "FAIL"
    assert any(x["issue_code"] == "text_all_empty" for x in issues)
    assert issue_rows == []
