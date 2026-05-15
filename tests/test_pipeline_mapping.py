from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.pipeline.map_labels import map_one_source


# Build: Normalized row structure that map_labels stage can process
def build_normalize_row(source_id: str, label_raw: str, row_index_raw: int = 0) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "row_index_raw": row_index_raw,
        "text_raw_source": "raw text",
        "text_raw": "raw text",
        "text_clean": "clean text",
        "label_raw": label_raw,
    }


# Create: Normalize input file for one source with test data
def write_normalize_file(repo_root: Path, source_id: str, rows: list[dict[str, Any]]) -> None:
    path = repo_root / "data" / "staging" / f"normalize_{source_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


# Test: Numeric raw labels are mapped to binary labels with pass status
def test_map_one_source_pass_numeric_labels(tmp_repo: Path) -> None:
    source_id = "src_num"
    rows = [
        build_normalize_row(source_id, "0", 0),
        build_normalize_row(source_id, "1", 1),
        build_normalize_row(source_id, "0", 2),
    ]
    write_normalize_file(tmp_repo, source_id, rows)

    source_cfg = {"source_id": source_id}
    mapping_cfg = {
        "global": {"label_binary_to_name": {0: "real", 1: "fake"}},
        "sources": {source_id: {"mapping": {"0": 0, "1": 1}}},
    }

    summary, issues = map_one_source(
        source_cfg=source_cfg,
        mapping_cfg=mapping_cfg,
        run_timestamp="2026-04-04T00:00:00+07:00",
        repo_root=tmp_repo,
    )

    out_df = pd.read_csv(tmp_repo / "data" / "staging" / f"map_labels_{source_id}.csv")
    assert summary["status"] == "PASS"
    assert summary["rows_output"] == 3
    assert issues == []
    assert set(out_df["label_binary"].tolist()) == {0, 1}
    assert set(out_df["label_name"].tolist()) == {"real", "fake"}


# Validate pass case for boolean-like raw labels mapped to binary labels.
def test_map_one_source_pass_bool_labels(tmp_repo: Path) -> None:
    source_id = "src_bool"
    rows = [
        build_normalize_row(source_id, "False", 0),
        build_normalize_row(source_id, "True", 1),
    ]
    write_normalize_file(tmp_repo, source_id, rows)

    source_cfg = {"source_id": source_id}
    mapping_cfg = {"sources": {source_id: {"mapping": {"False": 0, "True": 1}}}}

    summary, issues = map_one_source(
        source_cfg=source_cfg,
        mapping_cfg=mapping_cfg,
        run_timestamp="2026-04-04T00:00:00+07:00",
        repo_root=tmp_repo,
    )
    out_df = pd.read_csv(tmp_repo / "data" / "staging" / f"map_labels_{source_id}.csv")

    assert summary["status"] == "PASS"
    assert issues == []
    assert out_df["label_binary"].tolist() == [0, 1]
    assert out_df["label_name"].tolist() == ["real", "fake"]


# Validate fail case when source mapping entry is missing in config.
def test_map_one_source_missing_mapping(tmp_repo: Path) -> None:
    source_id = "src_missing_map"
    write_normalize_file(tmp_repo, source_id, [build_normalize_row(source_id, "0")])

    summary, issues = map_one_source(
        source_cfg={"source_id": source_id},
        mapping_cfg={"sources": {}},
        run_timestamp="2026-04-04T00:00:00+07:00",
        repo_root=tmp_repo,
    )

    assert summary["status"] == "FAIL"
    assert any(x["issue_code"] == "missing_label_mapping" for x in issues)


# Validate fail cases for null labels and labels outside mapping.
def test_map_one_source_label_null_and_out_of_mapping(tmp_repo: Path) -> None:
    source_id = "src_bad_labels"
    rows = [
        build_normalize_row(source_id, "", 0),
        build_normalize_row(source_id, "weird_label", 1),
    ]
    write_normalize_file(tmp_repo, source_id, rows)

    summary, issues = map_one_source(
        source_cfg={"source_id": source_id},
        mapping_cfg={"sources": {source_id: {"mapping": {"0": 0, "1": 1}}}},
        run_timestamp="2026-04-04T00:00:00+07:00",
        repo_root=tmp_repo,
    )

    issue_codes = {x["issue_code"] for x in issues}
    assert summary["status"] == "FAIL"
    assert "label_null" in issue_codes
    assert "label_out_of_mapping" in issue_codes