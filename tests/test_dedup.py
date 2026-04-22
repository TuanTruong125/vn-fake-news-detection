from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipeline import deduplicate
from src.utils.hashing import md5_text


# Build one quality_filter row with required deduplicate columns.
def build_quality_row(source_id: str, row_index_raw: int, text_clean: str) -> dict[str, object]:
    return {
        "source_id": source_id,
        "row_index_raw": row_index_raw,
        "text_clean": text_clean,
        "label_binary": 0,
        "label_name": "real",
    }


# Validate hash utility determinism for identical and different inputs.
def test_md5_text_is_deterministic() -> None:
    text = "Tin gia tieng Viet"
    assert md5_text(text) == md5_text(text)
    assert md5_text(text) != md5_text(text + " khac")


# Validate global dedup keeps first occurrence deterministically across sources.
def test_deduplicate_main_global_keep_first(
    tmp_repo: Path,
    write_yaml,
    write_csv,
    patch_module_repo_root,
) -> None:
    source_a = "source_a"
    source_b = "source_b"

    write_yaml(
        tmp_repo / "configs" / "data_sources.yaml",
        {
            "sources": [
                {"source_id": source_a, "enabled": True, "path": "data/raw/internal/source_a.csv"},
                {"source_id": source_b, "enabled": True, "path": "data/raw/internal/source_b.csv"},
            ]
        },
    )

    write_csv(
        tmp_repo / "data" / "staging" / f"quality_filter_{source_a}.csv",
        [
            build_quality_row(source_a, 0, "duplicate_text"),
            build_quality_row(source_a, 1, "unique_a"),
        ],
    )
    write_csv(
        tmp_repo / "data" / "staging" / f"quality_filter_{source_b}.csv",
        [
            build_quality_row(source_b, 0, "duplicate_text"),
            build_quality_row(source_b, 1, "unique_b"),
        ],
    )

    patch_module_repo_root(deduplicate, "src/pipeline/deduplicate.py")
    deduplicate.main()

    out_a = pd.read_csv(tmp_repo / "data" / "staging" / f"deduplicate_{source_a}.csv")
    out_b = pd.read_csv(tmp_repo / "data" / "staging" / f"deduplicate_{source_b}.csv")
    master = pd.read_csv(tmp_repo / "data" / "staging" / "deduplicate_master_internal.csv")
    summary = pd.read_csv(tmp_repo / "logs" / "deduplicate_summary.csv")
    removed = pd.read_csv(tmp_repo / "logs" / "deduplicate_removed_rows.csv")

    assert len(out_a) == 2
    assert len(out_b) == 1
    assert len(master) == 3
    assert master["hash_text"].is_unique
    assert (master["sample_id"] == master["hash_text"]).all()
    assert int(summary.loc[summary["source_id"] == "__ALL__", "duplicates_removed"].iloc[0]) == 1
    assert len(removed) == 1


# Validate stage failure when all expected quality_filter source files are missing.
def test_deduplicate_missing_input_file_fails(
    tmp_repo: Path,
    write_yaml,
    patch_module_repo_root,
) -> None:
    write_yaml(
        tmp_repo / "configs" / "data_sources.yaml",
        {
            "sources": [
                {"source_id": "missing_source", "enabled": True, "path": "data/raw/internal/missing.csv"}
            ]
        },
    )

    patch_module_repo_root(deduplicate, "src/pipeline/deduplicate.py")

    try:
        deduplicate.main()
    except SystemExit as exc:
        assert "No valid source data available" in str(exc)
    else:
        raise AssertionError("deduplicate.main() should fail when all input files are missing.")
