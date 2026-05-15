from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import split_data
from src.pipeline.split_data import SplitDataError, parse_split_config


# Build split config payload used by split_data main in tests.
def build_split_cfg() -> dict:
    return {
        "split": {
            "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            "random_state": 42,
            "shuffle": True,
            "stratify_by": "label_binary",
            "apply_to": "internal",
            "deduplicate_before_split": True,
            "leakage_key": "hash_text",
            "internal_split_values": ["train", "val", "test"],
            "external_split_value": "external_test",
        }
    }


# Build one master row with required split columns.
def build_master_row(idx: int, label_binary: int) -> dict:
    return {
        "sample_id": f"id_{idx}",
        "text_raw": f"raw_{idx}",
        "text_clean": f"text_{idx}",
        "hash_text": f"hash_{idx}",
        "label_binary": label_binary,
        "label_name": "real" if label_binary == 0 else "fake",
        "source_file": "src.csv",
        "source_domain": "example.com",
        "content_type": "news",
        "published_at": "",
        "label_confidence": 1.0,
        "text_length": 20,
        "split": "",
    }


# Validate happy path split ratios, split assignment, and zero leakage.
def test_split_main_pass(
    tmp_repo: Path,
    write_yaml,
    write_csv,
    patch_module_repo_root,
) -> None:
    write_yaml(tmp_repo / "configs" / "split.yaml", build_split_cfg())

    rows = [build_master_row(i, 0 if i < 50 else 1) for i in range(100)]
    write_csv(tmp_repo / "data" / "processed" / "master_dataset_v1.csv", rows)

    patch_module_repo_root(split_data, "src/pipeline/split_data.py")
    split_data.main()

    train_df = pd.read_csv(tmp_repo / "data" / "processed" / "train.csv")
    val_df = pd.read_csv(tmp_repo / "data" / "processed" / "val.csv")
    test_df = pd.read_csv(tmp_repo / "data" / "processed" / "test.csv")
    master_df = pd.read_csv(tmp_repo / "data" / "processed" / "master_dataset_v1.csv")
    summary_df = pd.read_csv(tmp_repo / "logs" / "split_summary.csv")

    assert len(train_df) + len(val_df) + len(test_df) == 100
    assert abs(len(train_df) / 100 - 0.8) <= 0.02
    assert abs(len(val_df) / 100 - 0.1) <= 0.02
    assert abs(len(test_df) / 100 - 0.1) <= 0.02
    assert set(master_df["split"].unique().tolist()) == {"train", "val", "test"}

    train_hash = set(train_df["hash_text"])
    val_hash = set(val_df["hash_text"])
    test_hash = set(test_df["hash_text"])
    assert len(train_hash & val_hash) == 0
    assert len(train_hash & test_hash) == 0
    assert len(val_hash & test_hash) == 0

    assert summary_df.iloc[0]["status"] == "PASS"


# Validate pre-split guard failure when leakage key is not unique.
def test_split_fails_when_hash_not_unique(
    tmp_repo: Path,
    write_yaml,
    write_csv,
    patch_module_repo_root,
) -> None:
    write_yaml(tmp_repo / "configs" / "split.yaml", build_split_cfg())

    rows = [build_master_row(i, 0 if i < 50 else 1) for i in range(100)]
    rows[1]["hash_text"] = rows[0]["hash_text"]
    write_csv(tmp_repo / "data" / "processed" / "master_dataset_v1.csv", rows)

    patch_module_repo_root(split_data, "src/pipeline/split_data.py")
    with pytest.raises(SystemExit):
        split_data.main()

    issues_df = pd.read_csv(tmp_repo / "logs" / "split_issues.csv")
    assert "hash_not_unique_before_split" in set(issues_df["issue_code"].tolist())


# Validate stratified split guard failure when minority class is too small.
def test_split_fails_when_min_class_too_small(
    tmp_repo: Path,
    write_yaml,
    write_csv,
    patch_module_repo_root,
) -> None:
    write_yaml(tmp_repo / "configs" / "split.yaml", build_split_cfg())

    rows = [build_master_row(i, 0) for i in range(20)] + [build_master_row(100 + i, 1) for i in range(2)]
    write_csv(tmp_repo / "data" / "processed" / "master_dataset_v1.csv", rows)

    patch_module_repo_root(split_data, "src/pipeline/split_data.py")
    with pytest.raises(SystemExit):
        split_data.main()

    issues_df = pd.read_csv(tmp_repo / "logs" / "split_issues.csv")
    assert "not_enough_samples_for_stratified_split" in set(issues_df["issue_code"].tolist())


# Validate split config parser rejects ratio sums different from 1.0.
def test_parse_split_config_fails_when_ratios_not_sum_to_one() -> None:
    bad_cfg = build_split_cfg()
    bad_cfg["split"]["ratios"] = {"train": 0.9, "val": 0.1, "test": 0.2}
    with pytest.raises(SplitDataError):
        parse_split_config(bad_cfg)