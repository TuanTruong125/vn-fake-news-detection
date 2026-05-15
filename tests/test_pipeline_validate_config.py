from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import validate_config


# Build a minimal valid configuration set for config-validation tests.
def build_config_payloads(tmp_repo: Path) -> tuple[dict, dict, dict, dict]:
    data_dir = tmp_repo / "data" / "raw" / "internal"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "source_a.csv"
    pd.DataFrame(columns=["content", "label", "published_at", "url", "domain"]).to_csv(
        csv_path, index=False, encoding="utf-8-sig"
    )

    data_sources = {
        "sources": [
            {
                "source_id": "source_a",
                "path": "data/raw/internal/source_a.csv",
                "text_columns": ["content"],
                "label_column": "label",
                "content_type": "news",
                "label_confidence": 1.0,
                "enabled": True,
            }
        ]
    }
    label_mapping = {"sources": {"source_a": {"label_column": "label", "mapping": {"0": 0, "1": 1}}}}
    schema = {
        "master_schema": {
            "columns": [{"name": name} for name in validate_config.EXPECTED_MASTER_COLUMNS]
        },
        "constraints": {"split_allowed": ["train", "val", "test", "external_test"], "label_binary_allowed": [0, 1]},
    }
    split = {
        "split": {
            "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
            "random_state": 42,
            "shuffle": True,
            "stratify_by": "label_binary",
            "deduplicate_before_split": True,
            "leakage_key": "hash_text",
        }
    }
    return data_sources, label_mapping, schema, split


# Validate each section individually for the happy path.
def test_validate_sections_happy_path(tmp_repo: Path) -> None:
    data_sources, label_mapping, schema, split = build_config_payloads(tmp_repo)

    declared, enabled, label_cols = validate_config.validate_data_sources(data_sources, tmp_repo)
    warnings = validate_config.validate_label_mapping(label_mapping, declared, enabled, label_cols)

    validate_config.validate_schema(schema)
    validate_config.validate_split(split)

    assert declared == ["source_a"]
    assert enabled == ["source_a"]
    assert warnings == []


# Validate main fails early when minimum source count is not met.
def test_validate_config_main_fails_with_too_few_sources(tmp_repo: Path, write_yaml, patch_module_repo_root) -> None:
    data_sources, label_mapping, schema, split = build_config_payloads(tmp_repo)
    write_yaml(tmp_repo / "configs" / "data_sources.yaml", data_sources)
    write_yaml(tmp_repo / "configs" / "label_mapping.yaml", label_mapping)
    write_yaml(tmp_repo / "configs" / "schema.yaml", schema)
    write_yaml(tmp_repo / "configs" / "split.yaml", split)

    patch_module_repo_root(validate_config, "src/pipeline/validate_config.py")

    with pytest.raises(validate_config.ConfigValidationError):
        validate_config.main()

