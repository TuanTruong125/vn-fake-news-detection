from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    from src.train.common.split_utils import (
        DatasetValidationError,
        assert_binary_labels,
        assert_no_id_overlap,
        assert_non_empty_split,
        assert_non_empty_text,
        assert_required_columns,
        assert_text_variant_allowed,
        assert_valid_ids,
        summarize_split,
    )
except ModuleNotFoundError:
    from split_utils import (  # type: ignore
        DatasetValidationError,
        assert_binary_labels,
        assert_no_id_overlap,
        assert_non_empty_split,
        assert_non_empty_text,
        assert_required_columns,
        assert_text_variant_allowed,
        assert_valid_ids,
        summarize_split,
    )


class DatasetLoaderError(Exception):
    pass


# Resolve repository root path from current file location.
def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# Load one YAML config file and validate top-level object type.
def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DatasetLoaderError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise DatasetLoaderError(f"Config must be a mapping object: {path}")
    return config


# Resolve train_ml config path using default project config when not provided.
def resolve_config_path(config_path: str | Path | None = None) -> Path:
    if config_path is None:
        return get_repo_root() / "configs" / "train_ml.yaml"
    return Path(config_path)


# Validate and return mandatory data config section from train_ml config.
def get_data_config(config: dict[str, Any]) -> dict[str, Any]:
    data_config = config.get("data")
    if not isinstance(data_config, dict):
        raise DatasetLoaderError("train_ml.yaml must define a 'data' mapping.")

    required_keys = ["train_path", "val_path", "test_path", "label_column", "id_column", "text_variants"]
    missing_keys = [key for key in required_keys if key not in data_config]
    if missing_keys:
        raise DatasetLoaderError(f"train_ml.yaml.data missing required keys: {missing_keys}")

    if not isinstance(data_config["text_variants"], list) or not data_config["text_variants"]:
        raise DatasetLoaderError("train_ml.yaml.data.text_variants must be a non-empty list.")

    return data_config


# Resolve split file paths from config relative to repository root.
def get_split_paths(data_config: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    split_paths = {
        "train": repo_root / str(data_config["train_path"]),
        "val": repo_root / str(data_config["val_path"]),
        "test": repo_root / str(data_config["test_path"]),
    }
    for split_name, split_path in split_paths.items():
        if not split_path.exists():
            raise DatasetLoaderError(f"{split_name}: split file not found: {split_path}")
    return split_paths


# Load one split CSV and run all required fail-fast validations.
def load_split(
    split_name: str,
    split_path: Path,
    text_col: str,
    label_col: str,
    id_col: str,
) -> pd.DataFrame:
    try:
        df = pd.read_csv(split_path)
    except Exception as exc:  # pragma: no cover - explicit CLI error path
        raise DatasetLoaderError(f"{split_name}: cannot read CSV {split_path}. Error: {exc}") from exc

    assert_non_empty_split(df, split_name)
    assert_required_columns(df, [id_col, text_col, label_col], split_name)

    normalized_ids = assert_valid_ids(df, id_col, split_name)
    normalized_text = assert_non_empty_text(df, text_col, split_name)
    normalized_labels = assert_binary_labels(df, label_col, split_name)

    loaded_df = df.copy()
    loaded_df[id_col] = normalized_ids
    loaded_df[text_col] = normalized_text
    loaded_df[label_col] = normalized_labels
    return loaded_df


# Convert one loaded split dataframe to X/y series for model training code.
def to_xy(df: pd.DataFrame, text_col: str, label_col: str) -> tuple[pd.Series, pd.Series]:
    x = df[text_col].fillna("").astype(str)
    y = df[label_col].astype(int)
    return x, y


# Load all train/val/test splits for one selected text variant.
def load_all_splits(
    text_variant: str,
    config_path: str | Path | None = None,
    strict_overlap: bool = True,
    print_summary: bool = True,
) -> dict[str, pd.DataFrame]:
    repo_root = get_repo_root()
    config = load_yaml(resolve_config_path(config_path))
    data_config = get_data_config(config)

    label_col = str(data_config["label_column"]).strip()
    id_col = str(data_config["id_column"]).strip()
    allowed_variants = [str(column).strip() for column in data_config["text_variants"]]
    assert_text_variant_allowed(text_variant, allowed_variants)

    split_paths = get_split_paths(data_config, repo_root)
    split_frames = {
        split_name: load_split(
            split_name=split_name,
            split_path=split_path,
            text_col=text_variant,
            label_col=label_col,
            id_col=id_col,
        )
        for split_name, split_path in split_paths.items()
    }

    assert_no_id_overlap(
        train_df=split_frames["train"],
        val_df=split_frames["val"],
        test_df=split_frames["test"],
        id_col=id_col,
        strict=strict_overlap,
    )

    for split_name, df in split_frames.items():
        summarize_split(
            df=df,
            split_name=split_name,
            label_col=label_col,
            text_col=text_variant,
            print_summary=print_summary,
        )

    return split_frames


# Load all splits and return dictionary with paired X/y tuples.
def get_xy_splits(
    text_variant: str,
    config_path: str | Path | None = None,
    strict_overlap: bool = True,
    print_summary: bool = True,
) -> dict[str, tuple[pd.Series, pd.Series]]:
    config = load_yaml(resolve_config_path(config_path))
    data_config = get_data_config(config)
    label_col = str(data_config["label_column"]).strip()

    split_frames = load_all_splits(
        text_variant=text_variant,
        config_path=config_path,
        strict_overlap=strict_overlap,
        print_summary=print_summary,
    )
    return {
        split_name: to_xy(df, text_variant, label_col)
        for split_name, df in split_frames.items()
    }


# Parse CLI args for quick local validation of dataset loader behavior.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and validate ML dataset splits.")
    parser.add_argument(
        "--text-variant",
        type=str,
        required=True,
        help="Text column variant to load, for example: text_ml_seg or text_ml_seg_lower.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional path to train_ml.yaml. Default: configs/train_ml.yaml",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="If set, overlap IDs across splits become warning-only.",
    )
    return parser.parse_args()


# Run dataset loading from CLI and print compact split diagnostics.
def main() -> None:
    args = parse_args()
    strict_overlap = not args.allow_overlap

    split_frames = load_all_splits(
        text_variant=args.text_variant,
        config_path=args.config_path,
        strict_overlap=strict_overlap,
        print_summary=True,
    )
    print(
        "[DATASET LOADER] Load success | "
        f"text_variant={args.text_variant} | "
        f"train={len(split_frames['train'])} val={len(split_frames['val'])} test={len(split_frames['test'])}"
    )


# Expose CLI-friendly loader entrypoint with fail-fast error reporting.
if __name__ == "__main__":
    try:
        main()
    except (DatasetLoaderError, DatasetValidationError) as exc:
        raise SystemExit(f"[DATASET LOADER ERROR] {exc}")
