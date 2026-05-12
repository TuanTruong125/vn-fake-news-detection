from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Any


try:
    from src.train.common.dataset_loader import DatasetLoaderError, get_xy_splits
    from src.train.common.split_utils import DatasetValidationError
    from src.train.ml.vectorizers import (
        VectorizerConfigError,
        build_feature_pipeline,
        get_feature_set_names,
        get_text_variants,
        load_train_ml_config,
        resolve_config_path,
        vectorizer_meta_to_json,
    )
except ModuleNotFoundError:
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    COMMON_DIR = CURRENT_DIR.parent / "common"
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    if str(COMMON_DIR) not in sys.path:
        sys.path.insert(0, str(COMMON_DIR))

    from dataset_loader import DatasetLoaderError, get_xy_splits  # type: ignore
    from split_utils import DatasetValidationError  # type: ignore
    from vectorizers import (  # type: ignore
        VectorizerConfigError,
        build_feature_pipeline,
        get_feature_set_names,
        get_text_variants,
        load_train_ml_config,
        resolve_config_path,
        vectorizer_meta_to_json,
    )


SUMMARY_COLUMNS = [
    "run_timestamp",
    "config_path",
    "text_variant",
    "feature_set",
    "status",
    "message",
    "train_rows",
    "val_rows",
    "test_rows",
    "train_features",
    "train_nnz",
    "val_nnz",
    "test_nnz",
    "fit_seconds",
    "transform_seconds",
    "vectorizer_meta_json",
]


# Error class for feature pipeline issues.
class FeaturePipelineError(Exception):
    pass


# Resolve repository root path from current file location.
def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# Build run timestamp string in local timezone-friendly format.
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# Parse CLI arguments for feature pipeline execution scope.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML feature pipeline fit/transform for dataset splits.")
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional path to train_ml.yaml. Default: configs/train_ml.yaml",
    )
    parser.add_argument(
        "--text-variant",
        type=str,
        default=None,
        help="Optional one text variant to run. If omitted, run all variants in config.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default=None,
        help="Optional one feature set to run. If omitted, run all feature sets in config.",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="If set, split overlap check becomes warning-only.",
    )
    return parser.parse_args()


# Validate requested run scope and return selected variants and feature sets.
def resolve_run_scope(
    config: dict[str, Any],
    selected_text_variant: str | None,
    selected_feature_set: str | None,
) -> tuple[list[str], list[str]]:
    all_text_variants = get_text_variants(config)
    all_feature_sets = get_feature_set_names(config)

    if selected_text_variant is None:
        final_text_variants = all_text_variants
    else:
        if selected_text_variant not in all_text_variants:
            raise FeaturePipelineError(
                f"text_variant='{selected_text_variant}' is not declared in train_ml config: {all_text_variants}"
            )
        final_text_variants = [selected_text_variant]

    if selected_feature_set is None:
        final_feature_sets = all_feature_sets
    else:
        if selected_feature_set not in all_feature_sets:
            raise FeaturePipelineError(
                f"feature_set='{selected_feature_set}' is not declared in train_ml config: {all_feature_sets}"
            )
        final_feature_sets = [selected_feature_set]

    return final_text_variants, final_feature_sets


# Fit feature pipeline on train and transform val/test while collecting run stats.
def run_one_combo(
    text_variant: str,
    feature_set: str,
    config: dict[str, Any],
    config_path: Path,
    strict_overlap: bool,
) -> dict[str, Any]:
    run_timestamp = now_iso()

    xy_splits = get_xy_splits(
        text_variant=text_variant,
        config_path=config_path,
        strict_overlap=strict_overlap,
        print_summary=False,
    )
    x_train, _ = xy_splits["train"]
    x_val, _ = xy_splits["val"]
    x_test, _ = xy_splits["test"]

    pipeline, vectorizer_meta = build_feature_pipeline(feature_set_name=feature_set, config=config)

    fit_start = time.perf_counter()
    x_train_vec = pipeline.fit_transform(x_train)
    fit_seconds = round(time.perf_counter() - fit_start, 6)

    transform_start = time.perf_counter()
    x_val_vec = pipeline.transform(x_val)
    x_test_vec = pipeline.transform(x_test)
    transform_seconds = round(time.perf_counter() - transform_start, 6)

    if x_train_vec.shape[0] != len(x_train):
        raise FeaturePipelineError("train vectorized row count mismatch.")
    if x_val_vec.shape[0] != len(x_val):
        raise FeaturePipelineError("val vectorized row count mismatch.")
    if x_test_vec.shape[0] != len(x_test):
        raise FeaturePipelineError("test vectorized row count mismatch.")

    return {
        "run_timestamp": run_timestamp,
        "config_path": str(config_path.as_posix()),
        "text_variant": text_variant,
        "feature_set": feature_set,
        "status": "PASS",
        "message": "",
        "train_rows": int(len(x_train)),
        "val_rows": int(len(x_val)),
        "test_rows": int(len(x_test)),
        "train_features": int(x_train_vec.shape[1]),
        "train_nnz": int(x_train_vec.nnz),
        "val_nnz": int(x_val_vec.nnz),
        "test_nnz": int(x_test_vec.nnz),
        "fit_seconds": fit_seconds,
        "transform_seconds": transform_seconds,
        "vectorizer_meta_json": vectorizer_meta_to_json(vectorizer_meta),
    }


# Build a failure summary row when one combination raises an error.
def build_fail_row(
    text_variant: str,
    feature_set: str,
    config_path: Path,
    message: str,
) -> dict[str, Any]:
    return {
        "run_timestamp": now_iso(),
        "config_path": str(config_path.as_posix()),
        "text_variant": text_variant,
        "feature_set": feature_set,
        "status": "FAIL",
        "message": message,
        "train_rows": "",
        "val_rows": "",
        "test_rows": "",
        "train_features": "",
        "train_nnz": "",
        "val_nnz": "",
        "test_nnz": "",
        "fit_seconds": "",
        "transform_seconds": "",
        "vectorizer_meta_json": "",
    }


# Write feature pipeline execution summary to logs CSV.
def write_summary_csv(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


# Execute selected combinations and return summary rows.
def run_feature_pipeline(
    config_path: str | Path | None = None,
    text_variant: str | None = None,
    feature_set: str | None = None,
    allow_overlap: bool = False,
) -> tuple[list[dict[str, Any]], Path]:
    resolved_config_path = resolve_config_path(config_path)
    config = load_train_ml_config(resolved_config_path)
    selected_text_variants, selected_feature_sets = resolve_run_scope(config, text_variant, feature_set)
    strict_overlap = not allow_overlap

    rows: list[dict[str, Any]] = []
    for current_text_variant in selected_text_variants:
        print(f"[ML FEATURES] Loading splits for text_variant={current_text_variant}")
        for current_feature_set in selected_feature_sets:
            print(
                f"[ML FEATURES] Running feature_set={current_feature_set} "
                f"on text_variant={current_text_variant}"
            )
            try:
                row = run_one_combo(
                    text_variant=current_text_variant,
                    feature_set=current_feature_set,
                    config=config,
                    config_path=resolved_config_path,
                    strict_overlap=strict_overlap,
                )
                rows.append(row)
                print(
                    f"[ML FEATURES] PASS | variant={current_text_variant} "
                    f"feature_set={current_feature_set} "
                    f"features={row['train_features']} "
                    f"fit={row['fit_seconds']}s transform={row['transform_seconds']}s"
                )
            except (
                DatasetLoaderError,
                DatasetValidationError,
                VectorizerConfigError,
                FeaturePipelineError,
                Exception,
            ) as exc:
                fail_row = build_fail_row(
                    text_variant=current_text_variant,
                    feature_set=current_feature_set,
                    config_path=resolved_config_path,
                    message=str(exc),
                )
                rows.append(fail_row)
                print(
                    f"[ML FEATURES] FAIL | variant={current_text_variant} "
                    f"feature_set={current_feature_set} | {exc}"
                )

    summary_path = get_repo_root() / "logs" / "ml_feature_pipeline_summary.csv"
    write_summary_csv(summary_path, rows)
    return rows, summary_path


# Run CLI entrypoint and fail if any feature combination fails.
def main() -> None:
    args = parse_args()
    rows, summary_path = run_feature_pipeline(
        config_path=args.config_path,
        text_variant=args.text_variant,
        feature_set=args.feature_set,
        allow_overlap=args.allow_overlap,
    )

    pass_count = sum(1 for row in rows if row.get("status") == "PASS")
    fail_count = sum(1 for row in rows if row.get("status") == "FAIL")
    print(
        f"[ML FEATURES] Done | pass={pass_count} fail={fail_count} | "
        f"summary={summary_path.as_posix()}"
    )
    if fail_count > 0:
        raise SystemExit("[ML FEATURES ERROR] One or more feature combinations failed.")


# Expose a CLI-friendly feature pipeline runner.
if __name__ == "__main__":
    main()
