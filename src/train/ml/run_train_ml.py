from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.train.common.dataset_loader import DatasetLoaderError
    from src.train.common.split_utils import DatasetValidationError
    from src.train.ml.models import ModelConfigError, ModelRuntimeError
    from src.train.ml.train import TrainRunError, train_single_run
    from src.train.ml.vectorizers import VectorizerConfigError
except ModuleNotFoundError:
    CURRENT_DIR = Path(__file__).resolve().parent
    COMMON_DIR = CURRENT_DIR.parent / "common"
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    if str(COMMON_DIR) not in sys.path:
        sys.path.insert(0, str(COMMON_DIR))

    from dataset_loader import DatasetLoaderError  # type: ignore
    from split_utils import DatasetValidationError  # type: ignore
    from models import ModelConfigError, ModelRuntimeError  # type: ignore
    from train import TrainRunError, train_single_run  # type: ignore
    from vectorizers import VectorizerConfigError  # type: ignore


# Parse CLI arguments for the single-run ML training entrypoint.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one end-to-end ML training run and save artifacts.")
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
        help="Optional text variant override, e.g. text_ml_seg_lower.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default=None,
        help="Optional feature set override, e.g. word_char.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional model override, e.g. logistic_regression.",
    )
    parser.add_argument(
        "--params-json",
        type=str,
        default=None,
        help='Optional params override JSON, requires --model-name. Example: \'{"C":1.0,"solver":"liblinear"}\'',
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="If set, split overlap check becomes warning-only.",
    )
    parser.add_argument(
        "--max-combos-per-model",
        type=int,
        default=0,
        help="Limit candidate param combos per model before selecting first combo (0 means all).",
    )
    parser.add_argument(
        "--save-as-best",
        action="store_true",
        help="If set, also copy artifacts to models/ml/best_* aliases.",
    )
    return parser.parse_args()


# Run one training job and print concise success output for terminal users.
def main() -> None:
    args = parse_args()
    print("[ML TRAIN] Starting single-run training...")
    run_record = train_single_run(
        config_path=args.config_path,
        text_variant=args.text_variant,
        feature_set=args.feature_set,
        model_name=args.model_name,
        params_json=args.params_json,
        allow_overlap=args.allow_overlap,
        max_combos_per_model=args.max_combos_per_model,
        save_as_best=args.save_as_best,
    )
    print(
        "[ML TRAIN] PASS | "
        f"run_id={run_record['run_id']} "
        f"model={run_record['model_name']} "
        f"feature_set={run_record['feature_set']} "
        f"text_variant={run_record['text_variant']} "
        f"val_f1_macro={run_record['val_f1_macro']} "
        f"test_f1_macro={run_record['test_f1_macro']}"
    )
    print("[ML TRAIN] runs.csv appended at experiments/ml/runs.csv")
    print("[ML TRAIN] artifacts saved under models/ml/")


# Expose CLI-friendly entrypoint for single ML training run.
if __name__ == "__main__":
    try:
        main()
    except (
        TrainRunError,
        ModelConfigError,
        ModelRuntimeError,
        DatasetLoaderError,
        DatasetValidationError,
        VectorizerConfigError,
    ) as exc:
        raise SystemExit(f"[ML TRAIN ERROR] {exc}")
