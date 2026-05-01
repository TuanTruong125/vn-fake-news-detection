from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from src.train.dl.train_phobert import DlTrainError, train_phobert
except ModuleNotFoundError:
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from train_phobert import DlTrainError, train_phobert  # type: ignore


# Parse CLI args for the single-run DL training entrypoint.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one end-to-end DL training run and save artifacts.")
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/train_dl.yaml",
        help="Path to train_dl.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional artifact root override (e.g., models/dl)",
    )
    return parser.parse_args()


# Run one training job and print concise success output for terminal users.
def main() -> None:
    args = parse_args()
    print("[DL TRAIN] Starting single-run training...")
    artifacts = train_phobert(
        config_path=args.config_path,
        output_root_override=args.output_root,
    )

    print(
        "[DL TRAIN] PASS | "
        f"run_id={artifacts.run_id} "
        f"model={artifacts.metrics['pretrained_name']} "
        "feature_set=transformer "
        "text_variant=text_clean "
        f"val_f1_macro={artifacts.metrics['metrics']['val']['f1_macro']:.6f} "
        f"test_f1_macro={artifacts.metrics['metrics']['test']['f1_macro']:.6f}"
    )
    print("[DL TRAIN] runs.csv appended at experiments/dl/runs.csv")
    print(f"[DL TRAIN] Artifacts: {artifacts.run_dir.as_posix()}")
    print("[DL TRAIN] artifacts saved under models/dl/")


if __name__ == "__main__":
    try:
        main()
    except DlTrainError as exc:
        raise SystemExit(f"[DL TRAIN ERROR] {exc}")
