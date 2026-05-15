from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = ["label_binary", "hash_text", "split"]
ISSUE_COLUMNS = ["run_timestamp", "severity", "issue_code", "count", "detail"]
RATIO_PASS_TOL = 0.02
RATIO_WARN_TOL = 0.05


# Error class for split stage issues.
class SplitDataError(Exception):
    pass


# Load one YAML file and return a dictionary object.
def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SplitDataError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SplitDataError(f"Config file must contain a mapping object: {path}")
    return data


# Append one standardized issue row to the issue list.
def add_issue(
    issues: list[dict[str, Any]],
    run_timestamp: str,
    severity: str,
    issue_code: str,
    count: int,
    detail: str,
) -> None:
    issues.append(
        {
            "run_timestamp": run_timestamp,
            "severity": severity,
            "issue_code": issue_code,
            "count": int(count),
            "detail": detail,
        }
    )


# Determine stage status based on issue severities.
def determine_status(issues: list[dict[str, Any]]) -> str:
    severities = {item["severity"] for item in issues}
    if "ERROR" in severities:
        return "FAIL"
    if "WARNING" in severities:
        return "WARNING"
    return "PASS"


# Read and validate split config values required by this stage.
def parse_split_config(split_cfg: dict[str, Any]) -> dict[str, Any]:
    split = split_cfg.get("split")
    if not isinstance(split, dict):
        raise SplitDataError("configs/split.yaml must define 'split' mapping.")

    ratios = split.get("ratios")
    if not isinstance(ratios, dict):
        raise SplitDataError("split.ratios must be a mapping.")
    train_ratio = float(ratios.get("train", 0))
    val_ratio = float(ratios.get("val", 0))
    test_ratio = float(ratios.get("test", 0))
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise SplitDataError("split.ratios train/val/test must all be > 0.")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise SplitDataError("split.ratios train+val+test must sum to 1.0.")

    random_state = split.get("random_state")
    if not isinstance(random_state, int):
        raise SplitDataError("split.random_state must be an integer.")
    shuffle = split.get("shuffle")
    if not isinstance(shuffle, bool):
        raise SplitDataError("split.shuffle must be a boolean.")

    stratify_by = split.get("stratify_by")
    if not isinstance(stratify_by, str) or not stratify_by.strip():
        raise SplitDataError("split.stratify_by must be a non-empty string.")
    leakage_key = split.get("leakage_key")
    if not isinstance(leakage_key, str) or not leakage_key.strip():
        raise SplitDataError("split.leakage_key must be a non-empty string.")

    split_values = split.get("internal_split_values")
    if not isinstance(split_values, list) or len(split_values) != 3:
        raise SplitDataError("split.internal_split_values must be a list with 3 values.")

    return {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_state": random_state,
        "shuffle": shuffle,
        "stratify_by": stratify_by.strip(),
        "leakage_key": leakage_key.strip(),
        "train_split_value": str(split_values[0]),
        "val_split_value": str(split_values[1]),
        "test_split_value": str(split_values[2]),
    }


# Build a one-row summary with split sizes, ratios, distributions, and status.
def build_summary(
    run_timestamp: str,
    status: str,
    total_rows: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    expected_train: float,
    expected_val: float,
    expected_test: float,
    leakage_train_val: int,
    leakage_train_test: int,
    leakage_val_test: int,
) -> dict[str, Any]:
    train_rows = int(len(train_df))
    val_rows = int(len(val_df))
    test_rows = int(len(test_df))
    train_ratio_actual = (train_rows / total_rows) if total_rows > 0 else 0.0
    val_ratio_actual = (val_rows / total_rows) if total_rows > 0 else 0.0
    test_ratio_actual = (test_rows / total_rows) if total_rows > 0 else 0.0

    def counts_and_ratios(df: pd.DataFrame) -> tuple[int, int, float, float]:
        c0 = int((df["label_binary"] == 0).sum())
        c1 = int((df["label_binary"] == 1).sum())
        n = len(df)
        r0 = (c0 / n) if n > 0 else 0.0
        r1 = (c1 / n) if n > 0 else 0.0
        return c0, c1, r0, r1

    train_0, train_1, train_r0, train_r1 = counts_and_ratios(train_df)
    val_0, val_1, val_r0, val_r1 = counts_and_ratios(val_df)
    test_0, test_1, test_r0, test_r1 = counts_and_ratios(test_df)

    return {
        "run_timestamp": run_timestamp,
        "total_rows": total_rows,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "train_ratio_expected": expected_train,
        "val_ratio_expected": expected_val,
        "test_ratio_expected": expected_test,
        "train_ratio_actual": train_ratio_actual,
        "val_ratio_actual": val_ratio_actual,
        "test_ratio_actual": test_ratio_actual,
        "train_label_0_count": train_0,
        "train_label_1_count": train_1,
        "train_label_0_ratio": train_r0,
        "train_label_1_ratio": train_r1,
        "val_label_0_count": val_0,
        "val_label_1_count": val_1,
        "val_label_0_ratio": val_r0,
        "val_label_1_ratio": val_r1,
        "test_label_0_count": test_0,
        "test_label_1_count": test_1,
        "test_label_0_ratio": test_r0,
        "test_label_1_ratio": test_r1,
        "leakage_train_val": leakage_train_val,
        "leakage_train_test": leakage_train_test,
        "leakage_val_test": leakage_val_test,
        "status": status,
    }


# Write split summary and issue logs to disk.
def write_logs(repo_root: Path, summary_row: dict[str, Any], issue_rows: list[dict[str, Any]]) -> None:
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary_row]).to_csv(logs_dir / "split_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(issue_rows, columns=ISSUE_COLUMNS).to_csv(logs_dir / "split_issues.csv", index=False, encoding="utf-8-sig")


# Execute split stage with stratified two-step split and leakage checks.
def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    run_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    issues: list[dict[str, Any]] = []

    split_cfg = load_yaml(repo_root / "configs" / "split.yaml")
    cfg = parse_split_config(split_cfg)

    master_path = repo_root / "data" / "processed" / "master_dataset_v1.csv"
    if not master_path.exists():
        raise SplitDataError(f"Missing input file: {master_path}")
    df = pd.read_csv(master_path)
    total_rows = int(len(df))

    missing_columns = sorted(set(REQUIRED_COLUMNS + [cfg["stratify_by"], cfg["leakage_key"]]) - set(df.columns))
    if missing_columns:
        add_issue(
            issues,
            run_timestamp,
            "ERROR",
            "missing_required_columns",
            len(missing_columns),
            f"missing={missing_columns}",
        )

    df["label_binary"] = pd.to_numeric(df["label_binary"], errors="coerce")
    invalid_label_binary_count = int((~df["label_binary"].isin([0, 1])).sum())
    if invalid_label_binary_count > 0:
        add_issue(
            issues,
            run_timestamp,
            "ERROR",
            "invalid_label_binary",
            invalid_label_binary_count,
            "label_binary must be in {0,1} before split",
        )

    # Guard stratified split by ensuring each class has enough samples for 3 partitions.
    class_counts = df[cfg["stratify_by"]].value_counts(dropna=False)
    min_class = int(class_counts.min()) if not class_counts.empty else 0
    if min_class < 3:
        add_issue(
            issues,
            run_timestamp,
            "ERROR",
            "not_enough_samples_for_stratified_split",
            min_class,
            f"min_class_count={min_class} < 3",
        )

    # Enforce unique leakage key before split when deduplicate_before_split=true.
    leakage_dups = int(df[cfg["leakage_key"]].duplicated(keep=False).sum())
    if leakage_dups > 0:
        add_issue(
            issues,
            run_timestamp,
            "ERROR",
            "hash_not_unique_before_split",
            leakage_dups,
            f"{cfg['leakage_key']} must be unique before split",
        )

    if determine_status(issues) == "FAIL":
        empty_df = pd.DataFrame(columns=df.columns)
        summary_row = build_summary(
            run_timestamp=run_timestamp,
            status="FAIL",
            total_rows=total_rows,
            train_df=empty_df,
            val_df=empty_df,
            test_df=empty_df,
            expected_train=cfg["train_ratio"],
            expected_val=cfg["val_ratio"],
            expected_test=cfg["test_ratio"],
            leakage_train_val=0,
            leakage_train_test=0,
            leakage_val_test=0,
        )
        write_logs(repo_root, summary_row, issues)
        raise SystemExit("[SPLIT ERROR] Pre-split validation failed. Check logs/split_issues.csv.")

    stratify_series = df[cfg["stratify_by"]]
    train_df, temp_df = train_test_split(
        df,
        train_size=cfg["train_ratio"],
        random_state=cfg["random_state"],
        shuffle=cfg["shuffle"],
        stratify=stratify_series,
    )

    # Use adjusted val ratio over the temp partition to preserve global target ratios.
    val_ratio_adjusted = cfg["val_ratio"] / (cfg["val_ratio"] + cfg["test_ratio"])
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio_adjusted,
        random_state=cfg["random_state"],
        shuffle=cfg["shuffle"],
        stratify=temp_df[cfg["stratify_by"]],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = cfg["train_split_value"]
    val_df["split"] = cfg["val_split_value"]
    test_df["split"] = cfg["test_split_value"]

    row_count_after_split = len(train_df) + len(val_df) + len(test_df)
    if row_count_after_split != len(df):
        add_issue(
            issues,
            run_timestamp,
            "ERROR",
            "row_count_mismatch_after_split",
            abs(row_count_after_split - len(df)),
            f"input={len(df)}, split_total={row_count_after_split}",
        )

    # Check leakage by intersection of leakage key values across split pairs.
    train_set = set(train_df[cfg["leakage_key"]].tolist())
    val_set = set(val_df[cfg["leakage_key"]].tolist())
    test_set = set(test_df[cfg["leakage_key"]].tolist())
    overlap_train_val = len(train_set.intersection(val_set))
    overlap_train_test = len(train_set.intersection(test_set))
    overlap_val_test = len(val_set.intersection(test_set))
    leakage_total = overlap_train_val + overlap_train_test + overlap_val_test
    if leakage_total > 0:
        add_issue(
            issues,
            run_timestamp,
            "ERROR",
            "leakage_detected",
            leakage_total,
            (
                f"train_val={overlap_train_val}, "
                f"train_test={overlap_train_test}, "
                f"val_test={overlap_val_test}"
            ),
        )

    # Apply ratio tolerance policy: <=0.02 pass, <=0.05 warning, >0.05 error.
    actual_ratios = {
        "train": len(train_df) / len(df) if len(df) > 0 else 0.0,
        "val": len(val_df) / len(df) if len(df) > 0 else 0.0,
        "test": len(test_df) / len(df) if len(df) > 0 else 0.0,
    }
    expected_ratios = {
        "train": cfg["train_ratio"],
        "val": cfg["val_ratio"],
        "test": cfg["test_ratio"],
    }
    for split_name in ["train", "val", "test"]:
        diff = abs(actual_ratios[split_name] - expected_ratios[split_name])
        if diff > RATIO_WARN_TOL:
            add_issue(
                issues,
                run_timestamp,
                "ERROR",
                "ratio_out_of_tolerance",
                1,
                f"split={split_name}, diff={diff:.6f}, expected={expected_ratios[split_name]:.6f}, actual={actual_ratios[split_name]:.6f}",
            )
        elif diff > RATIO_PASS_TOL:
            add_issue(
                issues,
                run_timestamp,
                "WARNING",
                "ratio_out_of_tolerance",
                1,
                f"split={split_name}, diff={diff:.6f}, expected={expected_ratios[split_name]:.6f}, actual={actual_ratios[split_name]:.6f}",
            )

    status = determine_status(issues)

    # Update split column in master while preserving original row order.
    split_series = pd.Series(index=df.index, dtype="object")
    split_series.loc[train_df.index] = cfg["train_split_value"]
    split_series.loc[val_df.index] = cfg["val_split_value"]
    split_series.loc[test_df.index] = cfg["test_split_value"]
    df_master_out = df.copy()
    df_master_out["split"] = split_series

    if status != "FAIL":
        processed_dir = repo_root / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        df_master_out.to_csv(processed_dir / "master_dataset_v1.csv", index=False, encoding="utf-8-sig")
        train_df.to_csv(processed_dir / "train.csv", index=False, encoding="utf-8-sig")
        val_df.to_csv(processed_dir / "val.csv", index=False, encoding="utf-8-sig")
        test_df.to_csv(processed_dir / "test.csv", index=False, encoding="utf-8-sig")

    summary_row = build_summary(
        run_timestamp=run_timestamp,
        status=status,
        total_rows=total_rows,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        expected_train=cfg["train_ratio"],
        expected_val=cfg["val_ratio"],
        expected_test=cfg["test_ratio"],
        leakage_train_val=overlap_train_val,
        leakage_train_test=overlap_train_test,
        leakage_val_test=overlap_val_test,
    )
    write_logs(repo_root, summary_row, issues)

    print(
        f"[SPLIT] status={status} total={total_rows} "
        f"train={len(train_df)} val={len(val_df)} test={len(test_df)}"
    )

    if status == "FAIL":
        raise SystemExit("[SPLIT ERROR] Split failed. Check logs/split_issues.csv.")


# Expose a CLI-friendly entrypoint for local runs and CI checks.
if __name__ == "__main__":
    main()
