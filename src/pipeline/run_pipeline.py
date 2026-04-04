from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


STAGES = [
    ("validate_config", "validate_config.py"),
    ("ingest", "ingest.py"),
    ("validate", "validate.py"),
    ("normalize", "normalize.py"),
    ("map_labels", "map_labels.py"),
    ("quality_filter", "quality_filter.py"),
    ("deduplicate", "deduplicate.py"),
    ("build_master", "build_master.py"),
    ("split_data", "split_data.py"),
    ("reporting", "reporting.py"),
]

SUMMARY_COLUMNS = [
    "run_id",
    "run_timestamp",
    "stage_name",
    "script_path",
    "status",
    "exit_code",
    "start_time",
    "end_time",
    "duration_seconds",
]


# Return an ISO-8601 timestamp in local timezone.
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# Keep only the tail of command output for compact logs.
def tail_text(text: str, max_chars: int = 3000) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


# Parse CLI args for full run, resume, and dry-run modes.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full fake-news data pipeline end-to-end.")
    parser.add_argument(
        "--from-stage",
        type=str,
        default=None,
        help="Resume pipeline from this stage name (e.g., normalize, split_data).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stage order and exit without executing scripts.",
    )
    return parser.parse_args()


# Validate a stage name provided by --from-stage.
def validate_from_stage(from_stage: str | None) -> None:
    if from_stage is None:
        return
    valid_names = [name for name, _ in STAGES]
    if from_stage not in valid_names:
        raise SystemExit(
            f"[RUNNER ERROR] Invalid --from-stage '{from_stage}'. "
            f"Valid stages: {valid_names}"
        )


# Resolve subset of stages when resuming from a specific stage.
def selected_stages(from_stage: str | None) -> list[tuple[str, str]]:
    if from_stage is None:
        return STAGES
    started = False
    out: list[tuple[str, str]] = []
    for stage in STAGES:
        if stage[0] == from_stage:
            started = True
        if started:
            out.append(stage)
    return out


# Run one stage script with the current Python interpreter.
def run_stage(
    repo_root: Path,
    run_id: str,
    run_timestamp: str,
    stage_name: str,
    script_name: str,
) -> dict[str, Any]:
    script_path = repo_root / "src" / "pipeline" / script_name
    if not script_path.exists():
        return {
            "run_id": run_id,
            "run_timestamp": run_timestamp,
            "stage_name": stage_name,
            "script_path": str(script_path.as_posix()),
            "status": "FAIL",
            "exit_code": -1,
            "start_time": now_iso(),
            "end_time": now_iso(),
            "duration_seconds": 0.0,
            "stdout_tail": "",
            "stderr_tail": f"Script not found: {script_path}",
        }

    start_time = now_iso()
    start_perf = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    end_perf = time.perf_counter()
    end_time = now_iso()
    duration = round(end_perf - start_perf, 6)
    status = "PASS" if proc.returncode == 0 else "FAIL"

    return {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "stage_name": stage_name,
        "script_path": str(script_path.relative_to(repo_root).as_posix()),
        "status": status,
        "exit_code": int(proc.returncode),
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration,
        "stdout_tail": tail_text(proc.stdout or ""),
        "stderr_tail": tail_text(proc.stderr or ""),
    }


# Write CSV summary with one row per attempted/skipped stage.
def write_summary_csv(logs_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_path = logs_dir / "pipeline_run_summary.csv"
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_COLUMNS})


# Write detailed JSON report for this pipeline run.
def write_run_report_json(
    logs_dir: Path,
    run_id: str,
    run_timestamp: str,
    overall_status: str,
    failed_stage: str | None,
    stages: list[dict[str, Any]],
) -> None:
    out_path = logs_dir / "pipeline_run_report.json"
    payload = {
        "run_metadata": {
            "run_id": run_id,
            "run_timestamp": run_timestamp,
            "overall_status": overall_status,
            "failed_stage": failed_stage,
            "python_executable": sys.executable,
            "cwd": str(Path.cwd().as_posix()),
        },
        "stages": stages,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# Write a short plain-text run digest for quick terminal review.
def write_last_run_txt(
    logs_dir: Path,
    run_id: str,
    run_timestamp: str,
    overall_status: str,
    failed_stage: str | None,
    stage_rows: list[dict[str, Any]],
) -> None:
    out_path = logs_dir / "pipeline_last_run.txt"
    lines = [
        "PIPELINE LAST RUN",
        f"run_id: {run_id}",
        f"run_timestamp: {run_timestamp}",
        f"overall_status: {overall_status}",
        f"failed_stage: {failed_stage or ''}",
        "",
        "STAGES",
    ]
    for row in stage_rows:
        lines.append(
            f"- {row['stage_name']}: {row['status']} "
            f"(exit={row.get('exit_code','')}, duration={row.get('duration_seconds','')}s)"
        )
    lines.append("")
    lines.append("Artifacts:")
    lines.append("- logs/pipeline_run_summary.csv")
    lines.append("- logs/pipeline_run_report.json")
    lines.append("- logs/pipeline_last_run.txt")
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# Print stage plan for dry-run mode.
def print_dry_run(stages: list[tuple[str, str]]) -> None:
    print("[RUNNER] Dry run mode. Stage order:")
    for idx, (name, script) in enumerate(stages, start=1):
        print(f"{idx}. {name} -> src/pipeline/{script}")


# Execute full pipeline runner with fail-fast behavior.
def main() -> None:
    args = parse_args()
    validate_from_stage(args.from_stage)
    repo_root = Path(__file__).resolve().parents[2]
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    stage_plan = selected_stages(args.from_stage)
    if args.dry_run:
        print_dry_run(stage_plan)
        return

    run_timestamp = now_iso()
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    stage_rows: list[dict[str, Any]] = []
    failed_stage: str | None = None

    print(f"[RUNNER] Starting pipeline run_id={run_id}")
    for stage_name, script_name in stage_plan:
        print(f"[RUNNER] Running stage: {stage_name}")
        row = run_stage(
            repo_root=repo_root,
            run_id=run_id,
            run_timestamp=run_timestamp,
            stage_name=stage_name,
            script_name=script_name,
        )
        stage_rows.append(row)
        print(
            f"[RUNNER] Stage {stage_name} status={row['status']} "
            f"exit={row['exit_code']} duration={row['duration_seconds']}s"
        )
        if row["status"] == "FAIL":
            failed_stage = stage_name
            break

    # Mark remaining stages as skipped when fail-fast stops execution.
    executed_names = {row["stage_name"] for row in stage_rows}
    for stage_name, script_name in stage_plan:
        if stage_name in executed_names:
            continue
        stage_rows.append(
            {
                "run_id": run_id,
                "run_timestamp": run_timestamp,
                "stage_name": stage_name,
                "script_path": f"src/pipeline/{script_name}",
                "status": "SKIPPED",
                "exit_code": "",
                "start_time": "",
                "end_time": "",
                "duration_seconds": "",
                "stdout_tail": "",
                "stderr_tail": "",
            }
        )

    overall_status = "FAIL" if failed_stage else "PASS"
    write_summary_csv(logs_dir, stage_rows)
    write_run_report_json(logs_dir, run_id, run_timestamp, overall_status, failed_stage, stage_rows)
    write_last_run_txt(logs_dir, run_id, run_timestamp, overall_status, failed_stage, stage_rows)

    print(
        f"[RUNNER] Completed run_id={run_id} overall_status={overall_status}. "
        "Runner logs written to logs/pipeline_run_*"
    )
    if overall_status == "FAIL":
        raise SystemExit(f"[RUNNER ERROR] Pipeline failed at stage: {failed_stage}")


# Expose a CLI-friendly entrypoint for local runs and CI checks.
if __name__ == "__main__":
    main()
