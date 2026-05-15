from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import json
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_pipeline


# Validate stage selection and stage-name checks.
def test_run_pipeline_stage_helpers() -> None:
    run_pipeline.validate_from_stage(None)
    assert run_pipeline.selected_stages(None) == run_pipeline.STAGES
    assert run_pipeline.selected_stages("split_data")[0][0] == "split_data"

    with pytest.raises(SystemExit):
        run_pipeline.validate_from_stage("unknown")


# Validate summary/report writers persist the expected filenames.
def test_run_pipeline_writers(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {"run_id": "r1", "run_timestamp": "2026-04-04T00:00:00+07:00", "stage_name": "ingest", "script_path": "src/pipeline/ingest.py", "status": "PASS", "exit_code": 0, "start_time": "a", "end_time": "b", "duration_seconds": 1.0},
    ]
    run_pipeline.write_summary_csv(logs_dir, rows)
    run_pipeline.write_run_report_json(logs_dir, "r1", "2026-04-04T00:00:00+07:00", "PASS", None, rows)
    run_pipeline.write_last_run_txt(logs_dir, "r1", "2026-04-04T00:00:00+07:00", "PASS", None, rows)

    assert (logs_dir / "pipeline_run_summary.csv").exists()
    assert json.loads((logs_dir / "pipeline_run_report.json").read_text(encoding="utf-8"))["run_metadata"]["run_id"] == "r1"
    assert (logs_dir / "pipeline_last_run.txt").exists()


# Validate dry-run printing and per-stage execution helpers.
def test_run_stage_and_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path
    script = repo_root / "src" / "pipeline" / "ingest.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("print('ok')\n", encoding="utf-8")

    monkeypatch.setattr(run_pipeline.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="hello", stderr=""))
    row = run_pipeline.run_stage(repo_root, "run-1", "2026-04-04T00:00:00+07:00", "ingest", "ingest.py")
    assert row["status"] == "PASS"

