from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

error_dl = pytest.importorskip("src.train.dl.error_analysis_dl")


# Validate split parser normalizes case, removes duplicates, and preserves order.
def test_parse_split_names_normalize_and_deduplicate() -> None:
    split_names = error_dl.parse_split_names("VAL,test,val")
    assert split_names == ["val", "test"]


# Validate split parser rejects unsupported split names.
def test_parse_split_names_reject_invalid() -> None:
    with pytest.raises(error_dl.DlErrorAnalysisError):
        error_dl.parse_split_names("train")


# Validate confidence score uses max(prob_fake, 1-prob_fake) for each row.
def test_add_confidence_scores_prob_fake_max() -> None:
    frame = pd.DataFrame({"prob_fake": [0.9, 0.4, 0.5]})
    out, method = error_dl.add_confidence_scores(frame)

    assert method == "prob_fake_max"
    assert out["confidence_score"].tolist() == [0.9, 0.6, 0.5]


# Validate confidence method description text is explicit and stable.
def test_describe_confidence_method_text() -> None:
    message = error_dl.describe_confidence_method("prob_fake_max")
    assert "max(P(fake), 1 - P(fake))" in message


# Validate target run resolver prefers requested run_id over best_config fallback.
def test_resolve_target_run_prefers_requested_run() -> None:
    rows = [
        {"run_id": "r1", "run_timestamp": "2026-05-01T00:00:00+07:00", "status": "PASS"},
        {"run_id": "r2", "run_timestamp": "2026-05-02T00:00:00+07:00", "status": "PASS"},
    ]
    best_cfg = {"best_run": {"run_id": "r2"}}

    target = error_dl.resolve_target_run(rows, best_cfg, "r1")
    assert target["run_id"] == "r1"


# Validate target run resolver falls back to latest PASS run when best_config is missing.
def test_resolve_target_run_fallback_latest_pass() -> None:
    rows = [
        {"run_id": "old", "run_timestamp": "2026-05-01T00:00:00+07:00", "status": "PASS"},
        {"run_id": "new", "run_timestamp": "2026-05-03T00:00:00+07:00", "status": "PASS"},
        {"run_id": "fail", "run_timestamp": "2026-05-04T00:00:00+07:00", "status": "FAIL"},
    ]

    target = error_dl.resolve_target_run(rows, None, None)
    assert target["run_id"] == "new"
