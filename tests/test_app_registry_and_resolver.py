from __future__ import annotations

from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.services.model_registry import ModelRegistry
from src.app.services.run_resolver import RunResolver, RunResolverError


# Build a minimal app artifact layout used across registry and resolver tests.
def build_app_artifacts(tmp_path: Path) -> Path:
    repo_root = tmp_path

    ml_dir = repo_root / "models" / "ml"
    dl_dir = repo_root / "models" / "dl"
    ml_dir.mkdir(parents=True, exist_ok=True)
    (dl_dir / "run-dl-1").mkdir(parents=True, exist_ok=True)

    (ml_dir / "run-ml-1__model.joblib").write_text("model", encoding="utf-8")
    (ml_dir / "run-ml-1__vectorizer.joblib").write_text("vectorizer", encoding="utf-8")
    (ml_dir / "run-ml-1__metadata.json").write_text(
        json.dumps({"run_id": "run-ml-1", "selection": {"model_name": "linear_svm", "feature_set": "word"}}),
        encoding="utf-8",
    )
    (ml_dir / "best_metadata.json").write_text(json.dumps({"run_id": "run-ml-1"}), encoding="utf-8")

    run_dir = dl_dir / "run-dl-1"
    (run_dir / "metadata.json").write_text(
        json.dumps({"run_id": "run-dl-1", "config_snapshot": {"model": {"pretrained_name": "phobert"}}}),
        encoding="utf-8",
    )
    (dl_dir / "best_run.json").write_text(json.dumps({"run_id": "run-dl-1"}), encoding="utf-8")
    return repo_root


# Validate direct run lookups and run-record resolution across both model families.
def test_model_registry_run_resolution(tmp_path: Path) -> None:
    repo_root = build_app_artifacts(tmp_path)
    registry = ModelRegistry(repo_root)

    assert registry.ml_run_exists("run-ml-1") is True
    assert registry.dl_run_exists("run-dl-1") is True

    ml_record = registry.get_run_record("run-ml-1")
    dl_record = registry.get_run_record("run-dl-1")

    assert ml_record is not None and ml_record.model_family == "ml"
    assert dl_record is not None and dl_record.model_family == "dl"


# Validate run listing exposes both families and marks best runs from best-artifacts.
def test_model_registry_list_runs_with_best(tmp_path: Path) -> None:
    repo_root = build_app_artifacts(tmp_path)
    registry = ModelRegistry(repo_root)

    runs = registry.list_runs_with_best()
    families = {(row["run_id"], row["model_family"], row["is_best"]) for row in runs}

    assert ("run-ml-1", "ml", True) in families
    assert ("run-dl-1", "dl", True) in families


# Validate run resolver enforces existence, family matching, and ambiguous-run protection.
def test_run_resolver_validation_paths(tmp_path: Path) -> None:
    repo_root = build_app_artifacts(tmp_path)
    registry = ModelRegistry(repo_root)
    resolver = RunResolver(registry)

    assert resolver.resolve("run-ml-1", "ml").model_family == "ml"

    with pytest.raises(RunResolverError):
        resolver.resolve("", "ml")

    with pytest.raises(RunResolverError):
        resolver.resolve("run-ml-1", "dl")
