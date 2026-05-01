from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# Container for run metadata including family and artifact path.
@dataclass(frozen=True)
class RunRecord:
    run_id: str
    model_family: Literal["ml", "dl"]
    metadata_path: Path


# Resolve run_id ownership and artifact existence across ML and DL stores.
class ModelRegistry:

    # Initialize the model registry with the root directory of the repository.
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root


    # Check if ML artifacts (model, vectorizer, metadata) exist for a run.
    def _ml_artifacts_exist(self, run_id: str) -> bool:
        model_dir = self.repo_root / "models" / "ml"
        model_path = model_dir / f"{run_id}__model.joblib"
        vectorizer_path = model_dir / f"{run_id}__vectorizer.joblib"
        metadata_path = model_dir / f"{run_id}__metadata.json"
        return model_path.exists() and vectorizer_path.exists() and metadata_path.exists()


    # Check if DL checkpoint and metadata exist for a run.
    def _dl_artifacts_exist(self, run_id: str) -> bool:
        run_dir = self.repo_root / "models" / "dl" / run_id
        metadata_path = run_dir / "metadata.json"
        return metadata_path.exists()


    # Retrieve run record by ID, returning family and metadata path or None if not found.
    def get_run_record(self, run_id: str) -> RunRecord | None:
        normalized = run_id.strip()
        if not normalized:
            return None

        if self._ml_artifacts_exist(normalized):
            return RunRecord(
                run_id=normalized,
                model_family="ml",
                metadata_path=self.repo_root / "models" / "ml" / f"{normalized}__metadata.json",
            )

        if self._dl_artifacts_exist(normalized):
            return RunRecord(
                run_id=normalized,
                model_family="dl",
                metadata_path=self.repo_root / "models" / "dl" / normalized / "metadata.json",
            )

        return None
    

    # Check if an ML run exists.
    def ml_run_exists(self, run_id: str) -> bool:
        normalized = run_id.strip()
        if not normalized:
            return False
        return self._ml_artifacts_exist(normalized)


    # Check if a DL run exists.
    def dl_run_exists(self, run_id: str) -> bool:
        normalized = run_id.strip()
        if not normalized:
            return False
        return self._dl_artifacts_exist(normalized)
