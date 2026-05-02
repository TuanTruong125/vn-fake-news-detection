from __future__ import annotations

import json
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


    # List available runs for UI selection with BEST markers.
    def list_runs_with_best(self) -> list[dict[str, str | bool]]:
        ml_dir = self.repo_root / "models" / "ml"
        dl_dir = self.repo_root / "models" / "dl"
        runs: list[dict[str, str | bool]] = []

        ml_run_ids: set[str] = set()
        for metadata_file in ml_dir.glob("*__metadata.json"):
            run_id = metadata_file.name.replace("__metadata.json", "").strip()
            if run_id and self._ml_artifacts_exist(run_id):
                ml_run_ids.add(run_id)

        dl_run_ids: set[str] = set()
        if dl_dir.exists():
            for run_dir in dl_dir.iterdir():
                if run_dir.is_dir() and (run_dir / "metadata.json").exists():
                    dl_run_ids.add(run_dir.name.strip())

        best_ml = self._resolve_best_ml_run_id()
        best_dl = self._resolve_best_dl_run_id()

        for run_id in sorted(ml_run_ids):
            runs.append({
                "run_id": run_id,
                "model_family": "ml",
                "is_best": run_id == best_ml,
            })
        for run_id in sorted(dl_run_ids):
            runs.append({
                "run_id": run_id,
                "model_family": "dl",
                "is_best": run_id == best_dl,
            })

        return runs


    # Resolve ML best run ID from best-config artifacts.
    def _resolve_best_ml_run_id(self) -> str | None:
        best_config_path = self.repo_root / "experiments" / "ml" / "best_config.json"
        if best_config_path.exists():
            try:
                payload = json.loads(best_config_path.read_text(encoding="utf-8"))
                best_run = payload.get("best_run", {})
                run_id = str(best_run.get("run_id", "")).strip()
                if run_id:
                    return run_id
            except Exception:
                pass

        best_metadata_path = self.repo_root / "models" / "ml" / "best_metadata.json"
        if best_metadata_path.exists():
            try:
                payload = json.loads(best_metadata_path.read_text(encoding="utf-8"))
                run_id = str(payload.get("run_id", "")).strip()
                if run_id:
                    return run_id
            except Exception:
                pass
        return None


    # Resolve DL best run ID from best-run artifacts.
    def _resolve_best_dl_run_id(self) -> str | None:
        best_run_json = self.repo_root / "models" / "dl" / "best_run.json"
        if best_run_json.exists():
            try:
                payload = json.loads(best_run_json.read_text(encoding="utf-8"))
                run_id = str(payload.get("run_id", "")).strip()
                if run_id:
                    return run_id
            except Exception:
                pass

        best_run_txt = self.repo_root / "models" / "dl" / "best_run_id.txt"
        if best_run_txt.exists():
            run_id = best_run_txt.read_text(encoding="utf-8").strip()
            if run_id:
                return run_id

        best_metadata = self.repo_root / "models" / "dl" / "best_metadata.json"
        if best_metadata.exists():
            try:
                payload = json.loads(best_metadata.read_text(encoding="utf-8"))
                run_id = str(payload.get("run_id", "")).strip()
                if run_id:
                    return run_id
            except Exception:
                pass
        return None
