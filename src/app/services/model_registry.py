from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


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
    def list_runs_with_best(self) -> list[dict[str, Any]]:
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
            metadata = self._load_ml_metadata(run_id)
            runs.append({
                "run_id": run_id,
                "model_family": "ml",
                "is_best": run_id == best_ml,
                "model_name": metadata.get("model_name"),
                "feature_set": metadata.get("feature_set"),
                "text_variant": metadata.get("text_variant"),
                "params": metadata.get("params"),
                "threshold": metadata.get("threshold"),
                "val_f1_macro": metadata.get("val_f1_macro"),
                "val_precision_macro": metadata.get("val_precision_macro"),
                "val_recall_macro": metadata.get("val_recall_macro"),
                "val_accuracy": metadata.get("val_accuracy"),
                "val_f1_fake": metadata.get("val_f1_fake"),
            })
        for run_id in sorted(dl_run_ids):
            metadata = self._load_dl_metadata(run_id)
            runs.append({
                "run_id": run_id,
                "model_family": "dl",
                "is_best": run_id == best_dl,
                "model_name": metadata.get("model_name"),
                "feature_set": metadata.get("feature_set"),
                "text_variant": metadata.get("text_variant"),
                "params": metadata.get("params"),
                "threshold": metadata.get("threshold"),
                "val_f1_macro": metadata.get("val_f1_macro"),
                "val_precision_macro": metadata.get("val_precision_macro"),
                "val_recall_macro": metadata.get("val_recall_macro"),
                "val_accuracy": metadata.get("val_accuracy"),
                "val_f1_fake": metadata.get("val_f1_fake"),
            })

        return runs


    # Load ML metadata from JSON file.
    def _load_ml_metadata(self, run_id: str) -> dict[str, Any]:
        metadata_path = self.repo_root / "models" / "ml" / f"{run_id}__metadata.json"
        if not metadata_path.exists():
            return {}
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            selection = data.get("selection", {}) if isinstance(data, dict) else {}
            val_metrics = data.get("metrics", {}).get("val", {}) if isinstance(data, dict) else {}
            return {
                "model_name": selection.get("model_name") or data.get("model_name"),
                "feature_set": selection.get("feature_set") or data.get("feature_set"),
                "text_variant": selection.get("text_variant") or data.get("text_variant"),
                "params": selection.get("params") or data.get("params"),
                "threshold": self._resolve_threshold(data),
                "val_f1_macro": self._safe_metric(val_metrics.get("f1_macro")),
                "val_precision_macro": self._safe_metric(val_metrics.get("precision_macro")),
                "val_recall_macro": self._safe_metric(val_metrics.get("recall_macro")),
                "val_accuracy": self._safe_metric(val_metrics.get("accuracy")),
                "val_f1_fake": self._safe_metric(val_metrics.get("f1_fake")),
            }
        except Exception:
            return {}


    # Load DL metadata from JSON file.
    def _load_dl_metadata(self, run_id: str) -> dict[str, Any]:
        metadata_path = self.repo_root / "models" / "dl" / run_id / "metadata.json"
        metrics_path = self.repo_root / "models" / "dl" / run_id / "metrics.json"
        if not metadata_path.exists():
            return {}
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            metrics_data: dict[str, Any] = {}
            if metrics_path.exists():
                try:
                    with metrics_path.open("r", encoding="utf-8") as f:
                        metrics_payload = json.load(f)
                    if isinstance(metrics_payload, dict):
                        epoch_history = metrics_payload.get("epoch_history", [])
                        if isinstance(epoch_history, list) and len(epoch_history) >= 3:
                            metrics_data = epoch_history[2].get("val_metrics", {}) if isinstance(epoch_history[2], dict) else {}
                        else:
                            metrics_data = metrics_payload.get("metrics", {}).get("val", {}) if isinstance(metrics_payload.get("metrics"), dict) else {}
                except Exception:
                    metrics_data = {}
            config_snapshot = data.get("config_snapshot", {}) if isinstance(data, dict) else {}
            model_snapshot = config_snapshot.get("model", {}) if isinstance(config_snapshot, dict) else {}
            training_snapshot = config_snapshot.get("training", {}) if isinstance(config_snapshot, dict) else {}
            return {
                "model_name": model_snapshot.get("pretrained_name") or data.get("model_name"),
                "feature_set": config_snapshot.get("feature_set") or data.get("feature_set"),
                "text_variant": config_snapshot.get("text_variant") or data.get("text_variant"),
                "params": {
                    **({"model": model_snapshot} if model_snapshot else {}),
                    **({"training": training_snapshot} if training_snapshot else {}),
                }
                or data.get("params"),
                "threshold": self._resolve_threshold(data),
                "val_f1_macro": self._safe_metric(metrics_data.get("f1_macro")),
                "val_precision_macro": self._safe_metric(metrics_data.get("precision_macro")),
                "val_recall_macro": self._safe_metric(metrics_data.get("recall_macro")),
                "val_accuracy": self._safe_metric(metrics_data.get("accuracy")),
                "val_f1_fake": self._safe_metric(metrics_data.get("f1_fake")),
            }
        except Exception:
            return {}


    # Safely normalize metric values so sorting can use numeric comparisons.
    def _safe_metric(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


    # Resolve the best threshold from metadata, preferring recommended_threshold and falling back to 0.5.
    def _resolve_threshold(self, metadata: Any) -> float:
        if not isinstance(metadata, dict):
            return 0.5

        threshold_calibration = metadata.get("threshold_calibration", {})
        if isinstance(threshold_calibration, dict):
            recommended = threshold_calibration.get("recommended_threshold")
            if recommended is not None:
                try:
                    return float(recommended)
                except (TypeError, ValueError):
                    pass

            default_threshold = threshold_calibration.get("default_threshold")
            if default_threshold is not None:
                try:
                    return float(default_threshold)
                except (TypeError, ValueError):
                    pass

        raw_threshold = metadata.get("threshold")
        if raw_threshold is not None:
            try:
                return float(raw_threshold)
            except (TypeError, ValueError):
                pass

        return 0.5


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
