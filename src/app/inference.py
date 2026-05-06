from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.app.inference_dl import DLInferenceEngine
from src.app.inference_common import InferenceError
from src.app.inference_ml import MLInferenceEngine


# Inference Service
class InferenceService:

    # Orchestrates ML and DL inference engines based on model family and run_id resolution.
    def __init__(self, repo_root: Path, app_config: dict[str, Any]) -> None:
        self.repo_root = repo_root
        self.app_config = app_config
        self.ml_engine = MLInferenceEngine(repo_root, app_config)
        self.dl_engine = DLInferenceEngine(repo_root, app_config)


    # Factory method to create an InferenceService instance from a config file path, with error handling for missing or invalid configs.
    @classmethod
    def from_config_path(cls, config_path: str | Path | None = None) -> "InferenceService":
        repo_root = Path(__file__).resolve().parents[2]
        if config_path is None:
            config_file = repo_root / "configs" / "app.yaml"
        else:
            config_file = Path(config_path)
            if not config_file.is_absolute():
                config_file = repo_root / config_file

        if not config_file.exists():
            raise InferenceError(f"Missing app config file: {config_file}", status_code=500)

        with config_file.open("r", encoding="utf-8") as f:
            app_config = yaml.safe_load(f)
        if not isinstance(app_config, dict):
            raise InferenceError("configs/app.yaml must be a mapping object.", status_code=500)

        return cls(repo_root=repo_root, app_config=app_config)


    # Preload the default ML model bundle into memory to optimize first prediction latency, while DL models are loaded on demand due to their larger size and resource requirements.
    def preload_default_bundle(self) -> Any:
        return self.ml_engine.preload_default_bundle()


    # Route prediction requests to the appropriate inference engine based on the specified model family, with validation for required parameters and error handling for invalid inputs.
    def predict(
        self,
        text: str,
        run_id: str,
        content_type: str,
        top_k: int | None,
        return_explanation: bool,
        model_family: str = "ml",
    ) -> dict[str, Any]:
        normalized_family = model_family.strip().lower()
        if normalized_family == "ml":
            return self.ml_engine.predict(
                text=text,
                run_id=run_id,
                content_type=content_type,
                top_k=top_k,
                return_explanation=return_explanation,
            )
        if normalized_family == "dl":
            if run_id is None:
                raise InferenceError("DL run_id is required.", status_code=422)
            return self.dl_engine.predict(
                text=text,
                run_id=run_id,
                content_type=content_type,
                top_k=top_k,
                return_explanation=return_explanation,
            )
        raise InferenceError("model_family must be one of: ml, dl.", status_code=422)


    # Aggregate health information from both ML and DL engines to provide a comprehensive status report, including model loading status, cache metrics, and default model context for the health check endpoint.
    def health(self) -> dict[str, Any]:
        ml_health = self.ml_engine.health()
        dl_health = self.dl_engine.health()
        ml_loaded = bool(ml_health.get("model_loaded"))
        dl_loaded = bool(dl_health.get("model_loaded"))
        status = "ok" if (ml_loaded or dl_loaded) else "degraded"

        return {
            "status": status,
            "default_run_id": ml_health.get("default_run_id") or dl_health.get("default_run_id"),
            "default_threshold_used": ml_health.get("default_threshold_used") or dl_health.get("default_threshold_used"),
            "default_score_method": ml_health.get("default_score_method") or dl_health.get("default_score_method"),
            "model_loaded": ml_loaded or dl_loaded,
            "cache_entries": int(ml_health.get("cache_entries", 0)) + int(dl_health.get("cache_entries", 0)),
            "cache_capacity": int(ml_health.get("cache_capacity", 0)) + int(dl_health.get("cache_capacity", 0)),
            "ml_model_loaded": ml_loaded,
            "dl_model_loaded": dl_loaded,
            "ml_cache_entries": int(ml_health.get("cache_entries", 0)),
            "dl_cache_entries": int(dl_health.get("cache_entries", 0)),
        }


__all__ = ["InferenceService", "InferenceError"]
