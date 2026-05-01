from __future__ import annotations

import json
import logging
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.app.explain_dl import build_dl_explanation
from src.app.inference_common import InferenceError
from src.pipeline.normalize import normalize_text


# Container for managing Deep Learning model artifacts and metadata.
@dataclass
class DLBundle:
    run_id: str
    model: Any
    tokenizer: Any
    metadata: dict[str, Any]
    model_path: Path
    tokenizer_path: Path
    metadata_path: Path
    model_name: str
    feature_set: str
    threshold_used: float
    score_method: str
    mtimes: dict[str, float]


# Deep Learning Inference Engine
class DLInferenceEngine:

    # Initialize engine and validate app configuration values.
    def __init__(self, repo_root: Path, app_config: dict[str, Any]) -> None:
        self.repo_root = repo_root
        self.app_config = app_config

        inf_cfg = app_config.get("inference", {})
        self.default_top_k = int(inf_cfg.get("default_top_k_per_direction", 5))
        self.max_input_chars = int(inf_cfg.get("max_input_chars", 20000))
        self.max_input_tokens = int(inf_cfg.get("max_input_tokens", 512))

        threshold_cfg = app_config.get("threshold", {})
        self.use_calibrated_threshold = bool(threshold_cfg.get("use_calibrated_threshold", True))
        self.fallback_threshold = float(threshold_cfg.get("fallback_threshold", 0.5))

        labels_cfg = app_config.get("labels", {0: "real", 1: "fake"})
        self.label_map = {int(k): str(v) for k, v in labels_cfg.items()}

        explanation_cfg = app_config.get("explanation", {})
        self.explanation_enabled = bool(explanation_cfg.get("enabled", True))
        self.include_decomposition = bool(explanation_cfg.get("include_decomposition", True))

        cache_cfg = app_config.get("cache", {})
        self.cache_enabled = bool(cache_cfg.get("enabled", True))
        self.cache_capacity = int(cache_cfg.get("max_entries", 3))
        self.reload_if_file_changed = bool(cache_cfg.get("reload_if_file_changed", True))
        self.cache: OrderedDict[str, DLBundle] = OrderedDict()

        if self.default_top_k <= 0:
            raise InferenceError("inference.default_top_k_per_direction must be > 0", status_code=500)
        if self.max_input_chars <= 0:
            raise InferenceError("inference.max_input_chars must be > 0", status_code=500)
        if self.max_input_tokens <= 0:
            raise InferenceError("inference.max_input_tokens must be > 0", status_code=500)
        if self.cache_capacity < 2 or self.cache_capacity > 5:
            raise InferenceError("cache.max_entries must be in range [2,5]", status_code=500)


    # Return file mtimes for given artifact paths (used for cache invalidation).
    def _bundle_file_mtimes(self, paths: dict[str, Path]) -> dict[str, float]:
        return {name: path.stat().st_mtime for name, path in paths.items() if path.exists()}


    # Resolve threshold for this bundle from metadata or fallback.
    def _resolve_bundle_threshold(self, metadata: dict[str, Any]) -> float:
        if self.use_calibrated_threshold:
            calibration = metadata.get("threshold_calibration")
            if isinstance(calibration, dict):
                candidate = calibration.get("recommended_threshold")
                if isinstance(candidate, (int, float)) and 0.0 <= float(candidate) <= 1.0:
                    return float(candidate)
                if candidate is not None:
                    logging.warning("Invalid DL recommended_threshold=%r. Fallback to configured threshold.", candidate)
        return float(self.fallback_threshold)


    # Locate metadata and checkpoint directory for a DL run_id.
    def _resolve_artifact_paths(self, run_id: str) -> tuple[str, Path, Path]:
        selected_run_id = run_id.strip()
        if not selected_run_id:
            raise InferenceError("DL run_id is empty.", status_code=422)

        run_dir = self.repo_root / "models" / "dl" / selected_run_id
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            raise InferenceError(f"Missing DL metadata artifact: {metadata_path}", status_code=404)

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        if not isinstance(metadata, dict):
            raise InferenceError(f"DL metadata is not a JSON object: {metadata_path}", status_code=500)

        checkpoint_dir = run_dir / "best_checkpoint"
        artifact_paths = metadata.get("artifact_paths")
        if isinstance(artifact_paths, dict):
            candidate = str(artifact_paths.get("best_checkpoint_dir", "")).strip()
            if candidate:
                checkpoint_candidate = Path(candidate)
                checkpoint_dir = checkpoint_candidate if checkpoint_candidate.is_absolute() else self.repo_root / checkpoint_candidate

        if not checkpoint_dir.exists():
            raise InferenceError(f"Missing DL checkpoint directory: {checkpoint_dir}", status_code=404)

        return selected_run_id, checkpoint_dir, metadata_path


    # Build a DLBundle by loading model, tokenizer and metadata from checkpoint.
    def _build_bundle(self, run_id: str) -> DLBundle:
        selected_run_id, checkpoint_dir, metadata_path = self._resolve_artifact_paths(run_id)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=False)

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        if not isinstance(metadata, dict):
            raise InferenceError(f"DL metadata is not a JSON object: {metadata_path}", status_code=500)

        config_snapshot = metadata.get("config_snapshot", {})
        if not isinstance(config_snapshot, dict):
            config_snapshot = {}
        model_cfg = config_snapshot.get("model", {}) if isinstance(config_snapshot.get("model"), dict) else {}

        model_name = str(metadata.get("pretrained_name", "")).strip() or str(model_cfg.get("pretrained_name", "vinai/phobert-base"))
        feature_set = str(config_snapshot.get("feature_set", "transformer")).strip() or "transformer"
        threshold_used = self._resolve_bundle_threshold(metadata)
        score_method = "prob_fake"

        candidate_paths = {
            "metadata": metadata_path,
            "config": checkpoint_dir / "config.json",
            "model_safetensors": checkpoint_dir / "model.safetensors",
            "pytorch_model": checkpoint_dir / "pytorch_model.bin",
            "tokenizer_config": checkpoint_dir / "tokenizer_config.json",
            "vocab": checkpoint_dir / "vocab.txt",
            "added_tokens": checkpoint_dir / "added_tokens.json",
            "bpe_codes": checkpoint_dir / "bpe.codes",
        }
        mtimes = self._bundle_file_mtimes(candidate_paths)

        tokenizer_path = checkpoint_dir / "tokenizer_config.json"
        if not tokenizer_path.exists():
            tokenizer_path = checkpoint_dir / "vocab.txt"

        model_path = checkpoint_dir / "model.safetensors"
        if not model_path.exists():
            model_path = checkpoint_dir / "pytorch_model.bin"

        return DLBundle(
            run_id=selected_run_id,
            model=model,
            tokenizer=tokenizer,
            metadata=metadata,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            metadata_path=metadata_path,
            model_name=model_name,
            feature_set=feature_set,
            threshold_used=threshold_used,
            score_method=score_method,
            mtimes=mtimes,
        )


    # Check whether a cached DL bundle is stale using mtimes.
    def _is_cache_entry_stale(self, bundle: DLBundle) -> bool:
        if not self.reload_if_file_changed:
            return False
        current = self._bundle_file_mtimes({
            "metadata": bundle.metadata_path,
            "model": bundle.model_path,
            "tokenizer": bundle.tokenizer_path,
        })
        for key, value in bundle.mtimes.items():
            if key in current and not math.isclose(current[key], value, rel_tol=0.0, abs_tol=0.0):
                return True
        return False


    # Put a bundle into the LRU cache and evict oldest entries.
    def _put_cache(self, key: str, bundle: DLBundle) -> None:
        if not self.cache_enabled:
            return
        self.cache[key] = bundle
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_capacity:
            self.cache.popitem(last=False)


    # Retrieve bundle from cache or load a fresh one.
    def _get_bundle(self, run_id: str) -> DLBundle:
        key = f"dl:run:{run_id.strip()}"
        if self.cache_enabled and key in self.cache:
            bundle = self.cache[key]
            if self._is_cache_entry_stale(bundle):
                self.cache.pop(key, None)
            else:
                self.cache.move_to_end(key)
                return bundle
        bundle = self._build_bundle(run_id)
        self._put_cache(key, bundle)
        return bundle


    # Convenience: preload a bundle for a specific DL run.
    def preload_bundle(self, run_id: str) -> DLBundle:
        return self._get_bundle(run_id)


    # Normalize raw input to the same text_clean used during DL training.
    def _normalize_for_dl(self, text: str) -> tuple[str, list[str], int]:
        warnings: list[str] = []
        original_length = len(text)
        input_text = text.strip()
        if not input_text:
            raise InferenceError("Input text is empty after trimming.", status_code=422)
        if len(input_text) > self.max_input_chars:
            input_text = input_text[: self.max_input_chars]
            warnings.append(f"Input exceeded max_input_chars={self.max_input_chars}. Applied hard safety truncation.")

        normalized = normalize_text(input_text)["text_clean"].strip()
        if not normalized:
            raise InferenceError("Input became empty after normalize_text. Please provide richer text.", status_code=422)
        return normalized, warnings, original_length


    # Compute prediction score and raw decision score (if available) from the model output.
    def _score_from_model(self, model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[float, float | None]:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        score = float(probs[1].detach().cpu().item())
        raw_decision_score = float(logits[1].detach().cpu().item()) if logits.shape[-1] > 1 else None
        return score, raw_decision_score


    # Run end-to-end DL prediction using tokenizer + model checkpoint.
    def predict(self, text: str, run_id: str, content_type: str, top_k: int | None, return_explanation: bool) -> dict[str, Any]:
        start = time.perf_counter()
        bundle = self._get_bundle(run_id)

        resolved_content_type = content_type.strip().lower()
        if resolved_content_type not in {"news", "social"}:
            raise InferenceError("content_type must be one of: news, social.", status_code=422)

        top_k_per_direction = top_k if top_k is not None else self.default_top_k
        if top_k_per_direction < 1 or top_k_per_direction > 30:
            raise InferenceError("top_k must be in range [1,30].", status_code=422)

        normalized_text, warnings, original_length_chars = self._normalize_for_dl(text)
        input_token_count = len(bundle.tokenizer.tokenize(normalized_text))

        dl_max_length = self.max_input_tokens
        config_snapshot = bundle.metadata.get("config_snapshot")
        if isinstance(config_snapshot, dict):
            model_cfg = config_snapshot.get("model")
            if isinstance(model_cfg, dict):
                candidate = model_cfg.get("max_length")
                if isinstance(candidate, (int, float)) and int(candidate) > 0:
                    dl_max_length = min(dl_max_length, int(candidate))

        encoded = bundle.tokenizer(
            [normalized_text],
            truncation=True,
            padding="max_length",
            max_length=dl_max_length,
            return_tensors="pt",
        )
        score, raw_decision_score = self._score_from_model(bundle.model, encoded["input_ids"], encoded["attention_mask"])

        if input_token_count > dl_max_length:
            warnings.append(f"Input tokens exceeded max_length={dl_max_length}. Applied token truncation.")

        label_id = 1 if score >= bundle.threshold_used else 0
        label_text = self.label_map.get(label_id, str(label_id))

        explanation_payload = build_dl_explanation(
            tokenizer=bundle.tokenizer,
            encoded=encoded,
            base_score=score,
            raw_decision_score=raw_decision_score,
            top_k_per_direction=top_k_per_direction,
            return_explanation=return_explanation,
            explanation_enabled=self.explanation_enabled,
            include_decomposition=self.include_decomposition,
            score_fn=lambda input_ids, attention_mask: self._score_from_model(bundle.model, input_ids, attention_mask),
        )
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return {
            "label_id": int(label_id),
            "label_text": label_text,
            "model_family": "dl",
            "threshold_used": float(bundle.threshold_used),
            "raw_score": float(score),
            "raw_decision_score": raw_decision_score,
            "confidence": float(score),
            "score_method": bundle.score_method,
            "confidence_type": "probability",
            "is_probability": True,
            "run_id": bundle.run_id,
            "model_name": bundle.model_name,
            "feature_set": bundle.feature_set,
            "text_variant": "text_clean",
            "original_length_chars": original_length_chars,
            "post_normalize_token_count": input_token_count,
            "truncated": input_token_count > dl_max_length,
            "truncated_tokens_from": input_token_count,
            "truncated_tokens_to": min(input_token_count, dl_max_length),
            "segmentation_backend": "transformer",
            "segmentation_fallback_used": False,
            "preprocessing_mode": "normalize->tokenize(transformer)->token_truncate",
            "warnings": warnings,
            "explanation_available": explanation_payload["explanation_available"],
            "explanation_reason": explanation_payload["explanation_reason"],
            "top_features_towards_fake": explanation_payload["top_features_towards_fake"],
            "top_features_towards_real": explanation_payload["top_features_towards_real"],
            "explanation_decomposition": explanation_payload["explanation_decomposition"],
            "processing_time_ms": elapsed_ms,
        }


    # Health summary for DL engine and cache.
    def health(self) -> dict[str, Any]:
        default_bundle = next(iter(self.cache.values()), None) if self.cache_enabled and self.cache else None
        return {
            "status": "ok" if default_bundle else "degraded",
            "default_run_id": default_bundle.run_id if default_bundle else None,
            "default_threshold_used": default_bundle.threshold_used if default_bundle else None,
            "default_score_method": default_bundle.score_method if default_bundle else None,
            "model_loaded": default_bundle is not None,
            "cache_entries": len(self.cache),
            "cache_capacity": self.cache_capacity,
        }
