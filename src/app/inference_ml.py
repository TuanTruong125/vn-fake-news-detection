from __future__ import annotations

import json
import logging
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from joblib import load

from src.app.explain_ml import explain_linear_prediction
from src.app.inference_common import InferenceError
from src.pipeline.normalize import normalize_text
from src.pipeline.prepare_ml_text import (
    build_segmenter,
    is_likely_presegmented,
    load_stopwords,
    parse_ml_config,
    pre_clean_text,
    tokenize_and_filter,
    load_yaml as load_ml_yaml,
)


# Data class representing a loaded ML model bundle with its artifacts and metadata.
@dataclass
class MLBundle:
    run_id: str
    model: Any
    vectorizer: Any
    metadata: dict[str, Any]
    model_path: Path
    vectorizer_path: Path
    metadata_path: Path
    model_name: str
    feature_set: str
    text_variant: str
    threshold_used: float
    score_method: str
    mtimes: dict[str, float]


# Result of the text preprocessing step.
@dataclass
class PreprocessResult:
    text_for_model: str
    warnings: list[str]
    original_length_chars: int
    post_normalize_token_count: int
    truncated: bool
    truncated_tokens_from: int
    truncated_tokens_to: int
    segmentation_backend: str
    segmentation_fallback_used: bool
    preprocessing_mode: str


# Machine Learning Inference Engine
class MLInferenceEngine:

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

        diagnostics_cfg = app_config.get("diagnostics", {})
        self.include_segmentation_impact_warning = bool(diagnostics_cfg.get("include_segmentation_impact_warning", True))
        self.include_warnings = bool(diagnostics_cfg.get("include_warnings", True))

        model_selection_cfg = app_config.get("model_selection", {})
        self.allow_run_id_override = bool(model_selection_cfg.get("allow_run_id_override", True))
        self.use_best_alias_first = bool(model_selection_cfg.get("use_best_alias_first", True))
        best_cfg_rel = str(model_selection_cfg.get("best_config_path", "experiments/ml/best_config.json"))
        self.best_config_path = self.repo_root / best_cfg_rel

        cache_cfg = app_config.get("cache", {})
        self.cache_enabled = bool(cache_cfg.get("enabled", True))
        self.cache_capacity = int(cache_cfg.get("max_entries", 3))
        self.reload_if_file_changed = bool(cache_cfg.get("reload_if_file_changed", True))
        self.cache: OrderedDict[str, MLBundle] = OrderedDict()

        if self.max_input_chars <= 0:
            raise InferenceError("inference.max_input_chars must be > 0", status_code=500)
        if self.max_input_tokens <= 0:
            raise InferenceError("inference.max_input_tokens must be > 0", status_code=500)
        if self.default_top_k <= 0:
            raise InferenceError("inference.default_top_k_per_direction must be > 0", status_code=500)
        if self.cache_capacity < 2 or self.cache_capacity > 5:
            raise InferenceError("cache.max_entries must be in range [2,5]", status_code=500)

        self._load_prepare_pipeline_config()


    # Load and configure the ML text preparation pipeline (segmenter, stopwords).
    def _load_prepare_pipeline_config(self) -> None:
        ml_cfg_raw = load_ml_yaml(self.repo_root / "configs" / "ml_text.yaml")
        self.prepare_cfg = parse_ml_config(ml_cfg_raw, self.repo_root)

        if self.prepare_cfg.get("remove_stopwords", False):
            stopwords_path = self.prepare_cfg.get("stopwords_path")
            if stopwords_path is None:
                raise InferenceError("remove_stopwords=true but stopwords_path is missing.", status_code=500)
            self.stopwords = load_stopwords(stopwords_path)
        else:
            self.stopwords = set()

        stage_warnings: list[str] = []
        self.segment_backend_actual, self.segment_fn = build_segmenter(self.prepare_cfg, stage_warnings)
        self.segment_backend_requested = str(self.prepare_cfg.get("segment_backend", "auto"))
        self.segmenter_fallback_used = self.segment_backend_actual == "regex" and self.segment_backend_requested != "regex"
        self.segmenter_warnings = list(stage_warnings)
        for warning in self.segmenter_warnings:
            print(f"[APP INFERENCE INTERNAL WARNING] {warning}")


    # Preload the default ML model bundle (used at startup to avoid cold starts).
    def preload_default_bundle(self) -> MLBundle:
        return self._get_bundle(run_id=None)


    # Return modification times for model/vectorizer/metadata files for cache invalidation.
    def _bundle_file_mtimes(self, model_path: Path, vectorizer_path: Path, metadata_path: Path) -> dict[str, float]:
        return {
            "model": model_path.stat().st_mtime,
            "vectorizer": vectorizer_path.stat().st_mtime,
            "metadata": metadata_path.stat().st_mtime,
        }


    # Read best run id from experiments best_config.json.
    def _resolve_run_id_from_best_config(self) -> str:
        if not self.best_config_path.exists():
            raise InferenceError(f"Missing best config file: {self.best_config_path}", status_code=500)
        with self.best_config_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise InferenceError("best_config.json must be a JSON object.", status_code=500)
        best_run = payload.get("best_run")
        if not isinstance(best_run, dict):
            raise InferenceError("best_config.json missing best_run object.", status_code=500)
        run_id = str(best_run.get("run_id", "")).strip()
        if not run_id:
            raise InferenceError("best_config.json best_run.run_id is empty.", status_code=500)
        return run_id


    # Resolve artifact file paths for a given run_id or fall back to best run.
    def _resolve_artifact_paths(self, run_id: str | None) -> tuple[str, Path, Path, Path]:
        model_dir = self.repo_root / "models" / "ml"
        if run_id:
            selected_run_id = run_id.strip()
            model_path = model_dir / f"{selected_run_id}__model.joblib"
            vectorizer_path = model_dir / f"{selected_run_id}__vectorizer.joblib"
            metadata_path = model_dir / f"{selected_run_id}__metadata.json"
            if not model_path.exists() or not vectorizer_path.exists() or not metadata_path.exists():
                raise InferenceError(
                    f"Missing artifacts for run_id={selected_run_id}. Required: model/vectorizer/metadata.",
                    status_code=404,
                )
            return selected_run_id, model_path, vectorizer_path, metadata_path

        if self.use_best_alias_first:
            best_model = model_dir / "best_model.joblib"
            best_vectorizer = model_dir / "best_vectorizer.joblib"
            best_metadata = model_dir / "best_metadata.json"
            if best_model.exists() and best_vectorizer.exists() and best_metadata.exists():
                with best_metadata.open("r", encoding="utf-8") as f:
                    metadata_payload = json.load(f)
                selected_run_id = str(metadata_payload.get("run_id", "best")).strip() or "best"
                return selected_run_id, best_model, best_vectorizer, best_metadata

        selected_run_id = self._resolve_run_id_from_best_config()
        model_path = model_dir / f"{selected_run_id}__model.joblib"
        vectorizer_path = model_dir / f"{selected_run_id}__vectorizer.joblib"
        metadata_path = model_dir / f"{selected_run_id}__metadata.json"
        if not model_path.exists() or not vectorizer_path.exists() or not metadata_path.exists():
            raise InferenceError(f"Missing artifacts for fallback best run_id={selected_run_id}.", status_code=500)
        return selected_run_id, model_path, vectorizer_path, metadata_path


    # Determine threshold from metadata calibration or fallback config.
    def _resolve_bundle_threshold(self, metadata: dict[str, Any]) -> float:
        if self.use_calibrated_threshold:
            calibration = metadata.get("threshold_calibration")
            if isinstance(calibration, dict):
                candidate = calibration.get("recommended_threshold")
                if isinstance(candidate, (int, float)) and 0.0 <= float(candidate) <= 1.0:
                    return float(candidate)
                if candidate is not None:
                    logging.warning("Invalid ML recommended_threshold=%r. Fallback to configured threshold.", candidate)
        return float(self.fallback_threshold)


    # Build an MLBundle by loading artifacts (model, vectorizer, metadata).
    def _build_bundle(self, run_id: str | None) -> MLBundle:
        selected_run_id, model_path, vectorizer_path, metadata_path = self._resolve_artifact_paths(run_id)
        model = load(model_path)
        vectorizer = load(vectorizer_path)
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        if not isinstance(metadata, dict):
            raise InferenceError(f"Metadata is not a JSON object: {metadata_path}", status_code=500)

        selection = metadata.get("selection")
        if not isinstance(selection, dict):
            raise InferenceError(f"Metadata missing selection object: {metadata_path}", status_code=500)

        model_name = str(selection.get("model_name", "")).strip() or "unknown"
        feature_set = str(selection.get("feature_set", "")).strip() or "unknown"
        text_variant = str(selection.get("text_variant", "")).strip()
        if text_variant not in {"text_ml_seg", "text_ml_seg_lower"}:
            raise InferenceError(
                f"Unsupported text_variant in metadata: {text_variant}. Expected one of: text_ml_seg, text_ml_seg_lower",
                status_code=500,
            )

        threshold_used = self._resolve_bundle_threshold(metadata)
        score_method = "predict_proba" if hasattr(model, "predict_proba") else "decision_function_sigmoid"
        return MLBundle(
            run_id=selected_run_id,
            model=model,
            vectorizer=vectorizer,
            metadata=metadata,
            model_path=model_path,
            vectorizer_path=vectorizer_path,
            metadata_path=metadata_path,
            model_name=model_name,
            feature_set=feature_set,
            text_variant=text_variant,
            threshold_used=threshold_used,
            score_method=score_method,
            mtimes=self._bundle_file_mtimes(model_path, vectorizer_path, metadata_path),
        )


    # Check whether a cached bundle is stale based on file mtimes.
    def _is_cache_entry_stale(self, bundle: MLBundle) -> bool:
        if not self.reload_if_file_changed:
            return False
        current = self._bundle_file_mtimes(bundle.model_path, bundle.vectorizer_path, bundle.metadata_path)
        return any(not math.isclose(current[key], bundle.mtimes[key], rel_tol=0.0, abs_tol=0.0) for key in current)


    # Put a bundle into the LRU cache and evict if capacity exceeded.
    def _put_cache(self, key: str, bundle: MLBundle) -> None:
        if not self.cache_enabled:
            return
        self.cache[key] = bundle
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_capacity:
            self.cache.popitem(last=False)


    # Load bundle from cache or build and cache it.
    def _get_bundle(self, run_id: str | None) -> MLBundle:
        key = f"run:{run_id.strip()}" if run_id else "default"
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


    # Compute classification score using predict_proba or decision_function.
    def _compute_score(self, model: Any, x_vec: Any) -> tuple[float, float | None, str, str, bool]:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_vec)
            classes = [int(x) for x in list(model.classes_)]
            if 1 not in classes:
                raise InferenceError(f"Model classes do not contain class 1: {classes}", status_code=500)
            idx = classes.index(1)
            return float(probs[0, idx]), None, "predict_proba", "probability", True
        if hasattr(model, "decision_function"):
            raw = model.decision_function(x_vec)
            raw_arr = np.asarray(raw)
            if raw_arr.ndim == 0:
                raw_decision_score = float(raw_arr)
            elif raw_arr.ndim == 1:
                raw_decision_score = float(raw_arr[0])
            else:
                classes = [int(x) for x in list(model.classes_)]
                if 1 not in classes:
                    raise InferenceError(f"Model classes do not contain class 1: {classes}", status_code=500)
                raw_decision_score = float(raw_arr[0, classes.index(1)])
            safe_score = max(min(raw_decision_score, 60.0), -60.0)
            score = 1.0 / (1.0 + math.exp(-safe_score))
            return float(score), float(raw_decision_score), "decision_function_sigmoid", "pseudo_probability", False
        raise InferenceError("Model does not support predict_proba or decision_function.", status_code=500)


    # Normalize, segment, tokenize and truncate input for ML models.
    def _normalize_and_tokenize(self, text: str, content_type: str, text_variant: str, feature_set: str) -> PreprocessResult:
        warnings: list[str] = []
        original_length = len(text)
        input_text = text.strip()
        if not input_text:
            raise InferenceError("Input text is empty after trimming.", status_code=422)
        if len(input_text) > self.max_input_chars:
            input_text = input_text[: self.max_input_chars]
            warnings.append(f"Input exceeded max_input_chars={self.max_input_chars}. Applied hard safety truncation.")

        normalized = normalize_text(input_text)["text_clean"]
        pre_cleaned = pre_clean_text(
            text=normalized,
            content_type=content_type,
            replace_underscore_with_space=self.prepare_cfg["replace_underscore_with_space"],
            remove_url_tokens=self.prepare_cfg["remove_url_tokens"],
            preserve_social_emoji=self.prepare_cfg["preserve_social_emoji"],
            remove_news_emoji=self.prepare_cfg["remove_news_emoji"],
        )
        if self.prepare_cfg["respect_presegmented_input"] and is_likely_presegmented(pre_cleaned):
            segmented_text = pre_cleaned
        else:
            segmented_text = self.segment_fn(pre_cleaned)

        if text_variant == "text_ml_seg":
            _, _, _, tokens = tokenize_and_filter(
                text_segmented=segmented_text,
                lowercase=False,
                keep_numbers=self.prepare_cfg["keep_numbers"],
                min_token_length=self.prepare_cfg["min_token_length"],
                remove_stopwords=self.prepare_cfg["remove_stopwords"],
                stopwords=self.stopwords,
            )
            preprocessing_mode = "normalize->pre_clean->segment->tokenize(text_ml_seg)->token_truncate"
        elif text_variant == "text_ml_seg_lower":
            _, _, _, tokens = tokenize_and_filter(
                text_segmented=segmented_text,
                lowercase=True,
                keep_numbers=self.prepare_cfg["keep_numbers"],
                min_token_length=self.prepare_cfg["min_token_length"],
                remove_stopwords=self.prepare_cfg["remove_stopwords"],
                stopwords=self.stopwords,
            )
            preprocessing_mode = "normalize->pre_clean->segment->tokenize(text_ml_seg_lower)->token_truncate"
        else:
            raise InferenceError(f"Unsupported text_variant='{text_variant}'. Expected text_ml_seg or text_ml_seg_lower.", status_code=500)

        post_normalize_token_count = len(tokens)
        truncated = False
        truncated_tokens_from = post_normalize_token_count
        truncated_tokens_to = post_normalize_token_count
        if post_normalize_token_count > self.max_input_tokens:
            tokens = tokens[: self.max_input_tokens]
            truncated = True
            truncated_tokens_to = len(tokens)
            warnings.append(f"Input tokens exceeded max_input_tokens={self.max_input_tokens}. Applied token truncation.")

        text_for_model = " ".join(tokens)
        if not text_for_model:
            raise InferenceError("Input became empty after normalize/segment/tokenize. Please provide richer text.", status_code=422)

        segmentation_fallback_used = self.segmenter_fallback_used
        if segmentation_fallback_used and self.include_segmentation_impact_warning:
            if feature_set in {"word", "word_char"}:
                warnings.append("Segmentation fallback may degrade performance for word-based models.")
            else:
                warnings.append("Segmentation fallback is active; impact is usually lower for char-based models.")

        return PreprocessResult(
            text_for_model=text_for_model,
            warnings=warnings if self.include_warnings else [],
            original_length_chars=original_length,
            post_normalize_token_count=post_normalize_token_count,
            truncated=truncated,
            truncated_tokens_from=truncated_tokens_from,
            truncated_tokens_to=truncated_tokens_to,
            segmentation_backend=self.segment_backend_actual,
            segmentation_fallback_used=segmentation_fallback_used,
            preprocessing_mode=preprocessing_mode,
        )


    # Run end-to-end ML prediction: preprocess -> vectorize -> score -> explain.
    def predict(self, text: str, run_id: str | None, content_type: str, top_k: int | None, return_explanation: bool) -> dict[str, Any]:
        start = time.perf_counter()
        if run_id and not self.allow_run_id_override:
            raise InferenceError("run_id override is disabled by app config.", status_code=422)
        bundle = self._get_bundle(run_id=run_id)
        resolved_content_type = content_type.strip().lower()
        if resolved_content_type not in {"news", "social"}:
            raise InferenceError("content_type must be one of: news, social.", status_code=422)
        top_k_per_direction = top_k if top_k is not None else self.default_top_k
        if top_k_per_direction < 1 or top_k_per_direction > 30:
            raise InferenceError("top_k must be in range [1,30].", status_code=422)

        prep = self._normalize_and_tokenize(text=text, content_type=resolved_content_type, text_variant=bundle.text_variant, feature_set=bundle.feature_set)
        x_vec = bundle.vectorizer.transform([prep.text_for_model])
        score, raw_decision_score, score_method, confidence_type, is_probability = self._compute_score(bundle.model, x_vec)
        fake_score = float(score)
        threshold_used = bundle.threshold_used
        label_id = 1 if fake_score >= threshold_used else 0
        label_text = self.label_map.get(label_id, str(label_id))
        prediction_confidence = max(fake_score, 1.0 - fake_score)

        explanation_payload = {
            "explanation_available": False,
            "explanation_reason": "Explanation disabled by request or config.",
            "top_features_towards_fake": [],
            "top_features_towards_real": [],
            "explanation_decomposition": None,
        }
        if self.explanation_enabled and return_explanation:
            explanation_payload = explain_linear_prediction(
                model=bundle.model,
                vectorizer=bundle.vectorizer,
                x_vec=x_vec,
                top_k_per_direction=top_k_per_direction,
                include_decomposition=self.include_decomposition,
                raw_decision_score=raw_decision_score,
            )

        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return {
            "label_id": int(label_id),
            "label_text": label_text,
            "model_family": "ml",
            "threshold_used": float(threshold_used),
            "raw_score": fake_score,
            "raw_decision_score": raw_decision_score,
            "confidence": float(prediction_confidence),
            "score_method": score_method,
            "confidence_type": confidence_type,
            "is_probability": bool(is_probability),
            "run_id": bundle.run_id,
            "model_name": bundle.model_name,
            "feature_set": bundle.feature_set,
            "text_variant": bundle.text_variant,
            "original_length_chars": prep.original_length_chars,
            "post_normalize_token_count": prep.post_normalize_token_count,
            "truncated": prep.truncated,
            "truncated_tokens_from": prep.truncated_tokens_from,
            "truncated_tokens_to": prep.truncated_tokens_to,
            "segmentation_backend": prep.segmentation_backend,
            "segmentation_fallback_used": prep.segmentation_fallback_used,
            "preprocessing_mode": prep.preprocessing_mode,
            "warnings": prep.warnings,
            "explanation_available": explanation_payload["explanation_available"],
            "explanation_reason": explanation_payload["explanation_reason"],
            "top_features_towards_fake": explanation_payload["top_features_towards_fake"],
            "top_features_towards_real": explanation_payload["top_features_towards_real"],
            "explanation_decomposition": explanation_payload["explanation_decomposition"],
            "processing_time_ms": elapsed_ms,
        }


    # Health summary for ML engine and cache.
    def health(self) -> dict[str, Any]:
        model_loaded = "default" in self.cache if self.cache_enabled else False
        default_bundle = self.cache.get("default") if self.cache_enabled else None
        return {
            "status": "ok" if model_loaded else "degraded",
            "default_run_id": default_bundle.run_id if default_bundle else None,
            "default_threshold_used": default_bundle.threshold_used if default_bundle else None,
            "default_score_method": default_bundle.score_method if default_bundle else None,
            "model_loaded": model_loaded,
            "cache_entries": len(self.cache),
            "cache_capacity": self.cache_capacity,
        }
