from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


# Config Error class for issues during vectorizer config parsing and validation.
class VectorizerConfigError(Exception):
    pass


# Resolve repository root path from current file location.
def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# Resolve train_ml config path with a default file under configs/.
def resolve_config_path(config_path: str | Path | None = None) -> Path:
    if config_path is None:
        return get_repo_root() / "configs" / "train_ml.yaml"
    return Path(config_path)


# Load train_ml config and validate top-level object type.
def load_train_ml_config(config_path: str | Path | None = None) -> dict[str, Any]:
    final_path = resolve_config_path(config_path)
    if not final_path.exists():
        raise VectorizerConfigError(f"Missing train ML config: {final_path}")
    with final_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise VectorizerConfigError("train_ml.yaml must be a mapping object.")
    return config


# Return declared text variants from config data section.
def get_text_variants(config: dict[str, Any]) -> list[str]:
    data = config.get("data")
    if not isinstance(data, dict):
        raise VectorizerConfigError("train_ml.yaml must define data section.")
    text_variants = data.get("text_variants")
    if not isinstance(text_variants, list) or not text_variants:
        raise VectorizerConfigError("data.text_variants must be a non-empty list.")
    normalized = [str(col).strip() for col in text_variants if str(col).strip()]
    if not normalized:
        raise VectorizerConfigError("data.text_variants must contain valid column names.")
    return normalized


# Return feature-set names declared in train_ml config.
def get_feature_set_names(config: dict[str, Any]) -> list[str]:
    feature_sets = config.get("feature_sets")
    if not isinstance(feature_sets, list) or not feature_sets:
        raise VectorizerConfigError("feature_sets must be a non-empty list.")
    names: list[str] = []
    for idx, feature_set in enumerate(feature_sets):
        if not isinstance(feature_set, dict):
            raise VectorizerConfigError(f"feature_sets[{idx}] must be a mapping.")
        name = feature_set.get("name")
        if not isinstance(name, str) or not name.strip():
            raise VectorizerConfigError(f"feature_sets[{idx}].name must be a non-empty string.")
        names.append(name.strip())
    return names


# Return one feature-set config mapping by name.
def get_feature_set_config(config: dict[str, Any], feature_set_name: str) -> dict[str, Any]:
    feature_sets = config.get("feature_sets")
    if not isinstance(feature_sets, list):
        raise VectorizerConfigError("feature_sets must be a list.")
    for entry in feature_sets:
        if isinstance(entry, dict) and str(entry.get("name", "")).strip() == feature_set_name:
            return entry
    raise VectorizerConfigError(f"Unknown feature_set: {feature_set_name}")


# Infer TF-IDF analyzer type from vectorizer key naming convention.
def infer_analyzer(vectorizer_key: str) -> str:
    lowered_key = vectorizer_key.lower()
    if "char" in lowered_key:
        return "char"
    return "word"


# Validate ngram range and cap char n-gram max size to 4 by feature requirement.
def normalize_ngram_range(raw_ngram_range: Any, analyzer: str, vectorizer_key: str) -> tuple[int, int, bool]:
    if (
        not isinstance(raw_ngram_range, list)
        or len(raw_ngram_range) != 2
        or not all(isinstance(value, int) for value in raw_ngram_range)
    ):
        raise VectorizerConfigError(
            f"vectorizers.{vectorizer_key}.ngram_range must be [min_n, max_n] integers."
        )

    min_n, max_n = raw_ngram_range
    if min_n <= 0 or max_n <= 0 or min_n > max_n:
        raise VectorizerConfigError(
            f"vectorizers.{vectorizer_key}.ngram_range must satisfy 1 <= min_n <= max_n."
        )

    clipped = False
    if analyzer == "char" and max_n > 4:
        max_n = 4
        clipped = True
        if min_n > max_n:
            min_n = max_n
    return min_n, max_n, clipped


# Build one TfidfVectorizer from config and return metadata for logging.
def build_single_vectorizer(vectorizer_key: str, config: dict[str, Any]) -> tuple[TfidfVectorizer, dict[str, Any]]:
    vectorizers = config.get("vectorizers")
    if not isinstance(vectorizers, dict):
        raise VectorizerConfigError("vectorizers must be a mapping.")

    vec_conf = vectorizers.get(vectorizer_key)
    if not isinstance(vec_conf, dict):
        raise VectorizerConfigError(f"Missing vectorizer config for key: {vectorizer_key}")

    analyzer = infer_analyzer(vectorizer_key)
    min_n, max_n, clipped = normalize_ngram_range(vec_conf.get("ngram_range"), analyzer, vectorizer_key)

    min_df = vec_conf.get("min_df")
    max_df = vec_conf.get("max_df")
    max_features = vec_conf.get("max_features")
    sublinear_tf = vec_conf.get("sublinear_tf")

    if not isinstance(min_df, (int, float)):
        raise VectorizerConfigError(f"vectorizers.{vectorizer_key}.min_df must be int or float.")
    if not isinstance(max_df, (int, float)):
        raise VectorizerConfigError(f"vectorizers.{vectorizer_key}.max_df must be int or float.")
    if not isinstance(max_features, int) or max_features <= 0:
        raise VectorizerConfigError(f"vectorizers.{vectorizer_key}.max_features must be a positive integer.")
    if not isinstance(sublinear_tf, bool):
        raise VectorizerConfigError(f"vectorizers.{vectorizer_key}.sublinear_tf must be boolean.")

    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=(min_n, max_n),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        lowercase=False,
    )
    meta = {
        "vectorizer_key": vectorizer_key,
        "analyzer": analyzer,
        "ngram_range": [min_n, max_n],
        "min_df": min_df,
        "max_df": max_df,
        "max_features": max_features,
        "sublinear_tf": sublinear_tf,
        "char_ngram_clipped_to_4": clipped,
    }
    return vectorizer, meta


# Build feature pipeline object for one feature set and return metadata.
def build_feature_pipeline(feature_set_name: str, config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    feature_set_conf = get_feature_set_config(config, feature_set_name)
    feature_set_type = str(feature_set_conf.get("type", "")).strip()
    if not feature_set_type:
        raise VectorizerConfigError(f"feature_set '{feature_set_name}' missing type.")

    if feature_set_type == "combined":
        parts = feature_set_conf.get("parts")
        if not isinstance(parts, list) or len(parts) < 2:
            raise VectorizerConfigError(
                f"feature_set '{feature_set_name}' with type=combined must define at least two parts."
            )

        transformers: list[tuple[str, Any]] = []
        part_meta: list[dict[str, Any]] = []
        for part in parts:
            if not isinstance(part, str) or not part.strip():
                raise VectorizerConfigError(
                    f"feature_set '{feature_set_name}' has invalid part entry: {part}"
                )
            vectorizer, meta = build_single_vectorizer(part.strip(), config)
            transformers.append((part.strip(), vectorizer))
            part_meta.append(meta)

        pipeline = FeatureUnion(transformer_list=transformers)
        metadata = {
            "feature_set_name": feature_set_name,
            "feature_set_type": "combined",
            "parts": part_meta,
        }
        return pipeline, metadata

    vectorizer, meta = build_single_vectorizer(feature_set_type, config)
    metadata = {
        "feature_set_name": feature_set_name,
        "feature_set_type": "single",
        "parts": [meta],
    }
    return vectorizer, metadata


# Convert vectorizer metadata to compact JSON for logs.
def vectorizer_meta_to_json(vectorizer_meta: dict[str, Any]) -> str:
    return json.dumps(vectorizer_meta, ensure_ascii=False, separators=(",", ":"))
