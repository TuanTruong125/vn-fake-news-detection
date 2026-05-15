from __future__ import annotations

from pathlib import Path
import sys

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.ml import models as ml_models


# Build: Compact ML model configuration with parameter grids for model registry tests
def build_models_config() -> dict[str, object]:
    return {
        "models": {
            "linear_svm": {
                "enabled": True,
                "params": {
                    "C": [1.0, 0.5],
                    "class_weight": ["balanced"],
                },
            },
            "multinomial_nb": {
                "enabled": True,
                "params": {
                    "alpha": [0.1],
                    "fit_prior": [True],
                },
            },
        }
    }


# Test: Model section extraction returns exact mapping from config
def test_load_models_section_pass() -> None:
    cfg = build_models_config()

    assert ml_models.load_models_section(cfg) == cfg["models"]


# Test: Enabled model extraction preserves order and rejects unsupported models
def test_get_enabled_model_names_and_validation() -> None:
    cfg = build_models_config()
    enabled = ml_models.get_enabled_model_names(cfg["models"])

    assert enabled == ["linear_svm", "multinomial_nb"]

    bad_models = {"unknown_model": {"enabled": True, "params": {}}}
    with pytest.raises(ml_models.ModelConfigError):
        ml_models.get_enabled_model_names(bad_models)


# Test: Parameter normalization converts scalars to lists, rejects null values
def test_normalize_param_values_and_reject_nulls() -> None:
    normalized = ml_models.normalize_param_values(
        "linear_svm",
        {
            "C": 1.0,
            "class_weight": ["balanced"],
        },
    )

    assert normalized == {"C": [1.0], "class_weight": ["balanced"]}

    with pytest.raises(ml_models.ModelConfigError):
        ml_models.normalize_param_values("linear_svm", {"C": [None]})


# Test: Cartesian product expansion generates one row per parameter combination
def test_expand_param_grid_cartesian_product() -> None:
    combos = ml_models.expand_param_grid({"C": [1.0, 0.5], "class_weight": ["balanced", None]})

    assert combos == [
        {"C": 1.0, "class_weight": "balanced"},
        {"C": 1.0, "class_weight": None},
        {"C": 0.5, "class_weight": "balanced"},
        {"C": 0.5, "class_weight": None},
    ]


# Test: Model combination builder respects max_combos_per_model limit per enabled model
def test_get_model_param_combinations_with_limit() -> None:
    cfg = build_models_config()
    combos = ml_models.get_model_param_combinations(cfg, max_combos_per_model=1)

    assert combos == [
        ("linear_svm", {"C": 1.0, "class_weight": "balanced"}),
        ("multinomial_nb", {"alpha": 0.1, "fit_prior": True}),
    ]


# Test: Random state injection and estimator construction work for supported models
def test_apply_random_state_and_build_model() -> None:
    final_params = ml_models.apply_random_state("linear_svm", {"C": 1.0}, random_state=7)
    assert final_params["random_state"] == 7

    nb_params = ml_models.apply_random_state("multinomial_nb", {"alpha": 1.0}, random_state=7)
    assert "random_state" not in nb_params

    model, built_params = ml_models.build_model("linear_svm", {"C": 1.0}, random_state=7)
    assert isinstance(model, LinearSVC)
    assert built_params["random_state"] == 7

    lr_model, _ = ml_models.build_model("logistic_regression", {"C": 1.0, "max_iter": 100})
    assert isinstance(lr_model, LogisticRegression)


# Validate score method selection prefers predict_proba, then decision_function, then none.
def test_get_score_method_priority() -> None:
    assert ml_models.get_score_method(LogisticRegression()) == "predict_proba"
    assert ml_models.get_score_method(LinearSVC()) == "decision_function"
    assert ml_models.get_score_method(object()) == "none"
