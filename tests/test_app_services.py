from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.services.explain_service import ExplainService
from src.app.services.predict_service import PredictService, PredictServiceError
from src.app.services.run_resolver import RunResolverError


# Build a tiny inference stub used by PredictService tests.
# Minimal inference stub that echoes routing inputs.
class FakeInferenceService:
    def predict(self, **kwargs):
        return {"model_family": kwargs["model_family"], "run_id": kwargs["run_id"], "ok": True}


# Build a tiny registry stub used by PredictService tests.
# Minimal registry stub that marks a single ML and DL run as present.
class FakeRegistry:
    def ml_run_exists(self, run_id: str) -> bool:
        return run_id == "run-ml"

    def dl_run_exists(self, run_id: str) -> bool:
        return run_id == "run-dl"


# Validate the explanation policy hook returns the requested boolean unchanged.
def test_explain_service_policy_hook() -> None:
    service = ExplainService()

    assert service.should_return_explanation(True) is True
    assert service.should_return_explanation(False) is False


# Validate PredictService normalizes text, resolves the run, and forwards the correct family.
def test_predict_service_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    service = PredictService(inference_service=FakeInferenceService(), repo_root=tmp_path)
    service.registry = FakeRegistry()
    service.run_resolver.resolve = lambda run_id, model_family: SimpleNamespace(run_id=run_id, model_family=model_family)

    result = service.predict(
        text="abc\r\nxyz",
        model_family="ml",
        run_id="run-ml",
        content_type="news",
        top_k=5,
        return_explanation=True,
    )

    assert result["model_family"] == "ml"
    assert result["run_id"] == "run-ml"


# Validate PredictService wraps resolver and inference errors into service errors.
def test_predict_service_error_translation(tmp_path: Path) -> None:
    service = PredictService(inference_service=FakeInferenceService(), repo_root=tmp_path)

    class BrokenResolver:
        def resolve(self, run_id: str, model_family: str):
            raise RunResolverError("resolver failed", status_code=404, error_code="RUN_ID_NOT_FOUND")

    service.run_resolver = BrokenResolver()

    with pytest.raises(PredictServiceError) as exc_info:
        service.predict(
            text="abc",
            model_family="ml",
            run_id="run-ml",
            content_type="news",
            top_k=5,
            return_explanation=False,
        )

    assert exc_info.value.error_code == "RUN_ID_NOT_FOUND"
