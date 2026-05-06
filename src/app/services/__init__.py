"""Services package exports.

This module exposes the high-level service classes used by the API
to keep imports compact (e.g. `from src.app.services import PredictService`).
"""

from .predict_service import PredictService, PredictServiceError

__all__ = ["PredictService", "PredictServiceError"]
