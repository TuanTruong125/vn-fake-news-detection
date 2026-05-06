from __future__ import annotations


# Common inference-related exceptions and orchestration logic shared across services.
class InferenceError(Exception):
    def __init__(self, message: str, status_code: int = 500, error_code: str = "INFERENCE_INTERNAL_ERROR") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
