from __future__ import annotations


# Centralized explanation-policy hook for predict flow orchestration.
class ExplainService:
    def should_return_explanation(self, requested: bool) -> bool:
        return bool(requested)
