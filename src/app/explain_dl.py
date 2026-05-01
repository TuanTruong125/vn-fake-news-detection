from __future__ import annotations

from typing import Any, Callable

import torch


# Centralized DL explanation construction logic for predict flow.
def build_dl_explanation(
    *,
    tokenizer: Any,
    encoded: Any,
    base_score: float,
    raw_decision_score: float | None,
    top_k_per_direction: int,
    return_explanation: bool,
    explanation_enabled: bool,
    include_decomposition: bool,
    score_fn: Callable[[torch.Tensor, torch.Tensor], tuple[float, float | None]],
) -> dict[str, Any]:
    if not explanation_enabled:
        return {
            "explanation_available": False,
            "explanation_reason": "Explanation disabled by config.",
            "top_features_towards_fake": [],
            "top_features_towards_real": [],
            "explanation_decomposition": None,
        }
    if not return_explanation:
        return {
            "explanation_available": False,
            "explanation_reason": "Explanation disabled by request.",
            "top_features_towards_fake": [],
            "top_features_towards_real": [],
            "explanation_decomposition": None,
        }

    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]
    seq_len = int(attention_mask.sum().item())
    if seq_len <= 2:
        return {
            "explanation_available": False,
            "explanation_reason": "Input too short for token-level explanation.",
            "top_features_towards_fake": [],
            "top_features_towards_real": [],
            "explanation_decomposition": None,
        }

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        mask_id = tokenizer.pad_token_id
    if mask_id is None:
        mask_id = tokenizer.unk_token_id
    if mask_id is None:
        return {
            "explanation_available": False,
            "explanation_reason": "Tokenizer does not expose a replacement token id for occlusion.",
            "top_features_towards_fake": [],
            "top_features_towards_real": [],
            "explanation_decomposition": None,
        }

    candidate_positions: list[int] = []
    for pos in range(seq_len):
        token_id = int(input_ids[pos].item())
        if token_id in special_ids:
            continue
        candidate_positions.append(pos)
    if not candidate_positions:
        return {
            "explanation_available": False,
            "explanation_reason": "No explainable non-special tokens found.",
            "top_features_towards_fake": [],
            "top_features_towards_real": [],
            "explanation_decomposition": None,
        }

    max_candidates = min(len(candidate_positions), max(8, top_k_per_direction * 2))
    candidate_positions = candidate_positions[:max_candidates]

    towards_fake: list[dict[str, Any]] = []
    towards_real: list[dict[str, Any]] = []
    sum_positive = 0.0
    sum_negative = 0.0

    for pos in candidate_positions:
        token_id = int(input_ids[pos].item())
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        masked_ids = input_ids.clone()
        masked_ids[pos] = mask_id
        masked_score, _ = score_fn(masked_ids.unsqueeze(0), encoded["attention_mask"])
        contribution = float(base_score - masked_score)
        row = {
            "feature": str(token),
            "value": 1.0,
            "weight": contribution,
            "contribution": contribution,
            "direction": "towards_fake" if contribution >= 0 else "towards_real",
        }
        if contribution >= 0:
            sum_positive += contribution
            towards_fake.append(row)
        else:
            sum_negative += contribution
            towards_real.append(row)

    towards_fake.sort(key=lambda item: item["contribution"], reverse=True)
    towards_real.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    decomposition = None
    if include_decomposition:
        decomposition = {
            "sum_positive_contrib": float(sum_positive),
            "sum_negative_contrib": float(sum_negative),
            "sum_total_contrib": float(sum_positive + sum_negative),
            "intercept": 0.0,
            "estimated_decision_score": float(base_score),
            "raw_decision_score": float(raw_decision_score) if raw_decision_score is not None else None,
            "decision_score_gap": None,
            "approximate_score_alignment_note": (
                "DL explanation uses token occlusion deltas on probability score; not directly additive like linear ML."
            ),
        }

    return {
        "explanation_available": bool(towards_fake or towards_real),
        "explanation_reason": None if (towards_fake or towards_real) else "No meaningful token contributions found.",
        "top_features_towards_fake": towards_fake[:top_k_per_direction],
        "top_features_towards_real": towards_real[:top_k_per_direction],
        "explanation_decomposition": decomposition,
    }

