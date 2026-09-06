"""Confidence computation. Implements docs/confidence.md exactly - if you change
the formula here, update that document in the same change.

`compute()` is shared infrastructure across ALL specialists - classical-CV and
real-model alike. What varies per specialist is `base_signal` (a real, computed
number - see docs/confidence.md's per-specialist section) and now, explicitly,
`method_version` + `basis_description`: the label and plain-English meaning of
that base_signal, so a UI can display an honest explanation without having to
infer it from which tool ran. The downgrade formula (validation warnings,
unknown modality, ambiguous routing) is genuinely common - the same penalties
apply whether base_signal came from a clustering separation score or from a
model's own generation-token probabilities - so it is not duplicated per
method.
"""
from __future__ import annotations

from typing import List

from routing.schemas import ConfidenceResult

METHOD_VERSION = "v0_classical"  # kept for backward compatibility with existing call sites
METHOD_CLASSICAL = "v0_classical"
METHOD_VLM_TOKEN_CONFIDENCE = "v1_vlm_mean_token_probability"

_CLASSICAL_BASIS_DESCRIPTION = (
    "Classical-CV signal (cluster separation or threshold quality) - not a "
    "calibrated probability from a trained deep model. See docs/confidence.md."
)

MAX_WARNING_DOWNGRADE = 0.30
WARNING_DOWNGRADE_PER_ITEM = 0.10
UNKNOWN_MODALITY_DOWNGRADE = 0.15
AMBIGUOUS_ROUTING_DOWNGRADE = 0.20
ROUTER_CONFIDENCE_AMBIGUOUS_THRESHOLD = 0.6


def compute(
    base_signal: float,
    validation_warnings: List[str],
    any_modality_unknown: bool,
    router_confidence: float,
    method_version: str = METHOD_CLASSICAL,
    basis_description: str = _CLASSICAL_BASIS_DESCRIPTION,
) -> ConfidenceResult:
    base_signal = max(0.0, min(1.0, base_signal))

    downgrades: List[str] = []
    total_downgrade = 0.0

    if any_modality_unknown:
        total_downgrade += UNKNOWN_MODALITY_DOWNGRADE
        downgrades.append(
            f"modality unknown for at least one input (-{UNKNOWN_MODALITY_DOWNGRADE:.2f})"
        )

    if validation_warnings:
        w = min(
            len(validation_warnings) * WARNING_DOWNGRADE_PER_ITEM,
            MAX_WARNING_DOWNGRADE,
        )
        total_downgrade += w
        downgrades.append(
            f"{len(validation_warnings)} validation warning(s) (-{w:.2f})"
        )

    if router_confidence < ROUTER_CONFIDENCE_AMBIGUOUS_THRESHOLD:
        total_downgrade += AMBIGUOUS_ROUTING_DOWNGRADE
        downgrades.append(
            f"router itself was not confident in the chosen branch "
            f"(router_confidence={router_confidence:.2f}) "
            f"(-{AMBIGUOUS_ROUTING_DOWNGRADE:.2f})"
        )

    total_downgrade = min(total_downgrade, 0.9)
    final = max(0.0, min(1.0, base_signal * (1 - total_downgrade)))

    return ConfidenceResult(
        value=round(final, 4),
        method_version=method_version,
        basis={
            "base_signal": round(base_signal, 4),
            "total_downgrade": round(total_downgrade, 4),
        },
        basis_description=basis_description,
        downgrades=downgrades,
    )
