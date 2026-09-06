"""Dispatches single-image VQA to the real deep-learning specialist
(`tool_single_image_vqa_smolvlm_v1`, SmolVLM-256M-Instruct) whenever it is
available, falling back to the classical baseline (`tool_single_image_vqa_v0`)
only when SmolVLM cannot load or run - and always disclosing, IN THE
RETURNED SpecialistOutput ITSELF (not just a log line), which one actually
produced the answer. `answer_text.tool_name` always names the specialist
that genuinely ran, and a fallback stamps `raw["smolvlm_attempted"]`,
`raw["smolvlm_fallback_reason"]`, and `raw["smolvlm_fallback_traceback"]`
with the real cause - never a silent swap.

`app/cli.py` (and the UI built on top of it) call THIS module for
`single_image_vqa`, never `specialists.vqa.run()` or
`specialists.vqa_smolvlm.run()` directly, so the fallback behavior lives in
exactly one place and can't be bypassed or duplicated.
"""
from __future__ import annotations

import traceback as _traceback

from ingestion.metadata import ImageMetadata
from routing.schemas import SpecialistOutput
from models import registry
from specialists import vqa as vqa_classical
from specialists import vqa_smolvlm

SMOLVLM_TOOL_NAME = "tool_single_image_vqa_smolvlm_v1"


def smolvlm_available() -> bool:
    """True only if the registry's dynamic dependency check (torch +
    transformers importable) found SmolVLM's specialist genuinely runnable
    in this environment - never assumed."""
    try:
        entry = registry.get(SMOLVLM_TOOL_NAME)
    except KeyError:
        return False
    return entry.status == registry.AVAILABLE


def run(
    array,
    meta: ImageMetadata,
    query: str,
    validation_warnings: list,
    router_confidence: float,
    evidence_out_path: str,
) -> SpecialistOutput:
    if not smolvlm_available():
        # Dependencies aren't importable in this environment at all - go
        # straight to the classical baseline. Nothing was attempted, so no
        # fallback disclosure is added (there is nothing to disclose a
        # fallback FROM).
        return vqa_classical.run(
            array, meta, query, validation_warnings, router_confidence, evidence_out_path
        )

    try:
        return vqa_smolvlm.run(
            array, meta, query, validation_warnings, router_confidence, evidence_out_path
        )
    except Exception as exc:  # noqa: BLE001 - deliberately broad: ANY real-model
        # failure (OOM, a download error, an unexpected API break on this
        # transformers version) falls back to the classical baseline rather
        # than crashing the whole query - but the fallback is always
        # disclosed in the output itself, never silent.
        full_tb = getattr(exc, "full_traceback", None) or _traceback.format_exc()
        reason = f"{type(exc).__name__}: {exc}"
        fallback_output = vqa_classical.run(
            array, meta, query, validation_warnings, router_confidence, evidence_out_path
        )
        # Explicit, first-class schema fields - a UI must never have to infer
        # fallback status from raw's key presence/absence.
        fallback_output.fallback_occurred = True
        fallback_output.fallback_reason = reason
        # Kept in raw too, for the technical-details panel / full traceback.
        fallback_output.raw["smolvlm_attempted"] = True
        fallback_output.raw["smolvlm_fallback_reason"] = reason
        fallback_output.raw["smolvlm_fallback_traceback"] = full_tb
        fallback_output.answer_text = (
            "[SmolVLM-256M-Instruct failed to produce an answer for this query - "
            "fell back to the classical baseline below. See "
            "raw.smolvlm_fallback_reason for the real cause.] "
        ) + fallback_output.answer_text
        return fallback_output
