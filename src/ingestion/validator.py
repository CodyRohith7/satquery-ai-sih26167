"""Input validation / compatibility checking.

Produces a ValidationResult that the router and CLI both consult. Validation never
raises for "this combination doesn't make sense" - it returns `ok=False` with a
human-readable reason, because an unsupported-input error is a normal, expected
outcome the UI has to render gracefully (PS requirement: "unsupported modality
detection", "incompatible-image detection", "missing-pair detection").
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

from ingestion.metadata import ImageMetadata


@dataclasses.dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = dataclasses.field(default_factory=list)
    warnings: List[str] = dataclasses.field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


MAX_REASONABLE_DIM = 20000
MIN_REASONABLE_DIM = 8


def validate_single(meta: ImageMetadata) -> ValidationResult:
    result = ValidationResult(ok=True)
    if meta.width < MIN_REASONABLE_DIM or meta.height < MIN_REASONABLE_DIM:
        result.add_error(
            f"Image too small to analyze ({meta.width}x{meta.height}); "
            f"minimum is {MIN_REASONABLE_DIM}x{MIN_REASONABLE_DIM}."
        )
    if meta.width > MAX_REASONABLE_DIM or meta.height > MAX_REASONABLE_DIM:
        result.add_warning(
            f"Very large image ({meta.width}x{meta.height}); consider tiling."
        )
    if meta.band_count < 1:
        result.add_error("Image reports zero bands - file is likely corrupt.")
    if meta.modality_confidence == "unknown":
        result.add_warning(
            "Modality could not be determined and was not declared; "
            "downstream confidence will be penalized (see confidence engine)."
        )
    return result


def validate_pair(
    meta_a: ImageMetadata,
    meta_b: ImageMetadata,
    intended_relationship: Optional[str] = None,
) -> ValidationResult:
    """intended_relationship: 'bitemporal' | 'cross_modal' | None (unspecified)."""
    result = ValidationResult(ok=True)
    for m in (meta_a, meta_b):
        sub = validate_single(m)
        result.errors.extend(sub.errors)
        result.warnings.extend(sub.warnings)
    if result.errors:
        result.ok = False

    if (meta_a.width, meta_a.height) != (meta_b.width, meta_b.height):
        result.add_warning(
            f"Image pair has mismatched dimensions "
            f"({meta_a.width}x{meta_a.height} vs {meta_b.width}x{meta_b.height}); "
            "pixel-level comparison will resize the second image to match the "
            "first, which reduces geometric precision."
        )

    if intended_relationship == "bitemporal":
        if not (meta_a.declared_date and meta_b.declared_date):
            result.add_warning(
                "Bi-temporal analysis requested but one or both acquisition dates "
                "were not declared; change results cannot be dated precisely."
            )
        elif meta_a.declared_date == meta_b.declared_date:
            result.add_error(
                "Bi-temporal change analysis requires two different acquisition "
                "dates; both inputs declared the same date."
            )

    if intended_relationship == "cross_modal":
        modalities = {meta_a.modality, meta_b.modality}
        declared_pair = meta_a.declared_modality and meta_b.declared_modality
        if not declared_pair:
            result.add_warning(
                "Optical+SAR fusion requested but modality was not explicitly "
                "declared for both inputs; routing is proceeding on an "
                "unconfirmed heuristic guess."
            )
        elif meta_a.declared_modality == meta_b.declared_modality:
            result.add_error(
                f"Optical+SAR fusion requires two different modalities; both "
                f"inputs were declared as '{meta_a.declared_modality}'."
            )

    return result
