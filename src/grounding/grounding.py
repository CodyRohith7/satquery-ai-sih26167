"""tool_grounding_v0

Locates a queried feature via HSV colour thresholding + contour extraction.
Supported target vocabulary is intentionally small and explicit (see
_TARGET_KEYWORDS) - if the query doesn't mention a recognized target, this
returns a low-confidence "target not recognized" result rather than guessing.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ingestion.metadata import ImageMetadata
from routing.schemas import ConfidenceResult, SpecialistOutput
from confidence import engine as confidence_engine
from evidence import composer

# (target label, keywords that trigger it, HSV lower bound, HSV upper bound)
_TARGET_KEYWORDS = [
    ("water", ("water", "lake", "river", "reservoir"), (90, 40, 40), (140, 255, 255)),
    ("vegetation", ("vegetation", "forest", "field", "crop", "green"), (35, 30, 30), (85, 255, 255)),
    ("urban_or_bare", ("urban", "built-up", "building", "road", "city"), (0, 0, 120), (180, 60, 220)),
]


def _find_target(query: str) -> Optional[Tuple[str, tuple, tuple]]:
    q = query.lower()
    for label, keywords, lo, hi in _TARGET_KEYWORDS:
        if any(kw in q for kw in keywords):
            return label, lo, hi
    return None


def run(
    array,
    meta: ImageMetadata,
    query: str,
    validation_warnings: list,
    router_confidence: float,
    evidence_out_path: str,
) -> SpecialistOutput:
    start = time.time()

    target = _find_target(query)
    img = array
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    rgb = img[..., :3].astype(np.uint8)

    if target is None:
        answer = (
            "The query did not clearly name a recognized target "
            f"({', '.join(t[0] for t in _TARGET_KEYWORDS)}), so no region can "
            "be grounded. (v0 baseline: fixed colour-keyword vocabulary only.)"
        )
        evidence_item = composer.save_boxes(array, [], "no target recognized", evidence_out_path)
        confidence = confidence_engine.compute(
            base_signal=0.0,
            validation_warnings=validation_warnings,
            any_modality_unknown=(meta.modality_confidence == "unknown"),
            router_confidence=router_confidence,
        )
        return SpecialistOutput(
            answer_text=answer,
            evidence=[evidence_item],
            confidence=confidence,
            raw={"target": None},
            tool_name="tool_grounding_v0",
            task_type="grounding",
            latency_seconds=round(time.time() - start, 4),
        )

    label, lo, hi = target
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    raw_mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    raw_area = int((raw_mask > 0).sum())

    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned_area = int((cleaned > 0).sum())

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 25]

    survival_ratio = (cleaned_area / raw_area) if raw_area > 0 else 0.0

    if boxes:
        answer = (
            f"Found {len(boxes)} candidate region(s) matching '{label}' via HSV "
            f"colour thresholding; largest region covers "
            f"{100 * cv2.contourArea(contours[0]) / array.shape[0] / array.shape[1]:.1f}% "
            "of the image. (v0 baseline: colour thresholding, not a learned "
            "grounding model - see docs/rs_adaptation.md.)"
        )
    else:
        answer = (
            f"Query asked to locate '{label}', but no coherent region survived "
            "colour thresholding + cleanup - the feature may not be present, or "
            "may not match this v0 baseline's colour assumptions."
        )

    evidence_item = composer.save_boxes(array, boxes, label, evidence_out_path)

    confidence = confidence_engine.compute(
        base_signal=survival_ratio if boxes else 0.05,
        validation_warnings=validation_warnings,
        any_modality_unknown=(meta.modality_confidence == "unknown"),
        router_confidence=router_confidence,
    )

    return SpecialistOutput(
        answer_text=answer,
        evidence=[evidence_item],
        confidence=confidence,
        raw={
            "target": label,
            "raw_area_px": raw_area,
            "cleaned_area_px": cleaned_area,
            "survival_ratio": survival_ratio,
            "num_boxes": len(boxes),
        },
        tool_name="tool_grounding_v0",
        task_type="grounding",
        latency_seconds=round(time.time() - start, 4),
    )
