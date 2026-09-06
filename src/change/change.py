"""tool_change_v0

Classic image-differencing change detection: grayscale absolute difference,
Otsu threshold, morphological cleanup, connected-component stats. This is a
real, well-established remote-sensing change-detection baseline technique - not
a deep CDVQA model (see docs/rs_adaptation.md), but not a toy either.
"""
from __future__ import annotations

import time
from typing import Tuple

import cv2
import numpy as np
from skimage.filters import threshold_otsu

from ingestion.metadata import ImageMetadata
from routing.schemas import SpecialistOutput
from confidence import engine as confidence_engine
from evidence import composer


def _to_gray(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array.astype(np.float64)
    rgb = array[..., :3].astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)


def _otsu_separation_score(diff: np.ndarray, thresh: float) -> float:
    """Between-class variance / total variance at the Otsu threshold - a
    standard measure of how bimodal (i.e. how confidently separable) the
    difference image is. 0 = no separation, 1 = perfectly bimodal."""
    below = diff[diff <= thresh]
    above = diff[diff > thresh]
    if below.size == 0 or above.size == 0:
        return 0.0
    total_var = diff.var()
    if total_var <= 1e-9:
        return 0.0
    w0, w1 = below.size / diff.size, above.size / diff.size
    between_class_var = w0 * w1 * (below.mean() - above.mean()) ** 2
    return float(min(1.0, between_class_var / total_var))


def run(
    before_array: np.ndarray,
    after_array: np.ndarray,
    meta_before: ImageMetadata,
    meta_after: ImageMetadata,
    query: str,
    validation_warnings: list,
    router_confidence: float,
    evidence_out_path: str,
) -> SpecialistOutput:
    start = time.time()

    gray_before = _to_gray(before_array)
    gray_after = _to_gray(after_array)
    if gray_before.shape != gray_after.shape:
        gray_after = cv2.resize(
            gray_after.astype(np.float32),
            (gray_before.shape[1], gray_before.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float64)

    blurred_before = cv2.GaussianBlur(gray_before.astype(np.float32), (5, 5), 0)
    blurred_after = cv2.GaussianBlur(gray_after.astype(np.float32), (5, 5), 0)
    diff = np.abs(blurred_after - blurred_before)

    try:
        thresh = float(threshold_otsu(diff))
    except ValueError:
        thresh = float(diff.mean())

    raw_mask = (diff > thresh).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    num_regions = max(0, num_labels - 1)  # exclude background label 0
    significant_regions = int((stats[1:, cv2.CC_STAT_AREA] > 20).sum()) if num_labels > 1 else 0

    changed_fraction = float((cleaned > 0).mean())
    separation_score = _otsu_separation_score(diff, thresh)

    date_before = meta_before.declared_date or "an undated earlier acquisition"
    date_after = meta_after.declared_date or "an undated later acquisition"
    answer = (
        f"Comparing {date_before} to {date_after}: {changed_fraction * 100:.1f}% "
        f"of the scene shows a significant intensity change, in "
        f"{significant_regions} distinct region(s) after cleanup. "
        "(v0 baseline: grayscale image differencing + Otsu threshold - a "
        "standard classical change-detection technique, not a learned "
        "CDVQA model; see docs/rs_adaptation.md.)"
    )

    evidence_item = composer.save_change_map(
        before_array, after_array, cleaned > 0, evidence_out_path
    )

    confidence = confidence_engine.compute(
        base_signal=separation_score,
        validation_warnings=validation_warnings,
        any_modality_unknown=(
            meta_before.modality_confidence == "unknown"
            or meta_after.modality_confidence == "unknown"
        ),
        router_confidence=router_confidence,
    )

    return SpecialistOutput(
        answer_text=answer,
        evidence=[evidence_item],
        confidence=confidence,
        raw={
            "changed_fraction": changed_fraction,
            "num_regions": significant_regions,
            "otsu_threshold": thresh,
            "otsu_separation_score": separation_score,
        },
        tool_name="tool_change_v0",
        task_type="bitemporal_change",
        latency_seconds=round(time.time() - start, 4),
    )
