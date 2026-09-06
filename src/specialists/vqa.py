"""tool_single_image_vqa_v0

Answers open-ended questions about a single image by clustering it into
land-cover-like regions (see land_cover_heuristics.py) and composing a
natural-language description from the cluster proportions. This does NOT do
free-form question answering - it always produces a scene-description-style
answer, honestly, regardless of the exact question wording. See
docs/rs_adaptation.md for the real VQA model this is a stand-in for.
"""
from __future__ import annotations

import time

from ingestion.metadata import ImageMetadata
from routing.schemas import ConfidenceResult, SpecialistOutput
from confidence import engine as confidence_engine
from evidence import composer
from specialists.land_cover_heuristics import cluster_scene, summarize_clusters


def run(
    array,
    meta: ImageMetadata,
    query: str,
    validation_warnings: list,
    router_confidence: float,
    evidence_out_path: str,
) -> SpecialistOutput:
    start = time.time()

    label_map, centers, silhouette = cluster_scene(array)
    fractions = summarize_clusters(label_map, centers)
    sorted_labels = sorted(fractions.items(), key=lambda kv: kv[1], reverse=True)

    parts = [f"{label.replace('_', ' ')} (~{frac * 100:.0f}%)" for label, frac in sorted_labels if frac > 0.02]
    answer = (
        "Based on colour/texture clustering, the dominant regions in this image "
        "are: " + ", ".join(parts) + ". "
        "(v0 classical baseline: RGB colour heuristics, not a calibrated "
        "multispectral or deep vision-language classification - see "
        "docs/rs_adaptation.md.)"
    )

    evidence_item = composer.save_segmentation(
        array, label_map, evidence_out_path,
        description="Colour/texture cluster map used to derive the scene description.",
    )

    base_signal = (silhouette + 1) / 2  # rescale [-1,1] -> [0,1] per docs/confidence.md
    confidence = confidence_engine.compute(
        base_signal=base_signal,
        validation_warnings=validation_warnings,
        any_modality_unknown=(meta.modality_confidence == "unknown"),
        router_confidence=router_confidence,
    )

    return SpecialistOutput(
        answer_text=answer,
        evidence=[evidence_item],
        confidence=confidence,
        raw={"cluster_fractions": fractions, "silhouette_score": silhouette},
        tool_name="tool_single_image_vqa_v0",
        task_type="single_image_vqa",
        latency_seconds=round(time.time() - start, 4),
    )
