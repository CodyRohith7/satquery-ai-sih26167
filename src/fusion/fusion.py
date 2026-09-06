"""tool_fusion_v0

Optical + SAR fusion. Builds a per-pixel feature vector from BOTH inputs
(normalized optical RGB + normalized SAR intensity) and clusters that JOINT
feature space - so the clustering genuinely depends on both modalities, not
just one with the other along for show. Labels combine the RGB colour
heuristic with a SAR-backscatter heuristic (water = low backscatter AND
blue-dominant; built-up = high backscatter AND low colour saturation), because
real SAR/optical fusion is valuable precisely because a single modality can be
ambiguous (e.g. a dark optical region could be water OR shadow; SAR backscatter
resolves that).
"""
from __future__ import annotations

import time
from typing import Dict

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ingestion.metadata import ImageMetadata
from routing.schemas import SpecialistOutput
from confidence import engine as confidence_engine
from evidence import composer
from specialists.land_cover_heuristics import label_cluster


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float64)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def _label_fused_cluster(mean_rgb_0_1: np.ndarray, mean_sar_0_1: float) -> str:
    r, g, b = mean_rgb_0_1
    if mean_sar_0_1 < 0.35 and b >= r:
        return "water (low SAR backscatter + blue-dominant optical)"
    if mean_sar_0_1 > 0.6 and abs(r - g) < 0.15 and abs(g - b) < 0.15:
        return "built-up (high SAR backscatter + low optical colour saturation)"
    if g > r and g > b:
        return "vegetation (optical colour; moderate SAR backscatter)"
    return "other / unclassified"


def run(
    optical_array: np.ndarray,
    sar_array: np.ndarray,
    meta_optical: ImageMetadata,
    meta_sar: ImageMetadata,
    query: str,
    validation_warnings: list,
    router_confidence: float,
    evidence_out_path: str,
    n_clusters: int = 3,
) -> SpecialistOutput:
    start = time.time()

    rgb = optical_array[..., :3] if optical_array.ndim == 3 else np.stack([optical_array] * 3, axis=-1)
    sar = sar_array if sar_array.ndim == 2 else sar_array[..., 0]

    if sar.shape != rgb.shape[:2]:
        sar = cv2.resize(sar.astype(np.float32), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    rgb_n = _normalize(rgb)
    sar_n = _normalize(sar)

    h, w = sar_n.shape
    stride = max(1, int(np.sqrt(h * w / 4000)))  # keep the KMeans fit fast
    fused = np.concatenate([rgb_n, sar_n[..., None]], axis=-1)
    sample = fused[::stride, ::stride].reshape(-1, 4)

    k = min(n_clusters, max(1, np.unique(sample.round(2), axis=0).shape[0]))
    if k < 2:
        labels_full = np.zeros((h, w), dtype=int)
        silhouette = 0.0
        centers = sample.mean(axis=0, keepdims=True)
    else:
        km = KMeans(n_clusters=k, n_init=4, random_state=42)
        sample_labels = km.fit_predict(sample)
        if sample.shape[0] > 2000:
            idx = np.random.default_rng(42).choice(sample.shape[0], 2000, replace=False)
            silhouette = float(silhouette_score(sample[idx], sample_labels[idx]))
        else:
            silhouette = float(silhouette_score(sample, sample_labels))
        labels_full = km.predict(fused.reshape(-1, 4)).reshape(h, w)
        centers = km.cluster_centers_

    fractions: Dict[str, float] = {}
    for cid in range(centers.shape[0]):
        frac = float((labels_full == cid).sum()) / labels_full.size
        sem = _label_fused_cluster(centers[cid, :3], centers[cid, 3])
        fractions[sem] = fractions.get(sem, 0.0) + frac

    sorted_labels = sorted(fractions.items(), key=lambda kv: kv[1], reverse=True)
    parts = [f"{label} (~{frac * 100:.0f}%)" for label, frac in sorted_labels if frac > 0.02]
    answer = (
        "Joint optical+SAR clustering identifies: " + "; ".join(parts) + ". "
        "Labels use SAR backscatter to disambiguate cases optical colour alone "
        "cannot (e.g. dark water vs. shadow). (v0 baseline: k-means over a "
        "hand-built [R,G,B,SAR] feature vector, not a learned fusion network - "
        "see docs/rs_adaptation.md.)"
    )

    evidence_item = composer.save_segmentation(
        optical_array, labels_full, evidence_out_path,
        description="Fused optical+SAR cluster map (colour = joint cluster assignment).",
    )

    confidence = confidence_engine.compute(
        base_signal=(silhouette + 1) / 2,
        validation_warnings=validation_warnings,
        any_modality_unknown=(
            meta_optical.modality_confidence == "unknown"
            or meta_sar.modality_confidence == "unknown"
        ),
        router_confidence=router_confidence,
    )

    return SpecialistOutput(
        answer_text=answer,
        evidence=[evidence_item],
        confidence=confidence,
        raw={"cluster_fractions": fractions, "silhouette_score": silhouette},
        tool_name="tool_fusion_v0",
        task_type="optical_sar_fusion",
        latency_seconds=round(time.time() - start, 4),
    )
