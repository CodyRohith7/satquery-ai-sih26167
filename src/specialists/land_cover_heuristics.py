"""Shared color-heuristic land-cover labeling, used by both the VQA and fusion
specialists so their vocabulary (and their honesty about limitations) stays
consistent.

This is a DOCUMENTED, DELIBERATE simplification: real land-cover classification
uses calibrated multispectral indices (NDVI, NDWI, etc.) computed from
radiometrically corrected bands. We have uncalibrated 8-bit RGB-ish arrays (see
docs/network_constraints.md - no real Sentinel data was reachable), so this
module uses plain RGB/HSV heuristics instead and says so in every answer it
produces. It is good enough to distinguish clearly-different synthetic regions
in the test fixtures, and is architected so a real per-band NDVI/NDWI
calculation slots in later without changing its callers (same signature).
"""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

LABELS = ("water", "vegetation", "urban_or_bare", "other")


def label_cluster(mean_bgr: np.ndarray) -> str:
    """mean_bgr: length-3 array, BGR order, 0-255 scale."""
    b, g, r = [float(x) for x in mean_bgr[:3]]
    total = b + g + r + 1e-6

    if b / total > 0.40 and b > r:
        return "water"
    if g / total > 0.40 and g > r and g > b:
        return "vegetation"
    if abs(r - g) < 20 and abs(g - b) < 20 and r > 90:
        # low colour saturation, mid-to-bright -> concrete/roads/built-up
        return "urban_or_bare"
    if r / total > 0.38 and r >= g >= b:
        return "urban_or_bare"  # soil / arid terrain grouped with bare/built-up
    return "other"


def cluster_scene(
    array: np.ndarray, n_clusters: int = 4, sample_stride: int = 2
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Returns (label_map[h,w] int, cluster_centers[k,3] BGR, silhouette_score).

    Uses scikit-learn KMeans on a downsampled pixel set for speed, then assigns
    every pixel in the full-resolution image to its nearest center - this keeps
    the evidence mask at full resolution while keeping the fit itself fast.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    img = array
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] > 3:
        img = img[..., :3]
    bgr = img[..., ::-1].astype(np.float64)

    sample = bgr[::sample_stride, ::sample_stride].reshape(-1, 3)
    # Guard against degenerate (near-constant) images.
    if np.unique(sample, axis=0).shape[0] < n_clusters:
        n_clusters = max(1, np.unique(sample, axis=0).shape[0])

    if n_clusters < 2:
        flat_labels = np.zeros(bgr.shape[:2], dtype=int)
        return flat_labels, sample.mean(axis=0, keepdims=True), 0.0

    km = KMeans(n_clusters=n_clusters, n_init=4, random_state=42)
    sample_labels = km.fit_predict(sample)

    # Silhouette on a further subsample for speed if the sample is large.
    if sample.shape[0] > 2000:
        idx = np.random.default_rng(42).choice(sample.shape[0], 2000, replace=False)
        sil = silhouette_score(sample[idx], sample_labels[idx])
    else:
        sil = silhouette_score(sample, sample_labels)

    full_flat = bgr.reshape(-1, 3)
    full_labels = km.predict(full_flat).reshape(bgr.shape[:2])
    return full_labels, km.cluster_centers_, float(sil)


def summarize_clusters(
    label_map: np.ndarray, centers: np.ndarray
) -> Dict[str, float]:
    """Returns {semantic_label: fraction_of_image}, merging clusters that map
    to the same semantic label."""
    total = label_map.size
    fractions: Dict[str, float] = {}
    for cluster_id in range(centers.shape[0]):
        frac = float((label_map == cluster_id).sum()) / total
        sem_label = label_cluster(centers[cluster_id])
        fractions[sem_label] = fractions.get(sem_label, 0.0) + frac
    return fractions
