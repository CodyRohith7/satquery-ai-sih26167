"""Draws and saves the visual evidence every specialist output must carry.

Nothing in this module invents an image - every function takes a real numpy
array (the loaded input) and real geometry/mask computed by a specialist, and
saves a genuine overlay PNG next to the other run artifacts.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from routing.schemas import EvidenceItem


def _to_bgr_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize an arbitrary-band array to a displayable 3-channel uint8 image."""
    if arr.ndim == 2:
        arr3 = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 1:
        arr3 = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] >= 3:
        arr3 = arr[..., :3]
    else:
        raise ValueError(f"Cannot render array with shape {arr.shape}")

    arr3 = arr3.astype(np.float64)
    lo, hi = np.percentile(arr3, 1), np.percentile(arr3, 99)
    if hi <= lo:
        hi = lo + 1.0
    arr3 = np.clip((arr3 - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    # RGB -> BGR for OpenCV writing
    return arr3[..., ::-1].copy()


def to_displayable_rgb(arr: np.ndarray) -> np.ndarray:
    """Public wrapper around the same percentile-stretch normalization every
    evidence image uses, exposed so a UI layer can render a RAW input array
    (e.g. the optical or SAR image the analyst uploaded) consistently with
    how evidence images look - without duplicating the stretch logic and
    without reaching into the private `_to_bgr_uint8` helper from outside
    this module. Returns RGB (not BGR - `_to_bgr_uint8` is oriented for
    `cv2.imwrite`, this is oriented for direct display, e.g. Streamlit's
    `st.image`)."""
    return _to_bgr_uint8(arr)[..., ::-1].copy()


def save_boxes(
    array: np.ndarray,
    boxes: Sequence[Tuple[int, int, int, int]],
    label: str,
    out_path: str,
) -> EvidenceItem:
    """boxes: list of (x, y, w, h) in pixel space."""
    img = _to_bgr_uint8(array)
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), max(2, img.shape[0] // 200))
    if boxes:
        x, y, w, h = boxes[0]
        cv2.putText(
            img, label, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
    return EvidenceItem(
        kind="bounding_box",
        description=f"{len(boxes)} box(es) for '{label}'",
        image_path=out_path,
        geometry={"boxes": [list(b) for b in boxes], "label": label},
    )


def save_change_map(
    before: np.ndarray,
    after: np.ndarray,
    change_mask: np.ndarray,
    out_path: str,
) -> EvidenceItem:
    """Renders a 3-panel before/after/change-highlighted composite, like the
    'before / after / change map' layout the frozen spec asked for."""
    b = _to_bgr_uint8(before)
    a = _to_bgr_uint8(after)
    if a.shape[:2] != b.shape[:2]:
        # Mismatched-dimension pairs are a supported, explicitly-warned-about
        # case (see ingestion/validator.py's "pixel-level comparison will
        # resize the second image to match the first" warning) - the change
        # specialist's own diff computation already resizes the after image
        # to the before image's shape before differencing (see change.py).
        # This evidence composite must honor the same contract, or
        # np.concatenate() below raises on any mismatched pair.
        a = cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_LINEAR)
    if change_mask.shape[:2] != a.shape[:2]:
        change_mask = cv2.resize(
            change_mask.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    highlighted = a.copy()
    highlighted[change_mask.astype(bool)] = (0, 0, 255)
    composite = np.concatenate([b, a, highlighted], axis=1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, composite)

    changed_fraction = float(change_mask.astype(bool).mean())
    return EvidenceItem(
        kind="change_map",
        description=(
            f"before | after | change-highlighted composite; "
            f"{changed_fraction * 100:.1f}% of pixels flagged as changed"
        ),
        image_path=out_path,
        geometry={"changed_fraction": changed_fraction},
    )


def save_source_image(
    array: np.ndarray,
    out_path: str,
    description: str,
) -> EvidenceItem:
    """For specialists that answer in free-form text with no computed
    box/mask/change-map to draw (e.g. SmolVLM VQA) - saves the REAL source
    image that was actually analyzed as the evidence artifact, rather than
    inventing a fake overlay or leaving evidence empty."""
    img = _to_bgr_uint8(array)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
    return EvidenceItem(
        kind="overlay_image",
        description=description,
        image_path=out_path,
        geometry=None,
    )


def save_segmentation(
    base_array: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    description: str,
) -> EvidenceItem:
    """labels: 2D int array of cluster ids, same H/W as base_array."""
    base = _to_bgr_uint8(base_array)
    n_labels = int(labels.max()) + 1
    rng = np.random.default_rng(42)  # fixed seed -> reproducible colour map
    palette = rng.integers(40, 255, size=(n_labels, 3))
    color_map = palette[labels]
    if color_map.shape[:2] != base.shape[:2]:
        color_map = cv2.resize(
            color_map.astype(np.uint8), (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    overlay = cv2.addWeighted(base, 0.5, color_map.astype(np.uint8), 0.5, 0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, overlay)
    return EvidenceItem(
        kind="mask",
        description=description,
        image_path=out_path,
        geometry={"num_clusters": n_labels},
    )
