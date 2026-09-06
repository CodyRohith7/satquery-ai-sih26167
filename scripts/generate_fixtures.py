#!/usr/bin/env python3
"""Generates SYNTHETIC test fixtures for the pipeline.

These are NOT real satellite imagery. Real Sentinel-2/BigEarthNet/SAR data was
unreachable from this development sandbox (see docs/network_constraints.md).
Every fixture here is procedurally generated with a fixed random seed so the
exact same files are reproduced by anyone who runs this script - that
reproducibility is the point: these fixtures exist to prove the PIPELINE works
end-to-end, not to demonstrate real-world land-cover accuracy.

Run: python3 scripts/generate_fixtures.py
Output: data/fixtures/*.png
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

SEED = 42
SIZE = 256
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "fixtures")


def _base_scene(rng: np.random.Generator) -> np.ndarray:
    """A synthetic 'landscape': brown soil background, a blue lake blob, a
    green vegetation patch, and a light-gray urban grid."""
    img = np.zeros((SIZE, SIZE, 3), dtype=np.float64)
    # Soil/background: warm brown with mild noise.
    img[:, :] = [139, 115, 85]
    img += rng.normal(0, 6, img.shape)

    yy, xx = np.mgrid[0:SIZE, 0:SIZE]

    # Lake: blue elliptical blob, bottom-left.
    lake_mask = ((xx - 70) / 55) ** 2 + ((yy - 180) / 40) ** 2 <= 1
    img[lake_mask] = [40, 70, 160]

    # Vegetation: green patch, top-right.
    veg_mask = ((xx - 180) / 60) ** 2 + ((yy - 70) / 50) ** 2 <= 1
    img[veg_mask] = [50, 120, 55]

    # Urban grid: light gray grid lines in a block, center.
    urban_block = (xx >= 100) & (xx <= 200) & (yy >= 120) & (yy <= 220)
    grid = ((xx % 12 < 3) | (yy % 12 < 3))
    img[urban_block & grid] = [190, 188, 182]
    img[urban_block & ~grid] = [150, 148, 142]

    return np.clip(img, 0, 255).astype(np.uint8)


def make_single_image(rng: np.random.Generator) -> np.ndarray:
    return _base_scene(rng)


def make_change_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    before = _base_scene(rng)
    after = before.astype(np.float64)
    # Simulate new construction: a gray rectangular block appears that wasn't
    # in `before`, entirely inside what was soil background.
    after[20:60, 190:240] = [170, 168, 162]
    after[20:60, 190:240] += rng.normal(0, 4, (40, 50, 3))
    return before, np.clip(after, 0, 255).astype(np.uint8)


def make_fusion_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Optical (RGB) + a plausible single-band SAR-like companion.

    SAR backscatter heuristic used here (documented, not claimed physically
    exact): water -> very low backscatter (smooth surface, specular
    reflection away from sensor); urban -> very high backscatter (corner/
    double-bounce reflection); vegetation/soil -> moderate, textured
    backscatter. Speckle noise (multiplicative, SAR-characteristic) is added.
    """
    optical = _base_scene(rng)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]

    sar = np.full((SIZE, SIZE), 90.0)  # moderate background backscatter
    sar += rng.normal(0, 8, sar.shape)  # textured background

    lake_mask = ((xx - 70) / 55) ** 2 + ((yy - 180) / 40) ** 2 <= 1
    sar[lake_mask] = 15.0

    urban_block = (xx >= 100) & (xx <= 200) & (yy >= 120) & (yy <= 220)
    sar[urban_block] = 220.0

    # Multiplicative speckle, characteristic of real SAR.
    speckle = rng.gamma(shape=4.0, scale=0.25, size=sar.shape)
    sar = sar * speckle

    return optical, np.clip(sar, 0, 255).astype(np.uint8)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    Image.fromarray(make_single_image(rng)).save(os.path.join(OUT_DIR, "single_image.png"))

    before, after = make_change_pair(rng)
    Image.fromarray(before).save(os.path.join(OUT_DIR, "change_before.png"))
    Image.fromarray(after).save(os.path.join(OUT_DIR, "change_after.png"))

    optical, sar = make_fusion_pair(rng)
    Image.fromarray(optical).save(os.path.join(OUT_DIR, "fusion_optical.png"))
    Image.fromarray(sar).save(os.path.join(OUT_DIR, "fusion_sar.png"))

    print(f"Wrote 5 synthetic fixture files to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
