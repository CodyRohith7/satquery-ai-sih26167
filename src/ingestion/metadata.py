"""Metadata / modality inspection.

Hard rule (from the frozen Phase 0 spec): never invent metadata. If we can't
determine something, the field is `None` and callers must render "Metadata
unavailable" rather than guessing a sensor name.

Modality is the clearest example of where it's tempting to guess. A single-band
image COULD be SAR, COULD be a panchromatic optical band, COULD be a DEM. Pixel
statistics alone cannot tell them apart reliably without calibration metadata this
sandbox has no way to read (no rasterio/GDAL tags, see raster_io.py). So:

  - if the caller (CLI/UI) explicitly DECLARES a modality, we record it as
    `declared` and trust it (that's the analyst's job, same as a real ISRO/SAC
    workflow: an analyst knows if they're handing the system a RISAT SAR product).
  - otherwise we report an `inferred_guess` ONLY for the cases with defensible
    pixel evidence (3-band file that decodes as plausible RGB -> "optical
    (unconfirmed)"), and everything else is `"unknown"` — explicitly, not silently.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from ingestion.raster_io import RasterImage


@dataclasses.dataclass
class ImageMetadata:
    path: str
    height: int
    width: int
    band_count: int
    dtype: str
    crs: Optional[str]
    transform: Optional[tuple]
    declared_modality: Optional[str]  # what the caller told us, if anything
    inferred_modality: str  # our best HONEST guess label, see module docstring
    modality_confidence: str  # "declared" | "heuristic" | "unknown"
    declared_date: Optional[str] = None  # ISO date string, if the caller supplied one

    @property
    def modality(self) -> str:
        return self.declared_modality or self.inferred_modality

    def metadata_summary(self) -> dict:
        """What we render in the UI's metadata panel. Absent fields say so."""
        return {
            "dimensions": f"{self.width}x{self.height}",
            "band_count": self.band_count,
            "dtype": self.dtype,
            "crs": self.crs or "Metadata unavailable",
            "transform": self.transform or "Metadata unavailable",
            "modality": self.modality,
            "modality_basis": self.modality_confidence,
            "acquisition_date": self.declared_date or "Metadata unavailable",
        }


def _heuristic_modality(arr: np.ndarray, band_count: int) -> str:
    """A defensible, documented heuristic - NOT a claim of certainty."""
    if band_count == 1:
        return "single-band (unconfirmed: could be SAR, panchromatic, or a derived index)"
    if band_count == 3:
        return "optical (unconfirmed: 3-band file consistent with RGB)"
    if band_count == 4:
        return "multispectral (unconfirmed: 4-band file consistent with RGB+NIR)"
    if band_count > 4:
        return f"multispectral (unconfirmed: {band_count}-band file)"
    return "unknown"


def inspect(
    raster: RasterImage,
    declared_modality: Optional[str] = None,
    declared_date: Optional[str] = None,
) -> ImageMetadata:
    inferred = _heuristic_modality(raster.array, raster.band_count)
    confidence = "declared" if declared_modality else (
        "heuristic" if inferred != "unknown" else "unknown"
    )
    return ImageMetadata(
        path=raster.path,
        height=raster.height,
        width=raster.width,
        band_count=raster.band_count,
        dtype=raster.dtype,
        crs=raster.crs,
        transform=raster.transform,
        declared_modality=declared_modality,
        inferred_modality=inferred,
        modality_confidence=confidence,
        declared_date=declared_date,
    )
