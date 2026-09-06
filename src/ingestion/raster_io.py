"""Image/raster loading.

Design note: this module is deliberately the *only* place that knows how to turn a
file on disk into a numpy array. Everything downstream (metadata, validation,
specialists) works on the `RasterImage` object this module produces, so that when
`rasterio`/GDAL become available (see docs/network_constraints.md) we only need to
extend `load()` here — nothing else in the codebase has to change.

Current backend: Pillow only. Pillow can read PNG/JPEG/TIFF pixel data but cannot
read geospatial metadata (CRS, affine transform, per-band wavelength/sensor info).
We do NOT fabricate that metadata — see `ingestion/metadata.py`.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Optional

import numpy as np
from PIL import Image

try:
    import rasterio  # type: ignore

    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False


class RasterLoadError(RuntimeError):
    """Raised when a file cannot be read as an image/raster at all."""


@dataclasses.dataclass
class RasterImage:
    """A loaded raster plus everything we could honestly determine about it.

    `crs` and `transform` are `None` unless rasterio is available AND the file
    actually carries that metadata — never guessed.
    """

    path: str
    array: np.ndarray  # shape (H, W) or (H, W, C), dtype as read from file
    band_count: int
    height: int
    width: int
    dtype: str
    crs: Optional[str] = None
    transform: Optional[tuple] = None
    backend: str = "pillow"


def _load_with_pillow(path: str) -> RasterImage:
    try:
        with Image.open(path) as img:
            img.load()
            arr = np.array(img)
    except Exception as exc:  # noqa: BLE001 - we re-raise as our own type
        raise RasterLoadError(f"Could not read '{path}' as an image: {exc}") from exc

    if arr.ndim == 2:
        band_count = 1
        height, width = arr.shape
    elif arr.ndim == 3:
        height, width, band_count = arr.shape
    else:
        raise RasterLoadError(
            f"Unsupported array shape {arr.shape} read from '{path}'"
        )

    return RasterImage(
        path=path,
        array=arr,
        band_count=band_count,
        height=height,
        width=width,
        dtype=str(arr.dtype),
        crs=None,
        transform=None,
        backend="pillow",
    )


def _load_with_rasterio(path: str) -> RasterImage:
    with rasterio.open(path) as src:  # type: ignore
        arr = src.read()  # (bands, H, W)
        arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        crs = str(src.crs) if src.crs else None
        transform = tuple(src.transform)[:6] if src.transform else None

    band_count = 1 if arr.ndim == 2 else arr.shape[-1]
    height, width = arr.shape[0], arr.shape[1]
    return RasterImage(
        path=path,
        array=arr,
        band_count=band_count,
        height=height,
        width=width,
        dtype=str(arr.dtype),
        crs=crs,
        transform=transform,
        backend="rasterio",
    )


def load(path: str) -> RasterImage:
    """Load an image/raster file. Raises RasterLoadError if it can't be read at all."""
    if not os.path.isfile(path):
        raise RasterLoadError(f"File not found: '{path}'")

    if _RASTERIO_AVAILABLE:
        try:
            return _load_with_rasterio(path)
        except Exception:
            # Fall through to Pillow - e.g. a plain PNG rasterio doesn't like.
            pass
    return _load_with_pillow(path)


def rasterio_available() -> bool:
    return _RASTERIO_AVAILABLE
