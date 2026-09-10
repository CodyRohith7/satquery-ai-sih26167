"""Standalone diagnostic for the SmolVLM ImportError fix (Issue 1, SIH26167).
Run this in the real 'satquery' conda env to get genuine proof the fix works
on this machine's actual transformers/torch install. Safe to delete anytime,
or keep as a permanent quick-check before a demo - it makes no repo changes.

Usage:  python scripts/verify_smolvlm_fix.py
Exit code: 0 = everything below succeeded for real. Non-zero = something
failed; read the printed traceback, it is never suppressed.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

print(f"Python: {sys.version}")

for _pkg in ("transformers", "torch", "PIL", "accelerate"):
    try:
        _mod = __import__(_pkg)
        _ver = getattr(_mod, "__version__", "unknown version")
        print(f"{_pkg}: {_ver}")
    except ImportError as exc:
        print(f"{_pkg}: NOT INSTALLED ({exc})")

print()
print("Checking `from transformers import AutoModelForImageTextToText` ...")
try:
    from transformers import AutoModelForImageTextToText  # noqa: F401
    print("  -> import OK")
except Exception as exc:  # noqa: BLE001
    print(f"  -> IMPORT FAILED: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    sys.exit(1)

print()
print("Running the real SmolVLM specialist end-to-end on a synthetic test image ...")
try:
    import numpy as np
    from PIL import Image as PILImage

    from ingestion.raster_io import RasterImage
    from ingestion.metadata import inspect as inspect_metadata
    from specialists import vqa_smolvlm

    fixture_path = os.path.join(os.path.dirname(__file__), "..", "data", "fixtures", "single_image.png")
    if os.path.exists(fixture_path):
        with PILImage.open(fixture_path) as img:
            img.load()
            array = np.array(img.convert("RGB"))
        img_path = fixture_path
    else:
        rng = np.random.default_rng(0)
        array = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
        img_path = "<synthetic in-memory image>"

    raster = RasterImage(
        path=img_path, array=array, band_count=3,
        height=array.shape[0], width=array.shape[1], dtype=str(array.dtype),
    )
    meta = inspect_metadata(raster, declared_modality="optical")

    t0 = time.perf_counter()
    result = vqa_smolvlm.run(
        array=array, meta=meta, query="What is the dominant land cover in this image?",
        validation_warnings=[], router_confidence=1.0,
        evidence_out_path="/tmp/verify_smolvlm_fix_evidence.png",
    )
    total_seconds = round(time.perf_counter() - t0, 3)

    print("  -> SUCCESS, model loaded and ran for real")
    print(f"  model_load_seconds: {result.raw.get('model_load_seconds')}")
    print(f"  inference_seconds:  {result.raw.get('inference_seconds')}")
    print(f"  total_seconds:      {total_seconds}")
    print(f"  answer_text:        {result.answer_text}")
    sys.exit(0)
except Exception as exc:  # noqa: BLE001 - report every failure, never swallow
    print(f"  -> FAILED: {type(exc).__name__}: {exc}")
    full_tb = getattr(exc, "full_traceback", None)
    print(full_tb if full_tb else traceback.format_exc())
    sys.exit(1)
