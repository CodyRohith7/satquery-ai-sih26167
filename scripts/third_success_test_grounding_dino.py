#!/usr/bin/env python3
"""THIRD SUCCESS TEST (Phase 2B, grounding-model swap): real, phrase-
distinguishing visual grounding via `IDEA-Research/grounding-dino-tiny`, on
the same real satellite image used for the Florence-2 attempt.

WHY THIS SCRIPT EXISTS: Florence-2-base was diagnosed (with real, sourced
evidence - see docs/rs_adaptation.md section 8 and
scripts/diagnose_grounding_quality.py's results) as collapsing to a
~99%-of-image box regardless of task or phrase, on this specific kind of
satellite imagery - a MODEL/DOMAIN limitation, not a software bug, per the
explicit instruction accepting that diagnosis. This script tries a
DIFFERENT grounding model rather than continuing to debug Florence-2.

WHAT THIS MODEL IS, HONESTLY: `grounding-dino-tiny` (IDEA-Research,
Apache-2.0, ~0.2B params) is a REAL, general-purpose OPEN-VOCABULARY OBJECT
DETECTOR, natively supported by transformers
(`AutoModelForZeroShotObjectDetection` - no trust_remote_code, no custom
modeling file, unlike Florence-2). It is NOT remote-sensing-specific and
was NOT trained or fine-tuned on satellite/aerial imagery. Its output must
never be described as "remote-sensing-adapted grounding" - it is a genuine
foundation-model detector applied, as-is, to an out-of-domain image type.
`compat_report`/`result["model_is_remote_sensing_specific"]` records this
explicitly so nothing downstream can silently drop that distinction.

Candidate comparison that led here (full version in docs/rs_adaptation.md):
  1. Remote-sensing-specific SMALL grounding model: none found that is
     both small enough for CPU/8GB and available as a ready pretrained
     checkpoint - current RS visual-grounding research (GeoGround,
     GeoPixel, LHRS-Bot) is built on 7B+ VLM backbones.
  2. Lightweight RS segmentation/detection model: DOTA-style aerial object
     detectors (planes/ships/vehicles) exist and are small, but their
     fixed classes don't include land-cover concepts like "farmland" or
     generic "buildings" at all - a vocabulary mismatch with this PS's
     query style, not just a quality gap. Land-cover segmentation
     datasets with the right classes exist (LandCover.ai has "buildings";
     DeepGlobe has "agriculture") but no well-documented, general-purpose
     pretrained checkpoint for either was found - only a narrow,
     single-city community checkpoint with unrelated classes. Doing this
     properly means training a small segmentation head ourselves, which is
     real future work (closer to the LoRA/adaptation phase), not today's
     fastest path.
  3. Classical CV + explicitly-labeled ML baseline: still available as a
     fallback (src/grounding/grounding.py, untouched by this script).

grounding-dino-tiny was picked over immediately defaulting to (3) because
it is a genuine foundation model (not classical CV), handles arbitrary
text phrases directly (unlike a fixed-class detector), needs no custom
modeling code (unlike Florence-2 - avoids repeating the last three
debugging cycles), and is small/licensed/documented enough to try today.
It is explicitly NOT claimed to be domain-adapted.

WHAT SUCCESS REQUIRES (this script does not judge that for you - it
records real numbers so you can): genuinely different boxes for different
phrases, each substantially smaller than the full image (a real
area-fraction is computed and printed per box - "the model emitted
<loc>-equivalent output" is not treated as success). This script does not
hardcode, perturb, or invent any box, and does not touch
src/grounding/grounding.py (the classical baseline stays separate and
unused here).

INPUT FORMAT FIX (after the real-hardware run that got past model load but
failed before inference): the first version of this script called
`processor(images=image, text=[[formatted]], return_tensors="pt")` - a
list-of-lists, matching a convenience input shape shown in transformers'
"main"-branch documentation for GroundingDINO
(`text=[["a cat", "a remote control"]]`). On the actual pinned
transformers==4.57.1 this raised, at that exact call, before any model
forward pass ran:

    TypeError: TextEncodeInput must be Union[TextInputSequence,
    Tuple[InputSequence, InputSequence]]

This is a tokenizer-level type error (raised by the underlying Rust
`tokenizers` library, not by GroundingDINO's own code), confirming the
failure was about the SHAPE of the `text` argument, not about the model or
about `trust_remote_code`/version pinning/monkeypatching - none of which
this fix touches. Two candidate explanations were considered: either the
list-of-lists convenience shape isn't supported by 4.57.1's
`GroundingDinoProcessor.__call__`, or it is supported but requires an
outer batch dimension shaped differently than the docs example implies.
Rather than guess between those from documentation alone (this project's
prior WebFetch lookups against large or version-pinned source files have
proven unreliable more than once - see the RecursionError and generation-
compatibility fixes above, both of which needed direct reproduction
instead of trusting a fetched summary), this fix sidesteps the ambiguity
entirely: `_format_grounding_dino_prompt()` now returns a single, already
period-terminated, lowercase STRING (e.g. `"farmland."`), and the main
loop passes it directly as `text=formatted` - no list, no list-of-lists.
A plain string is the one input shape every version of
`GroundingDinoProcessor`/its underlying tokenizer is documented to accept
unambiguously (it is exactly the `TextInputSequence` named in the
tokenizer's own error message), and since this script only ever sends ONE
candidate phrase per forward call (farmland, then buildings, then the
control phrase, each as its own `model(**inputs)` call), there was never a
need for the list-of-lists batching shape in the first place - that shape
exists to send multiple phrases in a single batched call, which this
script deliberately does not do (each phrase gets its own timed inference
call so `inference_seconds` and `boxes`/`scores` stay per-phrase and
comparable). See `_format_grounding_dino_prompt()`'s own docstring below
for the code-level detail, and
`tests/integration/test_third_success_test_grounding_dino.py` for the
regression test that reproduces the original list-of-lists failure against
a stub tokenizer and confirms the plain-string format is accepted.

Usage:
    python scripts/third_success_test_grounding_dino.py --image path\\to\\same_real_image.png --target-phrase "farmland" --target-phrase-2 "buildings"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

RESULT_SCHEMA_VERSION = 1


def _now_iso():
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"


def _write_result(path, result):
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {path}")
    print(json.dumps(result, indent=2))


def _format_grounding_dino_prompt(phrase: str) -> str:
    """GroundingDINO's documented text-prompt convention: lowercase, ending
    in a period. Returns a PLAIN STRING, not a list or list-of-lists.

    ROOT CAUSE of the TypeError this replaces (see this script's module
    docstring "INPUT FORMAT FIX" section for the full account): the
    transformers "main"-branch docs for grounding-dino-tiny show a
    convenience `text=[["a cat", "a remote control"]]` (list-of-lists)
    input format, and an earlier version of this function followed that
    exact shape. That either isn't supported by the actual pinned
    transformers==4.57.1 release, or is handled differently than the dev
    docs describe - either way, passing it through
    `GroundingDinoProcessor.__call__` reached the underlying Rust
    `tokenizers` library with a structure it doesn't recognize, raising
    `TypeError: TextEncodeInput must be Union[TextInputSequence,
    Tuple[InputSequence, InputSequence]]` - a tokenizer-level type error,
    confirming this was a text-shape problem, not a model problem.

    The fix uses the plainest, most fundamental input type the processor
    and tokenizer both unambiguously support in every transformers version
    that ships GroundingDINO: a single already-merged string
    (`TextInputSequence` in the tokenizer's own error message). Since this
    script always sends ONE candidate phrase per forward pass (never
    multiple phrases combined in one call), a plain string needs no
    period-joining logic beyond ensuring the trailing period - there is no
    list to construct in the first place."""
    formatted = phrase.strip().lower()
    if not formatted.endswith("."):
        formatted += "."
    return formatted


def _bbox_area_fraction(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    total = float(image_width * image_height)
    return round(area / total, 4) if total > 0 else None


def _draw_overlay(image, boxes_by_phrase, output_path):
    from PIL import ImageDraw

    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    colors = ["red", "lime", "cyan", "yellow", "magenta"]
    for i, (phrase, boxes) in enumerate(boxes_by_phrase.items()):
        color = colors[i % len(colors)]
        for box in boxes:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            draw.text((x1 + 3, max(0, y1 - 14)), phrase, fill=color)
    overlay.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--target-phrase", required=True)
    parser.add_argument("--target-phrase-2", default=None)
    parser.add_argument("--control-phrase", default="an airplane",
                         help="A phrase that should plausibly NOT be present, as a sanity check that "
                              "the model doesn't just return a fixed box regardless of input.")
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--output-json", default="third_success_test_grounding_dino_result.json")
    parser.add_argument("--output-image", default="third_success_test_grounding_dino_overlay.png")
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()

    phrases = [p for p in (args.target_phrase, args.target_phrase_2, args.control_phrase) if p]

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "test_name": "third_success_test_grounding_dino",
        "model_id": args.model_id,
        "model_is_remote_sensing_specific": False,
        "model_license": "apache-2.0",
        "image_path": args.image,
        "phrases_tested": phrases,
        "control_phrase": args.control_phrase,
        "started_at_utc": _now_iso(),
        "status": "NOT_STARTED",
        "model_load_success": None,
        "model_load_seconds": None,
        "backend": None,
        "torch_version": None,
        "transformers_version": None,
        "ram_baseline_mb": None,
        "ram_after_load_mb": None,
        "ram_after_inference_mb": None,
        "peak_working_set_mb": None,
        "image_size_wh": None,
        "per_phrase": {},  # phrase -> {inference_seconds, num_detections, boxes, scores, area_fractions}
        "boxes_differ_between_target_phrases": None,
        "overlay_image_path": None,
        "note": (
            "grounding-dino-tiny is a general-purpose open-vocabulary object "
            "detector, NOT remote-sensing-specific and NOT fine-tuned on "
            "satellite/aerial imagery. This test does not claim RS-domain "
            "adaptation - see this script's module docstring."
        ),
        "error": None,
        "traceback": None,
    }

    if not os.path.isfile(args.image):
        result["status"] = "FAILED"
        result["error"] = f"--image path does not exist: {args.image}"
        _write_result(args.output_json, result)
        return 1

    if ("fixtures" in args.image.replace("\\", "/")) and not args.allow_synthetic:
        result["status"] = "FAILED"
        result["error"] = "This looks like a synthetic fixture, not a real image. Pass --allow-synthetic only for a code-sanity check."
        _write_result(args.output_json, result)
        return 1

    try:
        import psutil
    except ImportError:
        result["status"] = "FAILED"
        result["error"] = "psutil not installed - see requirements-ml-minimal.txt"
        _write_result(args.output_json, result)
        return 1

    proc = psutil.Process(os.getpid())
    result["ram_baseline_mb"] = round(proc.memory_info().rss / (1024 ** 2), 1)

    missing = []
    for pkg in ("torch", "transformers", "PIL"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        result["status"] = "FAILED"
        result["error"] = f"missing dependency/dependencies: {missing}"
        _write_result(args.output_json, result)
        return 1

    import torch
    from PIL import Image
    import transformers as _tf
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    result["torch_version"] = torch.__version__
    result["transformers_version"] = _tf.__version__
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result["backend"] = device

    try:
        t0 = time.perf_counter()
        processor = AutoProcessor.from_pretrained(args.model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(device)
        model.eval()
        result["model_load_seconds"] = round(time.perf_counter() - t0, 3)
        result["model_load_success"] = True
    except Exception as exc:  # noqa: BLE001
        import traceback as _traceback
        result["model_load_success"] = False
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = _traceback.format_exc()
        result["ram_after_load_mb"] = round(proc.memory_info().rss / (1024 ** 2), 1)
        _write_result(args.output_json, result)
        return 1

    result["ram_after_load_mb"] = round(proc.memory_info().rss / (1024 ** 2), 1)

    image = Image.open(args.image).convert("RGB")
    result["image_size_wh"] = [image.width, image.height]
    boxes_by_phrase = {}

    try:
        for phrase in phrases:
            # Plain string, not a list/list-of-lists - see
            # _format_grounding_dino_prompt's docstring for exactly why.
            formatted = _format_grounding_dino_prompt(phrase)

            inputs = processor(images=image, text=formatted, return_tensors="pt").to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = model(**inputs)
            inference_seconds = round(time.perf_counter() - t0, 3)

            post = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                target_sizes=[image.size[::-1]],  # (height, width) per current transformers docs
            )[0]

            boxes = [[round(v, 3) for v in b.tolist()] for b in post["boxes"]]
            scores = [round(s.item(), 4) for s in post["scores"]]
            labels = list(post.get("text_labels", post.get("labels", [])))

            result["per_phrase"][phrase] = {
                "formatted_prompt": formatted,
                "inference_seconds": inference_seconds,
                "num_detections": len(boxes),
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "area_fractions": [_bbox_area_fraction(b, image.width, image.height) for b in boxes],
            }
            if boxes:
                boxes_by_phrase[phrase] = boxes
            print(f"[{phrase!r} -> {formatted!r}] inference={inference_seconds}s "
                  f"detections={len(boxes)} boxes={boxes} scores={scores}")

        result["status"] = "SUCCESS"

        target_phrases = [p for p in (args.target_phrase, args.target_phrase_2) if p]
        if len(target_phrases) == 2:
            b1 = result["per_phrase"][target_phrases[0]]["boxes"]
            b2 = result["per_phrase"][target_phrases[1]]["boxes"]
            result["boxes_differ_between_target_phrases"] = (b1 != b2)

        if boxes_by_phrase:
            _draw_overlay(image, boxes_by_phrase, args.output_image)
            result["overlay_image_path"] = args.output_image

    except Exception as exc:  # noqa: BLE001
        import traceback as _traceback
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = _traceback.format_exc()

    result["ram_after_inference_mb"] = round(proc.memory_info().rss / (1024 ** 2), 1)
    mem_info = proc.memory_info()
    peak = getattr(mem_info, "peak_wset", None)
    if peak is not None:
        result["peak_working_set_mb"] = round(peak / (1024 ** 2), 1)
    result["finished_at_utc"] = _now_iso()

    _write_result(args.output_json, result)
    return 0 if result["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
