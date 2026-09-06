#!/usr/bin/env python3
"""Grounding-QUALITY diagnostic for Florence-2-base on a real satellite
image, run after second_success_test_grounding.py reported: model load
VERIFIED, generation VERIFIED (no crash), but both "farmland" and
"buildings" produced the identical near-full-image bounding box
([1.6949999, 0.2259999, 676.9829, 451.3219]) - i.e. grounding QUALITY is not
verified. This script does NOT assume the cause. It runs a small, ordered
set of experiments and records raw evidence for each, so the actual
conclusion (A: our invocation/preprocessing bug, B: our post-processing
bug, or C: a real Florence-2-base/domain limitation) is read off real
output, not asserted.

Per the explicit instructions this script was written to satisfy:
  1. Uses the exact official Florence-2 model-card invocation
     (https://huggingface.co/microsoft/Florence-2-base "How to Get
     Started" / <CAPTION_TO_PHRASE_GROUNDING> example) as the FIRST thing
     tried for each task - max_new_tokens=1024, do_sample=False,
     num_beams=3 - not this project's compatibility fallback cascade,
     unless that literal call fails (in which case it falls back through
     generate_florence2()'s documented candidates and records which one
     was actually needed).
  2. Logs the real image dimensions fed to the processor and to
     post_process_generation's image_size, so a preprocessing/scaling
     mismatch would be visible directly rather than assumed absent.
  3. Records RAW generated text (skip_special_tokens=False - the raw
     <loc_NNNN> token stream) for every call, not just the parsed boxes -
     this is the single most direct way to see whether two different
     phrases actually produced different model output that happens to
     parse to the same box (a post-processing bug) versus genuinely
     identical output (the model not distinguishing the phrases at all).
  4. Runs <OD> (plain object detection, no phrase - the single simplest
     official Florence-2 example there is) on the SAME image first, as the
     baseline sanity check the instructions asked for: if Florence-2-base
     cannot find ANY differentiated sub-regions in this image at all, that
     is evidence pointing at C, independent of anything about how our
     phrase-grounding call is built.
  5. Also runs <DENSE_REGION_CAPTION> (no phrase) as a second, independent
     no-phrase baseline - multiple official Florence-2 tasks failing to
     localize anything in this specific image is stronger evidence than
     one.
  6. Tries BOTH a bare single-word phrase ("farmland") and a caption-style
     phrase ("a large area of farmland") for each concept - the official
     model-card example itself uses a full caption-style phrase ("A green
     car parked in front of a yellow building."), not a bare noun, so this
     directly tests whether phrase FORMAT (not phrase choice) explains the
     degenerate output.
  7. Computes each returned box's fraction of total image area
     numerically (not by eyeballing coordinates), so "is this basically
     the whole image" has an actual number attached.

Does NOT: hardcode any box, use src/grounding/grounding.py (the classical
baseline), perturb any returned box, or claim grounding works because
<loc_*> tokens were merely generated - success is judged by whether boxes
are genuinely different AND spatially smaller than the full image, read
from this script's real output.

Usage:
    python scripts/diagnose_grounding_quality.py --image path\\to\\same_real_image.png
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


def _bbox_area_fraction(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    total = float(image_width * image_height)
    return round(area / total, 4) if total > 0 else None


def _draw_overlay(image, boxes_by_label, output_path):
    from PIL import ImageDraw

    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    colors = ["red", "lime", "cyan", "yellow", "magenta", "orange"]
    for i, (label, result) in enumerate(boxes_by_label.items()):
        color = colors[i % len(colors)]
        for box in result.get("bboxes", []):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            draw.text((x1 + 3, max(0, y1 - 14)), label, fill=color)
    overlay.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-id", default="microsoft/Florence-2-base")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--output-json", default="diagnose_grounding_quality_result.json")
    parser.add_argument("--output-image", default="diagnose_grounding_quality_overlay.png")
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "test_name": "diagnose_grounding_quality",
        "model_id": args.model_id,
        "image_path": args.image,
        "started_at_utc": _now_iso(),
        "status": "NOT_STARTED",
        "model_load_success": None,
        "generation_diagnostics": None,
        "image_size_wh": None,
        "settled_generation_config": None,  # discovered once, reused for every experiment below
        "experiments": {},  # label -> {task_prompt, phrase, raw_generated_text, parsed, area_fraction(s), inference_seconds}
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

    missing = []
    for pkg in ("torch", "transformers", "einops", "timm", "PIL"):
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

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from models.florence2_compat import (
        load_florence2,
        Florence2LoadError,
        diagnose_generation_capability,
        generate_florence2,
        Florence2GenerationError,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    try:
        processor, model, compat_report = load_florence2(args.model_id, dtype, device)
        result["model_load_success"] = True
    except Florence2LoadError as exc:
        result["model_load_success"] = False
        result["status"] = "FAILED"
        result["error"] = str(exc)
        result["traceback"] = exc.full_traceback
        _write_result(args.output_json, result)
        return 1

    result["generation_diagnostics"] = diagnose_generation_capability(model)

    image = Image.open(args.image).convert("RGB")
    result["image_size_wh"] = [image.width, image.height]

    # (label, task_prompt, phrase_or_None)
    # Order matters: <OD> and <DENSE_REGION_CAPTION> run FIRST and take NO
    # phrase at all - they are the "can Florence-2-base localize anything
    # in this specific image" baseline the instructions asked for, run
    # before anything phrase-specific.
    experiments_to_run = [
        ("OD_simplest_official_baseline", "<OD>", None),
        ("DENSE_REGION_CAPTION_baseline", "<DENSE_REGION_CAPTION>", None),
        ("grounding_farmland_bareword", "<CAPTION_TO_PHRASE_GROUNDING>", "farmland"),
        ("grounding_buildings_bareword", "<CAPTION_TO_PHRASE_GROUNDING>", "buildings"),
        ("grounding_farmland_caption_style", "<CAPTION_TO_PHRASE_GROUNDING>", "a large area of farmland"),
        ("grounding_buildings_caption_style", "<CAPTION_TO_PHRASE_GROUNDING>", "a cluster of buildings"),
    ]

    settled_overrides = None
    boxes_by_label = {}

    try:
        for label, task_prompt, phrase in experiments_to_run:
            prompt = task_prompt if phrase is None else (task_prompt + phrase)
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

            t0 = time.perf_counter()
            if settled_overrides is None:
                # First call: use the exact official model-card invocation
                # (max_new_tokens=1024, do_sample=False, num_beams=3) as the
                # FIRST candidate generate_florence2 tries - it only falls
                # back to a different documented config if that literal
                # official call fails, and reports which one was used.
                generated_ids, generation_report = generate_florence2(
                    model, inputs, max_new_tokens=1024, num_beams=3, do_sample=False
                )
                settled_overrides = generation_report["generation_kwargs_used"]
                result["settled_generation_config"] = generation_report
            else:
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        **settled_overrides,
                    )
            inference_seconds = round(time.perf_counter() - t0, 3)

            raw_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(
                raw_text, task=task_prompt, image_size=(image.width, image.height)
            )
            parsed_for_task = parsed.get(task_prompt, {})

            entry = {
                "task_prompt": task_prompt,
                "phrase": phrase,
                "inference_seconds": inference_seconds,
                "raw_generated_text": raw_text,
                "parsed": parsed_for_task,
            }

            bboxes = parsed_for_task.get("bboxes") if isinstance(parsed_for_task, dict) else None
            if bboxes:
                entry["num_boxes"] = len(bboxes)
                entry["area_fractions"] = [
                    _bbox_area_fraction(b, image.width, image.height) for b in bboxes
                ]
                boxes_by_label[label] = parsed_for_task

            result["experiments"][label] = entry
            print(f"[{label}] inference={inference_seconds}s boxes={entry.get('num_boxes')} "
                  f"area_fractions={entry.get('area_fractions')} raw={raw_text[:200]!r}")

        result["status"] = "SUCCESS"

        raw_farmland = result["experiments"]["grounding_farmland_bareword"]["raw_generated_text"]
        raw_buildings = result["experiments"]["grounding_buildings_bareword"]["raw_generated_text"]
        result["farmland_vs_buildings_raw_text_identical"] = (raw_farmland == raw_buildings)

        if boxes_by_label:
            _draw_overlay(image, boxes_by_label, args.output_image)
            result["overlay_image_path"] = args.output_image

    except Florence2GenerationError as exc:
        result["status"] = "FAILED"
        result["error"] = str(exc)
        result["traceback"] = exc.full_traceback
    except Exception as exc:  # noqa: BLE001
        import traceback as _traceback
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = _traceback.format_exc()

    result["finished_at_utc"] = _now_iso()
    _write_result(args.output_json, result)
    return 0 if result["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
