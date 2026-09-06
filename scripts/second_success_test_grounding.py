#!/usr/bin/env python3
"""SECOND SUCCESS TEST (Phase 2B, hybrid strategy): real model-based
grounding via Florence-2-base's native phrase-grounding task, on the same
real image used for the VQA test. Must be run on the real target machine,
in a real terminal, with einops + timm installed on top of the core ML
dependencies (see requirements-optional.txt) - it cannot be run in a
network-isolated, GPU-less test environment (see docs/rs_adaptation.md
section 7 for why).

This deliberately does NOT import or fall back to this project's classical
grounding baseline (src/grounding/grounding.py) - that tool stays exactly
as-is, unaffected, as its own separate v0 baseline. This script is a
standalone real-model test, not a replacement wired into the app yet.

What "genuinely generated, not hardcoded" means here and how this script
proves it: pass --target-phrase and --target-phrase-2 (two DIFFERENT real
phrases about the same image). If the returned bounding boxes are actually a
function of the model's forward pass on the given text, they should differ
between the two prompts. This script runs both and reports whether the boxes
differ - that comparison, not a single run, is the actual evidence.

Florence-2 ships as custom modeling code (trust_remote_code=True) rather than
a class built into transformers itself - that is a real trust decision, not
a formality, and is called out here rather than silently added.

Known CPU compatibility issues this script works around (both handled by
src/models/florence2_compat.py, not inline here - see that module's
docstring for the full sourced explanation of each):
  1. Florence-2's default config requests flash_attn, uninstallable without
     CUDA.
  2. A confirmed transformers-side regression (~4.50+) where
     'PreTrainedModel' no longer provides a default '_supports_sdpa' class
     attribute that Florence-2's custom code still expects to inherit,
     raising AttributeError during load. The loader tries plain
     attn_implementation="eager" first and only applies the documented
     one-line compatibility patch if that specific AttributeError still
     occurs - see florence2_compat.py for exactly what is and isn't patched.

Usage:
    python scripts/second_success_test_grounding.py --image path\\to\\same_real_image.png --target-phrase "farmland" --target-phrase-2 "buildings"
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


def _draw_overlay(image, boxes_by_phrase, output_path):
    """boxes_by_phrase: {phrase: {"bboxes": [[x1,y1,x2,y2], ...], "labels": [...]}}
    Draws every phrase's boxes in a different color so multiple grounding
    results on one image stay visually distinguishable."""
    from PIL import ImageDraw

    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    colors = ["red", "lime", "cyan", "yellow", "magenta"]
    for i, (phrase, result) in enumerate(boxes_by_phrase.items()):
        color = colors[i % len(colors)]
        for box in result.get("bboxes", []):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            draw.text((x1 + 3, max(0, y1 - 14)), phrase, fill=color)
    overlay.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to the same real image used for the VQA test.")
    parser.add_argument("--target-phrase", required=True, help="Real grounding query phrase, e.g. 'farmland'.")
    parser.add_argument("--target-phrase-2", default=None,
                         help="A second, DIFFERENT phrase - run to prove the boxes are model-generated, not fixed.")
    parser.add_argument("--model-id", default="microsoft/Florence-2-base")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--output-json", default="second_success_test_grounding_result.json")
    parser.add_argument("--output-image", default="second_success_test_grounding_overlay.png")
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "test_name": "second_success_test_grounding",
        "model_id": args.model_id,
        "image_path": args.image,
        "target_phrases": [p for p in (args.target_phrase, args.target_phrase_2) if p],
        "started_at_utc": _now_iso(),
        "status": "NOT_STARTED",
        "model_load_success": None,
        "model_load_seconds": None,
        "compat_report": None,
        "generation_diagnostics": None,  # model class/MRO/GenerationMixin facts - see florence2_compat.py
        "generation_report": None,       # which generate() config actually worked, and every failed attempt
        "backend": None,
        "torch_version": None,
        "transformers_version": None,
        "ram_baseline_mb": None,
        "ram_after_load_mb": None,
        "ram_after_inference_mb": None,
        "peak_working_set_mb": None,
        "per_phrase": {},   # phrase -> {inference_seconds, raw_generated_text, parsed}
        "boxes_differ_between_phrases": None,
        "overlay_image_path": None,
        "note_this_is_not_the_classical_baseline": (
            "This test calls microsoft/Florence-2-base directly. It does not "
            "import or use src/grounding/grounding.py (the v0 classical-CV "
            "baseline), per the explicit instruction not to use that baseline "
            "for this test."
        ),
        "error": None,
        "traceback": None,  # full traceback text on any failure - never just str(exc)
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
    for pkg in ("torch", "transformers", "einops", "timm"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        result["status"] = "FAILED"
        result["error"] = f"missing dependency/dependencies: {missing}. Run: pip install einops timm"
        _write_result(args.output_json, result)
        return 1

    import torch
    from PIL import Image
    import transformers as _tf

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from models.florence2_compat import (
        load_florence2,
        Florence2LoadError,
        diagnose_generation_capability,
        generate_florence2,
        Florence2GenerationError,
    )

    result["torch_version"] = torch.__version__
    result["transformers_version"] = _tf.__version__
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result["backend"] = device
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    try:
        t0 = time.perf_counter()
        processor, model, compat_report = load_florence2(args.model_id, dtype, device)
        result["model_load_seconds"] = round(time.perf_counter() - t0, 3)
        result["model_load_success"] = True
        result["compat_report"] = compat_report
    except Florence2LoadError as exc:
        result["model_load_success"] = False
        result["status"] = "FAILED"
        result["error"] = str(exc)
        result["traceback"] = getattr(exc, "full_traceback", None)
        result["ram_after_load_mb"] = round(proc.memory_info().rss / (1024 ** 2), 1)
        _write_result(args.output_json, result)
        return 1

    result["ram_after_load_mb"] = round(proc.memory_info().rss / (1024 ** 2), 1)
    result["generation_diagnostics"] = diagnose_generation_capability(model)

    image = Image.open(args.image).convert("RGB")
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    boxes_by_phrase = {}
    # Discovered on the FIRST phrase (see generate_florence2's ordered
    # candidate search), then reused directly for subsequent phrases - both
    # so we don't re-pay the search cost per phrase, and so the evidence
    # shows ONE fixed, documented configuration working for both prompts,
    # not different configs happening to work per call.
    settled_overrides = None

    try:
        for phrase in result["target_phrases"]:
            prompt = task_prompt + phrase
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

            t0 = time.perf_counter()
            if settled_overrides is None:
                generated_ids, generation_report = generate_florence2(
                    model, inputs,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=False,
                )
                result["generation_report"] = generation_report
                settled_overrides = generation_report["generation_kwargs_used"]
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
            parsed_for_task = parsed.get(task_prompt, {"bboxes": [], "labels": []})

            result["per_phrase"][phrase] = {
                "inference_seconds": inference_seconds,
                "raw_generated_text": raw_text,
                "parsed": parsed_for_task,
            }
            boxes_by_phrase[phrase] = parsed_for_task

        result["status"] = "SUCCESS"

        phrases = result["target_phrases"]
        if len(phrases) == 2:
            b1 = boxes_by_phrase[phrases[0]].get("bboxes", [])
            b2 = boxes_by_phrase[phrases[1]].get("bboxes", [])
            result["boxes_differ_between_phrases"] = (b1 != b2)

        _draw_overlay(image, boxes_by_phrase, args.output_image)
        result["overlay_image_path"] = args.output_image

    except Florence2GenerationError as exc:
        result["status"] = "FAILED"
        result["error"] = str(exc)
        result["generation_report"] = {"failed_attempts": exc.attempts}
        # Every candidate's own traceback, not just the point where this
        # wrapper exception was raised - see Florence2GenerationError.
        result["traceback"] = exc.full_traceback
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
