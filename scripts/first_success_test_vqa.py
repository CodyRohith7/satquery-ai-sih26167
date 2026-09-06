#!/usr/bin/env python3
"""FIRST SUCCESS TEST (Phase 2B, hybrid strategy): one real image, one real
natural-language question, through SmolVLM-256M-Instruct, on THIS machine's
real CPU. Must be run on the actual target machine, in a real terminal, with
the ML dependencies (see requirements.txt) installed - it cannot be run in a
network-isolated, GPU-less test environment (see docs/rs_adaptation.md
section 7 for why).

This script does not train anything and does not claim adaptation. It exists
solely to answer, with real measurements and no assumptions: does this model
load on this machine, does it produce an answer, how long does it take, how
much RAM does it use, and what happens if something goes wrong. Every field
in the JSON it writes is either a real measured value or an honest null/error
string - never a guess, and never silently skipped.

Usage:
    python scripts/first_success_test_vqa.py --image path\\to\\real_satellite_image.png --question "What is the dominant land cover in this image?"

Refuses to silently accept one of this project's own synthetic fixtures as
the "real satellite image" (pass --allow-synthetic to override, only for a
pure code-sanity check - not a substitute for the real test).
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to ONE real image file (not a synthetic fixture).")
    parser.add_argument("--question", default="What is the dominant land cover or feature visible in this image?")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                         help="float32 is the safer default for CPU correctness; try bfloat16 if RAM is tight.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", default="first_success_test_vqa_result.json")
    parser.add_argument("--allow-synthetic", action="store_true",
                         help="Bypass the synthetic-fixture check. Only for a code-sanity run, not the real test.")
    args = parser.parse_args()

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "test_name": "first_success_test_vqa",
        "model_id": args.model_id,
        "image_path": args.image,
        "question": args.question,
        "started_at_utc": _now_iso(),
        "status": "NOT_STARTED",
        "model_load_success": None,
        "model_load_seconds": None,
        "inference_seconds": None,
        "output_text": None,
        "backend": None,
        "torch_version": None,
        "transformers_version": None,
        "ram_baseline_mb": None,
        "ram_after_load_mb": None,
        "ram_after_inference_mb": None,
        "peak_working_set_mb": None,
        "error": None,
    }

    if not os.path.isfile(args.image):
        result["status"] = "FAILED"
        result["error"] = f"--image path does not exist: {args.image}"
        _write_result(args.output, result)
        return 1

    if ("fixtures" in args.image.replace("\\", "/")) and not args.allow_synthetic:
        result["status"] = "FAILED"
        result["error"] = (
            "This path looks like one of this project's own synthetic fixtures "
            "(data/fixtures/...), not a real satellite image. The Phase 2B "
            "instruction is explicit: 'one REAL satellite image'. Pass "
            "--allow-synthetic only if you intend a pure code-sanity check, "
            "and say so when reporting results - it does not count as the "
            "real first success test."
        )
        _write_result(args.output, result)
        return 1

    try:
        import psutil
    except ImportError:
        result["status"] = "FAILED"
        result["error"] = "psutil not installed - see requirements-ml-minimal.txt"
        _write_result(args.output, result)
        return 1

    proc = psutil.Process(os.getpid())
    result["ram_baseline_mb"] = round(proc.memory_info().rss / (1024 ** 2), 1)

    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import transformers as _tf
    except ImportError as exc:
        result["status"] = "FAILED"
        result["error"] = f"missing dependency: {exc}. Install requirements-ml-minimal.txt first."
        _write_result(args.output, result)
        return 1

    result["torch_version"] = torch.__version__
    result["transformers_version"] = _tf.__version__
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result["backend"] = device
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    try:
        t0 = time.perf_counter()
        processor = AutoProcessor.from_pretrained(args.model_id)
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            _attn_implementation="eager",  # flash_attention_2 requires CUDA; this machine has none
        ).to(device)
        model.eval()
        result["model_load_seconds"] = round(time.perf_counter() - t0, 3)
        result["model_load_success"] = True
    except Exception as exc:  # noqa: BLE001 - deliberately broad: report ANY load failure honestly
        result["model_load_success"] = False
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["ram_after_load_mb"] = round(proc.memory_info().rss / (1024 ** 2), 1)
        _write_result(args.output, result)
        return 1

    result["ram_after_load_mb"] = round(proc.memory_info().rss / (1024 ** 2), 1)

    try:
        image = Image.open(args.image).convert("RGB")
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": args.question}],
        }]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        result["inference_seconds"] = round(time.perf_counter() - t0, 3)

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        result["output_text"] = generated_text
        result["status"] = "SUCCESS"
    except Exception as exc:  # noqa: BLE001
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["ram_after_inference_mb"] = round(proc.memory_info().rss / (1024 ** 2), 1)
    mem_info = proc.memory_info()
    peak = getattr(mem_info, "peak_wset", None)  # Windows-specific field; None elsewhere
    if peak is not None:
        result["peak_working_set_mb"] = round(peak / (1024 ** 2), 1)
    result["finished_at_utc"] = _now_iso()

    _write_result(args.output, result)
    return 0 if result["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
