"""tool_single_image_vqa_smolvlm_v1

Real deep-learning single-image VQA via `HuggingFaceTB/SmolVLM-256M-Instruct`,
using the exact loading/generation recipe verified on real CPU hardware in
`scripts/first_success_test_vqa.py` (real result: model load 46.891s,
inference 38.457s, peak working set 2882.7MB - see docs/rs_adaptation.md).

This is the REAL specialist for single-image VQA whenever torch/transformers
are importable (see `src/models/registry.py`'s dynamic availability check
for `tool_single_image_vqa_smolvlm_v1`) - it is not a demo or a fallback.
The classical KMeans-clustering baseline (`specialists/vqa.py`,
`tool_single_image_vqa_v0`) is the explicit, clearly-disclosed fallback for
when THIS model cannot load or run - see `specialists/vqa_dispatch.py`,
which is the module the router/CLI/UI actually call (never this module or
the classical one directly), so the fallback behavior lives in one place.

Model id, dtype, and attn_implementation are pinned to the exact
configuration verified on real hardware - do not change them without
re-verifying there; this is not incidental configuration, it's the
difference between "loads on this CPU" and "crashes."

CONFIDENCE: unlike the classical specialists (which derive base_signal from
a clustering separation score), this model doesn't produce one naturally -
so base_signal here is the mean max-softmax probability across the
generated tokens (from `model.generate(..., output_scores=True,
return_dict_in_generate=True)`), a real, model-native signal computed from
this specific answer's own generation - not a fabricated or fixed number.
"""
from __future__ import annotations

import time
from typing import Dict, Tuple

from ingestion.metadata import ImageMetadata
from routing.schemas import SpecialistOutput
from confidence import engine as confidence_engine
from evidence import composer

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
MAX_NEW_TOKENS = 64


class SmolVLMError(RuntimeError):
    """Raised on any load or inference failure. Always carries the full
    traceback (never just the exception message), per this project's
    established rule of reporting complete tracebacks on failure."""

    def __init__(self, message: str, full_traceback: str):
        super().__init__(message)
        self.full_traceback = full_traceback


_CACHE: Dict[str, Tuple] = {}  # model_id -> (processor, model, device, dtype)


def _adapt_unexpected_kwarg(exc: TypeError, kwargs: dict, renames: Dict[str, str]) -> bool:
    """Mutates `kwargs` in place to work around ONE unexpected-keyword
    TypeError from a from_pretrained() call whose accepted argument names
    differ across transformers releases - never used to hide any other
    kind of failure. Returns True if `kwargs` was changed (caller should
    retry the call with the new kwargs), False if this TypeError doesn't
    match the "unexpected keyword argument" shape or doesn't reference a
    kwarg we actually passed (caller should re-raise it unchanged)."""
    import re as _re

    match = _re.search(r"unexpected keyword argument '([^']+)'", str(exc))
    if not match:
        return False
    bad = match.group(1)
    if bad not in kwargs:
        return False
    value = kwargs.pop(bad)
    renamed = renames.get(bad)
    if renamed and renamed not in kwargs:
        kwargs[renamed] = value
    return True


def _load(model_id: str = MODEL_ID):
    """Loads once per process and caches - real load time is ~47s on the
    hardware this was verified on, so reloading per-query would make the UI
    unusable. Cache hits report model_load_seconds close to 0 in run()'s
    result, which is the honest number for that call, not a fabricated
    "always ~47s" claim."""
    if model_id in _CACHE:
        return _CACHE[model_id]

    import torch

    # transformers v5 removed AutoModelForVision2Seq in favor of
    # AutoModelForImageTextToText (confirmed root cause of a real, observed
    # "ImportError: cannot import name 'AutoModelForVision2Seq'" on a
    # transformers>=5 install - see docs/rs_adaptation.md). This is the
    # HF-documented current class for SmolVLM and has been available since
    # well before the >=4.46 floor this project already requires, so no
    # requirements.txt change is needed alongside this fix.
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
    except ImportError as exc:  # noqa: BLE001 - report ANY import failure honestly, never swallow
        import traceback as _traceback
        raise SmolVLMError(f"{type(exc).__name__}: {exc}", _traceback.format_exc()) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # verified default on CPU - see scripts/first_success_test_vqa.py

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        # Exact accepted from_pretrained() kwarg names have drifted across
        # transformers releases (dtype/torch_dtype, attn_implementation/
        # _attn_implementation). Rather than guess which this environment's
        # installed version wants, adapt to the REAL TypeError it actually
        # raises - renaming a kwarg once if a known replacement exists,
        # otherwise dropping it - and retry. Any other exception (or a
        # TypeError we don't recognize) still propagates below with a full
        # traceback; this never masks a genuine load failure.
        load_kwargs = {
            "dtype": dtype,
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",  # flash_attention_2 requires CUDA; this project targets CPU
        }
        renames = {"dtype": "torch_dtype", "attn_implementation": "_attn_implementation"}
        while True:
            try:
                model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs).to(device)
                break
            except TypeError as exc:
                if not _adapt_unexpected_kwarg(exc, load_kwargs, renames):
                    raise
        model.eval()
    except Exception as exc:  # noqa: BLE001 - report ANY load failure honestly, never swallow
        import traceback as _traceback
        raise SmolVLMError(f"{type(exc).__name__}: {exc}", _traceback.format_exc()) from exc

    _CACHE[model_id] = (processor, model, device, dtype)
    return _CACHE[model_id]


def _mean_token_confidence(scores) -> float:
    import torch

    if not scores:
        return 0.0
    per_token = [torch.softmax(s[0], dim=-1).max().item() for s in scores]
    return sum(per_token) / len(per_token)


def run(
    array,
    meta: ImageMetadata,
    query: str,
    validation_warnings: list,
    router_confidence: float,
    evidence_out_path: str,
) -> SpecialistOutput:
    start = time.time()

    import numpy as np
    import torch
    from PIL import Image

    was_cached = MODEL_ID in _CACHE
    t_load0 = time.perf_counter()
    processor, model, device, dtype = _load()
    load_seconds = round(time.perf_counter() - t_load0, 3)

    img_array = array
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    image = Image.fromarray(img_array[..., :3].astype(np.uint8)).convert("RGB")

    try:
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": query}],
        }]
        # tokenize=False is explicit (not relied on as a default) because
        # transformers v5 changed apply_chat_template's return shape when
        # tokenize=True (now a BatchEncoding dict); this code has always
        # assumed a plain string prompt to pass into processor(text=...).
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                output_scores=True,
                return_dict_in_generate=True,
            )
        inference_seconds = round(time.perf_counter() - t0, 3)

        # Decode only the newly generated continuation (excluding the input
        # prompt tokens the chat template built) - the standard way to avoid
        # echoing the prompt back as if it were part of the model's answer.
        new_tokens = gen_out.sequences[:, prompt_len:]
        generated_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        full_decoded_text = processor.batch_decode(gen_out.sequences, skip_special_tokens=True)[0]
        mean_confidence = _mean_token_confidence(gen_out.scores)
    except Exception as exc:  # noqa: BLE001
        import traceback as _traceback
        raise SmolVLMError(f"{type(exc).__name__}: {exc}", _traceback.format_exc()) from exc

    answer = (
        f"{generated_text} "
        "(SmolVLM-256M-Instruct, real CPU inference - a genuine "
        "vision-language model, not remote-sensing-domain-adapted; "
        "see docs/rs_adaptation.md.)"
    )

    evidence_item = composer.save_source_image(
        array, evidence_out_path,
        description="Source image passed to SmolVLM-256M-Instruct for this answer.",
    )

    num_generated_tokens = new_tokens.shape[1] if hasattr(new_tokens, "shape") else None
    confidence = confidence_engine.compute(
        base_signal=mean_confidence,
        validation_warnings=validation_warnings,
        any_modality_unknown=(meta.modality_confidence == "unknown"),
        router_confidence=router_confidence,
        method_version=confidence_engine.METHOD_VLM_TOKEN_CONFIDENCE,
        basis_description=(
            "Mean of the top predicted token's softmax probability across "
            f"all {num_generated_tokens if num_generated_tokens is not None else len(generated_text.split())} "
            "newly generated tokens in THIS answer's generation - a measure "
            "of how 'peaked'/certain SmolVLM's own next-token distribution "
            "was while writing this specific answer. This measures "
            "GENERATION certainty, NOT factual correctness: a fluently, "
            "confidently-worded wrong answer can still score high here, and "
            "a correct answer phrased with rarer wording can score lower. "
            "It is not a calibrated probability that the answer's content "
            "is accurate - see docs/confidence.md."
        ),
    )

    return SpecialistOutput(
        answer_text=answer,
        evidence=[evidence_item],
        confidence=confidence,
        raw={
            "model_id": MODEL_ID,
            "backend": device,
            "dtype": str(dtype),
            "model_was_already_loaded": was_cached,
            "model_load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "mean_token_confidence": round(mean_confidence, 4),
            "generated_text": generated_text,
            "full_decoded_text_including_prompt": full_decoded_text,
        },
        tool_name="tool_single_image_vqa_smolvlm_v1",
        task_type="single_image_vqa",
        latency_seconds=round(time.time() - start, 4),
        model_name=MODEL_ID,
        fallback_occurred=False,
        fallback_reason=None,
    )
