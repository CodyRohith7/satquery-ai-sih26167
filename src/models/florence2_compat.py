"""Compatibility loader for microsoft/Florence-2-* on CPU.

Florence-2 ships as custom modeling code (loaded via `trust_remote_code=True`)
that Microsoft has not updated to track every transformers internal refactor.
Two SPECIFIC, well-documented breaks are known to hit CPU-only setups, and
this module works around exactly those two - nothing related to Florence-2's
actual forward-pass logic is touched.

1. `flash_attn` import requirement. Florence-2's modeling file lists
   `flash_attn` as a required import even when it isn't actually needed
   (i.e. when NOT requesting `attn_implementation="flash_attention_2"`).
   `flash_attn` cannot be installed without CUDA, so this fails outright on
   a CPU-only machine. Source:
   https://huggingface.co/microsoft/Florence-2-base/discussions/4

2. `AttributeError: '...ForConditionalGeneration' object has no attribute
   '_supports_sdpa'`. Confirmed, widely reported against transformers
   versions from roughly 4.50 onward (title of one report is literally
   "Florence2 stopped working after upgrade to 4.50.0"):
   https://github.com/huggingface/transformers/issues/39974
   https://github.com/huggingface/transformers/issues/41622
   https://github.com/huggingface/transformers/discussions/36886
   Root cause: transformers' own `modeling_utils.py` checks
   `self._supports_sdpa` while resolving which attention implementation to
   use. Older versions of `PreTrainedModel` defined a default value for this
   class attribute, which Florence-2's custom subclass relied on inheriting
   rather than defining itself. Newer transformers no longer provides that
   default, so the attribute genuinely does not exist anywhere in Florence-2's
   class MRO under the newer library - hence the AttributeError, not a bug in
   Florence-2's own logic.

ROOT CAUSE of the RecursionError seen after fix #2 was first added (read this
before touching the patching logic below again):

The RecursionError ("maximum recursion depth exceeded while calling a Python
object") was NOT caused by patching `_supports_sdpa`, and was not a deeper
transformers-version incompatibility. It was a self-referential bug in THIS
module's own fix #1. The previous implementation of the flash_attn import
shim looked like this:

    def _fixed_get_imports(filename):
        from transformers.dynamic_module_utils import get_imports  # BUG
        ...
        imports = get_imports(filename)
        ...

That function is installed, via `unittest.mock.patch`, AS
`transformers.dynamic_module_utils.get_imports` for the duration of the load
attempt. Because the `from ... import get_imports` line re-resolves that name
EVERY time the function runs, and the patch is still active while it runs,
the name `get_imports` inside the function rebinds to `_fixed_get_imports`
itself - so calling `get_imports(filename)` calls `_fixed_get_imports`
again, forever. This was verified by reproducing it directly (with
transformers/torch stubbed out, since neither is installable in the
development sandbox): a single call into the patched function recurses
until Python's recursion limit is hit, producing the exact reported error.
This is why it appeared once fix #2's retry path started actually being
exercised - fix #1 was broken from the point it was refactored into this
module, and any invocation of it recurses, not just a retried one.

THE FIX: capture the real, original `get_imports` ONCE, before installing
the patch, and have the shim call that captured reference - never re-import
the name from the module while the patch on that same module attribute is
active. See `load_florence2` below.

GENERATION COMPATIBILITY (separate from load - this is what happens once
`.generate()` is actually called, after loading succeeds):

Confirmed, sourced issue: `AttributeError: 'NoneType' object has no
attribute 'shape'`, traceback ending in Florence-2's own
`prepare_inputs_for_generation`, at `past_key_values[0][0].shape[2]`. This is
NOT a bug hit by this project alone - it is an active, documented Florence-2
compatibility problem:
  https://huggingface.co/Rusted-Gold/Florence-2-large-ft-Transformers-Fix
    (a community fork patching several `past_key_value[0] is not None`
    guards into the attention layers for exactly this failure shape)
  https://github.com/chflame163/ComfyUI_LayerStyle/issues/575
    (same exact "'NoneType' object has no attribute 'shape'" report)

Root cause context (confirmed via the model's own source and HuggingFace's
own documented deprecation, not guessed): starting with transformers 4.50,
`PreTrainedModel` no longer automatically inherits `GenerationMixin` - a
model class must inherit it explicitly to keep working `.generate()`
support and, more importantly here, the modern cache-handling behavior that
`GenerationMixin.generate()` sets up before calling
`prepare_inputs_for_generation` on every step. Multiple Florence-2 HF repos
needed exactly this class-inheritance fix added after 4.50 shipped:
  https://huggingface.co/microsoft/Florence-2-base/discussions/22
    (merged: added `GenerationMixin` to the INNER
    `Florence2LanguageForConditionalGeneration` class)
  https://huggingface.co/microsoft/Florence-2-large-ft/commit/3502d39
    (added `GenerationMixin` to the OUTER `Florence2ForConditionalGeneration`
    wrapper class specifically - the class `AutoModelForCausalLM.
    from_pretrained(..., trust_remote_code=True)` actually returns and that
    `.generate()` is called on here)
Florence-2's own `prepare_inputs_for_generation` (shown in full in the
discussion above) is written in the OLD, pre-Cache-class style: it assumes
that whenever `past_key_values is not None`, `past_key_values[0][0]` is
already a populated tensor. Whether the outer wrapper class this project
loads (`microsoft/Florence-2-base`'s `Florence2ForConditionalGeneration`)
has received the same outer-class fix as `Florence-2-large-ft` could not be
confirmed from here - the file is too large for this project's available
fetch tooling to read past roughly its first third, and the outer class is
defined near the end. `generate_florence2()` below does not guess which
side of that gap is true; it diagnoses (`diagnose_generation_capability`)
and tries a small, ordered set of DOCUMENTED, standard `generate()` keyword
arguments (never edits to Florence-2's own code) - starting with the
literal request, then `num_beams=1` (plain greedy, per the explicit
instruction to try that before anything else), then `use_cache=False`
(which keeps `past_key_values` `None` on every step regardless of how it
would otherwise have been constructed, so the crashing line's `if
past_key_values is not None:` guard is simply never entered) - and reports
in full which one actually worked, or every traceback if none did. Nothing
here is presented as fixed until a real run confirms it.
"""
from __future__ import annotations

import traceback
from unittest.mock import patch


class Florence2LoadError(RuntimeError):
    """Raised when Florence-2 cannot be loaded even after both compatibility
    tiers are tried - carries the real underlying exception (via __cause__)
    AND the full original traceback text in `.full_traceback`, so a failure
    is never reported as just an exception message with no way to see where
    it actually originated."""

    def __init__(self, message: str, full_traceback: str | None = None):
        super().__init__(message)
        self.full_traceback = full_traceback


class Florence2GenerationError(RuntimeError):
    """Raised when Florence-2 loaded successfully but `.generate()` failed
    under every candidate configuration tried. `.attempts` lists every
    configuration tried with its full traceback - never just the last
    exception's message."""

    def __init__(self, message: str, attempts: list | None = None):
        super().__init__(message)
        self.attempts = attempts or []
        self.full_traceback = "\n\n".join(
            f"--- attempt: {a['label']} ---\n{a['traceback']}" for a in self.attempts
        )


def load_florence2(model_id: str, dtype, device: str):
    """Returns (processor, model, compat_report). compat_report records
    exactly which tier(s) were needed, so callers/tests can assert on it
    instead of just on success/failure - this is what
    tests/integration/test_florence2_compat.py checks."""
    from transformers import AutoProcessor, AutoModelForCausalLM
    # Captured HERE, before any patch is installed on this same attribute -
    # this is the one-line fix for the RecursionError (see module docstring
    # ROOT CAUSE section). Every call to _fixed_get_imports below uses THIS
    # captured reference, never a fresh lookup of the (possibly-patched)
    # module attribute.
    from transformers.dynamic_module_utils import get_imports as _real_get_imports
    import transformers as _tf

    compat_report = {
        "flash_attn_import_patch_applied": True,  # tier 0, always needed on CPU - see module docstring
        "sdpa_attribute_patch_applied": False,     # tier 2, only if tier 1 alone fails
        "attn_implementation_used": "eager",
        "transformers_version": _tf.__version__,
    }

    def _fixed_get_imports(filename):
        if not str(filename).endswith("modeling_florence2.py"):
            return _real_get_imports(filename)
        imports = _real_get_imports(filename)
        if "flash_attn" in imports:
            imports.remove("flash_attn")
        return imports

    def _try_load():
        with patch("transformers.dynamic_module_utils.get_imports", _fixed_get_imports):
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                attn_implementation="eager",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).to(device)
        return processor, model

    try:
        processor, model = _try_load()
    except AttributeError as exc:
        full_tb = traceback.format_exc()
        if "_supports_sdpa" not in str(exc):
            raise Florence2LoadError(
                f"Florence-2 load failed with an unexpected AttributeError "
                f"(not the known _supports_sdpa issue - report this traceback "
                f"rather than assuming the same fix applies): {exc}",
                full_traceback=full_tb,
            ) from exc

        # Tier 2: restore the missing default that Florence-2's custom code
        # expects to inherit. See module docstring for exactly why this is
        # scoped and honest, not a blind patch.
        import transformers.modeling_utils as _modeling_utils
        _modeling_utils.PreTrainedModel._supports_sdpa = True
        compat_report["sdpa_attribute_patch_applied"] = True

        try:
            processor, model = _try_load()
        except Exception as exc2:  # noqa: BLE001
            raise Florence2LoadError(
                f"Florence-2 still failed to load after the _supports_sdpa "
                f"compatibility patch: {type(exc2).__name__}: {exc2}",
                full_traceback=traceback.format_exc(),
            ) from exc2
    except Exception as exc:  # noqa: BLE001
        raise Florence2LoadError(
            f"{type(exc).__name__}: {exc}",
            full_traceback=traceback.format_exc(),
        ) from exc

    model.eval()
    return processor, model, compat_report


def diagnose_generation_capability(model) -> dict:
    """Reports facts about the loaded model relevant to the generation-time
    compatibility issue described in the module docstring - never a guess,
    just what's actually true of this model instance right now: its exact
    class, full MRO, and whether GenerationMixin is actually present in that
    MRO. This is the evidence that settles which side of the "was the outer
    wrapper class's GenerationMixin fix applied to this specific repo"
    question is true, instead of assuming either answer."""
    try:
        from transformers import GenerationMixin
    except Exception:  # noqa: BLE001
        GenerationMixin = None  # noqa: N806

    mro = [cls.__name__ for cls in type(model).__mro__]
    return {
        "model_class_name": type(model).__name__,
        "model_class_mro": mro,
        "generation_mixin_in_mro": ("GenerationMixin" in mro),
        "isinstance_generation_mixin": (
            isinstance(model, GenerationMixin) if GenerationMixin is not None else None
        ),
    }


# Ordered, documented, standard `generate()` keyword-argument overrides to
# try for the known past_key_values/prepare_inputs_for_generation failure -
# see module docstring GENERATION COMPATIBILITY section for why each is
# here and in this order. None of these edit Florence-2's own code.
_GENERATION_CONFIG_CANDIDATES = [
    {"label": "requested configuration (as given)", "overrides": {}},
    {"label": "greedy decoding (num_beams=1)", "overrides": {"num_beams": 1}},
    {"label": "use_cache=False", "overrides": {"use_cache": False}},
    {"label": "greedy decoding + use_cache=False", "overrides": {"num_beams": 1, "use_cache": False}},
]


def generate_florence2(model, inputs, *, max_new_tokens: int, num_beams: int, do_sample: bool = False):
    """Runs model.generate() for Florence-2, trying the ordered, documented
    generate() argument overrides in _GENERATION_CONFIG_CANDIDATES until one
    succeeds. Returns (generated_ids, generation_report). generation_report
    always records every attempt made (label, kwargs, and - for failures -
    the full traceback), so which configuration actually worked (or that
    none did) is never left to a caller's assumption.

    This is deliberately NOT a monkeypatch of Florence-2's own generation
    code - every attempt is a plain call to the model's own public
    `.generate()` with different, standard keyword arguments."""
    import torch

    attempts = []
    base_kwargs = dict(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
    )

    for candidate in _GENERATION_CONFIG_CANDIDATES:
        kwargs = {**base_kwargs, **candidate["overrides"]}
        loggable_kwargs = {k: v for k, v in kwargs.items() if k not in ("input_ids", "pixel_values")}
        try:
            with torch.no_grad():
                generated_ids = model.generate(**kwargs)
            return generated_ids, {
                "working_generation_config": candidate["label"],
                "generation_kwargs_used": loggable_kwargs,
                "failed_attempts": attempts,
            }
        except Exception as exc:  # noqa: BLE001
            attempts.append({
                "label": candidate["label"],
                "kwargs": loggable_kwargs,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })

    raise Florence2GenerationError(
        "Florence-2 generation failed under every candidate configuration "
        f"tried ({[a['label'] for a in attempts]}). This needs a real fix, "
        "not another configuration guess - see .attempts for the full "
        "traceback of each one.",
        attempts=attempts,
    )
