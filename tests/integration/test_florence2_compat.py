"""Regression tests for src/models/florence2_compat.py.

Background (full sourced explanation lives in florence2_compat.py's module
docstring - summarized here):

1. Florence-2 needs a flash_attn import stripped to load on CPU at all
   (`_fixed_get_imports`, installed via `unittest.mock.patch` as
   `transformers.dynamic_module_utils.get_imports` for the duration of the
   load attempt).
2. transformers >= 4.50 raises `AttributeError: '...' object has no
   attribute '_supports_sdpa'` when loading Florence-2. The loader retries
   once after restoring that attribute on `PreTrainedModel`.
3. A REAL bug (not a transformers-version issue) was introduced when fix #1
   was refactored into this module: `_fixed_get_imports` re-imported
   `get_imports` from `transformers.dynamic_module_utils` on every call -
   but since fix #1 is itself installed AS that same module attribute while
   it runs, that re-import rebound the name to itself, so calling it called
   itself, forever: `RecursionError: maximum recursion depth exceeded while
   calling a Python object`. This was confirmed by direct reproduction (with
   transformers/torch stubbed, since neither installs in the dev sandbox -
   no network to pypi.org here) before being fixed, not guessed at. The fix
   captures the real `get_imports` ONCE, before any patch is installed, and
   the shim always calls that captured reference.

`TestFixedGetImportsNoRecursion` is the test that would have caught this
exact bug: it exercises the patched shim being CALLED MULTIPLE TIMES while
the patch is active (mirroring both a plain load and the tier-2 retry path,
where `_try_load()` - and therefore the `with patch(...)` block - runs
twice), which is exactly the condition the RecursionError needed. It uses a
`RecursionError` timeout guard (`sys.setrecursionlimit` is NOT touched, per
the explicit instruction not to raise the recursion limit as a workaround)
so a regression here fails fast and clearly rather than hanging.

4. A SEPARATE, generation-time failure was hit after loading succeeded:
   `AttributeError: 'NoneType' object has no attribute 'shape'`, in
   Florence-2's own `prepare_inputs_for_generation`, at
   `past_key_values[0][0].shape[2]`. Confirmed as an active, documented
   Florence-2/transformers compatibility issue (not something specific to
   this project) - see florence2_compat.py's GENERATION COMPATIBILITY
   docstring section for the full sourced explanation, including the
   confirmed HuggingFace deprecation (transformers >= 4.50 no longer grants
   `GenerationMixin` automatically) and the fact that some Florence-2 HF
   repos needed an explicit fix for exactly this and others may not have
   received it yet. `generate_florence2()` tries an ordered list of
   documented, standard `generate()` keyword-argument overrides (never
   edits to Florence-2's own code) and reports exactly which one worked.
   `TestGenerationCompatibility` regression-tests this fallback mechanism
   deterministically (mocking `model.generate()`'s failure/success under
   specific kwargs), so it's meaningful regardless of which configuration
   the real model actually needs on any given machine.

`TestFixedGetImportsNoRecursion`, `TestUnrecoverableFailureReporting`, and
`TestGenerationCompatibility` all need transformers importable but do NOT
need network access or the real model weights - they mock the relevant
transformers/model calls deterministically. `TestFlorence2CpuCompat`
additionally loads the REAL microsoft/Florence-2-base model end to end,
including running real generation through `generate_florence2()`, and needs
network access to download it (~460MB, cached after the first run).
"""
from __future__ import annotations

import os
import socket
import unittest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _transformers_and_torch_available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _dependencies_available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import einops  # noqa: F401
        import timm  # noqa: F401
    except ImportError:
        return False
    return True


def _network_available():
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
        return True
    except OSError:
        return False


@unittest.skipUnless(
    _transformers_and_torch_available(),
    "torch/transformers not installed - see requirements-ml-minimal.txt "
    "(this test is expected to skip in the cloud sandbox; it must run for "
    "real wherever these are installed, since it mocks the model calls "
    "itself and needs no network or model weights)",
)
class TestFixedGetImportsNoRecursion(unittest.TestCase):
    """Regression test for the exact RecursionError incident: calls
    load_florence2 with from_pretrained mocked to internally re-invoke the
    patched get_imports shim (as real transformers' trust_remote_code
    loading does), and asserts this returns normally - no RecursionError -
    both on a clean load and across the tier-2 retry (two separate
    `with patch(...)` entries in one load_florence2 call)."""

    def setUp(self):
        import sys
        sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

    def test_get_imports_shim_does_not_recurse_on_clean_load(self):
        import torch
        from unittest.mock import patch as mock_patch
        from models.florence2_compat import load_florence2

        call_log = []

        def _fake_from_pretrained(*args, **kwargs):
            # Mirrors real transformers: trust_remote_code loading calls
            # get_imports on the (possibly patched) module attribute to
            # resolve the custom modeling file's required imports.
            from transformers.dynamic_module_utils import get_imports
            call_log.append(get_imports("modeling_florence2.py"))

            class _FakeModel:
                def eval(self):
                    return self

                def to(self, device):
                    return self

            return _FakeModel()

        with mock_patch("transformers.AutoProcessor.from_pretrained", return_value=object()), \
             mock_patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=_fake_from_pretrained):
            processor, model, compat_report = load_florence2(
                "microsoft/Florence-2-base", torch.float32, "cpu"
            )

        self.assertIsNotNone(processor)
        self.assertIsNotNone(model)
        self.assertEqual(len(call_log), 1)
        self.assertNotIn("flash_attn", call_log[0])
        self.assertFalse(compat_report["sdpa_attribute_patch_applied"])

    def test_get_imports_shim_does_not_recurse_across_sdpa_retry(self):
        """The condition that actually triggered the incident: the shim is
        entered a SECOND time (tier-2 retry re-enters `with patch(...)`)."""
        import torch
        import transformers.modeling_utils as modeling_utils
        from unittest.mock import patch as mock_patch
        from models.florence2_compat import load_florence2

        had_patch_before = hasattr(modeling_utils.PreTrainedModel, "_supports_sdpa")
        original_value = getattr(modeling_utils.PreTrainedModel, "_supports_sdpa", None)
        if had_patch_before:
            del modeling_utils.PreTrainedModel._supports_sdpa

        attempt = {"n": 0}
        call_log = []

        def _fake_from_pretrained(*args, **kwargs):
            from transformers.dynamic_module_utils import get_imports
            call_log.append(get_imports("modeling_florence2.py"))
            attempt["n"] += 1

            class _FakeModel:
                def eval(self):
                    return self

                def to(self, device):
                    return self

            if attempt["n"] == 1 and not hasattr(modeling_utils.PreTrainedModel, "_supports_sdpa"):
                raise AttributeError(
                    "'Florence2ForConditionalGeneration' object has no attribute '_supports_sdpa'"
                )
            return _FakeModel()

        try:
            with mock_patch("transformers.AutoProcessor.from_pretrained", return_value=object()), \
                 mock_patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=_fake_from_pretrained):
                processor, model, compat_report = load_florence2(
                    "microsoft/Florence-2-base", torch.float32, "cpu"
                )
        finally:
            if had_patch_before:
                modeling_utils.PreTrainedModel._supports_sdpa = original_value
            elif hasattr(modeling_utils.PreTrainedModel, "_supports_sdpa"):
                del modeling_utils.PreTrainedModel._supports_sdpa

        self.assertIsNotNone(processor)
        self.assertIsNotNone(model)
        # Exactly two attempts (initial failure + one retry) - proves the
        # shim was re-entered under a second `with patch(...)` and still
        # did not recurse.
        self.assertEqual(attempt["n"], 2)
        self.assertEqual(len(call_log), 2)
        self.assertTrue(compat_report["sdpa_attribute_patch_applied"])


@unittest.skipUnless(
    _transformers_and_torch_available(),
    "torch/transformers not installed - see requirements-ml-minimal.txt "
    "(this test is expected to skip in the cloud sandbox; it must run for "
    "real wherever these are installed, since it mocks the failure itself "
    "and needs no network or model weights)",
)
class TestUnrecoverableFailureReporting(unittest.TestCase):
    """A failure that ISN'T the known _supports_sdpa case, or that survives
    the retry, must still raise Florence2LoadError carrying the FULL
    original traceback - not just str(exc). This is the diagnostic gap that
    left the RecursionError impossible to pin down from the JSON result
    alone; it must not regress even now that the recursion itself is fixed."""

    def setUp(self):
        import sys
        sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

    def test_unexpected_attribute_error_carries_full_traceback(self):
        import torch
        from unittest.mock import patch as mock_patch
        from models.florence2_compat import load_florence2, Florence2LoadError

        def _raise_other(*a, **k):
            raise AttributeError("object has no attribute 'something_unrelated'")

        with mock_patch("transformers.AutoProcessor.from_pretrained", return_value=object()), \
             mock_patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=_raise_other):
            with self.assertRaises(Florence2LoadError) as ctx:
                load_florence2("microsoft/Florence-2-base", torch.float32, "cpu")

        err = ctx.exception
        self.assertIn("unexpected AttributeError", str(err))
        self.assertIsNotNone(err.full_traceback)
        self.assertIn("something_unrelated", err.full_traceback)

    def test_failure_surviving_the_sdpa_retry_carries_full_traceback(self):
        import torch
        import transformers.modeling_utils as modeling_utils
        from unittest.mock import patch as mock_patch
        from models.florence2_compat import load_florence2, Florence2LoadError

        had_patch_before = hasattr(modeling_utils.PreTrainedModel, "_supports_sdpa")
        original_value = getattr(modeling_utils.PreTrainedModel, "_supports_sdpa", None)
        if had_patch_before:
            del modeling_utils.PreTrainedModel._supports_sdpa

        def _always_fail(*a, **k):
            if not hasattr(modeling_utils.PreTrainedModel, "_supports_sdpa"):
                raise AttributeError(
                    "'Florence2ForConditionalGeneration' object has no attribute '_supports_sdpa'"
                )
            raise RuntimeError("out of memory (simulated)")

        try:
            with mock_patch("transformers.AutoProcessor.from_pretrained", return_value=object()), \
                 mock_patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=_always_fail):
                with self.assertRaises(Florence2LoadError) as ctx:
                    load_florence2("microsoft/Florence-2-base", torch.float32, "cpu")
        finally:
            if had_patch_before:
                modeling_utils.PreTrainedModel._supports_sdpa = original_value
            elif hasattr(modeling_utils.PreTrainedModel, "_supports_sdpa"):
                del modeling_utils.PreTrainedModel._supports_sdpa

        err = ctx.exception
        self.assertIn("still failed to load", str(err))
        self.assertIsNotNone(err.full_traceback)
        self.assertIn("out of memory (simulated)", err.full_traceback)


@unittest.skipUnless(
    _transformers_and_torch_available(),
    "torch/transformers not installed - see requirements-ml-minimal.txt "
    "(this test is expected to skip in the cloud sandbox; it must run for "
    "real wherever these are installed, since it mocks model.generate() "
    "itself and needs no network or model weights)",
)
class TestGenerationCompatibility(unittest.TestCase):
    """Regression tests for the generate_florence2() fallback mechanism
    (see florence2_compat.py's GENERATION COMPATIBILITY docstring section):
    the confirmed 'NoneType' object has no attribute 'shape' failure in
    Florence-2's own prepare_inputs_for_generation, hit on real hardware
    after loading succeeded. These are deterministic - they mock
    model.generate() to fail or succeed under specific keyword-argument
    combinations, so they verify the FALLBACK MECHANISM itself (correct
    cascade order, correct reporting, no monkeypatching of generation
    internals - only standard keyword arguments) regardless of which
    config the real model actually needs on any given machine."""

    def setUp(self):
        import sys
        sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

    def test_falls_back_to_use_cache_false_when_only_that_works(self):
        from models.florence2_compat import generate_florence2

        class _FakeModel:
            def generate(self, input_ids, pixel_values, max_new_tokens, num_beams, do_sample, use_cache=True):
                if use_cache:
                    raise AttributeError("'NoneType' object has no attribute 'shape'")
                return "OK"

        generated_ids, report = generate_florence2(
            _FakeModel(), {"input_ids": "x", "pixel_values": "y"},
            max_new_tokens=1024, num_beams=3,
        )
        self.assertEqual(generated_ids, "OK")
        self.assertEqual(report["working_generation_config"], "use_cache=False")
        # Must have actually tried (and recorded) the requested config and
        # greedy decoding first, in that order, before falling back further.
        self.assertEqual(
            [a["label"] for a in report["failed_attempts"]],
            ["requested configuration (as given)", "greedy decoding (num_beams=1)"],
        )
        for attempt in report["failed_attempts"]:
            self.assertIn("shape", attempt["traceback"])

    def test_succeeds_immediately_when_requested_config_already_works(self):
        """No incompatibility present (e.g. a fixed model repo, or a
        transformers version where this never broke) - must not report a
        fallback was needed when it wasn't."""
        from models.florence2_compat import generate_florence2

        class _FakeModel:
            def generate(self, **kwargs):
                return "OK"

        _, report = generate_florence2(
            _FakeModel(), {"input_ids": "x", "pixel_values": "y"},
            max_new_tokens=1024, num_beams=3,
        )
        self.assertEqual(report["working_generation_config"], "requested configuration (as given)")
        self.assertEqual(report["failed_attempts"], [])

    def test_raises_with_all_tracebacks_when_nothing_works(self):
        from models.florence2_compat import generate_florence2, Florence2GenerationError

        class _FakeModel:
            def generate(self, **kwargs):
                raise RuntimeError("unrelated, unfixable failure")

        with self.assertRaises(Florence2GenerationError) as ctx:
            generate_florence2(
                _FakeModel(), {"input_ids": "x", "pixel_values": "y"},
                max_new_tokens=1024, num_beams=3,
            )
        err = ctx.exception
        self.assertEqual(len(err.attempts), 4)  # every candidate was actually tried
        self.assertIn("unrelated, unfixable failure", err.full_traceback)

    def test_diagnose_generation_capability_detects_missing_generation_mixin(self):
        """This is the evidence that settles whether the outer wrapper
        class's GenerationMixin fix (confirmed applied to some Florence-2
        HF repos, unconfirmed for microsoft/Florence-2-base's outer class -
        see module docstring) is actually present for a given loaded model,
        instead of assuming either way."""
        from models.florence2_compat import diagnose_generation_capability
        import transformers

        class _WithMixin(transformers.GenerationMixin):
            pass

        class _WithoutMixin:
            pass

        with_diag = diagnose_generation_capability(_WithMixin())
        without_diag = diagnose_generation_capability(_WithoutMixin())
        self.assertTrue(with_diag["generation_mixin_in_mro"])
        self.assertFalse(without_diag["generation_mixin_in_mro"])
        self.assertIn("model_class_mro", with_diag)


@unittest.skipUnless(
    _dependencies_available(),
    "torch/transformers/einops/timm not installed - see requirements-ml-minimal.txt "
    "(this test is expected to skip in the cloud sandbox; it must run for real "
    "on the target machine before Florence-2 is trusted)",
)
@unittest.skipUnless(
    _network_available(),
    "no network access to huggingface.co - cannot download microsoft/Florence-2-base "
    "(this test is expected to skip in the cloud sandbox)",
)
class TestFlorence2CpuCompat(unittest.TestCase):
    """Loads the REAL microsoft/Florence-2-base model on CPU and asserts the
    exact previously-broken path now works, end to end, with no mocking of
    the model itself - only the two documented, narrow compatibility shims
    in florence2_compat.py are involved, exactly as they run in
    scripts/second_success_test_grounding.py."""

    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
        import torch
        cls.torch = torch

    def test_florence2_base_loads_on_cpu(self):
        from models.florence2_compat import load_florence2

        processor, model, compat_report = load_florence2(
            "microsoft/Florence-2-base", self.torch.float32, "cpu"
        )
        self.assertIsNotNone(processor)
        self.assertIsNotNone(model)
        self.assertIn("transformers_version", compat_report)
        self.assertIn("sdpa_attribute_patch_applied", compat_report)

    def test_grounding_task_runs_end_to_end_on_real_model(self):
        """Not a benchmark or accuracy check - only proves the loaded model
        can actually run its native grounding task and return a real,
        structured (bboxes, labels) result, on a tiny synthetic image (kept
        synthetic here deliberately, since this is a compatibility/plumbing
        test, not the real grounding evidence test - that is
        scripts/second_success_test_grounding.py, run on a real image)."""
        import numpy as np
        from PIL import Image
        from models.florence2_compat import load_florence2, generate_florence2

        processor, model, _ = load_florence2("microsoft/Florence-2-base", self.torch.float32, "cpu")
        image = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"))

        prompt = "<CAPTION_TO_PHRASE_GROUNDING>" + "an object"
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        # Goes through the same generate_florence2() fallback path the real
        # grounding script uses, rather than a bare model.generate() call -
        # this is what actually proves the generation-time compatibility fix
        # works end to end, not just that loading does.
        generated_ids, generation_report = generate_florence2(
            model, inputs, max_new_tokens=64, num_beams=1, do_sample=False
        )
        self.assertIsNotNone(generation_report["working_generation_config"])
        raw_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            raw_text, task="<CAPTION_TO_PHRASE_GROUNDING>", image_size=(image.width, image.height)
        )
        self.assertIn("<CAPTION_TO_PHRASE_GROUNDING>", parsed)
        self.assertIn("bboxes", parsed["<CAPTION_TO_PHRASE_GROUNDING>"])


if __name__ == "__main__":
    unittest.main()
