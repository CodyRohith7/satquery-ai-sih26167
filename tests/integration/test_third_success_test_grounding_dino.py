"""Regression tests for scripts/third_success_test_grounding_dino.py's
processor text-input format.

Background (full account in that script's module docstring's "INPUT FORMAT
FIX" section): the first version of this script called
`processor(images=image, text=[[formatted]], return_tensors="pt")` - a
list-of-lists, matching a convenience shape shown in transformers'
"main"-branch docs for GroundingDINO. On the real pinned
transformers==4.57.1, that raised, before any model forward pass ran:

    TypeError: TextEncodeInput must be Union[TextInputSequence,
    Tuple[InputSequence, InputSequence]]

- a tokenizer-level type error (from the underlying Rust `tokenizers`
library), confirming this was a text-SHAPE problem, not a model or
version-pinning problem. The fix changed `_format_grounding_dino_prompt()`
and the main loop to pass a single plain string (e.g. `"farmland."`)
instead of any list structure - the one shape every version of
`GroundingDinoProcessor` is documented to accept unambiguously, and the
only shape this script ever needs since it sends exactly one candidate
phrase per forward call.

`TestFormatGroundingDinoPrompt` exercises the pure formatting function
directly - no dependencies needed, always runs. `TestProcessorTextInputFormat`
reproduces the exact failure/fix contract against a stub processor that
mimics the real tokenizer's observed behavior (rejects a nested list,
accepts a plain string) - deterministic, no network or model weights
needed, mirroring how test_florence2_compat.py's
`TestGenerationCompatibility` mocks `model.generate()` to test a fallback
contract without depending on the real model.
"""
from __future__ import annotations

import importlib.util
import os
import unittest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "third_success_test_grounding_dino.py")


def _load_script_module():
    """third_success_test_grounding_dino.py lives in scripts/, which is not
    a package (no __init__.py), so it's loaded by file path rather than a
    normal import - the same approach used elsewhere in this project to
    exercise script modules deterministically without needing scripts/ on
    sys.path."""
    spec = importlib.util.spec_from_file_location(
        "third_success_test_grounding_dino", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULE = _load_script_module()


class TestFormatGroundingDinoPrompt(unittest.TestCase):
    """_format_grounding_dino_prompt() is a pure function (no torch/
    transformers import needed to exercise it), so these run unconditionally."""

    def test_returns_plain_string_not_a_list(self):
        result = _MODULE._format_grounding_dino_prompt("farmland")
        self.assertIsInstance(result, str)

    def test_lowercases_and_adds_trailing_period(self):
        self.assertEqual(_MODULE._format_grounding_dino_prompt("Farmland"), "farmland.")

    def test_preserves_existing_trailing_period(self):
        self.assertEqual(_MODULE._format_grounding_dino_prompt("buildings."), "buildings.")

    def test_strips_surrounding_whitespace(self):
        self.assertEqual(_MODULE._format_grounding_dino_prompt("  an airplane  "), "an airplane.")

    def test_multi_word_phrase_unchanged_besides_case_and_period(self):
        self.assertEqual(
            _MODULE._format_grounding_dino_prompt("a cluster of buildings"),
            "a cluster of buildings.",
        )


class _FakeTokenizerTypeError(TypeError):
    pass


class _FakeGroundingDinoProcessor:
    """Mimics the real, observed behavior of GroundingDinoProcessor.__call__
    on transformers==4.57.1 for the `text` argument specifically: a nested
    list (list-of-lists) reaches the underlying Rust tokenizer with a shape
    it does not recognize and raises the exact reported TypeError; a plain
    string (or a flat list of strings) is accepted. This is a deliberately
    narrow, deterministic stand-in for the real processor - it does not
    model image handling or tensor output, only the text-shape contract
    this regression test exists to pin down."""

    def __call__(self, images, text, return_tensors="pt"):
        if isinstance(text, list) and any(isinstance(item, list) for item in text):
            raise _FakeTokenizerTypeError(
                "TextEncodeInput must be Union[TextInputSequence, "
                "Tuple[InputSequence, InputSequence]]"
            )
        if not isinstance(text, (str, list)):
            raise _FakeTokenizerTypeError(
                "TextEncodeInput must be Union[TextInputSequence, "
                "Tuple[InputSequence, InputSequence]]"
            )
        return {"images": images, "text": text, "return_tensors": return_tensors}


class TestProcessorTextInputFormat(unittest.TestCase):
    """Reproduces the original bug against the fake processor, then proves
    the current script code path (plain string via
    _format_grounding_dino_prompt) does not hit it."""

    def test_old_list_of_lists_shape_reproduces_the_original_typeerror(self):
        processor = _FakeGroundingDinoProcessor()
        formatted = _MODULE._format_grounding_dino_prompt("farmland")
        with self.assertRaises(_FakeTokenizerTypeError):
            processor(images=object(), text=[[formatted]], return_tensors="pt")

    def test_current_plain_string_format_is_accepted(self):
        processor = _FakeGroundingDinoProcessor()
        formatted = _MODULE._format_grounding_dino_prompt("farmland")
        # Should not raise - this is exactly what the main loop in
        # third_success_test_grounding_dino.py now calls.
        result = processor(images=object(), text=formatted, return_tensors="pt")
        self.assertEqual(result["text"], "farmland.")

    def test_all_three_real_test_phrases_survive_the_fake_processor(self):
        processor = _FakeGroundingDinoProcessor()
        for phrase in ("farmland", "buildings", "an airplane"):
            formatted = _MODULE._format_grounding_dino_prompt(phrase)
            # Must not raise for any of the three phrases this script's
            # real run actually sends (target, target-2, control).
            processor(images=object(), text=formatted, return_tensors="pt")


if __name__ == "__main__":
    unittest.main()
