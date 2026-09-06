"""Real tests of `routing/failure_classification.py:classify()` against the
exact fixed set of failure-string prefixes `app/pipeline.run_query()` can
produce (see that module's docstring) - proving each maps to a stable,
documented code/category pair rather than a per-message invention."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from routing import failure_classification as fc  # noqa: E402


class TestFailureClassification(unittest.TestCase):
    def test_input_loading_failed_is_400_invalid_image(self):
        result = fc.classify("Input loading failed: cannot identify image file")
        self.assertEqual(result.code, 400)
        self.assertEqual(result.category, "INVALID_IMAGE")

    def test_validation_same_date_is_400_ambiguous_query(self):
        result = fc.classify(
            "Validation failed: Bi-temporal change analysis requires two different "
            "acquisition dates; both inputs declared the same date."
        )
        self.assertEqual(result.code, 400)
        self.assertEqual(result.category, "AMBIGUOUS_QUERY")

    def test_validation_same_modality_is_400_missing_modality(self):
        result = fc.classify(
            "Validation failed: Optical+SAR fusion requires two different modalities; "
            "both inputs were declared as 'optical'."
        )
        self.assertEqual(result.code, 400)
        self.assertEqual(result.category, "MISSING_MODALITY")

    def test_router_ambiguous_is_400_ambiguous_query(self):
        result = fc.classify("Router could not confidently determine the analysis branch...")
        self.assertEqual(result.code, 400)
        self.assertEqual(result.category, "AMBIGUOUS_QUERY")

    def test_specialist_execution_failed_is_500(self):
        result = fc.classify(
            "Specialist execution failed: boom\nTraceback (most recent call last):\n  ..."
        )
        self.assertEqual(result.code, 500)
        self.assertEqual(result.category, "INTERNAL_EXECUTION_ERROR")
        self.assertNotIn("Traceback", result.why)

    def test_unrecognized_prefix_falls_back_to_500_unexpected(self):
        result = fc.classify("Something odd happened")
        self.assertEqual(result.code, 500)
        self.assertEqual(result.category, "UNEXPECTED_ERROR")


if __name__ == "__main__":
    unittest.main()
