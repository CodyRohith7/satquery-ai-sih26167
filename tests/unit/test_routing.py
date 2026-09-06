import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ingestion.metadata import ImageMetadata
from routing import router


def _meta(**kwargs):
    defaults = dict(
        path="x.png", height=100, width=100, band_count=3, dtype="uint8",
        crs=None, transform=None, declared_modality=None,
        inferred_modality="optical (unconfirmed: 3-band file consistent with RGB)",
        modality_confidence="heuristic", declared_date=None,
    )
    defaults.update(kwargs)
    return ImageMetadata(**defaults)


class TestRouterSingleImage(unittest.TestCase):
    def test_grounding_query_routes_to_grounding(self):
        d = router.decide("locate the water body and highlight it", [_meta()])
        self.assertEqual(d.task_type, "grounding")
        self.assertEqual(d.tool_name, "tool_grounding_v0")

    def test_descriptive_query_routes_to_vqa(self):
        d = router.decide("describe the land cover in this scene", [_meta()])
        self.assertEqual(d.task_type, "single_image_vqa")
        # tool_name depends on whether SmolVLM's real dependencies (torch +
        # transformers) are importable here - the router picks the real
        # model when it can, the classical baseline otherwise (see
        # router._select_vqa_tool_name). Both are valid, honest outcomes;
        # what must NOT happen is claiming the real model when it isn't
        # actually runnable, or vice versa.
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            expected = "tool_single_image_vqa_smolvlm_v1"
        except ImportError:
            expected = "tool_single_image_vqa_v0"
        self.assertEqual(d.tool_name, expected)

    def test_confidence_is_a_real_probability(self):
        d = router.decide("what are the major features here", [_meta()])
        self.assertGreater(d.confidence, 0.0)
        self.assertLessEqual(d.confidence, 1.0)


class TestRouterPairedImages(unittest.TestCase):
    def test_different_modalities_route_to_fusion(self):
        a = _meta(declared_modality="optical", declared_date="2024-01-01")
        b = _meta(declared_modality="sar", declared_date="2024-01-01")
        d = router.decide("identify built-up and water regions", [a, b])
        self.assertEqual(d.task_type, "optical_sar_fusion")

    def test_different_dates_route_to_change(self):
        a = _meta(declared_date="2020-01-01")
        b = _meta(declared_date="2024-01-01")
        d = router.decide("what changed", [a, b])
        self.assertEqual(d.task_type, "bitemporal_change")

    def test_same_declared_date_still_routes_to_change_for_specific_error(self):
        a = _meta(declared_date="2020-01-01")
        b = _meta(declared_date="2020-01-01")
        d = router.decide("what changed", [a, b])
        self.assertEqual(d.task_type, "bitemporal_change")

    def test_no_distinguishing_signal_refuses_to_guess(self):
        a = _meta()
        b = _meta()
        d = router.decide("tell me something interesting", [a, b])
        self.assertEqual(d.task_type, "needs_clarification")
        self.assertEqual(d.tool_name, "none")

    def test_three_inputs_is_unsupported(self):
        d = router.decide("anything", [_meta(), _meta(), _meta()])
        self.assertEqual(d.task_type, "needs_clarification")

    def test_change_keyword_breaks_tie_when_same_modality_no_dates(self):
        a = _meta(declared_modality="optical")
        b = _meta(declared_modality="optical")
        d = router.decide("compare before and after", [a, b])
        self.assertEqual(d.task_type, "bitemporal_change")


if __name__ == "__main__":
    unittest.main()
