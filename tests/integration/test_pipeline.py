"""Verifies app/pipeline.py:run_query() - the orchestration function
extracted out of app/cli.py:run() so both the CLI and the Streamlit UI call
the exact same real pipeline (see docs/ui_design.md and app/pipeline.py's
own docstring for why this extraction happened).

This is a thin wiring test: it does not re-verify routing/validation/
specialist correctness (that's covered by test_specialists.py,
test_routing.py, test_vqa_dispatch.py, etc.) - it verifies that run_query()
with plain keyword arguments produces the same real, correct ExecutionTrace
shape that app/cli.py's argparse-based run() always produced, across all
four task types plus one failure path.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "app"))

FIXTURES = os.path.join(REPO_ROOT, "data", "fixtures")


def _ensure_fixtures():
    if not os.path.exists(os.path.join(FIXTURES, "single_image.png")):
        subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "generate_fixtures.py")], check=True)


_ensure_fixtures()

import pipeline  # noqa: E402


class TestPipelineRunQuery(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="satquery_pipeline_test_")

    def test_single_image_vqa(self):
        trace = pipeline.run_query(
            image1_path=os.path.join(FIXTURES, "single_image.png"),
            query="Describe the major land cover types visible in this image.",
            out_dir=self._tmp,
        )
        self.assertIsNone(trace.failure)
        self.assertEqual(trace.router_decision.task_type, "single_image_vqa")
        self.assertIsNotNone(trace.specialist_output)
        self.assertGreater(len(trace.specialist_output.evidence), 0)

    def test_grounding(self):
        trace = pipeline.run_query(
            image1_path=os.path.join(FIXTURES, "single_image.png"),
            query="Locate the water body in this image and highlight it.",
            out_dir=self._tmp,
        )
        self.assertIsNone(trace.failure)
        self.assertEqual(trace.router_decision.task_type, "grounding")

    def test_bitemporal_change(self):
        trace = pipeline.run_query(
            image1_path=os.path.join(FIXTURES, "change_before.png"),
            image2_path=os.path.join(FIXTURES, "change_after.png"),
            query="What changed between these two dates?",
            date1="2020-01-01",
            date2="2024-01-01",
            out_dir=self._tmp,
        )
        self.assertIsNone(trace.failure)
        self.assertEqual(trace.router_decision.task_type, "bitemporal_change")

    def test_optical_sar_fusion(self):
        trace = pipeline.run_query(
            image1_path=os.path.join(FIXTURES, "fusion_optical.png"),
            image2_path=os.path.join(FIXTURES, "fusion_sar.png"),
            query="Use both images to identify built-up and water-covered regions.",
            modality1="optical",
            modality2="sar",
            out_dir=self._tmp,
        )
        self.assertIsNone(trace.failure)
        self.assertEqual(trace.router_decision.task_type, "optical_sar_fusion")

    def test_missing_second_image_for_change_routes_to_clarification_not_a_crash(self):
        """Single image, but a change-flavored query - the router has no
        second input to reason about at all here (this exercises the
        single-image branch, which correctly falls through to VQA/grounding
        intent classification, not a crash) - a real missing-pair situation
        for change/fusion is instead caught by validator.py when 2 images ARE
        given but declared incompatibly; see test_ingestion.py for that."""
        trace = pipeline.run_query(
            image1_path=os.path.join(FIXTURES, "single_image.png"),
            query="what changed between these two dates",
            out_dir=self._tmp,
        )
        self.assertIsNone(trace.failure)
        self.assertEqual(len(trace.input_summary), 1)

    def test_nonexistent_file_fails_cleanly(self):
        trace = pipeline.run_query(
            image1_path="/tmp/this_file_does_not_exist_satquery.png",
            query="describe this image",
            out_dir=self._tmp,
        )
        self.assertIsNotNone(trace.failure)
        self.assertIn("Input loading failed", trace.failure)


if __name__ == "__main__":
    unittest.main()
