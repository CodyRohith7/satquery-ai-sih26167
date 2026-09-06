"""Regression tests for specialists/vqa_dispatch.py's fallback contract, and
for routing/router.py's dynamic VQA-backend selection.

Both are tested by mocking `models.registry.get()` directly, rather than by
installing real torch/transformers - this project's cloud development
sandbox has neither (see docs/network_constraints.md), so a test that only
passed when the real deep-learning path executed would silently skip here
and never actually verify the fallback logic. Mocking the registry lookup
lets these tests deterministically exercise BOTH the "SmolVLM available"
and "SmolVLM unavailable" branches on any machine, matching the pattern
used in test_florence2_compat.py's TestGenerationCompatibility (mock the
model call itself, not the whole environment).

What these tests do NOT cover: whether SmolVLM's own loading/generation
code (specialists/vqa_smolvlm.py) actually works against the real model -
that can only be verified on real hardware with torch/transformers
installed (see docs/RUN_ON_WINDOWS.md for the real-hardware verification
step). These tests cover the DISPATCH CONTRACT: given SmolVLM succeeds,
fails, or isn't available, does the right thing happen and is it disclosed
honestly - regardless of which underlying reason applies.
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

FIXTURES = os.path.join(REPO_ROOT, "data", "fixtures")


def _ensure_fixtures():
    if not os.path.exists(os.path.join(FIXTURES, "single_image.png")):
        subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "generate_fixtures.py")], check=True)


_ensure_fixtures()

from ingestion import raster_io, metadata as metadata_mod
from routing.schemas import SpecialistOutput, ConfidenceResult
from models import registry
from specialists import vqa_dispatch
from routing import router


def _load_single_image():
    r = raster_io.load(os.path.join(FIXTURES, "single_image.png"))
    m = metadata_mod.inspect(r)
    return r.array, m


def _fake_available_entry():
    return registry.ModelEntry(
        name=vqa_dispatch.SMOLVLM_TOOL_NAME,
        purpose="test double",
        input_modality="single image (RGB)",
        output="n/a",
        provenance="test double",
        framework="n/a",
        status=registry.AVAILABLE,
    )


def _fake_unavailable_entry():
    return registry.ModelEntry(
        name=vqa_dispatch.SMOLVLM_TOOL_NAME,
        purpose="test double",
        input_modality="single image (RGB)",
        output="n/a",
        provenance="test double",
        framework="n/a",
        status=registry.NOT_AVAILABLE_SANDBOX,
        unavailable_reason="test double: forced unavailable",
    )


def _fake_smolvlm_output():
    return SpecialistOutput(
        answer_text="a fake SmolVLM answer",
        evidence=[],
        confidence=ConfidenceResult(
            value=0.8,
            method_version="v1_vlm_mean_token_probability",
            basis={},
            basis_description="mean generated-token probability (test double)",
        ),
        raw={"model_id": "HuggingFaceTB/SmolVLM-256M-Instruct"},
        tool_name=vqa_dispatch.SMOLVLM_TOOL_NAME,
        task_type="single_image_vqa",
        latency_seconds=1.23,
        model_name="HuggingFaceTB/SmolVLM-256M-Instruct",
        fallback_occurred=False,
        fallback_reason=None,
    )


class TestVqaDispatchFallbackContract(unittest.TestCase):
    def test_uses_classical_directly_when_registry_entry_missing(self):
        arr, meta = _load_single_image()
        with patch("specialists.vqa_dispatch.registry.get", side_effect=KeyError(vqa_dispatch.SMOLVLM_TOOL_NAME)):
            out = vqa_dispatch.run(arr, meta, "describe this", [], 0.9, "/tmp/_it_vd_ev1.png")
        self.assertEqual(out.tool_name, "tool_single_image_vqa_v0")
        self.assertNotIn("smolvlm_attempted", out.raw)
        self.assertIsNone(out.model_name)
        self.assertFalse(out.fallback_occurred)
        self.assertIsNone(out.fallback_reason)
        self.assertEqual(out.confidence.method_version, "v0_classical")

    def test_uses_classical_directly_when_status_not_available(self):
        arr, meta = _load_single_image()
        with patch("specialists.vqa_dispatch.registry.get", return_value=_fake_unavailable_entry()):
            out = vqa_dispatch.run(arr, meta, "describe this", [], 0.9, "/tmp/_it_vd_ev2.png")
        self.assertEqual(out.tool_name, "tool_single_image_vqa_v0")
        self.assertNotIn("smolvlm_attempted", out.raw)
        self.assertIsNone(out.model_name)
        self.assertFalse(out.fallback_occurred)

    def test_uses_smolvlm_output_verbatim_when_available_and_it_succeeds(self):
        arr, meta = _load_single_image()
        with patch("specialists.vqa_dispatch.registry.get", return_value=_fake_available_entry()), \
             patch("specialists.vqa_dispatch.vqa_smolvlm.run", return_value=_fake_smolvlm_output()):
            out = vqa_dispatch.run(arr, meta, "describe this", [], 0.9, "/tmp/_it_vd_ev3.png")
        self.assertEqual(out.tool_name, vqa_dispatch.SMOLVLM_TOOL_NAME)
        self.assertEqual(out.answer_text, "a fake SmolVLM answer")
        # Schema-level disclosure fields - a UI must be able to read these
        # directly, never infer them from tool_name string matching.
        self.assertEqual(out.model_name, "HuggingFaceTB/SmolVLM-256M-Instruct")
        self.assertFalse(out.fallback_occurred)
        self.assertIsNone(out.fallback_reason)
        # The confidence methodology must be the VLM one, never silently
        # relabeled as the classical baseline's - this was the exact bug
        # found on real hardware (confidence showed method=v0_classical for
        # a SmolVLM-produced answer).
        self.assertEqual(out.confidence.method_version, "v1_vlm_mean_token_probability")
        self.assertNotEqual(out.confidence.method_version, "v0_classical")

    def test_falls_back_to_classical_with_full_disclosure_when_smolvlm_raises(self):
        arr, meta = _load_single_image()

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated SmolVLM failure")

        with patch("specialists.vqa_dispatch.registry.get", return_value=_fake_available_entry()), \
             patch("specialists.vqa_dispatch.vqa_smolvlm.run", side_effect=_boom):
            out = vqa_dispatch.run(arr, meta, "describe this", [], 0.9, "/tmp/_it_vd_ev4.png")

        # Must have actually fallen back to the classical implementation's
        # real output shape (cluster_fractions), not an invented one.
        self.assertEqual(out.tool_name, "tool_single_image_vqa_v0")
        self.assertIn("cluster_fractions", out.raw)
        # And the fallback must be disclosed, not silent - both as
        # first-class schema fields (never requiring the UI to infer status
        # from raw's keys) and in raw for the technical-details panel.
        self.assertTrue(out.fallback_occurred)
        self.assertIn("simulated SmolVLM failure", out.fallback_reason)
        self.assertIsNone(out.model_name)  # it's the classical output now - no model produced it
        self.assertTrue(out.raw["smolvlm_attempted"])
        self.assertIn("simulated SmolVLM failure", out.raw["smolvlm_fallback_reason"])
        self.assertIn("RuntimeError", out.raw["smolvlm_fallback_traceback"])
        self.assertIn("SmolVLM-256M-Instruct failed", out.answer_text)
        # Confidence methodology must honestly reflect what actually ran
        # (classical), not be left over from the failed SmolVLM attempt.
        self.assertEqual(out.confidence.method_version, "v0_classical")


class TestRouterVqaBackendSelection(unittest.TestCase):
    def test_router_selects_smolvlm_when_registry_reports_available(self):
        with patch("routing.router.registry_mod.get", return_value=_fake_available_entry()):
            self.assertEqual(router._select_vqa_tool_name(), vqa_dispatch.SMOLVLM_TOOL_NAME)

    def test_router_selects_classical_when_registry_reports_unavailable(self):
        with patch("routing.router.registry_mod.get", return_value=_fake_unavailable_entry()):
            self.assertEqual(router._select_vqa_tool_name(), "tool_single_image_vqa_v0")

    def test_router_selects_classical_when_registry_entry_missing(self):
        with patch("routing.router.registry_mod.get", side_effect=KeyError(vqa_dispatch.SMOLVLM_TOOL_NAME)):
            self.assertEqual(router._select_vqa_tool_name(), "tool_single_image_vqa_v0")

    def test_full_decide_call_reflects_mocked_availability(self):
        from ingestion.metadata import ImageMetadata

        meta = ImageMetadata(
            path="x.png", height=100, width=100, band_count=3, dtype="uint8",
            crs=None, transform=None, declared_modality=None,
            inferred_modality="optical (unconfirmed: 3-band file consistent with RGB)",
            modality_confidence="heuristic", declared_date=None,
        )
        with patch("routing.router.registry_mod.get", return_value=_fake_available_entry()):
            d = router.decide("describe the land cover in this scene", [meta])
        self.assertEqual(d.task_type, "single_image_vqa")
        self.assertEqual(d.tool_name, vqa_dispatch.SMOLVLM_TOOL_NAME)
        self.assertTrue(any("SmolVLM" in r for r in d.reasoning))


if __name__ == "__main__":
    unittest.main()
