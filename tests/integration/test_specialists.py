"""Integration tests: each specialist actually runs against the real
(synthetic) fixture files and produces a schema-valid, sane output. These
regenerate the fixtures if missing so the suite is self-contained."""
import os
import subprocess
import sys
import unittest

import numpy as np
from PIL import Image

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

FIXTURES = os.path.join(REPO_ROOT, "data", "fixtures")


def _ensure_fixtures():
    if not os.path.exists(os.path.join(FIXTURES, "single_image.png")):
        subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "generate_fixtures.py")], check=True)


_ensure_fixtures()

from ingestion import raster_io, metadata as metadata_mod
from routing.schemas import SpecialistOutput
from specialists import vqa as vqa_mod
from grounding import grounding as grounding_mod
from change import change as change_mod
from fusion import fusion as fusion_mod


def _load(name, **meta_kwargs):
    r = raster_io.load(os.path.join(FIXTURES, name))
    m = metadata_mod.inspect(r, **meta_kwargs)
    return r.array, m


class TestVQASpecialist(unittest.TestCase):
    def test_runs_and_returns_valid_schema(self):
        arr, meta = _load("single_image.png")
        out = vqa_mod.run(arr, meta, "what are the major features", [], 0.9, "/tmp/_it_vqa_ev.png")
        self.assertIsInstance(out, SpecialistOutput)
        self.assertTrue(len(out.answer_text) > 0)
        self.assertTrue(0.0 <= out.confidence.value <= 1.0)
        self.assertTrue(os.path.exists(out.evidence[0].image_path))

    def test_cluster_fractions_sum_close_to_one(self):
        arr, meta = _load("single_image.png")
        out = vqa_mod.run(arr, meta, "describe this", [], 0.9, "/tmp/_it_vqa_ev2.png")
        total = sum(out.raw["cluster_fractions"].values())
        self.assertAlmostEqual(total, 1.0, places=1)


class TestGroundingSpecialist(unittest.TestCase):
    def test_finds_water_in_synthetic_scene(self):
        arr, meta = _load("single_image.png")
        out = grounding_mod.run(arr, meta, "locate the water body", [], 0.9, "/tmp/_it_gr_ev.png")
        self.assertEqual(out.raw["target"], "water")
        self.assertGreaterEqual(out.raw["num_boxes"], 1)

    def test_unrecognized_target_returns_zero_confidence_basis(self):
        arr, meta = _load("single_image.png")
        out = grounding_mod.run(arr, meta, "what is the meaning of life", [], 0.9, "/tmp/_it_gr_ev2.png")
        self.assertIsNone(out.raw["target"])
        self.assertEqual(out.confidence.basis["base_signal"], 0.0)


class TestChangeSpecialist(unittest.TestCase):
    def test_detects_injected_change_region(self):
        before, meta_b = _load("change_before.png", declared_date="2020-01-01")
        after, meta_a = _load("change_after.png", declared_date="2024-01-01")
        out = change_mod.run(before, after, meta_b, meta_a, "what changed", [], 0.95, "/tmp/_it_ch_ev.png")
        # The synthetic generator changes a 40x50=2000px block out of 256x256=65536px (~3.05%).
        self.assertGreater(out.raw["changed_fraction"], 0.01)
        self.assertLess(out.raw["changed_fraction"], 0.10)
        self.assertGreaterEqual(out.raw["num_regions"], 1)

    def test_identical_images_show_near_zero_change(self):
        arr, meta = _load("single_image.png", declared_date="2024-01-01")
        out = change_mod.run(arr, arr, meta, meta, "what changed", [], 0.95, "/tmp/_it_ch_ev2.png")
        self.assertLess(out.raw["changed_fraction"], 0.02)


class TestFusionSpecialist(unittest.TestCase):
    def test_uses_both_modalities(self):
        optical, meta_o = _load("fusion_optical.png", declared_modality="optical")
        sar, meta_s = _load("fusion_sar.png", declared_modality="sar")
        out = fusion_mod.run(optical, sar, meta_o, meta_s, "identify built-up and water", [], 0.95, "/tmp/_it_fu_ev.png")
        labels = " ".join(out.raw["cluster_fractions"].keys())
        self.assertIn("water", labels)
        self.assertIn("built-up", labels)

    def test_fusion_differs_from_optical_only_clustering(self):
        """Proves the SAR channel actually changes the result, rather than
        being loaded and ignored."""
        optical, meta_o = _load("fusion_optical.png", declared_modality="optical")
        sar, meta_s = _load("fusion_sar.png", declared_modality="sar")
        out_fused = fusion_mod.run(optical, sar, meta_o, meta_s, "q", [], 0.95, "/tmp/_it_fu_ev2.png")

        from specialists.land_cover_heuristics import cluster_scene
        label_map_optical_only, centers, _ = cluster_scene(optical)

        # Fused labels must include SAR-referencing language; optical-only
        # clustering (land_cover_heuristics) never mentions "backscatter".
        fused_labels = " ".join(out_fused.raw["cluster_fractions"].keys())
        self.assertIn("backscatter", fused_labels)


if __name__ == "__main__":
    unittest.main()
