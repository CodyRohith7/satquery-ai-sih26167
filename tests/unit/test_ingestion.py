import os
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ingestion import raster_io, metadata, validator


class TestRasterIO(unittest.TestCase):
    def setUp(self):
        self.tmp_rgb = "/tmp/_test_rgb.png"
        self.tmp_gray = "/tmp/_test_gray.png"
        Image.fromarray((np.random.rand(40, 40, 3) * 255).astype("uint8")).save(self.tmp_rgb)
        Image.fromarray((np.random.rand(40, 40) * 255).astype("uint8")).save(self.tmp_gray)

    def test_loads_rgb(self):
        r = raster_io.load(self.tmp_rgb)
        self.assertEqual(r.band_count, 3)
        self.assertEqual((r.height, r.width), (40, 40))

    def test_loads_single_band(self):
        r = raster_io.load(self.tmp_gray)
        self.assertEqual(r.band_count, 1)

    def test_missing_file_raises(self):
        with self.assertRaises(raster_io.RasterLoadError):
            raster_io.load("/tmp/does_not_exist_xyz.png")

    def test_no_geospatial_metadata_is_none_not_guessed(self):
        r = raster_io.load(self.tmp_rgb)
        self.assertIsNone(r.crs)
        self.assertIsNone(r.transform)


class TestMetadataHonesty(unittest.TestCase):
    def setUp(self):
        self.tmp = "/tmp/_test_meta.png"
        Image.fromarray((np.random.rand(40, 40, 3) * 255).astype("uint8")).save(self.tmp)
        self.raster = raster_io.load(self.tmp)

    def test_undeclared_modality_is_heuristic_not_certain(self):
        m = metadata.inspect(self.raster)
        self.assertEqual(m.modality_confidence, "heuristic")
        self.assertIn("unconfirmed", m.inferred_modality)

    def test_declared_modality_is_trusted_and_labeled_as_declared(self):
        m = metadata.inspect(self.raster, declared_modality="sar")
        self.assertEqual(m.modality, "sar")
        self.assertEqual(m.modality_confidence, "declared")

    def test_missing_crs_renders_as_unavailable_not_a_guess(self):
        m = metadata.inspect(self.raster)
        summary = m.metadata_summary()
        self.assertEqual(summary["crs"], "Metadata unavailable")


class TestValidator(unittest.TestCase):
    def setUp(self):
        self.tmp = "/tmp/_test_val.png"
        Image.fromarray((np.random.rand(40, 40, 3) * 255).astype("uint8")).save(self.tmp)
        self.raster = raster_io.load(self.tmp)
        self.meta = metadata.inspect(self.raster, declared_modality="optical", declared_date="2024-01-01")

    def test_valid_single_image_passes(self):
        v = validator.validate_single(self.meta)
        self.assertTrue(v.ok)

    def test_tiny_image_fails(self):
        tiny = np.zeros((3, 3, 3), dtype="uint8")
        Image.fromarray(tiny).save("/tmp/_test_tiny.png")
        r = raster_io.load("/tmp/_test_tiny.png")
        m = metadata.inspect(r)
        v = validator.validate_single(m)
        self.assertFalse(v.ok)

    def test_bitemporal_same_date_is_an_error(self):
        meta2 = metadata.inspect(self.raster, declared_modality="optical", declared_date="2024-01-01")
        v = validator.validate_pair(self.meta, meta2, intended_relationship="bitemporal")
        self.assertFalse(v.ok)

    def test_fusion_same_modality_is_an_error(self):
        meta2 = metadata.inspect(self.raster, declared_modality="optical", declared_date="2024-06-01")
        v = validator.validate_pair(self.meta, meta2, intended_relationship="cross_modal")
        self.assertFalse(v.ok)


if __name__ == "__main__":
    unittest.main()
