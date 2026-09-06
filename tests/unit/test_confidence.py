import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from confidence import engine


class TestConfidenceEngine(unittest.TestCase):
    def test_perfect_signal_no_penalties(self):
        r = engine.compute(base_signal=1.0, validation_warnings=[], any_modality_unknown=False, router_confidence=0.95)
        self.assertEqual(r.value, 1.0)
        self.assertEqual(r.downgrades, [])

    def test_unknown_modality_downgrades(self):
        r = engine.compute(base_signal=1.0, validation_warnings=[], any_modality_unknown=True, router_confidence=0.95)
        self.assertLess(r.value, 1.0)
        self.assertTrue(any("modality unknown" in d for d in r.downgrades))

    def test_warnings_capped_downgrade(self):
        many_warnings = [f"warning {i}" for i in range(10)]
        r = engine.compute(base_signal=1.0, validation_warnings=many_warnings, any_modality_unknown=False, router_confidence=0.95)
        # capped at 0.30 downgrade regardless of how many warnings
        self.assertAlmostEqual(r.basis["total_downgrade"], 0.30, places=2)

    def test_ambiguous_routing_downgrades(self):
        r = engine.compute(base_signal=1.0, validation_warnings=[], any_modality_unknown=False, router_confidence=0.4)
        self.assertLess(r.value, 1.0)

    def test_zero_base_signal_stays_zero(self):
        r = engine.compute(base_signal=0.0, validation_warnings=[], any_modality_unknown=False, router_confidence=0.95)
        self.assertEqual(r.value, 0.0)

    def test_value_always_in_bounds(self):
        for base in (-1.0, 0.0, 0.5, 1.0, 2.0):
            r = engine.compute(base_signal=base, validation_warnings=["a"] * 5, any_modality_unknown=True, router_confidence=0.1)
            self.assertGreaterEqual(r.value, 0.0)
            self.assertLessEqual(r.value, 1.0)

    def test_method_version_is_labeled(self):
        r = engine.compute(base_signal=0.5, validation_warnings=[], any_modality_unknown=False, router_confidence=0.9)
        self.assertEqual(r.method_version, "v0_classical")

    def test_default_basis_description_names_it_classical(self):
        r = engine.compute(base_signal=0.5, validation_warnings=[], any_modality_unknown=False, router_confidence=0.9)
        self.assertIn("Classical-CV", r.basis_description)

    def test_caller_can_override_method_version_and_basis_description(self):
        """Regression test for the exact bug found on real hardware: SmolVLM's
        confidence was being stamped method_version='v0_classical' even
        though its base_signal came from real generation-token
        probabilities, not a classical clustering/threshold score. compute()
        must let a caller declare its own methodology label rather than
        always hardcoding the classical one."""
        r = engine.compute(
            base_signal=0.5,
            validation_warnings=[],
            any_modality_unknown=False,
            router_confidence=0.9,
            method_version=engine.METHOD_VLM_TOKEN_CONFIDENCE,
            basis_description="mean generated-token probability",
        )
        self.assertEqual(r.method_version, "v1_vlm_mean_token_probability")
        self.assertNotEqual(r.method_version, "v0_classical")
        self.assertEqual(r.basis_description, "mean generated-token probability")


if __name__ == "__main__":
    unittest.main()
