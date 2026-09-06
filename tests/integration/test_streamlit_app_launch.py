"""Real Streamlit smoke tests using `streamlit.testing.v1.AppTest`.

These are the tests that actually verify the app LAUNCHES and renders in a
real (headless) Streamlit runtime - something test_streamlit_app_logic.py's
fake-streamlit-module tests deliberately cannot do. They require streamlit
to be installed, which this project's cloud development sandbox cannot do
(no PyPI network egress - see docs/network_constraints.md), so they
`unittest.skipUnless` themselves out here and are meant to actually run on
the user's real machine (see docs/RUN_ON_WINDOWS.md for the command).

`AppTest` cannot easily simulate a real file_uploader with in-memory bytes
across streamlit versions, so these tests stick to what AppTest supports
robustly: does the app run at all without raising, and does it expose the
four mandatory demo-view tabs with the "use bundled demo fixture" checkboxes
that let a user drive a full analysis run through the UI itself. The
fixture-driven, full-pipeline logic (does clicking RUN actually call the
real pipeline and render a real result) is already covered for real,
end-to-end, by test_streamlit_app_logic.py - that coverage does not need to
be duplicated through AppTest's slower and more brittle simulation surface.
"""
from __future__ import annotations

import importlib.util
import os
import unittest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
APP_PATH = os.path.join(REPO_ROOT, "app", "streamlit_app.py")

_STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None


@unittest.skipUnless(_STREAMLIT_AVAILABLE, "streamlit is not installed in this environment")
class TestStreamlitAppLaunches(unittest.TestCase):
    def _app(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()
        return at

    def test_app_launches_without_exception(self):
        at = self._app()
        self.assertFalse(at.exception, f"App raised on launch: {[str(e) for e in at.exception]}")

    def test_all_four_mandatory_tabs_are_present(self):
        at = self._app()
        tab_labels = [t.label for t in at.tabs]
        self.assertEqual(len(tab_labels), 4)
        self.assertTrue(any("SINGLE IMAGE" in l for l in tab_labels))
        self.assertTrue(any("GROUNDING" in l for l in tab_labels))
        self.assertTrue(any("CHANGE" in l for l in tab_labels))
        self.assertTrue(any("OPTICAL" in l for l in tab_labels))

    def test_sample_image_checkboxes_present_on_every_tab(self):
        """Every capability tab must offer the clearly-labeled synthetic
        sample-image path (see docs/ui_design.md's demo-mode rule: never
        make synthetic data look real) so the app is fully driveable without
        the analyst supplying their own imagery."""
        at = self._app()
        checkbox_labels = [c.label for c in at.checkbox]
        sample_checkboxes = [l for l in checkbox_labels if "bundled sample" in l.lower()]
        self.assertEqual(len(sample_checkboxes), 4, checkbox_labels)


if __name__ == "__main__":
    unittest.main()
