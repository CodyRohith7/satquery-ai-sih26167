"""Verifies app/streamlit_app.py's LOGIC against a hand-built fake `streamlit`
module - control flow, widget wiring, and (critically) that the app calls the
REAL `pipeline.run_query()` with real arguments and renders the REAL
`ExecutionTrace` it returns, exactly like `app/cli.py` does.

Real `streamlit` cannot be installed in this project's cloud development
sandbox (no PyPI network egress - see docs/network_constraints.md), so this
test cannot exercise real widget rendering or CSS. That is intentionally
covered by a SEPARATE test - test_streamlit_app_launch.py's
`streamlit.testing.v1.AppTest`-based smoke tests, which `unittest.skipUnless`
gate on streamlit actually being importable, so they run for real on the
user's machine (see docs/RUN_ON_WINDOWS.md) and simply skip here.

This test's job is narrower and different: prove the app's Python control
flow is correct (buttons wire to the right session_state keys, the right
pipeline.run_query() arguments are built per tab, results survive reruns via
session_state, and render_result()/render_disclosure()/render_confidence()/
render_failure() all execute without exception against REAL SpecialistOutput/
ExecutionTrace objects produced by REAL specialist code) - by loading the
actual app/streamlit_app.py source with runpy against a fake `streamlit`
stub, the same "mock the boundary, not the whole environment" pattern used
in test_vqa_dispatch.py and test_florence2_compat.py.
"""
from __future__ import annotations

import contextlib
import datetime
import os
import runpy
import shutil
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
APP_PATH = os.path.join(REPO_ROOT, "app", "streamlit_app.py")
FIXTURES = os.path.join(REPO_ROOT, "data", "fixtures")

sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


def _ensure_fixtures():
    if not os.path.exists(os.path.join(FIXTURES, "single_image.png")):
        subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "generate_fixtures.py")], check=True)


_ensure_fixtures()


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeUploadedFile:
    """Stands in for streamlit's UploadedFile: a real file's bytes, under a
    real filename, exactly as app/streamlit_app.py's `_save_upload()` expects."""

    def __init__(self, path):
        self._path = path
        self.name = os.path.basename(path)

    def getvalue(self):
        with open(self._path, "rb") as f:
            return f.read()


def _make_fake_streamlit(session_state: dict, button_true_key=None):
    """A minimal fake `streamlit` module. Widget getters read from
    `session_state` (so a test can pre-seed exactly what a real user's
    inputs would produce); `button()` returns True only for the one key
    under test, simulating "the analyst clicked RUN ANALYSIS on this tab
    and no other" for one script rerun - matching Streamlit's own rerun
    model, where every tab's code runs every time but only one button call
    returns True per click."""
    calls = {"button_keys_seen": [], "images": [], "downloads": [], "button_types": {}}
    st = types.ModuleType("streamlit")
    st.session_state = session_state

    st.set_page_config = lambda **kwargs: None
    st.markdown = lambda text, *a, **kwargs: calls.setdefault("markdown", []).append(text)
    st.caption = lambda *a, **kwargs: None
    st.code = lambda *a, **kwargs: None
    st.json = lambda *a, **kwargs: None
    st.sidebar = _Ctx()
    st.tabs = lambda labels: [_Ctx() for _ in labels]
    st.columns = lambda spec, gap=None: [_Ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.checkbox = lambda label, key=None, value=False, **kw: session_state.get(key, value)
    st.file_uploader = lambda label, type=None, key=None, **kw: session_state.get(key)
    st.selectbox = lambda label, options, key=None, **kw: session_state.get(key, options[0])
    st.text_input = lambda label, value="", key=None, **kw: session_state.get(key, value)
    st.date_input = lambda label, value=None, key=None, **kw: session_state.get(key, value)

    def _button(label, key=None, disabled=False, on_click=None, args=(), **kw):
        calls["button_keys_seen"].append((key, disabled))
        calls["button_types"][key] = kw.get("type")
        clicked = (key == button_true_key) and not disabled
        if clicked and on_click is not None:
            # Real Streamlit invokes on_click callbacks BEFORE the script
            # body re-executes for the rerun triggered by this click - so
            # session_state is already updated by the time any widget reads
            # it later in this same simulated pass.
            on_click(*args)
        return clicked

    st.button = _button

    def _rerun():
        calls["rerun_count"] = calls.get("rerun_count", 0) + 1

    st.rerun = _rerun

    @contextlib.contextmanager
    def _spinner(msg):
        yield

    @contextlib.contextmanager
    def _expander(label):
        calls.setdefault("expanders", []).append(label)
        yield

    st.spinner = _spinner
    st.expander = _expander
    st.image = lambda img, use_container_width=None, **kw: calls["images"].append(img)
    st.download_button = lambda label, data, file_name=None, mime=None, key=None, **kw: calls["downloads"].append(
        (label, file_name, key, kw.get("type"))
    )
    st._calls = calls
    return st


@contextlib.contextmanager
def _installed_as_streamlit(fake_module):
    old = sys.modules.get("streamlit")
    sys.modules["streamlit"] = fake_module
    try:
        yield fake_module
    finally:
        if old is not None:
            sys.modules["streamlit"] = old
        else:
            sys.modules.pop("streamlit", None)


def _run_app(session_state: dict, button_true_key=None):
    fake = _make_fake_streamlit(session_state, button_true_key=button_true_key)
    with _installed_as_streamlit(fake):
        runpy.run_path(APP_PATH, run_name="__not_main__")
    return fake


def _run_app_ns(session_state: dict, button_true_key=None):
    """Like _run_app, but also returns the app module's real namespace
    (runpy.run_path returns the executed globals) - used to unit-test the
    app's own pure helper functions (e.g. _capability_backend_label)
    directly against the real registry, instead of scraping rendered
    markdown strings for them."""
    fake = _make_fake_streamlit(session_state, button_true_key=button_true_key)
    with _installed_as_streamlit(fake):
        ns = runpy.run_path(APP_PATH, run_name="__not_main__")
    return fake, ns


class TestStreamlitAppLogic(unittest.TestCase):
    def test_initial_load_has_all_run_buttons_disabled(self):
        """No inputs supplied yet on any tab -> every RUN ANALYSIS button
        must start disabled, never runnable against nothing. (Example-query
        chip buttons are a separate, always-enabled control - they only
        fill the query field and are excluded from this check.)"""
        fake = _run_app({})
        run_buttons = [(k, d) for k, d in fake._calls["button_keys_seen"] if k is not None and k.endswith("_run")]
        self.assertEqual(len(run_buttons), 4, fake._calls["button_keys_seen"])
        for key, disabled in run_buttons:
            self.assertTrue(disabled, f"{key} should start disabled with no input")

    def test_vqa_tab_runs_real_pipeline_and_renders_result(self):
        tmp_upload = os.path.join(REPO_ROOT, "data", "fixtures", "single_image.png")
        session_state = {
            "vqa_upload": _FakeUploadedFile(tmp_upload),
            "vqa_query": "Describe the major land cover types visible in this image.",
            "vqa_modality": "Not declared",
            "vqa_fixture": False,
        }
        fake = _run_app(session_state, button_true_key="vqa_run")
        self.assertIn("vqa_trace", session_state)
        trace = session_state["vqa_trace"]
        self.assertIsNone(trace.failure)
        self.assertEqual(trace.router_decision.task_type, "single_image_vqa")
        self.assertIsNotNone(trace.specialist_output)
        self.assertGreaterEqual(len(fake._calls["images"]), 1)
        # Both JSON and (reportlab is installed in this sandbox) PDF export
        # buttons must be offered for a successful result. Revision 4 makes
        # PDF the visually stronger/primary export action and renames the
        # JSON button to a plain "JSON" label - see render_export_row().
        download_labels = [d[0] for d in fake._calls["downloads"]]
        self.assertIn("JSON", download_labels)
        self.assertIn("PDF report", download_labels)

    def test_fusion_tab_shows_raw_optical_and_sar_plus_fused_evidence(self):
        """Regression test for the fusion evidence-shape decision documented
        in docs/ui_design.md: tool_fusion_v0's own evidence is a single fused
        cluster map, so the UI itself must additionally render the two raw
        inputs - three images total, not just the one fused evidence image."""
        session_state = {"fu_fixture": True, "fu_query": "Use both images to identify built-up and water-covered regions."}
        fake = _run_app(session_state, button_true_key="fu_run")
        self.assertIn("fu_trace", session_state)
        trace = session_state["fu_trace"]
        self.assertIsNone(trace.failure)
        self.assertGreaterEqual(len(fake._calls["images"]), 3)

    def test_change_tab_runs_real_pipeline(self):
        session_state = {
            "ch_fixture": True,
            "ch_query": "What changed between these two dates?",
            "ch_date1": datetime.date(2020, 1, 1),
            "ch_date2": datetime.date(2024, 1, 1),
        }
        _run_app(session_state, button_true_key="ch_run")
        self.assertIn("ch_trace", session_state)
        self.assertIsNone(session_state["ch_trace"].failure)

    def test_change_tab_shows_clean_no_change_state_for_identical_images(self):
        """Regression test for docs/ui_design.md Revision 2 rule 4: when the
        real changed_fraction from tool_change_v0 rounds to 0.0%, the UI adds
        an explicit 'No significant change detected' headline - comparing an
        image against itself is real input that produces real 0% change, not
        a mocked/invented result."""
        same_path = os.path.join(REPO_ROOT, "data", "fixtures", "single_image.png")
        session_state = {
            "ch_upload1": _FakeUploadedFile(same_path),
            "ch_upload2": _FakeUploadedFile(same_path),
            "ch_fixture": False,
            "ch_query": "What changed between these two dates?",
            "ch_date1": datetime.date(2020, 1, 1),
            "ch_date2": datetime.date(2024, 1, 1),
        }
        fake = _run_app(session_state, button_true_key="ch_run")
        trace = session_state["ch_trace"]
        self.assertIsNone(trace.failure)
        self.assertEqual(trace.specialist_output.raw.get("changed_fraction"), 0.0)
        self.assertTrue(any("No significant change detected" in m for m in fake._calls.get("markdown", [])))

    def test_grounding_tab_renders_professional_failure_state_not_a_crash(self):
        """A too-small image must produce a rendered failure panel (via
        render_failure()), not an unhandled exception in the UI layer -
        validator.py already raises this as a ValidationResult error; the UI
        must surface it, not choke on it."""
        from PIL import Image
        import numpy as np

        tiny_path = "/tmp/_test_streamlit_tiny.png"
        Image.fromarray((np.random.rand(4, 4, 3) * 255).astype("uint8")).save(tiny_path)
        session_state = {
            "gr_upload": _FakeUploadedFile(tiny_path),
            "gr_query": "Locate the water body in this image and highlight it.",
            "gr_modality": "Not declared",
            "gr_fixture": False,
        }
        _run_app(session_state, button_true_key="gr_run")
        self.assertIn("gr_trace", session_state)
        trace = session_state["gr_trace"]
        self.assertIsNotNone(trace.failure)
        self.assertIn("too small", trace.failure.lower())
        os.remove(tiny_path)

    def test_disclosure_shows_real_model_fields_when_smolvlm_is_available(self):
        """Regression test for the model/tool disclosure contract: when the
        registry reports SmolVLM AVAILABLE and it succeeds, the trace the UI
        stores must carry model_name/confidence.method_version exactly as
        specialists/vqa_dispatch.py produces them - the UI must never relabel
        or drop these fields. Mocked the same way test_vqa_dispatch.py does."""
        from models import registry
        from specialists import vqa_dispatch
        from routing.schemas import SpecialistOutput, ConfidenceResult

        def _fake_available_entry():
            return registry.ModelEntry(
                name=vqa_dispatch.SMOLVLM_TOOL_NAME, purpose="t", input_modality="single image (RGB)",
                output="n/a", provenance="t", framework="n/a", status=registry.AVAILABLE,
            )

        def _fake_smolvlm_output(*a, **k):
            return SpecialistOutput(
                answer_text="The dominant land cover in this image is farmland.",
                evidence=[],
                confidence=ConfidenceResult(
                    value=0.83, method_version="v1_vlm_mean_token_probability", basis={},
                    basis_description="mean generated-token probability (test double)",
                ),
                raw={"model_id": "HuggingFaceTB/SmolVLM-256M-Instruct"},
                tool_name=vqa_dispatch.SMOLVLM_TOOL_NAME, task_type="single_image_vqa", latency_seconds=29.9,
                model_name="HuggingFaceTB/SmolVLM-256M-Instruct", fallback_occurred=False, fallback_reason=None,
            )

        tmp_upload = os.path.join(REPO_ROOT, "data", "fixtures", "single_image.png")
        session_state = {
            "vqa_upload": _FakeUploadedFile(tmp_upload),
            "vqa_query": "Describe the major land cover types visible in this image.",
            "vqa_modality": "Not declared",
            "vqa_fixture": False,
        }
        with patch("specialists.vqa_dispatch.registry.get", return_value=_fake_available_entry()), \
             patch("specialists.vqa_dispatch.vqa_smolvlm.run", side_effect=_fake_smolvlm_output):
            _run_app(session_state, button_true_key="vqa_run")
        out = session_state["vqa_trace"].specialist_output
        self.assertEqual(out.model_name, "HuggingFaceTB/SmolVLM-256M-Instruct")
        self.assertFalse(out.fallback_occurred)
        self.assertEqual(out.confidence.method_version, "v1_vlm_mean_token_probability")

    # --- FINAL PRODUCT UI/UX PASS regression coverage -----------------------

    def test_example_chip_click_fills_query_field_without_running_analysis(self):
        """Clicking an example-query chip must only set session_state for the
        query field (via the real on_click callback wiring) - it must not
        itself be mistaken for the RUN ANALYSIS button, i.e. no trace is
        produced from a chip click alone."""
        session_state = {}
        fake = _run_app(session_state, button_true_key="vqa_query_chip_0")
        self.assertEqual(session_state.get("vqa_query"), "Describe the major land cover types visible in this image.")
        self.assertNotIn("vqa_trace", session_state)

    def test_registry_backend_labels_distinguish_real_model_classical_and_unavailable(self):
        """The sidebar's capability registry must report one of exactly
        three real, registry-derived states - never invent a fourth, and
        never label a classical-only capability (grounding/change/fusion)
        as a real model."""
        _, ns = _run_app_ns({})
        backend_label = ns["_capability_backend_label"]
        real_label, real_ready = backend_label([ns["SMOLVLM_TOOL_NAME"], ns["CLASSICAL_VQA_TOOL_NAME"]])
        self.assertIn(real_label, ("REAL MODEL", "CLASSICAL", "UNAVAILABLE"))
        ground_label, ground_ready = backend_label(["tool_grounding_v0"])
        self.assertIn(ground_label, ("CLASSICAL", "UNAVAILABLE"))  # never REAL MODEL - classical-only tool
        unavailable_label, unavailable_ready = backend_label(["tool_that_does_not_exist"])
        self.assertEqual((unavailable_label, unavailable_ready), ("UNAVAILABLE", False))

    def test_followup_question_reruns_pipeline_and_records_conversation_history(self):
        """A follow-up question must call the real pipeline again with the
        SAME cached image path (no re-upload) and the NEW query text, and
        the previous (query, trace) pair must be preserved in real
        conversation history - not discarded and not fabricated."""
        tmp_upload = os.path.join(REPO_ROOT, "data", "fixtures", "single_image.png")
        first_query = "Describe the major land cover types visible in this image."
        session_state = {
            "vqa_upload": _FakeUploadedFile(tmp_upload),
            "vqa_query": first_query,
            "vqa_modality": "Not declared",
            "vqa_fixture": False,
        }
        _run_app(session_state, button_true_key="vqa_run")
        self.assertIn("vqa_trace", session_state)
        first_trace = session_state["vqa_trace"]
        cached_inputs = session_state["vqa_inputs"]
        self.assertTrue(os.path.exists(cached_inputs["image1_path"]))

        followup_text = "Is there any visible water in this image?"
        session_state["vqa_followup"] = followup_text
        _run_app(session_state, button_true_key="vqa_followup_run")

        second_trace = session_state["vqa_trace"]
        self.assertIsNot(second_trace, first_trace)
        self.assertEqual(second_trace.query, followup_text)
        history = session_state.get("vqa_history", [])
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0][0], first_query)
        self.assertIs(history[0][1], first_trace)

    def test_change_tab_shows_before_and_after_panels_in_addition_to_change_map(self):
        """Per the BEFORE|AFTER-over-CHANGE-MAP evidence layout requirement,
        the change tab must render the two raw input images separately, in
        addition to the specialist's own before|after|highlighted composite
        - three real images total for one successful run, not just one."""
        session_state = {
            "ch_fixture": True,
            "ch_query": "What changed between these two dates?",
            "ch_date1": datetime.date(2020, 1, 1),
            "ch_date2": datetime.date(2024, 1, 1),
        }
        fake = _run_app(session_state, button_true_key="ch_run")
        self.assertIn("ch_trace", session_state)
        self.assertIsNone(session_state["ch_trace"].failure)
        self.assertGreaterEqual(len(fake._calls["images"]), 3)
        self.assertIn("ch_extra", session_state)
        captions = [c for c, _arr in session_state["ch_extra"]]
        self.assertEqual(captions, ["Before", "After"])

    def test_failure_panel_shows_numeric_error_code_and_category(self):
        """A validation failure must render with a real ERROR <code> ·
        <CATEGORY> badge derived from routing/failure_classification.py -
        not just the old free-text WHAT/WHY/NEXT panel with no code."""
        from PIL import Image
        import numpy as np

        tiny_path = "/tmp/_test_streamlit_tiny_2.png"
        Image.fromarray((np.random.rand(4, 4, 3) * 255).astype("uint8")).save(tiny_path)
        session_state = {
            "gr_upload": _FakeUploadedFile(tiny_path),
            "gr_query": "Locate the water body in this image and highlight it.",
            "gr_modality": "Not declared",
            "gr_fixture": False,
        }
        fake = _run_app(session_state, button_true_key="gr_run")
        os.remove(tiny_path)
        self.assertTrue(any("ERROR 400" in m and "INVALID_IMAGE" in m for m in fake._calls.get("markdown", [])))

    def test_agent_process_timeline_expander_renders_for_a_successful_run(self):
        session_state = {"fu_fixture": True, "fu_query": "Use both images to identify built-up and water-covered regions."}
        fake = _run_app(session_state, button_true_key="fu_run")
        self.assertIn("Agent process timeline", fake._calls.get("expanders", []))
        self.assertIn("Technical details", fake._calls.get("expanders", []))

    def test_run_and_ask_buttons_are_primary_while_example_chips_are_secondary(self):
        """Section 7 (REQUIRED, Revision 4): execution actions (Run analysis,
        Ask a follow-up) must render with the amber "primary" Streamlit
        button kind; example-query chips must NOT, so they read as a
        distinct, non-executing accent color rather than looking like a
        primary action."""
        session_state = {"fu_fixture": True, "fu_query": "Use both images to identify built-up and water-covered regions."}
        fake = _run_app(session_state, button_true_key="fu_run")
        button_types = fake._calls["button_types"]
        self.assertEqual(button_types.get("vqa_run"), "primary")
        self.assertEqual(button_types.get("gr_run"), "primary")
        self.assertEqual(button_types.get("ch_run"), "primary")
        self.assertEqual(button_types.get("fu_run"), "primary")
        chip_keys = [k for k in button_types if k and k.startswith("fu_query_chip_")]
        self.assertTrue(chip_keys, "expected at least one example-query chip for the fusion tab")
        for k in chip_keys:
            self.assertNotEqual(button_types[k], "primary")

    def test_followup_ask_button_is_primary(self):
        """The follow-up "Ask" button is an execution action (it re-runs the
        real pipeline), so it must carry the same primary/amber styling as
        "Run analysis" - not the secondary styling used for suggestions."""
        session_state = {"fu_fixture": True, "fu_query": "Use both images to identify built-up and water-covered regions."}
        fake = _run_app(session_state, button_true_key="fu_run")
        self.assertEqual(fake._calls["button_types"].get("fu_followup_run"), "primary")

    def test_return_to_analysis_clears_failed_tab_state_and_reruns(self):
        """Section 19's "[ Return to analysis ]" control must be a real,
        functional reset - not decorative copy - so a failed run's leftover
        state does not linger once the analyst asks to start over, and the
        app actually reruns so the tab's empty state reappears immediately."""
        from PIL import Image
        import numpy as np

        tiny_path = "/tmp/_test_streamlit_return_to_analysis.png"
        Image.fromarray((np.random.rand(4, 4, 3) * 255).astype("uint8")).save(tiny_path)
        session_state = {
            "gr_upload": _FakeUploadedFile(tiny_path),
            "gr_query": "Locate the water body in this image and highlight it.",
            "gr_modality": "Not declared",
            "gr_fixture": False,
        }
        _run_app(session_state, button_true_key="gr_run")
        os.remove(tiny_path)
        self.assertIn("gr_trace", session_state)
        self.assertIsNotNone(session_state["gr_trace"].failure)

        fake = _run_app(session_state, button_true_key="gr_return")
        for suffix in ("_trace", "_rundir", "_inputs", "_query_used", "_extra", "_history"):
            self.assertNotIn(f"gr{suffix}", session_state, f"gr{suffix} should have been cleared")
        self.assertEqual(fake._calls.get("rerun_count", 0), 1)

    def test_header_status_badge_reflects_real_registry_state(self):
        """The header's SYSTEM READY / PARTIAL SYSTEM badge must be computed
        live from the same registry-backed check the sidebar's "System" rail
        uses (_capability_ready over the real registry) - never a fixed
        string - so it never claims readiness that isn't real."""
        fake, ns = _run_app_ns({})
        ready, total = ns["_system_status"]()
        markdown_blob = "\n".join(fake._calls.get("markdown", []))
        if ready >= total:
            self.assertIn("SYSTEM READY", markdown_blob)
        else:
            self.assertIn("PARTIAL SYSTEM", markdown_blob)
            self.assertIn(f"{ready}/{total} READY", markdown_blob)


if __name__ == "__main__":
    unittest.main()
