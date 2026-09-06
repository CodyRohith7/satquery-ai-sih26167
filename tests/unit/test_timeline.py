"""Real, direct tests of `routing/timeline.py:derive()` against real
`ExecutionTrace` objects - proving the checklist never marks a step done
that did not happen, for every real failure branch `pipeline.run_query()`
can produce plus the full-success path."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from routing import timeline  # noqa: E402
from routing.schemas import (  # noqa: E402
    ConfidenceResult,
    EvidenceItem,
    ExecutionTrace,
    RouterDecision,
    SpecialistOutput,
)


def _decision(task_type="single_image_vqa", tool_name="tool_single_image_vqa_v0"):
    return RouterDecision(
        task_type=task_type,
        tool_name=tool_name,
        signals={"num_inputs": 1},
        reasoning=["1 input supplied."],
        confidence=0.9,
    )


def _output():
    return SpecialistOutput(
        answer_text="A field of grass.",
        evidence=[EvidenceItem(kind="overlay_image", description="source", image_path=None)],
        confidence=ConfidenceResult(value=0.7, method_version="v0_classical", basis={"x": 0.7}),
        raw={},
        tool_name="tool_single_image_vqa_v0",
        task_type="single_image_vqa",
        latency_seconds=1.23,
    )


class TestTimelineDerivation(unittest.TestCase):
    def test_full_success_marks_every_step_done(self):
        trace = ExecutionTrace(
            query="q",
            input_summary=[{}],
            validation={"ok": True, "errors": [], "warnings": []},
            router_decision=_decision(),
            specialist_output=_output(),
            started_at=0.0,
            finished_at=1.0,
            failure=None,
        )
        steps = timeline.derive(trace)
        self.assertEqual(len(steps), 7)
        self.assertTrue(all(s.status == timeline.STATUS_DONE for s in steps))

    def test_input_load_failure_only_first_step_fails_rest_not_reached(self):
        trace = ExecutionTrace(
            query="q",
            input_summary=[],
            validation={"ok": False, "errors": ["bad file"], "warnings": []},
            router_decision=None,
            specialist_output=None,
            started_at=0.0,
            finished_at=1.0,
            failure="Input loading failed: bad file",
        )
        steps = timeline.derive(trace)
        self.assertEqual(steps[0].status, timeline.STATUS_FAILED)
        self.assertTrue(all(s.status == timeline.STATUS_NOT_REACHED for s in steps[1:]))

    def test_validation_failure_after_load_succeeds(self):
        trace = ExecutionTrace(
            query="q",
            input_summary=[{}, {}],
            validation={"ok": False, "errors": ["same date"], "warnings": []},
            router_decision=_decision(task_type="bitemporal_change", tool_name="tool_change_v0"),
            specialist_output=None,
            started_at=0.0,
            finished_at=1.0,
            failure="Validation failed: same date",
        )
        steps = timeline.derive(trace)
        self.assertEqual(steps[0].status, timeline.STATUS_FAILED)
        self.assertTrue(all(s.status == timeline.STATUS_NOT_REACHED for s in steps[1:]))

    def test_needs_clarification_stops_at_specialist_selected(self):
        decision = RouterDecision(
            task_type="needs_clarification",
            tool_name="none",
            signals={},
            reasoning=["ambiguous"],
            confidence=0.0,
        )
        trace = ExecutionTrace(
            query="q",
            input_summary=[{}, {}],
            validation={"ok": True, "errors": [], "warnings": []},
            router_decision=decision,
            specialist_output=None,
            started_at=0.0,
            finished_at=1.0,
            failure="Router could not confidently determine...",
        )
        steps = timeline.derive(trace)
        self.assertEqual(steps[0].status, timeline.STATUS_DONE)
        self.assertEqual(steps[1].status, timeline.STATUS_DONE)
        self.assertEqual(steps[2].status, timeline.STATUS_FAILED)
        self.assertTrue(all(s.status == timeline.STATUS_NOT_REACHED for s in steps[3:]))

    def test_specialist_execution_failure(self):
        trace = ExecutionTrace(
            query="q",
            input_summary=[{}],
            validation={"ok": True, "errors": [], "warnings": []},
            router_decision=_decision(),
            specialist_output=None,
            started_at=0.0,
            finished_at=1.0,
            failure="Specialist execution failed: boom\nTraceback (most recent call last):\n...",
        )
        steps = timeline.derive(trace)
        self.assertEqual(steps[0].status, timeline.STATUS_DONE)
        self.assertEqual(steps[1].status, timeline.STATUS_DONE)
        self.assertEqual(steps[2].status, timeline.STATUS_DONE)
        self.assertEqual(steps[3].status, timeline.STATUS_FAILED)
        self.assertTrue(all(s.status == timeline.STATUS_NOT_REACHED for s in steps[4:]))


if __name__ == "__main__":
    unittest.main()
