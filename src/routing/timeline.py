"""Derives a human-readable, checklist-style summary of what actually
happened while producing an `ExecutionTrace`, strictly from fields already
present on that trace (see `routing/schemas.py`).

This is a READ-ONLY summarization layer, not a new stage of the pipeline: it
computes nothing new, calls no model, and never marks a step "done" that
did not really happen. It exists so the Streamlit UI's "Agent process
timeline" panel and the PDF report's "Agent execution timeline" section
render the identical, real sequence of events instead of two hand-written
descriptions that could quietly drift apart. This is the one small,
additive backend module the "FINAL PRODUCT UI/UX PASS" instructions
explicitly allow ("a small backend change... genuinely required to support
the requested UI behavior") - it does not touch routing, validation,
specialists, confidence, or evidence computation.
"""
from __future__ import annotations

import dataclasses
from typing import List

from routing.schemas import ExecutionTrace

STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_NOT_REACHED = "not_reached"

STEP_LABELS = (
    "Input validated",
    "Query interpreted",
    "Specialist selected",
    "Analysis executed",
    "Evidence generated",
    "Confidence computed",
    "Result ready",
)


@dataclasses.dataclass
class TimelineStep:
    label: str
    status: str  # done | failed | not_reached
    detail: str = ""


def _remaining(labels, from_index: int) -> List[TimelineStep]:
    return [TimelineStep(label, STATUS_NOT_REACHED) for label in labels[from_index:]]


def derive(trace: ExecutionTrace) -> List[TimelineStep]:
    """Walk the same branch order `app/pipeline.py:run_query()` actually
    executes in, and report, for each named stage, whether it really
    completed, really failed, or was never reached - never a guess."""
    labels = STEP_LABELS
    steps: List[TimelineStep] = []

    load_failed = bool(trace.failure) and trace.failure.startswith("Input loading failed")
    if load_failed:
        steps.append(TimelineStep(labels[0], STATUS_FAILED, trace.failure or ""))
        steps.extend(_remaining(labels, 1))
        return steps

    validation_ok = bool(trace.validation.get("ok")) if trace.validation else False
    steps.append(
        TimelineStep(
            labels[0],
            STATUS_DONE if validation_ok else STATUS_FAILED,
            "" if validation_ok else "; ".join(trace.validation.get("errors", [])),
        )
    )
    if not validation_ok:
        steps.extend(_remaining(labels, 1))
        return steps

    decision = trace.router_decision
    if decision is None:
        steps.extend(_remaining(labels, 1))
        return steps

    steps.append(
        TimelineStep(
            labels[1],
            STATUS_DONE,
            f"Routed as {decision.task_type} (router confidence {decision.confidence:.2f})",
        )
    )

    if decision.task_type == "needs_clarification":
        steps.append(TimelineStep(labels[2], STATUS_FAILED, "Router refused to guess - see reasoning below."))
        steps.extend(_remaining(labels, 3))
        return steps

    steps.append(TimelineStep(labels[2], STATUS_DONE, decision.tool_name))

    output = trace.specialist_output
    exec_failed = bool(trace.failure) and trace.failure.startswith("Specialist execution failed")
    if output is None:
        steps.append(TimelineStep(labels[3], STATUS_FAILED if exec_failed else STATUS_NOT_REACHED, trace.failure or ""))
        steps.extend(_remaining(labels, 4))
        return steps

    steps.append(TimelineStep(labels[3], STATUS_DONE, f"{output.latency_seconds:.2f}s"))
    steps.append(
        TimelineStep(
            labels[4],
            STATUS_DONE if output.evidence else STATUS_NOT_REACHED,
            f"{len(output.evidence)} item(s)",
        )
    )
    steps.append(
        TimelineStep(
            labels[5],
            STATUS_DONE if output.confidence is not None else STATUS_NOT_REACHED,
            f"{output.confidence.value:.2f}" if output.confidence is not None else "",
        )
    )
    steps.append(TimelineStep(labels[6], STATUS_DONE if trace.failure is None else STATUS_FAILED))
    return steps
