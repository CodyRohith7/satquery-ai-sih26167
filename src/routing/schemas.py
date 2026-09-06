"""Output schemas shared across the whole pipeline.

These are the objects that get serialized straight into the UI's trace panel and
the exported report. There is exactly ONE code path that builds an
`ExecutionTrace` (see `router.py` / `app/cli.py`), and the UI renders that object
directly - it never has a way to print a narrated string instead of the real
decision, which is the specific failure mode identified in the competitor
analysis (Phase 0) that this schema is designed to make impossible.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional


TASK_TYPES = (
    "single_image_vqa",
    "grounding",
    "bitemporal_change",
    "optical_sar_fusion",
    "needs_clarification",
)


@dataclasses.dataclass
class RouterDecision:
    """The router's actual decision. Rendered verbatim in the UI - never re-typed."""

    task_type: str
    tool_name: str
    signals: Dict[str, Any]  # the actual inputs the decision was based on
    reasoning: List[str]  # short factual statements, e.g. "2 inputs supplied"
    confidence: float  # router's OWN confidence in the routing choice, 0-1

    def __post_init__(self) -> None:
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"Unknown task_type: {self.task_type}")


@dataclasses.dataclass
class EvidenceItem:
    kind: str  # "bounding_box" | "mask" | "change_map" | "overlay_image" | "none"
    description: str
    image_path: Optional[str] = None  # path to a saved evidence PNG, if any
    geometry: Optional[Dict[str, Any]] = None  # e.g. box coords, in pixel space


@dataclasses.dataclass
class ConfidenceResult:
    value: float  # 0.0 - 1.0
    method_version: str  # which base_signal methodology produced this - see docs/confidence.md
    basis: Dict[str, float]  # named sub-scores that were combined - see docs/confidence.md
    basis_description: str = ""  # plain-English explanation of what base_signal means for
    # THIS result (e.g. "classical clustering separation score" vs. "mean generated-token
    # probability - measures generation certainty, not factual correctness") - populated so
    # a UI can display it verbatim, never having to infer meaning from method_version alone.
    downgrades: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class SpecialistOutput:
    answer_text: str
    evidence: List[EvidenceItem]
    confidence: ConfidenceResult
    raw: Dict[str, Any]  # tool-specific raw numbers, for the "technical details" panel
    tool_name: str
    task_type: str
    latency_seconds: float
    model_name: Optional[str] = None  # human-readable model identity (e.g.
    # "SmolVLM-256M-Instruct") when this output came from a real pretrained model;
    # None for classical-CV specialists, which are techniques, not models - a UI
    # must never have to infer "was this a real model" from tool_name string matching.
    fallback_occurred: bool = False  # explicit, always present (not just inferable from
    # raw's presence/absence of specific keys) - True only when a preferred real-model
    # specialist was attempted and failed, and a different specialist's output is
    # being returned instead.
    fallback_reason: Optional[str] = None  # the real exception/message that caused the
    # fallback, when fallback_occurred is True; None otherwise.


@dataclasses.dataclass
class ExecutionTrace:
    """The complete, real record of one query. This whole object is what the UI
    renders as the 'execution trace' - not a hand-written summary of it."""

    query: str
    input_summary: List[Dict[str, Any]]
    validation: Dict[str, Any]
    router_decision: RouterDecision
    specialist_output: Optional[SpecialistOutput]
    started_at: float
    finished_at: Optional[float] = None
    failure: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        return d

    def total_latency_seconds(self) -> Optional[float]:
        if self.finished_at is None:
            return None
        return round(self.finished_at - self.started_at, 4)


def now() -> float:
    return time.time()
