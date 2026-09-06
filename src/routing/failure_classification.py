"""Categorizes the small, fixed set of real failure-string prefixes that
`app/pipeline.run_query()` can ever produce (see that module) into an
HTTP-style numeric code and a named category, purely for display.

This does not invent new failure modes or change what failure occurs - the
full original `ExecutionTrace.failure` string is always preserved and shown
verbatim alongside the classification; this module only names the small,
already-fixed set of prefixes that pipeline.py is documented to produce.
Shared by the Streamlit UI's failure panel and the PDF report's
"Limitations & warnings" section so the two never disagree.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class FailureClassification:
    code: int
    category: str
    what: str
    why: str
    next_step: str


def classify(raw: str) -> FailureClassification:
    headline = raw.split("\nTraceback (most recent call last):")[0]

    if raw.startswith("Input loading failed"):
        return FailureClassification(
            code=400,
            category="INVALID_IMAGE",
            what="Unsupported or unreadable input",
            why=headline,
            next_step="Confirm the file is a valid PNG/JPEG/TIFF image and try again.",
        )

    if raw.startswith("Validation failed"):
        if "same acquisition date" in raw or "same date" in raw:
            category = "AMBIGUOUS_QUERY"
        elif "requires two different modalities" in raw:
            category = "MISSING_MODALITY"
        elif "too small to analyze" in raw or "zero bands" in raw:
            category = "INVALID_IMAGE"
        else:
            category = "VALIDATION_ERROR"
        return FailureClassification(
            code=400,
            category=category,
            what="Input validation failed",
            why=headline,
            next_step="Review the message above and adjust the modality, date, or image.",
        )

    if raw.startswith("Router could not confidently determine"):
        return FailureClassification(
            code=400,
            category="AMBIGUOUS_QUERY",
            what="Ambiguous query or insufficient input",
            why=headline,
            next_step="Declare a modality and/or date for both images, or rephrase the query.",
        )

    if raw.startswith("Specialist execution failed"):
        return FailureClassification(
            code=500,
            category="INTERNAL_EXECUTION_ERROR",
            what="Analysis failed to complete",
            why=headline.split("\n")[0],
            next_step="See technical details below. This is not hidden or silently ignored.",
        )

    return FailureClassification(
        code=500,
        category="UNEXPECTED_ERROR",
        what="Unexpected failure",
        why=headline,
        next_step="See technical details below.",
    )
