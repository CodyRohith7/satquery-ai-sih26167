"""Shared orchestration logic: INPUT -> VALIDATE -> ROUTE -> SPECIALIST -> TRACE.

This module exists so the CLI (`app/cli.py`) and the Streamlit UI
(`app/streamlit_app.py`) call exactly the SAME code path to go from raw
inputs to an `ExecutionTrace` - there is no second, UI-specific
reimplementation of routing/validation/dispatch logic anywhere. `run_query()`
below is a straight extraction of what used to be `cli.run()`'s body, with
`argparse.Namespace` attribute access replaced by plain keyword parameters so
non-CLI callers (like Streamlit) don't need to construct a fake Namespace.

Per the frozen "do not rewrite the backend architecture" instruction, nothing
about ingestion/routing/validation/specialists/evidence/confidence/export
changes here - this module only relocates orchestration glue that already
existed in `app/cli.py`.
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingestion import raster_io, metadata as metadata_mod, validator  # noqa: E402
from routing import router as router_mod  # noqa: E402
from routing.schemas import ExecutionTrace, now  # noqa: E402
from models import registry  # noqa: E402
from specialists import vqa_dispatch as vqa_mod  # dispatches to SmolVLM (real model) or the classical fallback - see specialists/vqa_dispatch.py  # noqa: E402
from grounding import grounding as grounding_mod  # noqa: E402
from change import change as change_mod  # noqa: E402
from fusion import fusion as fusion_mod  # noqa: E402


def run_query(
    image1_path: str,
    query: str,
    image2_path: str = None,
    modality1: str = None,
    modality2: str = None,
    date1: str = None,
    date2: str = None,
    out_dir: str = "runs/latest",
) -> ExecutionTrace:
    """Run one query through the real pipeline and return the real
    `ExecutionTrace`. Every branch below is identical in behavior to the
    original `app/cli.py:run()` - see that file's history for provenance.
    """
    started = now()
    os.makedirs(out_dir, exist_ok=True)

    input_summary = []
    images_meta = []
    arrays = []

    try:
        raster1 = raster_io.load(image1_path)
        meta1 = metadata_mod.inspect(raster1, modality1, date1)
        images_meta.append(meta1)
        arrays.append(raster1.array)
        input_summary.append(meta1.metadata_summary())

        if image2_path:
            raster2 = raster_io.load(image2_path)
            meta2 = metadata_mod.inspect(raster2, modality2, date2)
            images_meta.append(meta2)
            arrays.append(raster2.array)
            input_summary.append(meta2.metadata_summary())
    except raster_io.RasterLoadError as exc:
        return ExecutionTrace(
            query=query,
            input_summary=input_summary,
            validation={"ok": False, "errors": [str(exc)], "warnings": []},
            router_decision=None,
            specialist_output=None,
            started_at=started,
            finished_at=now(),
            failure=f"Input loading failed: {exc}",
        )

    # --- routing (real branch selection - see router.py for the decision logic) ---
    decision = router_mod.decide(query, images_meta)

    # --- validation appropriate to the chosen branch ---
    if len(images_meta) == 1:
        vres = validator.validate_single(images_meta[0])
    else:
        rel = "bitemporal" if decision.task_type == "bitemporal_change" else (
            "cross_modal" if decision.task_type == "optical_sar_fusion" else None
        )
        vres = validator.validate_pair(images_meta[0], images_meta[1], intended_relationship=rel)
    validation_dict = {"ok": vres.ok, "errors": vres.errors, "warnings": vres.warnings}

    if not vres.ok:
        return ExecutionTrace(
            query=query,
            input_summary=input_summary,
            validation=validation_dict,
            router_decision=decision,
            specialist_output=None,
            started_at=started,
            finished_at=now(),
            failure=f"Validation failed: {'; '.join(vres.errors)}",
        )

    if decision.task_type == "needs_clarification":
        return ExecutionTrace(
            query=query,
            input_summary=input_summary,
            validation=validation_dict,
            router_decision=decision,
            specialist_output=None,
            started_at=started,
            finished_at=now(),
            failure=(
                "Router could not confidently determine the analysis branch and "
                "refused to guess. See router_decision.reasoning for exactly why. "
                "Declare modality and/or acquisition date for both images to "
                "disambiguate."
            ),
        )

    # --- dispatch to the real specialist the router selected ---
    evidence_path = os.path.join(out_dir, "evidence.png")
    try:
        registry.get(decision.tool_name)  # raises if not AVAILABLE - never silently faked
        if decision.task_type == "single_image_vqa":
            output = vqa_mod.run(arrays[0], images_meta[0], query, vres.warnings, decision.confidence, evidence_path)
        elif decision.task_type == "grounding":
            output = grounding_mod.run(arrays[0], images_meta[0], query, vres.warnings, decision.confidence, evidence_path)
        elif decision.task_type == "bitemporal_change":
            output = change_mod.run(arrays[0], arrays[1], images_meta[0], images_meta[1], query, vres.warnings, decision.confidence, evidence_path)
        elif decision.task_type == "optical_sar_fusion":
            optical_idx = 0 if images_meta[0].modality == "optical" else 1
            sar_idx = 1 - optical_idx
            output = fusion_mod.run(
                arrays[optical_idx], arrays[sar_idx], images_meta[optical_idx], images_meta[sar_idx],
                query, vres.warnings, decision.confidence, evidence_path,
            )
        else:
            raise RuntimeError(f"Unhandled task_type: {decision.task_type}")
    except Exception as exc:  # noqa: BLE001 - deliberately broad: report, don't crash silently
        return ExecutionTrace(
            query=query,
            input_summary=input_summary,
            validation=validation_dict,
            router_decision=decision,
            specialist_output=None,
            started_at=started,
            finished_at=now(),
            failure=f"Specialist execution failed: {exc}\n{traceback.format_exc()}",
        )

    return ExecutionTrace(
        query=query,
        input_summary=input_summary,
        validation=validation_dict,
        router_decision=decision,
        specialist_output=output,
        started_at=started,
        finished_at=now(),
        failure=None,
    )
