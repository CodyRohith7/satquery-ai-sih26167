"""Export the execution trace as JSON (always) and a PDF audit report (when
reportlab is available - it is, in this environment; see requirements.txt).

The PDF report ("FINAL PRODUCT UI/UX PASS" requirement) is a scientific/
aerospace-style document with ten named sections - Analysis Summary, Input
Data, Analysis Result, Visual Evidence, Active Specialist, Confidence, Agent
Execution Timeline, Technical Trace, Limitations & Warnings, and Export
Metadata - plus page numbers and a running header/footer. Every value in it
is read directly off the real `ExecutionTrace` object (or the two small,
shared derivation helpers `routing/timeline.py` and
`routing/failure_classification.py`, which are themselves pure read-only
summaries of that same trace) - there are no hardcoded report values.
"""
from __future__ import annotations

import datetime
import json
import os
from typing import Any, Dict

from xml.sax.saxutils import escape as _xml_escape

from routing import failure_classification as failure_mod
from routing import timeline as timeline_mod
from routing.schemas import ExecutionTrace

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Image as RLImage,
        Table,
        TableStyle,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.enums import TA_RIGHT

    _REPORTLAB_AVAILABLE = True
except ImportError:
    _REPORTLAB_AVAILABLE = False

BUILD_LABEL = "v0 (MVP baseline)"
REPORT_TITLE = "SatQuery AI — Analysis Report"

_TASK_TYPE_LABELS = {
    "single_image_vqa": "Single-image VQA",
    "grounding": "Feature grounding",
    "bitemporal_change": "Bi-temporal change",
    "optical_sar_fusion": "Optical + SAR fusion",
    "needs_clarification": "Needs clarification (ambiguous input)",
}

_TIMELINE_STATUS_LABEL = {
    timeline_mod.STATUS_DONE: "Done",
    timeline_mod.STATUS_FAILED: "Failed",
    timeline_mod.STATUS_NOT_REACHED: "Not reached",
}


def export_json(trace: ExecutionTrace, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(trace.to_dict(), f, indent=2, default=str)
    return out_path


def _safe_para_text(text: Any) -> str:
    """Escapes free-form/uncontrolled text (a real model's generated answer,
    a raw exception message, a full traceback) before it reaches reportlab's
    Paragraph mini-XML parser. Paragraph() interprets a small set of tag
    names (<b>, <i>, <br/>, <font>, ...) as real markup - unescaped dynamic
    text that happens to contain one of those tag names unmatched (e.g. a
    VLM-generated "<b>" with no closing tag, or a Python exception/repr
    string containing a stray angle bracket) raises an uncaught ValueError
    and aborts PDF generation entirely. Every other value placed in this
    report (query, tool names, modality labels, fixed template strings) is
    either rendered inside a Table cell (reportlab does not markup-parse
    plain Table cell strings - only Paragraph()) or drawn from this
    project's own small fixed vocabularies, so this escaping is applied only
    at call sites that carry real free text - see docs/confidence.md and
    routing/schemas.py for which fields are ever true free text."""
    return _xml_escape(str(text))


def _table(rows, col_widths=(6 * cm, 11 * cm)):
    t = Table(rows, colWidths=list(col_widths))
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return t


def _make_header_footer(generated_at: str):
    def _draw(canvas, doc):
        canvas.saveState()
        width, height = A4
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(2 * cm, height - 1.2 * cm, "SatQuery AI — Analysis Report")
        canvas.drawRightString(width - 2 * cm, height - 1.2 * cm, "SIH26167 · ISRO · Space Technology Programme")
        canvas.line(2 * cm, height - 1.3 * cm, width - 2 * cm, height - 1.3 * cm)
        canvas.drawString(2 * cm, 1.2 * cm, f"Generated {generated_at}")
        canvas.drawRightString(width - 2 * cm, 1.2 * cm, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    return _draw


def export_pdf(trace: ExecutionTrace, out_path: str) -> str:
    if not _REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab is not available; cannot export PDF.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    styles = getSampleStyleSheet()
    small_right = ParagraphStyle("SmallRight", parent=styles["Normal"], alignment=TA_RIGHT, fontSize=8, textColor=colors.grey)
    story = []

    generated_dt = datetime.datetime.now()
    generated_at = generated_dt.strftime("%Y-%m-%d %H:%M:%S")
    started_dt = datetime.datetime.fromtimestamp(trace.started_at)
    total_latency = trace.total_latency_seconds()
    overall_status = "FAILED" if trace.failure else "SUCCESS"
    decision = trace.router_decision
    out = trace.specialist_output

    # --- Title -----------------------------------------------------------
    story.append(Paragraph(REPORT_TITLE, styles["Title"]))
    story.append(Paragraph("SIH26167 · ISRO Space Technology Programme", small_right))
    story.append(Spacer(1, 0.4 * cm))

    # --- 1. Analysis Summary ----------------------------------------------
    story.append(Paragraph("1. Analysis Summary", styles["Heading2"]))
    story.append(
        _table(
            [
                ["Query", trace.query],
                ["Analysis type", _TASK_TYPE_LABELS.get(decision.task_type, decision.task_type) if decision else "N/A"],
                ["Started at", started_dt.strftime("%Y-%m-%d %H:%M:%S")],
                ["Status", overall_status],
                ["Total latency", f"{total_latency:.3f}s" if total_latency is not None else "N/A"],
            ]
        )
    )
    story.append(Spacer(1, 0.3 * cm))

    # --- 2. Input Data ------------------------------------------------------
    story.append(Paragraph("2. Input Data", styles["Heading2"]))
    if trace.input_summary:
        rows = [["Image", "Modality", "Modality basis", "Acquisition date"]]
        for i, s in enumerate(trace.input_summary, start=1):
            rows.append(
                [
                    f"Input {i}",
                    str(s.get("modality", "unknown")),
                    str(s.get("modality_basis", "unknown")),
                    str(s.get("acquisition_date", "Metadata unavailable")),
                ]
            )
        story.append(_table(rows, col_widths=(3 * cm, 3.5 * cm, 3.5 * cm, 7 * cm)))
    else:
        story.append(Paragraph("No input could be loaded for this run.", styles["BodyText"]))
    story.append(Spacer(1, 0.3 * cm))

    # --- 3. Analysis Result ---------------------------------------------
    story.append(Paragraph("3. Analysis Result", styles["Heading2"]))
    if out is not None:
        story.append(Paragraph(_safe_para_text(out.answer_text), styles["BodyText"]))
    elif trace.failure:
        story.append(Paragraph("Analysis did not complete - see Limitations & warnings below.", styles["BodyText"]))
    else:
        story.append(Paragraph("N/A", styles["BodyText"]))
    story.append(Spacer(1, 0.3 * cm))

    # --- 4. Visual Evidence -----------------------------------------------
    story.append(Paragraph("4. Visual Evidence", styles["Heading2"]))
    if out is not None and out.evidence:
        for ev in out.evidence:
            story.append(Paragraph(f"<b>{_safe_para_text(ev.kind)}:</b> {_safe_para_text(ev.description)}", styles["BodyText"]))
            if ev.image_path and os.path.exists(ev.image_path):
                try:
                    reader = ImageReader(ev.image_path)
                    iw, ih = reader.getSize()
                    target_w = 14 * cm
                    target_h = target_w * ih / iw
                    story.append(RLImage(ev.image_path, width=target_w, height=target_h))
                except Exception:
                    story.append(Paragraph("(evidence image could not be embedded)", styles["BodyText"]))
            story.append(Spacer(1, 0.2 * cm))
    else:
        story.append(Paragraph("No visual evidence was produced for this run.", styles["BodyText"]))
    story.append(Spacer(1, 0.1 * cm))

    # --- 5. Active Specialist -----------------------------------------
    story.append(Paragraph("5. Active Specialist", styles["Heading2"]))
    if out is not None:
        rows = [
            ["Tool ID", out.tool_name],
            ["Model", out.model_name if out.model_name else "N/A - classical CV technique, no pretrained model"],
            ["Latency", f"{out.latency_seconds:.3f}s"],
        ]
        if out.fallback_occurred:
            rows.append(["Fallback occurred", f"Yes - {out.fallback_reason}"])
        story.append(_table(rows))
    elif decision is not None:
        story.append(Paragraph(f"Router selected tool: {decision.tool_name} (not executed - see Limitations).", styles["BodyText"]))
    else:
        story.append(Paragraph("No specialist was selected (input failed to load).", styles["BodyText"]))
    story.append(Spacer(1, 0.3 * cm))

    # --- 6. Confidence ---------------------------------------------------
    story.append(Paragraph("6. Confidence", styles["Heading2"]))
    if out is not None:
        conf = out.confidence
        rows = [["Value", f"{conf.value:.2f}"], ["Method", conf.method_version]]
        if conf.basis_description:
            rows.append(["What this measures", conf.basis_description])
        for k, v in conf.basis.items():
            rows.append([k, f"{v:.3f}" if isinstance(v, float) else str(v)])
        if conf.downgrades:
            rows.append(["Downgrades applied", "; ".join(conf.downgrades)])
        story.append(_table(rows))
    else:
        story.append(Paragraph("No confidence score was computed for this run.", styles["BodyText"]))
    story.append(Spacer(1, 0.3 * cm))

    # --- 7. Agent Execution Timeline --------------------------------------
    story.append(Paragraph("7. Agent Execution Timeline", styles["Heading2"]))
    rows = [["Step", "Status", "Detail"]]
    for step in timeline_mod.derive(trace):
        rows.append([step.label, _TIMELINE_STATUS_LABEL[step.status], step.detail or ""])
    story.append(_table(rows, col_widths=(4.5 * cm, 2.5 * cm, 10 * cm)))
    story.append(Spacer(1, 0.3 * cm))

    # --- 8. Technical Trace ------------------------------------------------
    story.append(Paragraph("8. Technical Trace", styles["Heading2"]))
    if decision is not None:
        story.append(Paragraph(f"Router decision: task_type={decision.task_type}, tool_name={decision.tool_name}, "
                                f"router confidence={decision.confidence:.2f}", styles["BodyText"]))
        for r in decision.reasoning:
            story.append(Paragraph(f"&bull; {_safe_para_text(r)}", styles["BodyText"]))
    else:
        story.append(Paragraph("No router decision was produced (input failed to load before routing).", styles["BodyText"]))
    if trace.failure:
        story.append(Spacer(1, 0.15 * cm))
        story.append(Paragraph("<b>Raw failure record</b> (full text, unedited):", styles["BodyText"]))
        story.append(Paragraph(_safe_para_text(trace.failure).replace("\n", "<br/>"), ParagraphStyle("Mono", parent=styles["Code"], fontSize=7)))
    story.append(Spacer(1, 0.3 * cm))

    # --- 9. Limitations & Warnings -----------------------------------------
    story.append(Paragraph("9. Limitations & Warnings", styles["Heading2"]))
    any_limitation = False
    warnings = trace.validation.get("warnings") if trace.validation else None
    if warnings:
        any_limitation = True
        for w in warnings:
            story.append(Paragraph(f"&bull; {_safe_para_text(w)}", styles["BodyText"]))
    if out is not None and out.fallback_occurred:
        any_limitation = True
        story.append(Paragraph(
            f"&bull; Fallback occurred: a preferred real-model specialist failed and this output came from the "
            f"classical fallback instead. Reason: {_safe_para_text(out.fallback_reason)}", styles["BodyText"],
        ))
    if decision is not None and decision.task_type in ("grounding", "bitemporal_change", "optical_sar_fusion"):
        any_limitation = True
        story.append(Paragraph(
            "&bull; This capability uses a classical computer-vision baseline technique, not a trained "
            "foundation model - see docs/rs_adaptation.md for the adaptation research that led to this decision.",
            styles["BodyText"],
        ))
    if trace.failure:
        any_limitation = True
        cls = failure_mod.classify(trace.failure)
        story.append(Paragraph(
            f"&bull; ERROR {cls.code} · {cls.category}: {cls.what}. {cls.next_step}", styles["BodyText"],
        ))
    if not any_limitation:
        story.append(Paragraph("No warnings or limitations were recorded for this run.", styles["BodyText"]))
    story.append(Spacer(1, 0.3 * cm))

    # --- 10. Export Metadata ------------------------------------------------
    story.append(Paragraph("10. Export Metadata", styles["Heading2"]))
    story.append(
        _table(
            [
                ["Report generated", generated_at],
                ["Report generator", "SatQuery AI report.py (reportlab)"],
                ["Build", BUILD_LABEL],
                ["Source trace file", os.path.basename(out_path).replace(".pdf", ".json")],
            ]
        )
    )

    doc = SimpleDocTemplate(out_path, pagesize=A4, topMargin=1.8 * cm, bottomMargin=1.8 * cm)
    header_footer = _make_header_footer(generated_at)
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return out_path


def reportlab_available() -> bool:
    return _REPORTLAB_AVAILABLE
