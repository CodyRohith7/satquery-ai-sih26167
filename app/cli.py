#!/usr/bin/env python3
"""SatQuery AI - thin end-to-end CLI.

This was the entire user interface through the CORE CAPABILITY FREEZE
milestone. A Streamlit UI now exists (`app/streamlit_app.py`) and calls the
exact same orchestration function this CLI calls - `app/pipeline.py:
run_query()` - so there is one real pipeline, not two. This CLI remains as
the scriptable/headless entry point and as the thing every prior
verification run (docs/RUN_ON_WINDOWS.md, docs/rs_adaptation.md) was
measured against.

Examples
--------
Single-image VQA:
  python app/cli.py --image1 data/fixtures/single_image.png \\
      --query "what are the major features in this image"

Grounding:
  python app/cli.py --image1 data/fixtures/single_image.png \\
      --query "locate the water body in this image and highlight it"

Bi-temporal change:
  python app/cli.py --image1 data/fixtures/change_before.png --date1 2020-01-01 \\
      --image2 data/fixtures/change_after.png --date2 2024-01-01 \\
      --query "what changed between these two dates"

Optical+SAR fusion:
  python app/cli.py --image1 data/fixtures/fusion_optical.png --modality1 optical \\
      --image2 data/fixtures/fusion_sar.png --modality2 sar \\
      --query "use both images to identify built-up and water-covered regions"
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from routing.schemas import ExecutionTrace  # noqa: E402
from export import report  # noqa: E402
import pipeline  # noqa: E402 - shared orchestration, also used by streamlit_app.py


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SatQuery AI thin end-to-end CLI (v0)")
    p.add_argument("--image1", required=True, help="Path to the first (or only) input image")
    p.add_argument("--image2", default=None, help="Path to a second input image (for change/fusion)")
    p.add_argument("--query", required=True, help="Natural-language query")
    p.add_argument("--modality1", default=None, choices=[None, "optical", "sar"], help="Declared modality of image1")
    p.add_argument("--modality2", default=None, choices=[None, "optical", "sar"], help="Declared modality of image2")
    p.add_argument("--date1", default=None, help="Declared acquisition date of image1 (ISO format)")
    p.add_argument("--date2", default=None, help="Declared acquisition date of image2 (ISO format)")
    p.add_argument("--out-dir", default="runs/latest", help="Directory to write evidence/trace/report into")
    return p


def run(args: argparse.Namespace) -> ExecutionTrace:
    """Thin wrapper: all real orchestration logic now lives in
    `app/pipeline.py:run_query()` so the Streamlit UI can call the exact same
    code path instead of a second, UI-specific reimplementation. This
    function's behavior is unchanged from before the extraction - see
    pipeline.py for the real logic."""
    return pipeline.run_query(
        image1_path=args.image1,
        query=args.query,
        image2_path=args.image2,
        modality1=args.modality1,
        modality2=args.modality2,
        date1=args.date1,
        date2=args.date2,
        out_dir=args.out_dir,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    trace = run(args)

    json_path = report.export_json(trace, os.path.join(args.out_dir, "trace.json"))
    print(f"\n=== SatQuery AI (v0) ===")
    print(f"Query: {trace.query}")
    if trace.router_decision:
        d = trace.router_decision
        print(f"Router decision: task={d.task_type} tool={d.tool_name} router_confidence={d.confidence:.2f}")
        for r in d.reasoning:
            print(f"  - {r}")
    if trace.failure:
        print(f"\nFAILURE: {trace.failure}")
    elif trace.specialist_output:
        out = trace.specialist_output
        print(f"\nAnswer: {out.answer_text}")
        print(f"Confidence: {out.confidence.value:.2f} (method={out.confidence.method_version})")
        if out.confidence.downgrades:
            print(f"  Downgrades: {out.confidence.downgrades}")
        for ev in out.evidence:
            print(f"Evidence: {ev.description} -> {ev.image_path}")
        print(f"Latency: {out.latency_seconds:.3f}s")

        if report.reportlab_available():
            pdf_path = report.export_pdf(trace, os.path.join(args.out_dir, "audit_report.pdf"))
            print(f"PDF report: {pdf_path}")
    print(f"JSON trace: {json_path}")


if __name__ == "__main__":
    main()
