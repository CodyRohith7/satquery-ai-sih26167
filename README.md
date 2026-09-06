# SatQuery AI

## Overview

SatQuery AI is a remote-sensing analysis assistant built for **SIH26167**
(Smart India Hackathon, ISRO — Space Technology problem statement). Given
one or two satellite/aerial images and a natural-language question, it
routes the query to the right analysis capability, runs it, generates
visual evidence, computes an honest confidence score, and produces a full
execution trace plus an exportable report (JSON and PDF) — through a
Streamlit "analyst console" UI or a CLI.

The system's guiding principle is **honesty about what actually ran**:
every answer discloses which model/technique produced it, every confidence
number states what it measures (and what it doesn't), and nothing is
presented as a trained deep-learning result unless it genuinely is one.

## Problem Statement

**SIH26167** — ISRO, Space Technology problem statement, calling for an
AI assistant capable of interpreting remote-sensing imagery (single-image
question answering, object/feature grounding, bi-temporal change
detection, and optical+SAR fusion) with a generic, non-remote-sensing-
adapted vision-language model explicitly called out as insufficient for
the domain.

## Solution

SatQuery AI implements the full pipeline the problem statement asks for —
ingestion, validation, intent routing, specialist execution, evidence
generation, confidence scoring, execution tracing, and export — using a
combination of a real pretrained vision-language model (for single-image
VQA) and classical computer-vision baselines (for grounding, change
detection, and fusion), with every component's real nature disclosed
rather than implied. The remote-sensing-domain adaptation work that would
close the gap between "a generic pretrained VLM" and a genuinely
RS-adapted one was designed, evaluated, and documented in detail (see
`docs/rs_adaptation.md`), but was not completed — larger candidate models
were found impractical on the available CPU-only hardware.

## Key Capabilities

1. **Single-image VQA** — Answers a natural-language question about one
   image. Real model: `HuggingFaceTB/SmolVLM-256M-Instruct`
   (`tool_single_image_vqa_smolvlm_v1`), a genuine, general-purpose
   vision-language model — **not** remote-sensing-domain-adapted — run on
   CPU. Falls back to a classical color/texture clustering baseline
   (`tool_single_image_vqa_v0`) if the model is unavailable or fails, with
   the fallback fully disclosed in the output. **Limitation**: a small
   (256M-parameter), general-purpose VLM trades some accuracy for
   CPU-feasibility, and it has no remote-sensing-specific training.
2. **Grounding** — Locates a described feature in an image and draws a
   bounding box. Method: classical computer vision
   (`tool_grounding_v0` — HSV color thresholding + contour extraction),
   not a deep-learning or foundation-model detector. **Limitation**: two
   real foundation-model candidates (Florence-2-base,
   `grounding-dino-tiny`) were evaluated on real hardware and both
   collapsed to near-full-image boxes on this kind of imagery — a
   documented model/domain limitation, not a bug — so the classical
   baseline remains the shipped capability (see `docs/rs_adaptation.md`).
3. **Bi-temporal change detection** — Compares two images of the same
   scene at different dates and highlights what changed. Method:
   classical CV (`tool_change_v0` — grayscale differencing + Otsu
   thresholding + morphological cleanup), not a trained change-detection
   model. **Limitation**: sensitive to illumination/registration
   differences between the two images that aren't real scene change.
4. **Optical+SAR fusion** — Jointly analyzes an optical and a SAR image of
   the same scene. Method: classical CV (`tool_fusion_v0` — k-means
   clustering over a stacked `[R, G, B, SAR]` feature vector), not a
   trained multi-modal fusion model. **Limitation**: no learned
   cross-modal representation; a hand-built joint feature space only.
5. **Intelligent routing** — A rule-based router combined with a trained
   intent classifier picks the right capability from the query text and
   the declared inputs (single image vs. pair, modality, dates), and
   explicitly refuses to guess (`needs_clarification`) when the query is
   genuinely ambiguous. **Limitation**: rule-based routing can misroute
   unusual phrasing; it is not a learned end-to-end agent.
6. **Evidence generation** — Every result includes a real, computed visual
   evidence artifact (bounding box overlay, change-map composite,
   fused cluster map, or the source image for VQA) — never a placeholder
   image. **Limitation**: evidence quality is bounded by the same
   classical-CV or small-VLM limitations described above.
7. **Confidence scoring** — Every result carries a confidence score
   derived from a real, specialist-specific computed signal (documented
   formula in `docs/confidence.md`), explicitly labeled by
   `method_version` (`v0_classical` heuristic signal vs.
   `v1_vlm_mean_token_probability` for the real model) so it is never
   mistaken for a calibrated correctness probability. **Limitation**:
   neither method version is a calibrated "this answer is factually
   correct" probability — that would require ground-truth-labeled
   evaluation data this project does not have.
8. **Execution trace** — Every run produces a complete, inspectable
   `ExecutionTrace` (query → validation → routing decision → specialist
   output → confidence → evidence), shown as a human-readable timeline and
   as raw JSON, both in the UI and in exports. **Limitation**: the trace
   reflects one pipeline run; it is not a persisted multi-session audit
   log.
9. **Contextual follow-up** — A follow-up question re-runs the same real
   pipeline against the same cached input image(s) without re-uploading,
   and keeps a real conversation history of prior (question, answer)
   pairs. **Limitation**: this is independent re-querying, not a
   memory-augmented conversational model — the VLM itself has no
   persistent context across turns.
10. **PDF + JSON reporting** — Exports a ten-section PDF report (Analysis
    Summary, Input Data, Analysis Result, Visual Evidence, Active
    Specialist, Confidence, Agent Execution Timeline, Technical Trace,
    Limitations & Warnings, Export Metadata) and the raw JSON trace, both
    generated directly from the same `ExecutionTrace` object rendered on
    screen. **Limitation**: PDF generation requires `reportlab`; without
    it, only JSON export is available.

## Architecture

```
Input (image(s) + query)
   │
   ▼
Validation      — src/ingestion/  (raster load, metadata, bi-temporal/cross-modal checks)
   │
   ▼
Router          — src/routing/router.py + intent_classifier.py
   │
   ▼
Specialist      — src/specialists/, src/grounding/, src/change/, src/fusion/
   │
   ▼
Evidence        — src/evidence/composer.py
   │
   ▼
Confidence      — src/confidence/engine.py
   │
   ▼
Execution Trace — src/routing/schemas.py (ExecutionTrace)
   │
   ▼
Report          — src/export/report.py (JSON + PDF)
```

`app/pipeline.py:run_query()` is the single orchestration path both
`app/cli.py` and `app/streamlit_app.py` call — there is no separate
UI-specific reimplementation of routing/validation/dispatch logic.

## Project Structure

```
satquery-ai/
├── app/                  # CLI (cli.py) and Streamlit UI (streamlit_app.py) + shared pipeline.py
├── src/
│   ├── ingestion/        # raster I/O, metadata inspection, input validation
│   ├── routing/          # router, intent classifier, schemas, timeline, failure classification
│   ├── specialists/      # VQA (classical + SmolVLM dispatch)
│   ├── grounding/        # classical grounding baseline
│   ├── change/           # classical bi-temporal change baseline
│   ├── fusion/           # classical optical+SAR fusion baseline
│   ├── evidence/         # evidence-image composition
│   ├── confidence/       # confidence engine
│   ├── models/           # model/tool availability registry, Florence-2 compat shim
│   ├── export/           # JSON + PDF report generation
│   └── evaluation/       # benchmark interfaces (VRSBench/RSVQA/CDVQA/ISRO-SAC — NOT_YET_EVALUATED)
├── scripts/              # one-off verification/research scripts (see docs/rs_adaptation.md)
├── configs/              # rs_adaptation.yaml (not-yet-executed LoRA adaptation config)
├── data/
│   ├── fixtures/         # synthetic, seeded demo images (see data/fixtures/README.md)
│   ├── manifests/        # real-dataset manifest template (none downloaded yet)
│   ├── processed/, raw/, samples/   # empty, reserved for real imagery
├── docs/                 # architecture/design/research documentation, evidence/
├── tests/                # unit/ and integration/ test suites
├── requirements.txt              # core runtime dependencies
├── requirements-optional.txt     # rasterio + research-script-only dependencies
└── .gitignore
```

## Installation

Tested on **Windows with Miniforge/Conda, Python 3.10.x**:

```powershell
conda create -n satquery python=3.10 -y
conda activate satquery
cd C:\path\to\satquery-ai
pip install -r requirements.txt
```

Install `requirements-optional.txt` only if you need real GeoTIFF I/O
(`rasterio`) or want to re-run the research scripts in `scripts/`:

```powershell
pip install -r requirements-optional.txt
```

## Run

```powershell
conda activate satquery
streamlit run app\streamlit_app.py
```

Opens the "SatQuery AI — Analysis Console" at `http://localhost:8501`,
with four tabs (Single-image VQA, Grounding, Bi-temporal Change,
Optical+SAR Fusion) and a bundled-synthetic-fixture option on every tab so
it can be run without hunting for real imagery.

A CLI is also available:

```powershell
python app\cli.py --image1 "C:\path\to\your_image.jpg" --query "What is the dominant land cover or feature visible in this image?"
```

## Example Workflow

1. Open the **Single-image VQA** tab, check "Use bundled synthetic demo
   fixture", and ask: *"What is the dominant land cover or feature visible
   in this image?"*
2. Click **Run analysis**. The router breadcrumb shows
   `single_image_vqa → tool_single_image_vqa_smolvlm_v1 → EVIDENCE`, the
   model/tool disclosure panel states `REAL MODEL —
   HuggingFaceTB/SmolVLM-256M-Instruct`, and a confidence value labeled
   "Generation certainty" appears alongside the source image.
3. Ask a follow-up question (e.g. *"Is there any visible water in this
   image?"*) without re-uploading — a new answer appears, and a
   conversation history expander shows the exchange.
4. Expand **Technical details** to see the full execution trace, then
   export the result as **PDF report** or **JSON**.

## Exports

- **JSON** — the complete raw `ExecutionTrace` (query, validation,
  routing decision, specialist output, confidence, evidence references).
- **PDF report** — a ten-section formatted report (Analysis Summary,
  Input Data, Analysis Result, Visual Evidence, Active Specialist,
  Confidence, Agent Execution Timeline, Technical Trace, Limitations &
  Warnings, Export Metadata) with page numbers and a running header/footer.

## Testing

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Verified in the local sandbox mirror (classical/CPU-fallback paths only,
no network egress for model downloads in that environment):
**100 tests collected, 87 executed and passed, 13 skipped, 0 failed.**
The 13 skips are tests gated on `torch`/`transformers`/`streamlit` being
importable (real-model and real-Streamlit `AppTest` smoke tests) — on a
machine with the full `requirements.txt` installed (e.g. the target
Windows/conda environment), those run for real instead of skipping.

## Performance

Measured on real CPU hardware (target Windows machine, no GPU):

- **VQA (SmolVLM-256M-Instruct) model load**: ~47 seconds.
- **VQA inference**: typically **~35–60 seconds** per query on CPU; one
  measured run took **128.47 seconds**, attributed to CPU contention on
  that run rather than a regression (see the project's engineering
  history for the investigation). This is **not real-time inference** —
  plan for tens of seconds of latency per VQA query.
- **Grounding/change/fusion (classical CV)**: sub-second to a few seconds,
  since these do not load a neural network.

## Limitations

- **Grounding, change detection, and fusion are classical computer-vision
  baselines**, not deep-learning or foundation-model-based — two real
  foundation-model grounding candidates were tried on real hardware and
  both failed to produce spatially meaningful boxes on this kind of
  imagery (a documented model/domain limitation).
- **The VQA model is a general-purpose vision-language model, not
  remote-sensing-domain-adapted.** A LoRA adaptation plan targeting a
  larger remote-sensing-oriented VLM was designed and documented but not
  executed (see `docs/rs_adaptation.md`).
- **CPU-only inference is slow** (tens of seconds per VQA query, ~47s
  model load) — there is no GPU acceleration path in the current setup.
- **Small VLM size (256M parameters)** trades some answer accuracy and
  robustness for CPU feasibility.
- **No calibrated correctness probability** — confidence scores measure
  signal quality (classical) or generation certainty (VLM), never
  factual-correctness probability, since no ground-truth-labeled
  evaluation data exists for this project yet (`NOT_YET_EVALUATED` in
  `src/evaluation/benchmarks.py`).
- **No real satellite imagery is bundled** — the fixtures in
  `data/fixtures/` are synthetic, seeded demo images, clearly labeled as
  such (`data/fixtures/README.md`); results on them describe the
  synthetic pattern, not a real place.
- **Optional `rasterio` dependency** — without it, raster loading falls
  back to Pillow (PNG/JPEG only, no CRS/band metadata); real GeoTIFF
  ingestion with full geospatial metadata requires installing
  `requirements-optional.txt`.

## Research / Model Notes

Larger remote-sensing-oriented VLM candidates — notably **PaliGemma 2**
(`google/paligemma2-3b-pt-448`, the original LoRA-adaptation target in
`configs/rs_adaptation.yaml`) — were evaluated and ultimately not pursued
as impractical for the available CPU-only hardware (no GPU, limited
RAM). Two general-purpose grounding foundation models
(Florence-2-base, `grounding-dino-tiny`) were also evaluated as
grounding-capability upgrades and rejected after real-hardware testing
showed both collapsing to near-full-image boxes on this kind of imagery.
The full evaluation history, real measured numbers, and the reasoning
behind each decision are in `docs/rs_adaptation.md`; the exact commands
used to reproduce every step on real hardware are in
`docs/RUN_ON_WINDOWS.md`.

## License

**No license currently exists for this repository.** No `LICENSE` file
is present. Until one is added, this code is not licensed for reuse,
modification, or redistribution beyond what applies by default under
copyright law — please add a license before treating this repository as
open source.
