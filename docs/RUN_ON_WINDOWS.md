# Running Phase 2B for real, on your Windows machine

**Superseded plan notice**: this file originally targeted `google/paligemma2-3b-pt-448`.
After you ran `scripts/verify_environment.py` for real and reported back no
NVIDIA GPU, no CUDA, and 8.24GB total RAM, that plan was replaced with the
hybrid small-model strategy below (approved). See `docs/rs_adaptation.md`
section 8 for the full reasoning. Steps here now match that revised plan.

This project is at `<repo-root>\satquery-ai` (e.g. `C:\path\to\satquery-ai`)
on your machine. Everything below must be run by you, in your own real terminal —
this session's only access to your computer is a small, GPU-less,
network-blocked companion VM, not your actual Windows/conda environment
(confirmed in `docs/rs_adaptation.md` section 7). It cannot run any of this
for you; it can only prepare the code and read back whatever JSON result
files you produce.

## FIRST SUCCESS TEST (VQA) — DONE, verified

`HuggingFaceTB/SmolVLM-256M-Instruct` on real hardware: model load 46.891s,
inference 38.457s, peak working set 2882.7MB, CPU backend, real output on a
real aerial photo. See `docs/rs_adaptation.md` section 8 for the full record.

## Compatibility fix applied (after the real `_supports_sdpa` AttributeError)

Your run hit a second, confirmed transformers-vs-Florence-2 compatibility
issue beyond the flash_attn one — see `docs/rs_adaptation.md` section 8 for
the full sourced explanation. Both issues are now handled in one place,
`src/models/florence2_compat.py`, which `scripts/second_success_test_grounding.py`
now uses instead of loading the model inline. No dependency or model change
was needed — same `einops`/`timm` as before, same `microsoft/Florence-2-base`.

## Fix applied for the RecursionError you hit next

Your next report was `RecursionError: maximum recursion depth exceeded while
calling a Python object` in place of the `_supports_sdpa` error. This was
**not** a transformers-version issue or anything about `_supports_sdpa`
itself: it was a real bug in this project's own flash_attn compatibility
shim, introduced when it was consolidated into `florence2_compat.py`. The
shim re-imported `get_imports` from `transformers.dynamic_module_utils`
every time it ran, but it is itself installed as that exact module attribute
while it runs, so the re-import rebound the name to itself, and calling it
called itself, forever.

This was confirmed by directly reproducing it (transformers/torch stubbed
out, since neither installs in the development sandbox - no network to
pypi.org there) before being fixed, not guessed at. See the "ROOT CAUSE"
section of `src/models/florence2_compat.py`'s module docstring for the full
account, and `tests/integration/test_florence2_compat.py`'s
`TestFixedGetImportsNoRecursion` for the regression test that exercises the
exact failing pattern - the shim being entered twice, once for the initial
load attempt and once for the `_supports_sdpa` retry, the same shape as your
actual run. The fix is a one-line change (capture the real `get_imports`
once, before any patch is installed, instead of re-importing it while the
patch is active). No dependency, environment, or model change was needed,
and the `_supports_sdpa` patch from the previous fix is unchanged.

## Fix applied for the generation-time `'NoneType' object has no attribute 'shape'` error

After loading succeeded (`model_load_success: true`, confirmed on real
hardware), generation itself failed with `AttributeError: 'NoneType' object
has no attribute 'shape'`, traceback ending in Florence-2's own
`prepare_inputs_for_generation` at `past_key_values[0][0].shape[2]`. This is
a separately confirmed, actively-documented Florence-2 compatibility issue
(not specific to this project) - see `docs/rs_adaptation.md` section 8 and
`src/models/florence2_compat.py`'s "GENERATION COMPATIBILITY" docstring
section for the full sourced explanation.

Rather than guess at one fix, `scripts/second_success_test_grounding.py`
now calls `generate_florence2()`, which tries an ordered list of
**documented, standard** `model.generate()` keyword-argument combinations
(never any edit to Florence-2's own code) - the requested configuration
first, then plain greedy decoding (`num_beams=1`), then `use_cache=False`,
then both together - and records in the result JSON's new
`"generation_report"` field exactly which one worked, or every attempt's
full traceback if none did. It also records `"generation_diagnostics"`:
the loaded model's exact class and full MRO, and whether `GenerationMixin`
is actually present in it - this is the evidence that settles a real,
sourced open question (some Florence-2 HF repos needed an explicit
`GenerationMixin` fix added after transformers 4.50; whether
`microsoft/Florence-2-base`'s outer wrapper class has received it could not
be confirmed from here).

No new dependencies, no model change - re-run the same command as before.

## Grounding QUALITY issue: both phrases returned the same near-full-image box

Model load and generation both now run cleanly on real hardware, but
`"farmland"` and `"buildings"` returned the identical box
(`[1.6949999, 0.2259999, 676.9829, 451.3219]`) - i.e. Florence-2 is running
without crashing, but its grounding output isn't usable yet. Rather than
guess why, run the new diagnostic script - it runs the exact official
Florence-2 model-card invocation first, then a small ordered set of
experiments, on the same image:

```powershell
python scripts\diagnose_grounding_quality.py --image "C:\path\to\your_image.jpg"
```

It runs, in order: `<OD>` (plain object detection, no phrase - the single
simplest official Florence-2 example, run first as a baseline for whether
the model can localize ANYTHING in this specific image at all);
`<DENSE_REGION_CAPTION>` (also no phrase, a second independent baseline);
then `<CAPTION_TO_PHRASE_GROUNDING>` for `"farmland"` and `"buildings"` as
bare words (matching the previous run); then the SAME two concepts phrased
as captions (`"a large area of farmland"`, `"a cluster of buildings"`) -
because the official model-card example itself uses a caption-style phrase
("A green car parked in front of a yellow building."), not a bare noun,
so this directly tests whether phrase FORMAT explains the degenerate
output. Every experiment records the RAW generated text
(`skip_special_tokens=False`, before any parsing) and each returned box's
fraction of total image area - not just the parsed coordinates - so this
one run's JSON should make it possible to tell apart: our own
invocation/preprocessing (A), a post-processing/coordinate-scaling bug (B),
or a genuine Florence-2-base/domain limitation on this kind of imagery (C).

**Report back**: the full JSON, plus the overlay PNG if one was written.
Still no LoRA/UI/GitHub/PPT until this is resolved.

## Florence-2-base grounding: diagnosed and closed (model/domain limitation)

Your `diagnose_grounding_quality.py` results were reviewed and accepted:
raw generated text genuinely changes per phrase (not an invocation bug),
but every grounding box - including plain `<OD>` with no phrase at all -
collapses to ~99% of the image. This is classified as a Florence-2-base
model/domain limitation, not a software bug. Per the explicit instruction,
Florence-2 grounding debugging is now STOPPED - no more patches, no box
manipulation, no claiming grounding works because tokens were generated.
Full comparison of alternatives considered is in `docs/rs_adaptation.md`
section 8.

## Current step: try a different grounding model - `grounding-dino-tiny`

`IDEA-Research/grounding-dino-tiny` (Apache-2.0, ~0.2B params) was picked
as the fastest defensible path to real, phrase-distinguishing grounding.
It is a general-purpose open-vocabulary object detector - **not**
remote-sensing-specific, and the script's own output is explicit about
that (`model_is_remote_sensing_specific: false`) so it's never mistaken
for RS-domain adaptation. It needs **no new dependencies** - natively
supported by the transformers/torch already installed, no
`trust_remote_code`, no custom modeling file (this sidesteps the whole
class of compatibility bugs Florence-2 needed three rounds to fix).

```powershell
cd C:\path\to\satquery-ai
conda activate satquery
python scripts\third_success_test_grounding_dino.py --image "C:\path\to\your_image.jpg" --target-phrase "farmland" --target-phrase-2 "buildings"
```

This runs `"farmland."` and `"buildings."` (GroundingDINO's documented
lowercase + trailing-period convention) plus a control phrase
(`"an airplane."`, which should find nothing or low-confidence detections
- a sanity check against a fixed/default-box failure mode, the same class
of bug Florence-2 had). It writes:

- `docs/evidence/third_success_test_grounding_dino_result.json` - load/inference
  time, RAM, and per-phrase real boxes + confidence scores + each box's
  fraction of total image area.
- `docs/evidence/third_success_test_grounding_dino_overlay.png` - the image
  with all detected boxes drawn on it, one color per phrase.

**Report back**: the full JSON and the overlay PNG. Success is genuinely
different, non-full-image boxes for "farmland" vs "buildings" - not just
that boxes were returned. Still no LoRA/UI/GitHub/PPT.

## Fix applied for the processor `TypeError` you hit on the first grounding-dino-tiny run

Model load succeeded (`model_load_success: true`, load time 96.626s, CPU,
peak working set 739.1MB, real weights downloaded), but the test failed
*before* any inference ran, at
`processor(images=image, text=text_labels, return_tensors="pt")`, with:

```
TypeError: TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
```

This was a text-input-**shape** bug in the script, not a model or
environment problem - a tokenizer-level type error confirms that. The
script had been sending a list-of-lists (`text=[[formatted]]`), matching a
convenience shape shown in transformers' "main"-branch docs; the real
pinned `transformers==4.57.1` on your machine doesn't accept it that way.
The fix (see the "INPUT FORMAT FIX" section of
`scripts/third_success_test_grounding_dino.py`'s module docstring for the
full account) sends a single **plain string** per phrase instead (e.g.
`"farmland."`) - no list, no list-of-lists - since this script only ever
sends one phrase per forward call anyway. No model change, no transformers
version change, no monkeypatch, no hardcoded detections, and the classical
grounding baseline (`src/grounding/grounding.py`) is untouched. A new
regression test,
`tests/integration/test_third_success_test_grounding_dino.py` (8 tests),
reproduces the exact `TypeError` against a stub that mimics the real
tokenizer's rejection of nested-list input and confirms the plain-string
format is accepted - all passing.

Re-run the exact same command as before - no new dependencies, same model:

```powershell
cd C:\path\to\satquery-ai
conda activate satquery
python scripts\third_success_test_grounding_dino.py --image "C:\path\to\your_image.jpg" --target-phrase "farmland" --target-phrase-2 "buildings"
```

**Report back**: the full JSON and the overlay PNG. Success requires
actual forward inference completing for all three phrases (farmland,
buildings, and the "an airplane" control), real boxes/scores, and
genuinely different, non-full-image boxes for "farmland" vs "buildings".
If it fails again after this fix, send the complete traceback - still no
LoRA/UI/GitHub/PPT.

## GROUNDING INVESTIGATION CLOSED (real-hardware result received and accepted)

After the input-format fix, `grounding-dino-tiny` ran successfully end to
end: model load, real inference for all three phrases, real boxes and
scores - no crash. But the boxes are not spatially meaningful: farmland
99.09%, buildings 99.18%, airplane (negative control) 98.82% of the image
area. All three, including the control, collapse to essentially the full
image - the same degenerate pattern Florence-2-base showed. This is a
second independent foundation model confirming the same domain limitation,
not a bug in this project's code (both the invocation format and the
inference path are now verified working).

**Decision**: neither Florence-2-base nor grounding-dino-tiny will be used
as the primary grounding capability, and no further time will be spent
debugging generic foundation-model grounding on this imagery. The existing
classical baseline, `tool_grounding_v0`
(`src/grounding/grounding.py`), stays the MVP's grounding capability -
already built, already `AVAILABLE`, already tested - explicitly labeled
as a `v0_classical` baseline, never as foundation-model or RS-adapted
grounding. Both foundation-model attempts and their real numbers are kept
in `docs/rs_adaptation.md` as engineering evidence, not deleted.

## Bi-temporal change: deep-learning upgrade evaluated and declined

Two Hugging Face candidates were researched (`DarthReca/actu-change-detection`
- water-only, ~200M params, GPU-implied; `deepang/adaptformer-LEVIR-CD` -
building-change-only, 12.5M params) - both need `trust_remote_code=True`,
the same custom-modeling-code risk that cost three debugging rounds on
Florence-2. Full comparison in `docs/rs_adaptation.md` section 9.
**Decision: do not pursue either.** `tool_change_v0` (already built,
`AVAILABLE`, tested, and labeled `v0_classical` in its own output) is the
accepted bi-temporal change capability going forward - no real-hardware
run needed for this milestone since nothing new is being added.

## Optical+SAR fusion: deep-learning upgrade evaluated and declined

Three candidates researched under an explicit hard time/scope limit (full
comparison table in `docs/rs_adaptation.md` section 10): `allenai/satlas-
pretrain` (separate optical/SAR backbones, no fusion head - would need us
to build and likely train one), `Yusin2Chen/SARoptical_fusion` (purpose-
built for this, but **no pretrained weights published at all** - training
code only), and the Clay Foundation Model v1.5 (632M params, genuinely
joint S1+S2 embeddings, but a masked autoencoder that only produces
embeddings out of the box - a usable answer needs fine-tuning a downstream
head on real labeled data, plus a non-`transformers` "Lightning"-based
loading path). None is a load-and-run checkpoint that produces a usable
fused answer without new modeling code or real training - both out of
scope for a small model swap. **Decision: reject all three.**
`tool_fusion_v0` (already built, `AVAILABLE`, tested, labeled
`v0_classical`) remains the shipped optical+SAR fusion capability.

## CORE CAPABILITY FREEZE: SmolVLM wired into the live app for real (gap found and fixed)

Before building the UI, a real gap was caught: SmolVLM had only ever run in
the standalone `scripts/first_success_test_vqa.py` proof script - the
actual app (`app/cli.py` / router / registry) was still routing every
`single_image_vqa` query to the classical KMeans baseline. Fixed now, not
shipped as a UI with a mislabeled model-disclosure panel. Full account in
`docs/rs_adaptation.md` section 11. Summary:

- `src/specialists/vqa_smolvlm.py` (new) - the real specialist, using the
  exact verified load/generate recipe.
- `src/specialists/vqa_dispatch.py` (new) - the only module the CLI/UI
  call for VQA; tries SmolVLM when the registry says it's genuinely
  available, falls back to the classical baseline on ANY failure with full
  disclosure in the output itself (`raw["smolvlm_attempted"]`,
  `raw["smolvlm_fallback_reason"]`, `raw["smolvlm_fallback_traceback"]`).
- `src/models/registry.py` / `src/routing/router.py` - SmolVLM's
  availability is a real, live dependency check, never hardcoded.
- 8 new tests (`tests/integration/test_vqa_dispatch.py`), all passing;
  full suite now 61/61 (was 53).

**Please run this real-hardware test before any further UI work** (per
the explicit instruction to verify the integrated path before proceeding):

```powershell
cd C:\path\to\satquery-ai
conda activate satquery
python app\cli.py --image1 "C:\path\to\your_image.jpg" --query "What is the dominant land cover or feature visible in this image?"
```

**Report back**: the full console output and `runs/latest/trace.json`.
Success requires: `router_decision.tool_name` =
`tool_single_image_vqa_smolvlm_v1` (proves the router picked the real
model, not the fallback), a real SmolVLM-generated answer (not the
"colour/texture clustering" classical phrasing), `specialist_output.raw`
showing real `model_load_seconds`/`inference_seconds`/
`mean_token_confidence`, and no `smolvlm_attempted`/fallback fields (those
would mean it fell back - if that happens, send the full
`smolvlm_fallback_traceback` instead so it can be fixed for real, same as
every other failure in this project).

## Confirmed working - then a real confidence-labeling bug found and fixed

The above ran successfully for real: `tool_single_image_vqa_smolvlm_v1`
selected, real SmolVLM answer ("...the image is farmland."), 47.082s real
CPU inference. But the trace showed `Confidence: 0.71
(method=v0_classical)` for a SmolVLM answer - `compute()` had no way to
label a caller's own confidence methodology, so it always stamped
`"v0_classical"`, even for a real neural network's output. Fixed: see
`docs/rs_adaptation.md` section 11 and `docs/confidence.md` for the full
account. Summary: `confidence_engine.compute()` now takes explicit
`method_version`/`basis_description` parameters (the four classical
specialists are unaffected - they get the same defaults as before);
SmolVLM's confidence is now honestly labeled
`method_version: "v1_vlm_mean_token_probability"` with a
`basis_description` explaining exactly what it measures (generation
certainty, not factual correctness); `SpecialistOutput` gained explicit
`model_name` / `fallback_occurred` / `fallback_reason` fields so the UI
never has to infer any of this from `tool_name` or from `raw`'s keys. 63
tests passing (was 61).

**Please re-run the same command** to confirm nothing regressed and the
confidence label is now correct:

```powershell
cd C:\path\to\satquery-ai
conda activate satquery
python app\cli.py --image1 "C:\path\to\your_image.jpg" --query "What is the dominant land cover or feature visible in this image?"
```

**Report back**: the console output and `runs/latest/trace.json`. Success
now additionally requires: `specialist_output.confidence.method_version`
= `"v1_vlm_mean_token_probability"` (never `"v0_classical"`),
`specialist_output.confidence.basis_description` present and explaining
the mean-token-probability meaning, `specialist_output.model_name` =
`"HuggingFaceTB/SmolVLM-256M-Instruct"`, and
`specialist_output.fallback_occurred` = `false` - alongside everything
verified in the previous run (real answer, real timing, SmolVLM tool
selected, no classical fallback).

## Earlier step (superseded): re-run the Florence-2-base grounding test

```powershell
cd C:\path\to\satquery-ai
conda activate satquery
pip install einops timm
```

Only these two — not the rest of `requirements-future.txt`, and don't start
the LoRA/adaptation experiment yet.

Run the grounding test on the **same real image** used for the VQA test, with
two different target phrases so the result shows whether the boxes actually
change with the query (proof they're model-generated, not fixed):

```powershell
python scripts\second_success_test_grounding.py --image "C:\path\to\the_same_real_image.png" --target-phrase "farmland" --target-phrase-2 "buildings"
```

This does **not** use `src/grounding/grounding.py` (the classical baseline) —
it calls `microsoft/Florence-2-base` directly via its native
`<CAPTION_TO_PHRASE_GROUNDING>` task. It writes two files:

- `docs/evidence/second_success_test_grounding_result.json` — model load time,
  backend, peak working set, and per-phrase inference time + the raw
  generated text + the parsed bounding boxes/labels for each phrase, plus
  `boxes_differ_between_phrases` (true/false) as the direct "not hardcoded"
  check.
- `docs/evidence/second_success_test_grounding_overlay.png` — the actual
  image with the returned boxes drawn on it (each phrase in a different
  color).

If it fails (missing dependency, OOM, a Florence-2 loading or generation
quirk we haven't hit yet), it writes `"status": "FAILED"` with the real
error AND a `"traceback"` field with the full original traceback text (not
just the exception message - this is what was missing when the
RecursionError could only be reported as a one-line message) — bring that
back too, along with `"generation_report"` and `"generation_diagnostics"`
if present.

**Report back**: the JSON content, and send/commit the overlay PNG back so
it can actually be looked at, not just described. Stop there — no LoRA
experiment, no UI, no GitHub, no PPT until this is reported and reviewed.

## CORE CAPABILITY FREEZE: Streamlit UI built — run and verify it for real

The backend is frozen (VQA → SmolVLM with disclosed classical fallback;
grounding/change/fusion → classical v0 baselines) and a Streamlit analyst
workstation UI now exists on top of the exact same pipeline the CLI uses
(`app/pipeline.py:run_query()` — extracted from `app/cli.py:run()` so there
is one real orchestration path, not two). It was designed and its Python
control flow was verified with a hand-built fake `streamlit` module in the
cloud sandbox (`tests/integration/test_streamlit_app_logic.py` — 5 tests,
all real pipeline executions, no invented data), because streamlit itself
cannot be installed there (no PyPI network egress — see
`docs/network_constraints.md`). **It has not yet been launched or looked at
in a real browser.** That verification can only happen on your machine.

Install streamlit into the same conda env everything else is in:

```powershell
conda activate satquery
pip install streamlit
```

Then launch it from the repo root:

```powershell
streamlit run app\streamlit_app.py
```

It should open a browser tab at `http://localhost:8501` titled "SatQuery AI
- Analysis Console": a dark, monospace-accented "analyst console" interface
with a sidebar listing every registered specialist's live `AVAILABLE` /
`NOT_AVAILABLE_SANDBOX` status (this should now show `tool_single_image_
vqa_smolvlm_v1` as `AVAILABLE` on your machine, since torch/transformers are
installed there), and four tabs: `01 · SINGLE-IMAGE VQA`, `02 · GROUNDING`,
`03 · BI-TEMPORAL CHANGE`, `04 · OPTICAL + SAR FUSION`.

**What to check on each tab** (every tab has a "Use bundled synthetic demo
fixture" checkbox so you can run all four without hunting for real
imagery — it is explicitly labeled as synthetic, not real satellite data):

1. Check the box (or upload your own image(s)), leave or edit the example
   query, click `RUN ANALYSIS`.
2. A router breadcrumb should appear (`QUERY → <TASK_TYPE> → <TOOL_NAME> →
   EVIDENCE`) with a real router confidence value and an expandable
   "Router reasoning" section with real sentences (not placeholder text).
3. On the VQA tab specifically: the "MODEL / TOOL DISCLOSURE" panel should
   say `REAL MODEL`, list `MODEL: HuggingFaceTB/SmolVLM-256M-Instruct`, and
   `CONFIDENCE BASIS` should mention generated-token probability / generation
   certainty — **not** "Classical CV". The other three tabs should always
   say `CLASSICAL BASELINE` with `TOOL: tool_grounding_v0` /
   `tool_change_v0` / `tool_fusion_v0` respectively — grounding's evidence
   caption should explicitly read "Visual evidence - classical
   remote-sensing baseline", never implying a foundation model.
4. The fusion tab should show three images: the raw optical input, the raw
   SAR input, and the fused cluster-map evidence — the raw pair is
   rendered by the UI itself, not by `tool_fusion_v0` (see
   `docs/ui_design.md`'s note on why fusion's own evidence artifact is a
   single image, not a triptych, and how the UI compensates without
   changing `fusion.py`).
5. Expand "EXECUTION TRACE" and confirm it's the real, complete
   `ExecutionTrace` object (`st.json` of `trace.to_dict()`), then try both
   `EXPORT JSON` and `EXPORT PDF` download buttons.
6. Deliberately break one tab (e.g. upload a tiny/corrupt image, or on the
   change tab set both dates equal) and confirm you get the red "ANALYSIS
   FAILED" panel with WHAT HAPPENED / WHY / WHAT TO DO NEXT — not a raw
   Python traceback in the main interface (the real traceback, when one
   exists, should only appear inside the collapsed "Technical details"
   expander).

Also run the full test suite on your machine and compare the total against
what the cloud sandbox reports as of this freeze: 78 tests collected, 65
actually executed and passed, 13 skipped (0 failures) — on your machine
several of those 13 skips (the ones gated on torch/transformers/streamlit
not being importable there) should instead run for real and pass, so your
passing count should be higher than 65, never lower, and your failing count
should be 0:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

**Report back**: whether the app launched, a screenshot of each of the four
tabs after a successful run (fixture or real imagery, your choice), the
full test count/pass/fail/skip breakdown, and anything that looked visually
wrong (CSS targets Streamlit's `data-testid` attributes, which can shift
between versions — if a widget looks unstyled, tell me your `streamlit
--version` and what looked off, rather than trying to patch the CSS
yourself). Do not start PPT/GitHub/video yet — that instruction is still in
force until you confirm this.

## UI REFINEMENT PASS: visual/UX redesign, no functional changes

The functional UI from the previous section was approved, and a
presentation-only redesign pass followed it — same `pipeline.run_query()`
calls, same session data, only the layout/typography/hierarchy/microcopy
changed. Full rationale and the before/after wireframe: `docs/ui_design.md`
("Revision 2"). Re-verified in the sandbox the same way as before (fake-
`streamlit` stub tests, since real streamlit still can't install here): 79
tests collected, 66 executed and passed, 13 skipped, 0 failures.

**What changed, briefly**: the sidebar shrank to a compact capability-
readiness rail (VQA/Ground/Change/Fusion — READY, computed live from the
registry) with raw internal tool IDs moved into a collapsed "Technical
system status"; monospace type is now restricted to model/tool identifiers
and the raw JSON trace — everything else (headings, labels, the answer
text) is a system sans-serif stack with no external font dependency, so it
should look correct even offline; confidence and model/tool disclosure are
now compact by default (a number + a short label) with the full technical
text one click away in a "Details" expander; the router breadcrumb moved
from sitting above every answer into the collapsed "Execution trace"
section, replaced up top by one quiet line ("Analysis type: ..."); VQA now
shows its source image beside the answer instead of below it; bi-temporal
change shows an explicit "No significant change detected" headline when the
real changed-fraction rounds to 0%; tab labels and button/checkbox copy
were reworded to remove developer jargon ("RUN ANALYSIS" → "Run analysis",
"demo fixture" → "bundled sample image", etc.).

**Please re-run the same launch command** (`streamlit run
app\streamlit_app.py`) and go through this checklist — same four tabs as
before, but now also check at two window sizes:

1. Header reads "SatQuery AI / Remote sensing analysis / SIH26167 · ISRO ·
   Space Technology Programme" — no large "v0" badge anywhere prominent.
2. Sidebar is noticeably narrower than before, shows four capability rows
   (VQA/Ground/Change/Fusion, each READY), and a collapsed "Technical
   system status" section underneath with the raw tool IDs.
3. Run each of the four tabs (sample image/pair is fine). Confirm: the
   answer text is the dominant element, not a wall of monospace; VQA shows
   the source image directly beside the answer; confidence shows a number
   + short label (e.g. "Generation certainty" for the real model, "Signal
   quality heuristic" for classical) with a "Confidence details" expander
   underneath holding the full real explanation; model/tool shows a
   compact block with a "Method details" expander for the raw tool ID.
4. On the change tab, compare an image against itself (same file for
   before/after, different dates) and confirm you see "No significant
   change detected" as a clear headline.
5. Confirm "Execution trace" is collapsed by default and, when expanded,
   reads as a plain Query → Router → Specialist → Evidence → Export flow
   before you reach the raw JSON (nested one level deeper).
6. Resize the browser window to roughly 1366×768 and then 1920×1080 (or
   your OS's window-snap sizes) and confirm nothing requires horizontal
   scrolling and the sidebar stays usable at both.
7. Tab through the controls with the keyboard and confirm you can see a
   visible focus outline on buttons and inputs.
8. Deliberately trigger a failure (tiny/corrupt image) and confirm the red
   failure panel still uses plain WHAT HAPPENED / WHY / NEXT STEP language,
   with the raw traceback only inside "Technical details".

**Report back**: screenshots of all four tabs at both window sizes, the
full test count from your machine, and anything that still looks
unfinished or visually wrong — CSS still targets Streamlit's `data-testid`
attributes (see the caveat at the top of `app/streamlit_app.py`), so name
your `streamlit --version` if something looks unstyled. Still holding on
GitHub/PPT/video until you confirm this pass.

## FINAL PRODUCT UI/UX PASS: registry/timeline/follow-up/reporting, no functional pipeline changes

Full rationale: `docs/ui_design.md` "Revision 3". Test count on this
machine before syncing: **96 collected, 83 executed and passed, 13
skipped** (the 13 are `test_streamlit_app_launch.py`'s real-Streamlit
`AppTest` smoke tests, which always skip in the cloud sandbox this project
is built in and are meant to run for real here — see step 0 below).
Two new unit-test files were added: `tests/unit/test_timeline.py` and
`tests/unit/test_failure_classification.py`, covering the two small,
additive, read-only backend helpers this pass required
(`src/routing/timeline.py`, `src/routing/failure_classification.py`) —
neither touches routing, validation, specialist dispatch, confidence, or
evidence computation; see their docstrings.

**0. Run the full suite for real** (`python -m unittest discover -s tests`
from the repo root) and confirm the real `AppTest`-based tests in
`test_streamlit_app_launch.py` now execute (not skip) and pass on your
machine — that is the one thing this sandbox categorically cannot verify.

**1. Sidebar.** Confirm the SYSTEM rail now says `REAL MODEL` (not just
`READY`) for VQA when SmolVLM is available on your machine, `CLASSICAL`
for grounding/change/fusion, and `UNAVAILABLE` for anything not installed.
Confirm a new `ACTIVE` section beneath it starts as all "—", and a new
`SESSION` section shows "Analyses run: 0". Expand "Technical system
status" and confirm the long tool-ID strings (e.g.
`tool_rs_vlm_adapted_v1` / `NOT_AVAILABLE_SANDBOX`) now read as normal
wrapped text, not character-by-character fragments — this was one of the
two real bugs from your screenshots; it's a best-effort fix I could not
verify without a real browser, so please confirm directly.

**2. File uploader.** On any tab without the sample checkbox, open the
upload widget and confirm it no longer shows duplicated "upload upload" /
"uploadUpload" text — the other real bug from your screenshots. Also
best-effort; please confirm.

**3. Per-mode input guidance.** On each tab, confirm one short caption
line appears above the inputs, an "About this analysis mode" expander
gives the fuller why/format/then breakdown, and 2–3 "Try:" example-query
buttons sit below the query field. Click one and confirm it fills the
query field WITHOUT running analysis (you still have to click "Run
analysis").

**4. Validation states.** On the change or fusion tab, upload only ONE of
the two required images and confirm a small `MISSING SECOND IMAGE` note
appears above the (still-disabled) Run button, distinct from the generic
`INSUFFICIENT INPUTS` note shown when nothing is uploaded at all.

**5. Run a real VQA query**, then:
   - confirm the run's result now shows a small `RUN <time> · <MODALITY>`
     badge above the answer;
   - confirm the panel below Confidence is now labeled "Active specialist"
     (same content as the old "Model / tool" panel);
   - expand the new "Agent process timeline" (collapsed by default) and
     confirm all seven steps (Input validated → Query interpreted →
     Specialist selected → Analysis executed → Evidence generated →
     Confidence computed → Result ready) show a done (✓) glyph;
   - scroll to "Ask a follow-up", type a second, different question about
     the same image (e.g. "Is there any visible water in this image?"),
     click Ask, and confirm: (a) you did NOT have to re-upload the image,
     (b) a new answer appears for the new question, (c) a "Conversation
     history (1)" expander appears showing your first question and a
     short excerpt of its answer;
   - confirm "Execution trace" is now labeled "Technical details" (same
     content: human-readable flow + raw JSON one level deeper).

**6. Trigger a real failure** (a 4×4 pixel image, or any file the
validator rejects) and confirm the red panel's label now reads
`Analysis failed · ERROR 400 · INVALID_IMAGE` (or the matching category
for your specific failure) above the existing WHAT HAPPENED / WHY / NEXT
STEP lines, with the raw traceback still confined to "Technical details".

**7. Bi-temporal change tab.** Run a real comparison and confirm you now
see two separate "Before" / "After" panels ABOVE the existing before|
after|highlighted composite image (labeled "Change map (before | after |
highlighted)").

**8. Export PDF** (requires `pip install reportlab` if it isn't already on
your machine — your earlier screenshot showed it wasn't) and open the
generated file. Confirm it has page numbers and a running header/footer,
and ten numbered sections: Analysis Summary, Input Data, Analysis Result,
Visual Evidence, Active Specialist, Confidence, Agent Execution Timeline,
Technical Trace, Limitations & Warnings, Export Metadata.

**9. Empty state.** Before running anything on a tab, confirm you see a
real "AWAITING INPUT — ..." message instead of a bare "No analysis run
yet." caption.

**10. Tab label.** Confirm the third tab now reads
"03 · BI-TEMPORAL CHANGE".

**Report back**: confirm items 0–10 above, plus screenshots of all four
tabs (including the new Agent process timeline expanded, the follow-up
conversation, and one 400-style failure panel), your real
`streamlit --version`, and the real test count from step 0. Still holding
on GitHub/PPT/video until you confirm this pass.

## Revision 4 checklist — FINAL UI REFINEMENT (visual-only pass)

This is the final visual pass on top of the Revision 3 checklist above —
no backend file changed, so every real-inference/real-baseline/real-report
behavior verified above is unchanged; this checklist only covers what
looks and behaves differently.

**0. Run the full test suite** (`python -m unittest discover`) and confirm
100 tests collected, 87 executed, 0 failed, 13 skipped (the 13 skips are
the real-Streamlit `AppTest` smoke tests gated on `streamlit` being
importable — they run for real on your machine, not in the cloud sandbox
this was built in).

**1. Header.** Confirm the header now shows a status badge to the right of
the title — "SYSTEM READY" if all four capabilities are available, or
"PARTIAL SYSTEM · N/4 READY" otherwise (this reads the real registry, so
if SmolVLM isn't installed it should still say READY, since the classical
VQA baseline keeps that capability available).

**2. Button colors (REQUIRED check).** Confirm "Run analysis" (all four
tabs) and "Ask" (follow-up) render in AMBER/gold. Confirm the "SUGGESTED
QUERIES" chips (was "TRY") and the "JSON" export button render in a
distinct cool muted cyan/blue outline — NOT the same amber. This is the
single most important visual check for this pass.

**3. Query experience order.** On any tab, confirm "SUGGESTED QUERIES"
(chips) now appears ABOVE "ASK YOUR QUESTION" (the query text field) —
reversed from the previous pass. Click a chip and confirm it still only
fills the field without running analysis.

**4. Section headers.** Confirm each tab's input column now shows a small
"01 · INPUT" style eyebrow with a one-line descriptor (e.g. "Single image
— Ask & interpret") instead of a bare "Input" label, and the result column
shows a plain "RESULT" eyebrow.

**5. Agent process timeline.** Expand it on a successful run and confirm
it now renders as a connected vertical line of dots (not a flat glyph
list) — completed steps in the cool accent color, with the same seven
real step labels as before. Trigger a real failure and confirm the failed
step's dot is red and later steps read "Not reached".

**6. Error panel + Return to analysis (REQUIRED check).** Trigger a real
failure (e.g. a 4×4 pixel image). Confirm the panel now reads as
"ERROR <code> · <CATEGORY>" / a plain-English title / a why line / a
"WHAT TO DO" label / the next-step line, and confirm a real "Return to
analysis" button appears below it. Click it and confirm the tab actually
resets to its empty state (not just a visual no-op) — you should be able
to start a fresh run on that tab immediately after.

**7. Conversation.** After a follow-up question, expand "Analysis session
(1 prior exchange)" and confirm it now reads as a "You: ... / SatQuery:
..." transcript rather than a Q1/A1 key-value table.

**8. Empty state.** Before running anything on a tab, confirm you see
"READY FOR ANALYSIS" with a short sentence on what to provide, plus the
four capability choices shown as small labels with the current tab's
choice highlighted.

**9. Export row.** Confirm "PDF report" now appears first/left and styled
as the stronger (amber) action, with "JSON" second/right in the muted
secondary style — reversed from the previous JSON-first, same-style
layout. Open the PDF and confirm it is unchanged in content (still the
same ten numbered sections).

**Report back**: confirm items 0–9 above, plus fresh screenshots of all
four tabs (empty state, a successful run, and one failure panel with
"Return to analysis" visible), and the real test count from step 0. Still
holding on GitHub/PPT/video until you confirm this pass — per the
project's standing rule, this is stated as the final visual refinement,
not "minor polish remaining."

## What comes after (not yet — for reference only)

1. Acquire a small, legitimate VRSBench subset and print its actual schema
   before writing any adaptation code against assumed column names.
2. The smallest real LoRA experiment, with a fixed seed, a saved checkpoint,
   real training loss, elapsed time, and `run_metadata.json`.
3. Compare base vs. adapted model on the same real examples before the
   registry entry for either model changes from `NOT_AVAILABLE_SANDBOX`.

Each of these gets its own script and its own report, the same way the first
two did — nothing here should be assumed working ahead of a real, measured
run.
