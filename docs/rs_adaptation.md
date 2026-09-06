# Remote-Sensing Adaptation: Status, Model, Dataset, and Reproducible Script

**Status: NOT EXECUTED. NOT TRAINED. NOT VERIFIED.**

## Section 8: SUPERSEDED — model choice revised for real hardware (Phase 2B)

Everything below this notice (sections 1-7) proposed `google/paligemma2-3b-pt-448`
as the base model. That plan is **superseded**, not deleted, because the
reasoning trail matters: sections 1-6 were written and approved before the
real target machine's hardware was measured. Once it was (real test, not
assumed): no NVIDIA GPU, no CUDA, 8.24GB total system RAM, Intel Iris Xe
integrated graphics only — a 3B-parameter model doesn't fit in memory for
training, full stop, independent of speed.

**Revised, approved model stack**:

- **Single-image VQA**: `HuggingFaceTB/SmolVLM-256M-Instruct` (~0.3B params,
  Apache 2.0, ungated) — small enough that a real LoRA/full fine-tune on this
  exact machine is a genuine multi-hour CPU job, not a fantasy.
- **Grounding**: `microsoft/Florence-2-base` (0.23B params, MIT, ungated) —
  has *native* `<CAPTION_TO_PHRASE_GROUNDING>` / `<OD>` task tokens, so it
  produces real bounding boxes without training being a precondition for a
  working answer at all.
- **Bi-temporal change** and **optical+SAR fusion**: keep the existing v0
  classical-CV feature extraction, add a genuinely trained lightweight
  classifier (scikit-learn logistic regression / small MLP) on top of the
  existing features, replacing the hand-tuned threshold / unsupervised
  clustering with something actually fit to data. No viable small pretrained
  open-source model exists for either task to just download instead.
- **Routing**: unchanged.

This keeps the PS-mandated principle intact — real pretrained models,
genuinely adapted, not a generic-VLM-only or classical-CV-only system — just
sized to hardware that was verified rather than assumed. Sections 1-7 remain
below as the historical record of the original (now superseded) plan and its
reasoning, not as the current status.

**FIRST SUCCESS TEST — VERIFIED**, on the real target machine (not this
session's sandbox), via `scripts/first_success_test_vqa.py`:
`HuggingFaceTB/SmolVLM-256M-Instruct`, CPU backend, model load 46.891s,
inference 38.457s, peak working set 2882.7MB, on a real aerial photograph of
farmland — output: *"The dominant land cover or feature in this image is
farmland."* This is real single-image VQA genuinely executing on the target
hardware. It is not yet adapted (no fine-tuning has happened) — this only
proves the base model loads and runs here at all.

**SECOND SUCCESS TEST — in progress**: real model-based grounding via
`microsoft/Florence-2-base`'s native `<CAPTION_TO_PHRASE_GROUNDING>` task,
using `scripts/second_success_test_grounding.py` (not the classical
`src/grounding/grounding.py` baseline, which is untouched). Requires
`einops` + `timm` on top of the FIRST SUCCESS TEST's dependencies.

Real hardware run hit a second, different CPU-compatibility issue beyond the
`flash_attn` one (both now handled in one place, `src/models/
florence2_compat.py`, rather than inline in the test script):

- **`flash_attn` import requirement** — Florence-2's modeling file lists it
  as required even when not requested. Sourced fix:
  [microsoft/Florence-2-base discussion #4](https://huggingface.co/microsoft/Florence-2-base/discussions/4).
- **`AttributeError: '...ForConditionalGeneration' object has no attribute
  '_supports_sdpa'`** — hit for real on the target machine (torch 2.14.0+cpu,
  transformers 4.57.1). This is a confirmed, widely-reported transformers-side
  regression, not a Florence-2 logic bug or something specific to this
  project: transformers' own `modeling_utils.py` checks `self._supports_sdpa`
  while resolving the attention implementation, and newer transformers no
  longer provides the default value on `PreTrainedModel` that Florence-2's
  custom code (written against an older transformers version) expects to
  inherit. Confirmed via multiple independent reports:
  [transformers#39974](https://github.com/huggingface/transformers/issues/39974),
  [transformers#41622](https://github.com/huggingface/transformers/issues/41622),
  [transformers#36886](https://github.com/huggingface/transformers/issues/36886)
  ("Florence2 stopped working after upgrade to 4.50.0").
  Fix applied, in order (only as much as necessary — no edits to Florence-2's
  own modeling file, unlike some community patches for unrelated bugs):
  first try `attn_implementation="eager"` plain; only if that specific
  `_supports_sdpa` AttributeError still occurs, restore the missing default
  by setting `transformers.modeling_utils.PreTrainedModel._supports_sdpa =
  True` (honestly true — CPU genuinely supports scaled-dot-product attention
  in modern PyTorch — this restores a default the library used to provide,
  it does not claim a false capability) and retry. Implemented once in
  `src/models/florence2_compat.py::load_florence2()`, used by both the test
  script and `tests/integration/test_florence2_compat.py`, which is a new
  regression test guarding specifically against this exact failure
  recurring. The script runs two different target phrases on the same image
and reports whether the returned boxes differ, as direct evidence the output
is a function of the model's forward pass rather than a fixed value.

**Second real-hardware failure and its fix — `RecursionError`, root-caused,
not patched around**: applying the `_supports_sdpa` fix above produced a
*different* failure on the target machine: `RecursionError: maximum
recursion depth exceeded while calling a Python object`. This was
investigated properly rather than guessed at a third time:

- Multiple sourced hypotheses (a transformers-version mismatch broader than
  `_supports_sdpa` alone; pinning `transformers==4.49.0`, the last release
  before that regression, reported working by several independent users in
  [microsoft/Florence-2-large-ft discussion #40](https://huggingface.co/microsoft/Florence-2-large-ft/discussions/40))
  were researched first, but none was adopted without verification, per the
  explicit instruction not to add another blind monkeypatch.
- The actual cause was found by direct reproduction: transformers/torch were
  stubbed out (neither installs in this development sandbox — no egress to
  pypi.org here) and `src/models/florence2_compat.py`'s own flash_attn
  compatibility shim (`_fixed_get_imports`) was exercised the way real
  `trust_remote_code=True` loading exercises it. It recursed immediately.
  Root cause: the shim did `from transformers.dynamic_module_utils import
  get_imports` **inside its own body**, but the shim is itself installed, via
  `unittest.mock.patch`, as `transformers.dynamic_module_utils.get_imports`
  for the duration of the load attempt — so that internal re-import rebound
  the local name to the shim itself, and calling it called itself, forever.
  This was a real bug in this project's own code (introduced when the
  flash_attn fix was consolidated into `florence2_compat.py` earlier in this
  phase), not a transformers-version incompatibility and not anything to do
  with `_supports_sdpa` itself — the `_supports_sdpa` retry only happened to
  be what made the shim run a second time.
- **Fix**: capture the real `get_imports` once, before any patch is
  installed on that attribute, and have the shim always call that captured
  reference rather than re-resolving the (possibly-patched) module
  attribute. One-line change, no dependency/version/environment change, and
  the `_supports_sdpa` tier-2 patch is unchanged and untouched.
- Verified by direct reproduction of both the failure (with the old code)
  and the fix (with the new code) using stubbed transformers/torch, then by
  the full project test suite (`python -m unittest discover -s tests`,
  41 tests, 6 skipped honestly for missing ML dependencies in this sandbox,
  0 failures) — not yet by an actual run on the target machine, which is the
  next step.
- New regression tests in `tests/integration/test_florence2_compat.py`
  (`TestFixedGetImportsNoRecursion`) reproduce the exact failing call
  pattern (the shim entered twice — once for the initial load, once for the
  `_supports_sdpa` retry, the same shape as the real failure) and assert it
  completes without recursing, so this exact bug cannot silently regress.
  `TestUnrecoverableFailureReporting` guards the related diagnostic gap: any
  load failure must carry the full original traceback (`Florence2LoadError.
  full_traceback`), not just an exception message — this is what made the
  RecursionError hard to pin down from the JSON result alone in the first
  place, since neither `florence2_compat.py` nor
  `scripts/second_success_test_grounding.py` previously captured more than
  `str(exc)`. Both now do (`result["traceback"]` in the script's JSON
  output).

**Third real-hardware milestone and third failure — model load VERIFIED,
generation fails separately**: the `RecursionError` fix above was confirmed
correct on real hardware: `model_load_success = true`, load 9.636s, CPU
backend, RAM after load 1220.9MB, peak working set 1663.0MB (transformers
4.57.1, torch 2.14.0+cpu). Generation itself then failed with a
**different, separately confirmed** issue: `AttributeError: 'NoneType'
object has no attribute 'shape'`, traceback ending in Florence-2's own
`prepare_inputs_for_generation`, at `past_key_values[0][0].shape[2]`.

- Confirmed as an active, documented Florence-2 compatibility problem, not
  something specific to this project:
  [Rusted-Gold/Florence-2-large-ft-Transformers-Fix](https://huggingface.co/Rusted-Gold/Florence-2-large-ft-Transformers-Fix)
  (a community fork patching `past_key_value[0] is not None` guards into
  several attention-layer locations for this exact failure shape) and
  [ComfyUI_LayerStyle#575](https://github.com/chflame163/ComfyUI_LayerStyle/issues/575)
  (same exact error reported independently).
- Root cause context, confirmed via Florence-2's own source and
  HuggingFace's own documented deprecation: starting with transformers
  4.50, `PreTrainedModel` no longer automatically inherits `GenerationMixin`
  - a model class needs it explicitly for `.generate()`'s modern
  cache-handling setup to work correctly. Multiple Florence-2 HF repos
  needed exactly this fix added, in TWO places - the inner
  `Florence2LanguageForConditionalGeneration` class
  ([microsoft/Florence-2-base discussion #22](https://huggingface.co/microsoft/Florence-2-base/discussions/22),
  confirmed merged into the exact repo this project uses) and the OUTER
  `Florence2ForConditionalGeneration` wrapper class - the class actually
  returned by `AutoModelForCausalLM.from_pretrained(...,
  trust_remote_code=True)` and that `.generate()` is called on here
  ([microsoft/Florence-2-large-ft commit 3502d39](https://huggingface.co/microsoft/Florence-2-large-ft/commit/3502d39104483126d97ce784a9b3f70fd445c025)).
  Whether `microsoft/Florence-2-base`'s outer class has received the same
  fix could not be confirmed from this sandbox - the file is too large for
  the available fetch tooling to read past roughly its first third, and the
  outer class is defined near the end (referenced line numbers in the
  community fix go past 2800).
- Rather than guess which single fix applies, `src/models/
  florence2_compat.py` gained `diagnose_generation_capability()` (reports
  the loaded model's exact class, full MRO, and whether `GenerationMixin`
  is actually present - real evidence instead of an assumption either way)
  and `generate_florence2()`, which tries an ORDERED list of documented,
  standard `generate()` keyword-argument combinations - the requested
  config, then plain greedy decoding (`num_beams=1`, tried before anything
  else per the explicit instruction), then `use_cache=False` (which keeps
  `past_key_values` `None` on every step, so the crashing line's guard is
  never entered regardless of the underlying cache-object mechanism), then
  both together - and reports exactly which one worked, or every attempt's
  full traceback if none did. No edits to Florence-2's own generation code.
- New regression tests in `tests/integration/test_florence2_compat.py`
  (`TestGenerationCompatibility`) verify this fallback mechanism
  deterministically, by mocking `model.generate()`'s failure/success under
  specific keyword-argument combinations - meaningful regardless of which
  configuration the real model turns out to need.
- Verified so far: the fallback mechanism's own logic, by direct
  reproduction with stubbed transformers/torch, and the full project test
  suite (45 tests, 10 skipped honestly for missing ML dependencies in this
  sandbox, 0 failures). NOT yet verified: which candidate configuration
  (or whether any of them) actually resolves the failure on the real
  target machine - that is the next real-hardware run, and this document
  will be updated with the actual answer once it's reported back, not
  before.

**Fourth real-hardware result: generation VERIFIED, grounding QUALITY NOT
verified**: on real hardware, load (7.363s) and generation for both phrases
completed with real `<loc_*>` tokens and a real overlay image - the
generation-compatibility fix above worked. But `"farmland"` and
`"buildings"` returned the IDENTICAL near-full-image box
(`[1.6949999, 0.2259999, 676.9829, 451.3219]`,
`boxes_differ_between_phrases = false`). Per the explicit instruction, this
is **not** treated as grounding working - "the model produced `<loc>`
tokens" is not the same claim as "the model produced genuine, phrase-
distinguishing localization," and only the second one is what the PS
requirement needs.

Investigation is not yet complete - `scripts/diagnose_grounding_quality.py`
was written to distinguish three real possibilities without guessing which
is true: (A) our own invocation/preprocessing, (B) our post-processing/
coordinate-scaling, or (C) a genuine Florence-2-base/domain limitation on
this kind of imagery. It runs the exact official Florence-2 model-card
invocation first, `<OD>` and `<DENSE_REGION_CAPTION>` (no phrase - the
official baseline for whether the model can localize anything in this
specific image at all) before any phrase-grounding call, then both bare-
word (`"farmland"`, `"buildings"`) and caption-style
(`"a large area of farmland"`, `"a cluster of buildings"`) phrasings of the
same two concepts - because the official model-card example itself uses a
caption-style phrase, not a bare noun. It records raw (pre-parsing)
generated text and each box's fraction of total image area for every
experiment. Verified so far only by stub-based dry run (transformers/torch
mocked, since neither installs in this sandbox) confirming the script's own
logic (JSON structure, area-fraction math, overlay drawing) is correct
given a scenario mirroring the reported bug - not yet run against the real
model. That real run, and its answer, is the next step.

**Diagnostic verdict: Florence-2-base classified as a MODEL/DOMAIN
limitation, not a software bug** - real-hardware results from
`diagnose_grounding_quality.py`: raw generated text genuinely changes with
the requested phrase (rules out "the phrase is silently ignored" - an
invocation bug), but every grounding box collapses to ~99% of the image
regardless of phrase, and even `<OD>` (plain object detection, no phrase at
all - the official baseline for "can this model find anything in this
image") produces a ~99.6% full-image box, and `<DENSE_REGION_CAPTION>`
finds nothing. This is accepted as root cause C, per the explicit
instruction: STOP debugging Florence-2 - no more monkeypatches, no box
manipulation, no claiming grounding works because tokens were generated.

**Grounding model swap: candidate comparison and selection.** Researched
(not assumed) before picking one:

1. *Remote-sensing-specific SMALL grounding model* - **not available in
   practice**. Current RS visual-grounding research
   ([GeoGround](https://github.com/VisionXLab/GeoGround),
   [GeoPixel](https://arxiv.org/html/2501.13925v1)) is built on 7B+ VLM
   backbones - structurally infeasible on CPU/8GB, and not available as a
   small, ready pretrained checkpoint.
2. *Lightweight RS segmentation/detection model* - real small models exist
   (DOTA-style aerial object detectors: planes/ships/vehicles), but their
   fixed classes don't include land-cover concepts like "farmland" or a
   generic "buildings" class at all - a vocabulary mismatch with this PS's
   query style, not a quality gap. Land-cover segmentation datasets with
   the right classes exist
   ([LandCover.ai](https://arxiv.org/abs/2005.02264) has "buildings";
   DeepGlobe has "agriculture"), but no well-documented, general-purpose
   pretrained checkpoint for either was found on Hugging Face - only
   dataset uploads and one narrow, single-city community checkpoint
   ([Pranilllllll/segformer-satellite-segementation](https://huggingface.co/Pranilllllll/segformer-satellite-segementation),
   trained on ~400 Kathmandu Valley tiles with unrelated classes:
   background/residential/road/river/forest/unused-land). Doing this
   properly means training a small segmentation head ourselves on
   LandCover.ai or DeepGlobe - real, legitimate future work, but it is
   dataset acquisition + training (closer to the LoRA/adaptation phase),
   not today's fastest path.
3. *Classical CV + explicitly-labeled ML baseline* - still available
   (`src/grounding/grounding.py`), untouched, as the honest fallback.

**Picked: `IDEA-Research/grounding-dino-tiny`** (Apache-2.0, ~0.2B params) -
outside the strict three-tier list above (it is a real foundation model,
not classical CV, but also not remote-sensing-specific), chosen because it
is the most defensible option actually available today:

| dimension | grounding-dino-tiny |
|---|---|
| size | ~0.2B params (comparable to Florence-2-base) |
| license | Apache-2.0 |
| CPU feasibility | natively supported by `transformers` (`AutoModelForZeroShotObjectDetection`), **no `trust_remote_code`, no custom modeling file** - avoids repeating the flash_attn/`_supports_sdpa`/RecursionError/generation-cache debugging this project just went through with Florence-2 |
| RS-domain suitability | **none** - general web-image object detector, not trained on satellite/aerial imagery; explicitly labeled as such in the script's output (`model_is_remote_sensing_specific: false`) so this is never mistaken for RS-domain adaptation |
| availability | ready pretrained checkpoint on Hugging Face, official transformers docs example |
| expected inference speed | not yet measured on this hardware; real user reports (e.g. [transformers#31533](https://github.com/huggingface/transformers/issues/31533), [IDEA-Research/GroundingDINO#31](https://github.com/IDEA-Research/GroundingDINO/issues/31)) confirm CPU use is common, in the same rough seconds-per-image range already measured for Florence-2/SmolVLM on this machine - to be confirmed by measurement, not assumed |
| output type | real boxes + scores + text labels per phrase, directly from a single forward pass (`model(**inputs)`, not autoregressive generation - no cache/generation-config class of bug is even possible here) |

Proof test: `scripts/third_success_test_grounding_dino.py` - runs
`"farmland."` / `"buildings."` (GroundingDINO's documented lowercase +
trailing-period convention) plus a control phrase (`"an airplane."`,
expected to find nothing or low-confidence detections, as a sanity check
against a fixed/default-box failure mode) on the same real image, using the
exact official transformers usage pattern. Records real boxes, scores,
per-box area-of-image fraction, and inference time per phrase - success
is read from genuinely different, non-full-image boxes, not asserted.
Verified so far only by stub-based dry run (transformers/torch mocked,
covering both the success path and a model-load-failure path) confirming
the script's own logic is correct - not yet run against the real model,
which is the next step. Needs no new dependencies beyond what's already
installed for the earlier tests.

**Real-hardware result: model load VERIFIED, then a processor input-format
bug, now fixed.** `grounding-dino-tiny` loaded successfully on real
hardware: `model_load_success = true`, load time 96.626s, CPU backend,
peak working set 739.1MB, weights downloaded from Hugging Face. The test
then failed **before any inference ran**, at
`processor(images=image, text=text_labels, return_tensors="pt")`, with:

```
TypeError: TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
```

This was a tokenizer-level type error (from the underlying Rust
`tokenizers` library), not a model or environment problem - confirming it
was specifically about the *shape* of the `text` argument. The script had
been sending a list-of-lists (`text=[[formatted]]`), matching a convenience
input shape shown in transformers' "main"-branch documentation for
GroundingDINO (`text=[["a cat", "a remote control"]]`). That shape either
isn't supported by the actual pinned `transformers==4.57.1`, or is handled
differently than the dev docs describe - the ground-truth runtime error is
what settled it, not further speculation about which documentation to
trust (this project's WebFetch lookups against large or version-pinned
source files had already proven unreliable more than once earlier in this
build, for both Florence-2's `modeling_florence2.py` and an attempt to
re-verify against the v4.57.1-tagged GroundingDINO processor source).

**Fix**: `_format_grounding_dino_prompt()` in
`scripts/third_success_test_grounding_dino.py` now returns a single,
already period-terminated, lowercase **plain string** (e.g.
`"farmland."`), and the main loop passes it directly as `text=formatted` -
no list, no list-of-lists. A plain string is the one input shape every
transformers version that ships `GroundingDinoProcessor` is documented to
accept unambiguously (it is exactly the `TextInputSequence` named in the
tokenizer's own error message), and since this script only ever sends ONE
candidate phrase per forward call, the list-of-lists batching shape was
never actually needed. No model change, no transformers version change,
no monkeypatch, no hardcoded detections, and the classical grounding
baseline remains untouched, per the explicit constraints for this fix.
Verified via a stub processor that reproduces the exact TypeError for a
nested-list `text` argument and accepts a plain string, matching the real
observed contract (`tests/integration/test_third_success_test_grounding_dino.py`,
8 tests, all passing) - plus unit tests on `_format_grounding_dino_prompt`
itself (lowercasing, trailing-period handling, whitespace stripping).

**FINAL REAL-HARDWARE RESULT, and GROUNDING INVESTIGATION CLOSED.** After
the input-format fix, `grounding-dino-tiny` ran end to end for real on
`C:\path\to\your_image.jpg`: model load
succeeded, inference succeeded for all three phrases, and real boxes/
scores were returned - no crash, no exception, genuine forward passes.
But the returned boxes are not spatially meaningful:

| phrase | box area (% of image) |
|---|---|
| farmland | 99.09% |
| buildings | 99.18% |
| an airplane (negative control) | 98.82% |

All three - including the negative control, which should have found
nothing or a small low-confidence region - collapse to essentially the
full image, the same degenerate pattern Florence-2-base showed. This rules
out an invocation/input-format bug (that class of bug is now fixed and
verified separately, above) and points to the same underlying cause as
Florence-2: **a general-purpose, web-image-trained open-vocabulary
detector applied to satellite/aerial imagery it was never trained on
tends to fall back to "the whole scene is the object," regardless of
which foundation model is used.**

**Decision (accepted): do NOT use either Florence-2-base or
grounding-dino-tiny as the primary satellite grounding capability.** No
further time will be spent debugging generic foundation-model grounding on
this imagery. The existing classical baseline, `tool_grounding_v0`
(`src/grounding/grounding.py` - HSV colour thresholding + contour
extraction, already `AVAILABLE` in `src/models/registry.py`, already
covered by `tests/integration/test_specialists.py::TestGroundingSpecialist`)
remains the MVP's grounding capability, explicitly and consistently
labeled everywhere as a `v0_classical` baseline - never as a foundation-
model or RS-adapted result. Both Florence-2 and grounding-dino-tiny's real
results are kept in this document and in `docs/RUN_ON_WINDOWS.md` as
genuine engineering evidence (two independent foundation models tested,
both real code paths verified working, both shown to be unreliable on
this specific domain) - they are not deleted or hidden, but they are not
shipped as a working grounding feature either.

## 9. Bi-temporal change: real deep-learning upgrade evaluated and declined

Following the grounding closure, the same question was asked for
bi-temporal change: is there a real, CPU-feasible deep-learning model
worth adding on top of `tool_change_v0` (the classical grayscale-
differencing + Otsu-threshold baseline, already `AVAILABLE` and tested -
see section 4 of `docs/milestone_report_phase1_2.md`)? Two candidates were
found on Hugging Face:

| dimension | `DarthReca/actu-change-detection` | `deepang/adaptformer-LEVIR-CD` |
|---|---|---|
| params | ~200M | 12.5M |
| license | OpenRAIL | MIT |
| loading | `trust_remote_code=True` (custom ConvNeXtV2+ConvLSTM modeling code) | `trust_remote_code=True` (custom AdaptFormer modeling code) |
| domain scope | water only (MNDWI-difference-based); needs Sentinel-2 time series + optional DEM/climate inputs - a much more complex input pipeline than a simple before/after pair | building change only (LEVIR-CD dataset); accepts a plain before/after image pair |
| CPU feasibility | not confirmed; GPU strongly implied by architecture/size | not confirmed either way, but small enough (12.5M params, smaller than grounding-dino-tiny) to be plausible |

Both require `trust_remote_code=True` - the exact bug class that cost
three separate debugging rounds on Florence-2 in this project
(RecursionError, `_supports_sdpa` AttributeError, and the generation-time
`past_key_values` AttributeError, all documented above). No model was
found that does general-purpose bi-temporal change detection through
`transformers`' native, non-custom-code path the way `grounding-dino-tiny`
did for grounding - change detection is enough of a niche task that every
ready checkpoint found ships its own modeling code.

**Decision (explicitly made by the project owner, not assumed): do NOT
pursue either candidate.** `tool_change_v0` remains the shipped bi-temporal
change capability - already built, already `AVAILABLE`, already tested
(`tests/integration/test_specialists.py::TestChangeSpecialist`), and
already honestly labeled in its own answer text as "v0 baseline: grayscale
image differencing + Otsu threshold - a standard classical
change-detection technique, not a learned CDVQA model." No further time
will be spent evaluating or debugging deep-learning change-detection
models unless this decision is revisited.

## 10. Optical+SAR fusion: real deep-learning upgrade evaluated and declined

Same question, asked for `tool_fusion_v0` (the classical joint-KMeans-
clustering baseline over a hand-stacked `[R, G, B, SAR]` feature vector,
already `AVAILABLE` and tested - `TestFusionSpecialist` in
`tests/integration/test_specialists.py`, including the dedicated test
proving the SAR channel actually changes the output). Researched under an
explicit hard time/scope limit ("only practical small models for our 8GB
Intel Iris Xe CPU-only machine, 2-3 candidates, do not spend time on large
research models"):

| dimension | `allenai/satlas-pretrain` | `Yusin2Chen/SARoptical_fusion` | Clay Foundation Model v1.5 |
|---|---|---|---|
| model size | not confirmed per-backbone from the model card; multiple backbone sizes offered | not stated | 632M total / 311M encoder (~1.25GB on disk) |
| license | ODC-BY | MIT | Apache-2.0 (code + weights) |
| RS relevance | genuine RS foundation model, but trains **separate** Sentinel-2 and Sentinel-1 backbones - no fusion head exists out of the box | purpose-built for exactly this (SAR+optical contrastive fusion), trained on a DFC2020 subset | genuine, explicitly multi-sensor (S1+S2+Landsat+others) joint embeddings in one forward pass |
| data availability | pretrained weights published; Sentinel-1/2 inputs freely available via Copernicus | **no pretrained weights published at all** - only training scripts (`train_DCCA.py`, `train_Efusion.py`); would require training from scratch, real dataset acquisition, real training run | pretrained weights published (Hugging Face); Sentinel-1/2 freely available |
| CPU feasibility | unconfirmed, and moot given the implementation-difficulty problem below | unconfirmed; disqualified regardless (no weights) | unconfirmed; 311M-parameter encoder is plausibly CPU-feasible per this project's own precedent (Florence-2-base's 0.23B ran at 1.6GB peak RAM), but see below |
| implementation difficulty | would require designing and likely training a fusion head ourselves to combine the two separate backbones' features - real modeling work, not a load-and-run swap | disqualified: no checkpoint to load at all | it is a **masked autoencoder** - out of the box it only produces embeddings, not a task answer (built-up/water classification); getting a usable output requires fine-tuning a downstream head on labeled data - real training work, and loading it needs a separate "Lightning"-based framework from the official repo, not native `transformers` |

**None of the three is a genuine load-and-run pretrained model that
produces fused, task-relevant optical+SAR output on CPU without further
training** - unlike `grounding-dino-tiny` for grounding or SmolVLM for
VQA, where a single `from_pretrained()` call plus a forward pass produced
a real, directly usable answer. All three would require either building
new modeling code (a fusion head) or actually training/fine-tuning on a
real, separately-acquired dataset - both explicitly out of scope for a
"small model swap" and squarely in LoRA/adaptation-phase territory
instead.

**Decision (explicitly made by the project owner): reject all three
candidates.** `tool_fusion_v0` remains the shipped optical+SAR fusion
capability - already built, already `AVAILABLE`, already tested, and
already honestly labeled as a `v0_classical` joint-clustering baseline. No
further time will be spent evaluating or debugging deep-learning fusion
models unless this decision is revisited.

## 11. CORE CAPABILITY FREEZE, and SmolVLM wired into the live app

The specialist strategy was declared FINAL by the project owner: VQA real
(SmolVLM), grounding/change/fusion classical (foundation-model attempts
documented above and rejected), router/evidence/confidence/trace/export
already built in Phase 1/2. Before productizing this into a UI, a real gap
was found and fixed: **SmolVLM had only ever been run in the standalone
`scripts/first_success_test_vqa.py` proof script - it was never wired into
`src/models/registry.py`, `src/routing/router.py`, or `app/cli.py` as the
actual specialist that executes for `single_image_vqa` queries.** The live
app was still calling `tool_single_image_vqa_v0` (the classical KMeans
baseline) for every VQA query, even though the freeze declared SmolVLM
shipped. This was caught and fixed here, not shipped as a UI that would
have silently mislabeled its own model-disclosure panel.

**Fix - real integration, not a relabeling:**

- `src/specialists/vqa_smolvlm.py` (new): wraps the EXACT verified loading/
  generation recipe from `scripts/first_success_test_vqa.py` (same model
  id, dtype, `_attn_implementation="eager"`) into the `SpecialistOutput`
  schema. Caches the loaded model per-process (real load time is ~47s -
  reloading per query would make the UI unusable); a cache hit honestly
  reports `model_was_already_loaded: true` and a near-zero
  `model_load_seconds` for that call, rather than fabricating a constant
  "always ~47s." Confidence's `base_signal` is the mean max-softmax
  probability across the generated tokens (from
  `model.generate(..., output_scores=True, return_dict_in_generate=True)`)
  - a real, model-native signal computed from that specific answer's own
  generation, the same spirit as using a classical specialist's
  silhouette/separation score.
- `src/specialists/vqa_dispatch.py` (new): the ONLY module the CLI/UI call
  for `single_image_vqa`. Checks the registry's live availability status
  for SmolVLM; if unavailable, goes straight to the classical baseline (no
  attempt, no fallback note - there's nothing to disclose a fallback
  from). If available, tries SmolVLM; on ANY failure (load or inference),
  falls back to the classical baseline but stamps
  `raw["smolvlm_attempted"]`, `raw["smolvlm_fallback_reason"]`, and
  `raw["smolvlm_fallback_traceback"]` with the real cause, and prefixes
  `answer_text` with an explicit disclosure - never a silent swap.
- `src/models/registry.py`: new entry `tool_single_image_vqa_smolvlm_v1`,
  with `status` set by a REAL dependency check (`import torch; import
  transformers`) at bootstrap time, not hardcoded - `AVAILABLE` wherever
  those are importable, `NOT_AVAILABLE_SANDBOX` otherwise (e.g. in this
  cloud development sandbox, which is why this repo's own test/CI runs
  always exercise the classical fallback path, honestly).
- `src/routing/router.py`: `_decide_single()` now calls
  `_select_vqa_tool_name()` for `single_image_vqa`, which queries the
  registry's live status and picks `tool_single_image_vqa_smolvlm_v1` only
  when it's genuinely `AVAILABLE` - the router's own `reasoning` list says
  which backend was picked and why, so the trace panel shows this
  decision, not just the final tool name.
- `app/cli.py`: now calls `specialists.vqa_dispatch.run()` for
  `single_image_vqa` instead of the classical module directly.

**Verification**: `tests/integration/test_vqa_dispatch.py` (8 new tests,
all passing) mocks the registry lookup directly - not real torch/
transformers - to deterministically exercise all four real branches
(registry entry missing, status not available, SmolVLM available and
succeeds, SmolVLM available but raises) without depending on this sandbox
having ML dependencies installed; this is the same reasoning as
`test_florence2_compat.py`'s `TestGenerationCompatibility`, which mocks
the model call rather than the whole environment. `vqa_smolvlm.py`'s own
load/generate/confidence logic was additionally verified via a stub-based
reproduction (fake torch/transformers/PIL modules mimicking the real
APIs' shapes) confirming prompt-trimming, per-process caching, and the
mean-token-confidence computation are all correct before trusting them on
real hardware - the same methodology used throughout this project for
Florence-2 and grounding-dino-tiny. Full suite: 61/61 passing (was 53; +8),
0 regressions. **Not yet run against the real model on real hardware** -
that is the explicit next step before further UI work, per the project
owner's instruction.

**Real-hardware result: SUCCESS, with one honest bug caught before UI
work started.** The integrated CLI ran for real: router selected
`tool_single_image_vqa_smolvlm_v1`, SmolVLM-256M-Instruct produced a real
answer ("The dominant land cover or feature in this image is farmland.")
in 47.082s real CPU inference, evidence and trace written. But the trace
showed `Confidence: 0.71 (method=v0_classical)` for a SmolVLM-produced
answer - `compute()` had no way for a caller to declare its own
methodology label, so every specialist's confidence, real model or
classical alike, was unconditionally stamped `"v0_classical"`. This is
exactly the confusion `docs/confidence.md`'s original "Known limitation"
section was written to prevent, just inverted: a real neural network's
output being mislabeled as a classical heuristic.

**Fix**: `confidence_engine.compute()` now takes `method_version` and
`basis_description` parameters (defaulted to the classical values, so the
four existing classical specialist call sites needed no changes).
`vqa_smolvlm.py` passes `method_version=METHOD_VLM_TOKEN_CONFIDENCE`
(`"v1_vlm_mean_token_probability"`) and a `basis_description` documenting
exactly what that number means: the mean top-token softmax probability
across the answer's generated tokens - a real signal, computed from that
specific generation, but one that measures generation certainty, not
factual correctness (a fluent wrong answer can still score high). Per the
explicit instruction, no confidence number is invented or fabricated -
this is the same honest mean-token-confidence signal already computed in
the original integration, now correctly labeled instead of mislabeled.
`SpecialistOutput` also gained three explicit fields so a UI never has to
infer methodology or fallback status from `tool_name` string-matching or
from which keys happen to exist in `raw`: `model_name` (the real model's
identity, or `None` for classical techniques), `fallback_occurred`, and
`fallback_reason` - all always present. `vqa_dispatch.py`'s fallback path
now sets these explicitly (in addition to the existing `raw` fields kept
for the technical-details panel). `src/export/report.py`'s PDF now
renders Specialist/Model/Fallback and the confidence `basis_description`
too, so the exported audit report carries the same honesty as the live
trace. Full account and the exact schema fields: `docs/confidence.md`.

**Verification**: `tests/unit/test_confidence.py` gained 2 tests
(overriding method_version/basis_description; confirming the classical
default). `tests/integration/test_vqa_dispatch.py` was extended so every
existing scenario (classical-direct, SmolVLM-succeeds, SmolVLM-fails) now
also asserts `model_name`, `fallback_occurred`, `fallback_reason`, and
`confidence.method_version` are all correct and mutually consistent - in
particular that a fallback to classical carries `method_version:
"v0_classical"` (never left over as the VLM's label) and that a genuine
SmolVLM success never carries `"v0_classical"`. `vqa_smolvlm.py`'s own
schema-field population was additionally verified via the same
stub-based-reproduction methodology used throughout this project. Full
suite: 63/63 passing (was 61; +2), 0 regressions. **Not yet re-run against
the real model on real hardware after this fix** - that is the next step.

## 12. CORE CAPABILITY FREEZE declared, and the Streamlit UI built

With the confidence-methodology fix verified on real hardware (section 11),
the specialist strategy was declared FINAL and frozen: VQA →
`tool_single_image_vqa_smolvlm_v1` (real SmolVLM, disclosed classical
fallback); grounding → `tool_grounding_v0` (classical, Florence-2/
GroundingDINO rejected per section 8); bi-temporal change → `tool_change_v0`
(classical, alternatives rejected per section 9); optical+SAR fusion →
`tool_fusion_v0` (classical, three candidates rejected per section 10). No
further model shopping for any of these four. Router/evidence/confidence/
trace/export were already real. Instruction: proceed to UI productization
around this frozen core, without rewriting the backend.

**Shared-pipeline extraction.** `app/cli.py:run(args)`'s orchestration body
(load → route → validate → dispatch → trace) was extracted verbatim into
`app/pipeline.py:run_query(...)`, replacing `argparse.Namespace` attribute
access with plain keyword parameters. `cli.py`'s `run()` became a three-line
wrapper. This was done so the new Streamlit UI and the existing CLI call
the *same* function - never two independently-maintained copies of routing/
validation/dispatch logic. Verified safe via `grep -rn "cli\.run\|app\.cli"
tests/ scripts/` returning no hits (nothing depended on `cli.run`'s internal
structure), then verified behaviorally by re-running all four capability
branches through `app/cli.py` after the extraction and diffing output
against pre-extraction runs - identical task routing, answers, confidence
values, and evidence paths in every branch.

**UI design process.** Per the explicit "use the available design skills"
instruction, `taste-skill:brutalist-skill` (dark "Tactical Telemetry"
archetype - matches the brief's own "analyst console / aerospace tooling /
restrained dark interface" language) and `frontend-design:frontend-design`
(anti-genericism guidance, plan-then-critique process) were invoked and
their guidance was synthesized into a concrete, written design-token plan
*before* any UI code was written - see `docs/ui_design.md` for the full
color/type/layout plan and the reasoning for rejecting the other available
design skills (image-generation-only skills, `stitch-skill`, `soft-skill`,
etc.) for this brief.

**The fusion evidence-shape question.** The frozen UI spec asks for an
OPTICAL / SAR / FUSED three-panel view for the fusion capability, but
`tool_fusion_v0`'s own evidence artifact (`composer.save_segmentation()`) is
a single fused cluster-map image, not a triptych. Rather than changing that
specialist's output shape (which would be a backend change, against the
explicit "do not rewrite the backend architecture" instruction), the
Streamlit UI itself independently loads and displays the two raw uploaded
optical/SAR arrays alongside the specialist's own fused evidence image. One
small, additive, public helper was added to support this cleanly:
`evidence.composer.to_displayable_rgb(arr)` - a thin public wrapper around
the same percentile-stretch normalization every evidence image already uses
(`_to_bgr_uint8`), so the raw-input display looks visually consistent with
real evidence images without duplicating normalization logic or reaching
into a private function from outside its module. This mirrors the precedent
set earlier by `composer.save_source_image()` (added for VQA's evidence).

**Verification without a running Streamlit.** streamlit cannot be installed
in this cloud sandbox (`pip install streamlit --break-system-packages`
fails with "Could not find a version that satisfies the requirement
streamlit (from versions: none)" - zero PyPI egress, consistent with every
other network-dependent finding in this project). So `app/streamlit_app.py`
was verified two ways: (1) a hand-built fake `streamlit` module
(`tests/integration/test_streamlit_app_logic.py`, 5 tests) that runs the
real app source via `runpy` across multiple simulated reruns, proving the
control flow is correct AND that clicking each tab's RUN button actually
invokes the real `pipeline.run_query()` and stores/renders the real
`ExecutionTrace` it returns - including the SmolVLM-available "REAL MODEL"
disclosure branch (mocked the same way `test_vqa_dispatch.py` mocks it) and
a deliberate failure path (a too-small image) rendering the professional
failure panel instead of crashing; (2) `tests/integration/
test_streamlit_app_launch.py`, real `streamlit.testing.v1.AppTest`-based
smoke tests that `unittest.skipUnless` themselves out here (no streamlit
installed) and are meant to run for real on the user's machine - see
`docs/RUN_ON_WINDOWS.md`. Full suite in this sandbox: 78 tests collected (was
63; +15: 6 pipeline wiring tests, 6 fake-streamlit logic tests, 3 real
AppTest launch tests), of which 65 actually executed and passed and 13 were
skipped (10 pre-existing skips for optional Florence-2/GroundingDINO
dependencies + the 3 new AppTest tests, since streamlit itself is one of
those unavailable dependencies here) - 0 failures, 0 regressions.

**What has NOT been verified yet, honestly**: the app has never actually
been launched in a real browser. Its visual appearance, whether the CSS
renders correctly against the user's installed Streamlit version, and
whether the four capability tabs behave correctly end-to-end with a human
clicking through them, are unverified until the user runs it per
`docs/RUN_ON_WINDOWS.md` and reports back. No GitHub push, PPT, or video
yet - those remain gated on that report.

---

This document exists to satisfy the PS requirement that the system include "a
component that has undergone remote-sensing-specific adaptation/fine-tuning
(e.g., via LoRA/PEFT) rather than relying solely on a generic pretrained
vision-language model." Per the explicit build instruction — *"do not claim
adaptation until it is actually executed and verified"* — nothing in this
document should be read as a claim that adaptation has happened. It has not.
`src/models/registry.py` marks `tool_rs_vlm_adapted_v1` as `NOT_AVAILABLE_SANDBOX`
for exactly this reason, and every specialist branch in the running system
today (`vqa`, `grounding`, `change`, `fusion`) uses the classical-CV v0 tools
instead, never this one.

What follows is (1) the exact model and exact dataset subset selected for the
adaptation, with the reasoning; (2) a config file and a training script that
are reproducible in principle — pinned versions, fixed seed, explicit
hyperparameters — but have never been run, because this development sandbox
has zero internet egress (`docs/network_constraints.md`) and none of
`torch`/`transformers`/`peft`/`datasets` are installed here; and (3) the exact
metadata a real training run must record before this tool's status can change
from `NOT_AVAILABLE_SANDBOX` to `AVAILABLE`.

## 1. Exact base model

### Model re-verification (Phase 2B)

Before touching the plan, the choice was re-checked against current
documentation rather than assumed. Findings, each backed by a fetched
source (see Sources at the end of this section):

- **`google/paligemma2-3b-pt-448` (PaliGemma 2) exists and is a real
  improvement over the original PaliGemma** — its own Hugging Face model
  card reports higher VQA and RefCOCO-style localization scores than
  PaliGemma 1, and explicitly names remote-sensing question answering as an
  intended fine-tuning target. Since the original doc named
  `google/paligemma-3b-pt-448` (PaliGemma **1**) without checking whether a
  newer, better-documented-for-this-exact-use-case successor existed, this
  is a real correction, not a cosmetic one.
- **Decision: switch the base model from `google/paligemma-3b-pt-448` to
  `google/paligemma2-3b-pt-448`.** Same size class (3B), same SigLIP vision
  encoder + native `<locNNNN>` localization-token output format (so the
  grounding-over-captioning reasoning below is unchanged), same access
  mechanics (see gating note below) — this is not a silent swap of
  architecture family, only an upgrade to the newer checkpoint of the same
  family, made explicit here rather than applied quietly.
- **Both PaliGemma and PaliGemma 2 are gated on Hugging Face**: whoever runs
  the adaptation must be logged into a Hugging Face account, click "agree
  and access repository" on the model page (Gemma license terms), and
  authenticate locally (`huggingface-cli login` with an access token) before
  any download will succeed. This was not previously documented and is now
  the first step in the run instructions (`docs/RUN_ON_WINDOWS.md`).
- **Alternative considered and rejected for now**: `Qwen/Qwen2.5-VL`
  (2B/7B-Instruct) is Apache-2.0 licensed (no gating, no login required) and
  has active community LoRA tooling, which is a genuine operational
  advantage. It was not selected because, unlike PaliGemma's model card, no
  equivalent documentation was found tying it specifically to remote-sensing
  fine-tuning, and its grounding output convention would need to be
  re-verified from scratch rather than reusing the location-token format
  already reasoned about below. If the Gemma license/login step becomes a
  real blocker when the team actually runs this, Qwen2.5-VL-2B-Instruct is
  the documented fallback — not a silent alternative, a named one.

**Selected: `google/paligemma2-3b-pt-448`** (PaliGemma 2, Google; Gemma
license, gated; SigLIP-So400m vision encoder + Gemma 2 decoder; 448x448
input resolution checkpoint).

Why this model and not a larger generic VLM (LLaVA-1.5-7B, Qwen-VL, etc.):

- **Grounding is native to its output format.** PaliGemma models are
  pretrained to emit `<locNNNN>` location tokens directly in text output for
  detection and referring-expression tasks, prompted as `detect {object}` or
  `segment {object}`. Since the team locked "grounding over captioning" as a
  Phase-1 decision, a base model whose pretraining objective already includes
  localization is a better adaptation target than a caption-only VLM — LoRA
  only needs to shift the *domain* (RGB/aerial photography → satellite/SAR
  imagery), not teach the model a new output grammar for boxes.
- **The `-pt-448` checkpoint is the pretrained-but-not-task-mixed variant**,
  which is what should be adapted, not the `-mix-448` checkpoint (already
  instruction-tuned across many tasks). Adapting `-mix` would confound "did
  our LoRA adapt it to remote sensing" with "was it already generically good
  at VQA/detection."
- **3B parameters is a realistic LoRA target on hackathon-grade hardware**
  (a single consumer or single cloud GPU, a few hours), unlike 7B+ models,
  which keeps "reproducible" honest rather than aspirational.
- It is a genuinely open-weights model with a permissive-enough license for
  a student competition (Gemma license — not a black-box API), matching the
  PS's "pretrained + adapted, not merely an API call to a closed LLM" intent.

**Sources checked this phase** (all fetched live, not from training-data
recall alone): [google/paligemma2-3b-pt-448 model card](https://huggingface.co/google/paligemma2-3b-pt-448),
[google/paligemma-3b-pt-448 model card](https://huggingface.co/google/paligemma-3b-pt-448),
[Qwen2.5-VL fine-tuning guide](https://datature.io/blog/how-to-fine-tune-qwen2-5-vl).

## 2. Exact dataset subset

**VRSBench** (Li et al., *"VRSBench: A Versatile Vision-Language Benchmark
for Remote Sensing Image Understanding"*, 2024) — specifically its **VQA
split** and **referring-expression (grounding) split**, both derived from the
DOTA-v2 / DIOR source imagery VRSBench is built on.

- We use VRSBench, not BigEarthNet, as the primary adaptation dataset because
  VRSBench is annotated for exactly the two task types we are adapting the
  model for (open-ended VQA and referring-expression grounding with
  bounding boxes). BigEarthNet is a multi-label land-cover classification
  dataset with no question/answer or referring-expression annotations, so it
  is a better fit for a future land-cover classification specialist than for
  this VLM adaptation. VRSBench is one of the four benchmarks the PS names
  explicitly, and `src/evaluation/benchmarks.py` already has a
  `VRSBenchTask` stub for it — evaluation and adaptation are meant to share
  the same dataset.
- **Named subset for this adaptation run**: the VQA and referring-expression
  training splits only (never the VRSBench test split, which is reserved for
  `VRSBenchTask` evaluation in `src/evaluation/benchmarks.py` — training and
  evaluating on the same images would invalidate the benchmark result).
- **Scale is explicitly uncertain and must be measured, not assumed.** Two
  different-looking figures have surfaced for this dataset and neither has
  been confirmed by us loading the data directly: the original arXiv paper
  abstract describes roughly 29k images with tens of thousands of QA/
  referring pairs, while a live fetch of the dataset's Hugging Face page
  during this phase reported 29,614 images but **3,123,221** VQA pairs — an
  order of magnitude larger than the paper-level figure, and the page's own
  dataset viewer was showing a backend error at fetch time, which is reason
  enough to distrust the exact number until it's loaded for real. Neither
  number goes into `configs/rs_adaptation.yaml` as fact. `scripts/
  adapt_rs_vlm.py` prints and logs the actual counts it loads at runtime
  specifically so this uncertainty gets replaced with a measured number on
  the first real run, not resolved by picking whichever figure sounds more
  impressive.
- **Hosting and access**: `xiang709/VRSBench` on the Hugging Face Hub
  (`datasets.load_dataset("xiang709/VRSBench", ...)`); a GitHub repo
  ([lx709/VRSBench](https://github.com/lx709/VRSBench)) and a project site
  ([vrsbench.github.io](https://vrsbench.github.io/)) also exist and are the
  right place to double check the exact split/field names before writing
  the data-loading code for real, since the exact `load_dataset` call and
  column names were not independently confirmed here (see caveat above).
- **License: CC BY-NC 4.0 (non-commercial)**, per the Hugging Face dataset
  page. Fine for a student competition demonstration; would need a different
  dataset if this project were ever commercialized. Record this in
  `data/manifests/` alongside the source URL and download date once real
  data is pulled, per the DATA requirements in the Phase 2B brief.
- Config for reproducing the fetch is in `configs/rs_adaptation.yaml` under
  `dataset:` — image resolution, split names, and expected local cache path.

**Sources checked this phase**: [xiang709/VRSBench dataset card](https://huggingface.co/datasets/xiang709/VRSBench),
[lx709/VRSBench GitHub repo](https://github.com/lx709/VRSBench),
[VRSBench project site](https://vrsbench.github.io/),
[VRSBench arXiv paper](https://arxiv.org/html/2406.12384v1).

## 3. What "adapted" will mean here

"Adapted" will mean, concretely and only once true: a LoRA adapter (rank,
alpha, and target modules pinned in `configs/rs_adaptation.yaml`) trained
with `peft` on top of the frozen `paligemma2-3b-pt-448` weights, using the
VRSBench VQA + referring-expression training split, for a fixed number of
epochs, checkpointed with the exact metadata described in Section 5, and
then loaded back through `src/models/registry.py` as `tool_rs_vlm_adapted_v1`
with `status=AVAILABLE` — at which point (and not before) it becomes a real
alternative the router can select for `single_image_vqa` and `grounding`,
alongside (not silently replacing) the classical-CV v0 tools. "Adapted" does
NOT mean: downloading the base model and calling it as-is (that is a generic
pretrained VLM, exactly what the PS says is insufficient), and does not mean
prompt-engineering the base model without any weight update.

## 4. Reproducible-but-not-executed script and config

- `configs/rs_adaptation.yaml` — every hyperparameter (seed, LoRA rank 16 /
  alpha 32 / dropout 0.05 / target modules, learning rate, batch size, epoch
  count, image resolution, dataset paths) pinned in one place so a future run
  is fully specified before it starts, not tuned ad hoc.
- `scripts/adapt_rs_vlm.py` — loads the config, loads `paligemma2-3b-pt-448`
  and the VRSBench splits, wraps the model with a `peft` `LoraConfig`, trains,
  and writes a checkpoint plus the metadata file described below. It imports
  `torch`, `transformers`, `peft`, and `datasets` at the top; the script
  itself checks for their presence and exits with a clear
  `RS_ADAPTATION_NOT_EXECUTABLE_HERE` message rather than partially running —
  see the guard at the top of that file. It has never completed a run in any
  environment as part of this project; it is reviewed for correctness by
  reading, not by execution.

## 5. Checkpoint / run metadata contract

The training script is required to write a `run_metadata.json` next to any
checkpoint it produces, with (at minimum) these fields — and
`src/models/registry.py` must not flip `tool_rs_vlm_adapted_v1` to
`AVAILABLE` until a real file matching this shape exists on disk and has been
loaded once to confirm the adapter actually changes model output versus the
frozen base:

```json
{
  "base_model": "google/paligemma2-3b-pt-448",
  "adapter_type": "LoRA",
  "lora_config": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]},
  "dataset": "VRSBench",
  "dataset_splits_used": ["vqa_train", "referring_train"],
  "dataset_image_count_measured": null,
  "dataset_local_hash": null,
  "seed": 42,
  "epochs_completed": 0,
  "train_loss_curve_path": null,
  "started_at_utc": null,
  "completed_at_utc": null,
  "git_commit": null,
  "executed_on": null,
  "status": "NOT_EXECUTED"
}
```

Every `null`/`0`/`"NOT_EXECUTED"` value above is a placeholder that a real
run must fill in with a measured value — never with an invented one. Until
that file exists with real values, any report to the user or a judge must
describe this component exactly as this document does: designed, not run.

## 6. What would have to change for this to run

Per `docs/network_constraints.md`: a networked machine (or this sandbox with
egress unblocked) with `pip install -r requirements-future.txt` succeeding,
enough disk/VRAM for a 3B-parameter model plus LoRA training state, and a
successful download of VRSBench (or a locally-supplied copy). None of these
are available in the current development sandbox, which is why this
component's status is `NOT_AVAILABLE_SANDBOX` rather than `AVAILABLE`, and
why every specialist branch a judge can actually run today uses the
classical-CV v0 tools instead.

## 7. Phase 2B: a remote-bridged development environment was checked, and it
   is not a way to run this either

A bridge to the target Windows machine was available during development
(a connected project folder), and it was considered as a way to run the
real LoRA training there. Before writing anything further, that bridge was
tested directly rather than assumed to work, and the result is another
honest constraint, not a workaround:

- The bridge confirmed a real Windows machine with a working conda/Miniforge
  installation and the project folder visible on it — i.e. the target
  machine does appear to have a real conda installation.
- However, the shell used to run commands through that bridge does **not**
  run on the native Windows OS. It reports a Linux userspace, a non-Windows
  home path, and **no GPU** (no VGA/NVIDIA device detected), only a couple
  of CPUs and a few GB of RAM/disk, and the same kind of blocked network
  egress as the rest of this development environment (`huggingface.co` and
  `pypi.org` both returned HTTP 403 from a proxy). In other words: that
  bridge shell is a small, isolated companion environment, not a window
  into the target machine's actual Windows Python/conda/GPU environment.
- The one thing that bridge *does* give real access to is the **connected
  project folder itself** — files written there land on the real disk of
  the target machine, because folder mounting bridges the isolated shell to
  the real filesystem even though the shell executing inside it is not the
  real shell. That's how this project's full source tree was placed on the
  target machine at its real project path — a real file operation — even
  though no training command can be *run* from that bridge.

**Conclusion**: the actual LoRA adaptation run must be executed by the user,
in their own real terminal (Anaconda Prompt / PowerShell with their conda
environment activated), on their own real GPU and real internet connection —
not through the development bridge used to prepare this repository.
`docs/RUN_ON_WINDOWS.md` gives the exact commands for that. Once run, the
resulting checkpoint and `run_metadata.json` should be committed to the
project folder so they can be reviewed and verified for real before
`tool_rs_vlm_adapted_v1` is ever marked `AVAILABLE`.
