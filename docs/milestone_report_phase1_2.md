# Milestone Report — Phase 1/2 (Core Foundation + Thin End-to-End MVP)

> **Historical snapshot — not the current project state.** This report
> captures the project exactly as it stood at the end of Phase 1/2 (35
> tests, CLI-only, no VQA model wired in yet, `google/paligemma-3b-pt-448`
> still the tentative RS-VLM adaptation target). Every later phase —
> documented in `docs/rs_adaptation.md` and `docs/RUN_ON_WINDOWS.md` —
> superseded parts of it: `HuggingFaceTB/SmolVLM-256M-Instruct` became the
> real, working single-image VQA specialist; a PaliGemma 2 / VRSBench LoRA
> adaptation was evaluated and found impractical on available CPU-only
> hardware; grounding/change/fusion stayed classical-CV baselines by
> deliberate decision, not by default; and a Streamlit UI was built and
> verified. See `README.md` for the current-state summary and current test
> counts. Kept here unedited as an honest development record.

Format follows the 9-point communication protocol from the Phase 1/2 approval
message exactly. Every claim below is backed by a real, re-runnable artifact
in this repo (`runs/*/trace.json`, `python3 -m unittest`, or a named file) —
nothing here is asserted without something on disk to check it against.

## 1. What was built

- **Core foundation** (Phase 1): raster/metadata ingestion honest about
  unknown CRS/modality/date (`src/ingestion/`), input validator with specific
  bi-temporal/cross-modal error messages (`src/ingestion/validator.py`), the
  frozen schema set (`src/routing/schemas.py`), a trained intent classifier +
  rule-based pair router (`src/routing/`), a confidence engine implementing
  `docs/confidence.md` exactly (`src/confidence/engine.py`), a model/tool
  registry with explicit AVAILABLE/NOT_AVAILABLE_SANDBOX status
  (`src/models/registry.py`), and evidence-image composers
  (`src/evidence/composer.py`).
- **All 5 mandatory branches** (Phase 2): single-image VQA
  (`src/specialists/vqa.py`), grounding (`src/grounding/grounding.py`),
  bi-temporal change (`src/change/change.py`), optical+SAR fusion
  (`src/fusion/fusion.py`), and agentic routing that actually branches
  execution (`src/routing/router.py` — see point 2).
- A CLI orchestrator (`app/cli.py`) standing in for Streamlit as the thin
  interface (Streamlit itself is uninstallable in this sandbox — see point 7).
- Benchmark interfaces for VRSBench/RSVQA/CDVQA/ISRO-SAC
  (`src/evaluation/benchmarks.py`), all correctly returning
  `NOT_YET_EVALUATED`.
- JSON + PDF export (`src/export/report.py`).
- 5 synthetic, seeded, clearly-labeled test fixtures
  (`scripts/generate_fixtures.py`, `data/fixtures/README.md`).
- `docs/rs_adaptation.md` + `configs/rs_adaptation.yaml` +
  `scripts/adapt_rs_vlm.py`: the named model, named dataset subset, and a
  reproducible adaptation script — confirmed to exit with an explicit
  `RS_ADAPTATION_NOT_EXECUTABLE_HERE` error rather than run (see point 8).
- `docs/network_constraints.md`: full record of why real RS-VLM
  adaptation and real satellite data are unreachable here.

## 2. What actually executes (proof of real agentic routing, not narration)

Ran manually via `app/cli.py` against real (synthetic) files. Five distinct
executions, five distinct code paths, captured in `runs/*/trace.json`:

| Case | Router output | What ran |
|---|---|---|
| `"describe this scene"` on 1 optical image | `single_image_vqa`, confidence 0.63 | `tool_single_image_vqa_v0` executed, returned real cluster fractions |
| `"locate the water body"` on 1 optical image | `grounding`, confidence 0.61 | `tool_grounding_v0` executed, returned a real bounding box |
| 2 images, declared dates 2020/2024 | `bitemporal_change`, confidence 0.95 | `tool_change_v0` executed, detected the real injected change region |
| 1 optical + 1 SAR image, no dates | `optical_sar_fusion`, confidence 0.95 | `tool_fusion_v0` executed, used both channels jointly |
| 2 images, no distinguishing modality or date | `needs_clarification`, confidence 0.0 | **No specialist ran** — the CLI exited with a specific reason, no fake answer |

This is not printed narration: `RouterDecision` is a dataclass consumed
directly by `app/cli.py`; changing `task_type` on the object changes which
Python function is called next (`app/cli.py`'s dispatch is a dict lookup on
`decision.task_type`, not an if/else pretending to be a dispatch). The
`needs_clarification` row is the clearest proof — there is no code path from
that `task_type` to any specialist, so it is structurally impossible for the
system to fabricate an answer when it isn't sure.

A same-date edge case was also verified: two images both declared
`2020-01-01` route to `bitemporal_change` (confidence 0.5, reasoning:
"dates were explicitly provided, signalling bi-temporal intent") and then
fail validation with the specific message *"Bi-temporal change analysis
requires two different acquisition dates; both inputs declared the same
date."* — not a generic ambiguity refusal. (`runs/same_date2/trace.json`)

## 3. What was tested

Full suite run just now:

```
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py" -v
...
Ran 35 tests in 0.854s
OK
```

35 real tests, 0 failures, 0 errors, across 4 files:

- `tests/unit/test_routing.py` — 9 tests (single-image branch selection,
  pair branch selection by modality/date, the same-date regression test,
  the 3-input rejection, the ambiguous-refusal case).
- `tests/unit/test_ingestion.py` — 11 tests (raster loading, "never invent
  metadata" honesty checks, validator pass/fail cases).
- `tests/unit/test_confidence.py` — 7 tests (formula bounds, downgrade
  triggers, the 0.30 warning cap, method-version labeling).
- `tests/integration/test_specialists.py` — 8 tests (each specialist run
  against real fixture files end to end, including a dedicated test proving
  the fusion specialist's output actually changes when the SAR channel is
  used, not merely loaded and ignored).

This is the first and only run of the complete suite; no test was edited to
pass after a prior failure without being re-run (the one router bug found
during manual verification — see point 7 — was fixed in the router itself
and locked in with a new test, not worked around in a test).

## 4. Exact models used

Nothing described as "AI model" is running today. What is genuinely
executing (`status=AVAILABLE` in `src/models/registry.py`), for all 4
specialist branches, is classical computer vision / statistics — no
pretrained weights, no external download:

- `tool_single_image_vqa_v0` — scikit-learn KMeans + silhouette score over
  color/texture features.
- `tool_grounding_v0` — OpenCV HSV threshold + contour extraction.
- `tool_change_v0` — grayscale differencing + Otsu threshold + morphological
  cleanup.
- `tool_fusion_v0` — scikit-learn KMeans over a hand-built `[R, G, B, SAR]`
  joint feature vector.

The intended deep-learning component, `tool_rs_vlm_adapted_v1`, is
registered with `status=NOT_AVAILABLE_SANDBOX` and has never executed. Its
target is named precisely in `docs/rs_adaptation.md`: base model
**`google/paligemma-3b-pt-448`**, adapted via LoRA
(`r=16, alpha=32, target_modules=[q_proj, v_proj]`, pinned in
`configs/rs_adaptation.yaml`). Nothing about it should be represented to a
judge as running, benchmarked, or trained — it is a documented, reviewable
plan, not a result.

## 5. Exact datasets used

The only data that has actually been loaded and processed is 5 synthetic
fixtures generated by `scripts/generate_fixtures.py` with a fixed seed
(`seed=42`, fully reproducible): `single_image.png`, `change_before.png`,
`change_after.png`, `fusion_optical.png`, `fusion_sar.png`. These are numpy
scenes with painted regions standing in for water/vegetation/urban/bare-soil
and a deliberately injected 40×50px change block — **not real satellite
imagery**, and labeled as such in `data/fixtures/README.md`. Real
Sentinel-2/SAR data and the named benchmark datasets (VRSBench, RSVQA,
CDVQA, BigEarthNet) were not reachable from this sandbox (`docs/
network_constraints.md`) and have not been used anywhere in this codebase.

## 6. Test count / result

**35 / 35 passing, 0 failures, 0 errors** (`python3 -m unittest discover`,
run just now, output pasted in point 3 above). This is the real, current
number — not a target or an estimate.

## 7. What remains

- Real RS-VLM adaptation (`tool_rs_vlm_adapted_v1`) — designed and
  documented, not trained; blocked on network access, not on missing design
  work.
- Real benchmark evaluation (VRSBench/RSVQA/CDVQA/ISRO-SAC) — interfaces
  exist and correctly report `NOT_YET_EVALUATED`; running them needs the
  same network access as above.
- Streamlit UI — the CLI currently proves all 5 branches work; Streamlit
  itself is not installable here (`requirements-future.txt`), so the UI
  layer is still to be built on a networked machine, using the exact same
  schemas the CLI already renders (`RouterDecision`, `SpecialistOutput`,
  `ConfidenceResult`, `ExecutionTrace`) so nothing about the underlying logic
  needs to change for the UI to exist.
- One self-found and already-fixed router bug (see below) — no other known
  defects at this time, but the specialists have only been exercised against
  synthetic fixtures, not the full variety of real remote-sensing imagery.

**Self-found and fixed during this phase** (not user-reported): the router
originally sent same-date bi-temporal requests to a generic
`needs_clarification` refusal instead of routing to `bitemporal_change` and
letting the validator produce its specific same-date error message. Found by
deliberately testing that exact case, fixed in `src/routing/router.py`
(`_decide_pair`), verified by re-running both the same-date and
different-date cases plus the true-ambiguous case, and locked in with
`test_same_declared_date_still_routes_to_change_for_specific_error`.

## 8. Biggest technical risk

**Zero network egress in the development sandbox** is the single risk that
threatens the PS's core differentiator: without it, the genuinely
RS-adapted VLM the PS explicitly requires ("a generic pretrained VLM without
remote-sensing adaptation is insufficient") cannot be trained, run, or
verified here, and neither can evaluation against the named public
benchmarks. The mitigation in place is architectural, not cosmetic: the
model registry, router, and every schema were built so that swapping
`tool_rs_vlm_adapted_v1` from `NOT_AVAILABLE_SANDBOX` to a real, loaded
model requires no change to the router, the contracts, or the UI/export
layer — only a new `_loader` in `src/models/registry.py` once training
happens on a networked machine. The secondary risk is presentational: a
judge could mistake the classical-CV v0 tools for more than they are unless
every answer, trace, and report keeps stating plainly (as they do today)
that these are `v0_classical` baselines and not the RS-adapted model.

## 9. Next milestone

Two things need a decision before proceeding, consistent with the "do not
proceed into deep UI productization until the branches are proven" rule —
that gate is now met (all 5 branches executed and verified above), so the
proposal is:

1. **Confirm the 5 branches meet the bar** — every input/query/model/output/
   evidence/confidence/trace/failure-behavior combination the approval
   message asked for is in this report's point 2 and in `runs/*/trace.json`
   for independent inspection.
2. **Decide the path for real RS adaptation**: either get network access
   unblocked in this environment, or run `scripts/adapt_rs_vlm.py` on a
   separate networked machine and bring the resulting checkpoint + verified
   `run_metadata.json` back in — no further design work is needed to start
   this, only executing it somewhere with internet access.
3. Pending that decision, the next build milestone is the Streamlit UI
   (Phase 6 per the original build order), built directly on the existing
   schemas so the trace panel renders `ExecutionTrace` verbatim rather than
   separately authored copy — still gated on your go-ahead, and still no
   GitHub push, PPT, or marketing copy until you say so.
