# UI design plan (Streamlit analyst console)

Written before `app/streamlit_app.py`, per the "plan, then review against the
brief, then build" process from the `frontend-design` skill, and using the
dark "Tactical Telemetry" archetype from the `brutalist-skill` (chosen over
its light "Swiss Industrial Print" archetype because the frozen brief
explicitly asked for a dark, restrained analyst-console interface, not a
light print-derived one).

## Why these two skills, and not the others

Invoked: `taste-skill:brutalist-skill` (its dark/monospace/zero-radius/
high-density "Tactical Telemetry" archetype matches the brief's own language
almost verbatim: "restrained dark interface", "analyst console", "aerospace/
geospatial tooling") and `frontend-design:frontend-design` (general
anti-genericism guidance: ground the design in the real subject matter,
avoid the specific "AI-generated design tells" it lists, work in a plan ->
critique -> build sequence).

Not invoked: the image-generation-only skills (`brandkit`,
`imagegen-frontend-mobile/web` - they produce images, not code, not
applicable to a Streamlit app); `stitch-skill` (a Google-Stitch-specific
tool); `soft-skill` (a "premium agency" aesthetic that trends toward
rounded/shadowed/decorative surfaces - directly contradicts the brief's
"no glassmorphism, no rounded cards" instruction); `minimalist-skill` /
`taste-skill` (kept as secondary references only, not primary, since
brutalist-skill's own brief-match was closer).

## Color

One base palette, dark substrate only (never mixed with a light substrate,
per brutalist-skill's own rule):

| Token | Hex | Use |
|---|---|---|
| `--bg` | `#0A0A0C` | Page background (deactivated-CRT black, not pure `#000`) |
| `--panel` | `#111114` | Compartment background (cards/sections) |
| `--panel-alt` | `#17171B` | Nested compartment (e.g. evidence image frame) |
| `--border` | `#2B2B30` | Hairline dividers between every compartment |
| `--border-strong` | `#45454C` | Active/focused compartment border |
| `--text` | `#E8E8E6` | Primary text (phosphor white, not pure `#FFF`) |
| `--text-dim` | `#8B8B90` | Secondary/metadata text |
| `--accent` | `#C98A3A` | The ONE accent color - amber, an avionics/HUD reference distinct from the brutalist skill's suggested red, chosen specifically so red is free to mean exactly one thing (below) |
| `--danger` | `#D6493F` | Reserved EXCLUSIVELY for failure states, validation errors, and fallback-occurred banners - never decorative, never reused for anything that isn't actually a problem |

No separate "success green" - a real result is communicated by the amber
accent and real numbers, not a decorative status color. This directly serves
the brief's "no decorative statistics" instruction: color carries meaning
(amber = live real data, red = something actually failed) and nothing else.

## Type

- Data, labels, metadata, all numeric fields, and every ALL-CAPS structural
  label: `'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`.
- Section headers and prose (the answer text, error explanations): `'Inter',
  -apple-system, 'Segoe UI', sans-serif`.
- ALL-CAPS is used only for fixed structural labels that repeat identically
  across every result (`TASK`, `TOOL`, `MODEL`, `CONFIDENCE`, `EVIDENCE`,
  `TRACE`) - i.e. instrument-panel labeling, not a one-off decorative
  eyebrow above a heading (the specific tell `frontend-design` warns
  against). The answer text itself, error prose, and reasoning strings are
  always sentence case, because they are real generated content, not UI
  chrome.
- No large "hero" display type (no `clamp(4rem, 10vw, 15rem)` treatment from
  the brutalist skill's own spec) - this is a working analysis tool where
  imagery is the visual hero, not a marketing page where oversized type is
  the hero; headers stay modest (1.1-1.5rem).

## Layout

```
+----------------+---------------------------------------------------+
| SYSTEM STATUS   | SATQUERY AI  //  REMOTE SENSING ANALYSIS CONSOLE  |
| (registry:      +---------------------------------------------------+
|  live AVAILABLE | [01 VQA] [02 GROUNDING] [03 CHANGE] [04 FUSION]   |
|  / NOT_AVAIL    +---------------------------------------------------+
|  per specialist)| INPUT          | RESULT                            |
|                 | - upload(s)    | QUERY -> TASK -> TOOL -> EVIDENCE  |
|                 | - query text   | (breadcrumb, real fields)          |
|                 | - modality/    |------------------------------------|
|                 |   date fields  | ANSWER (prose)                    |
|                 | - [RUN]        | MODEL/TOOL disclosure block        |
|                 |                | CONFIDENCE block                   |
|                 |                | EVIDENCE imagery (hero-sized)       |
|                 |                | [> EXECUTION TRACE] (expander)     |
|                 |                | [EXPORT JSON] [EXPORT PDF]         |
+----------------+---------------------------------------------------+
```

Left-aligned throughout (not centered/justified) - this is a dense
instrument panel read top-to-bottom and left-to-right, not a marketing page
read as a centered column. Narrow input column (~1/3) + wide result column
(~2/3) inside each tab, so evidence imagery has room to be the visual hero
per the brief. Zero `border-radius` anywhere; every compartment boundary is
a real 1px `--border` hairline, generated as CSS Grid `gap` per the
brutalist skill's grid-determinism technique where practical, plain borders
elsewhere (Streamlit's own component structure doesn't expose a single grid
container to hang the gap-trick on everywhere).

## Principles specific to this brief

1. Every number on screen must trace to a real field on `ExecutionTrace` /
   `SpecialistOutput` / `ConfidenceResult` / `RouterDecision` - the UI layer
   never authors a stand-in string for data it could instead read from the
   real object. This is the same discipline already enforced in the
   backend's own docs (`docs/confidence.md`); the UI must not undo it by
   inventing a friendlier-looking number anywhere.
2. Amber marks "this is live and real" (the active tab indicator, the
   router breadcrumb, the confidence value, the RUN button). Red marks
   "something needs the analyst's attention" (validation errors, execution
   failures, `fallback_occurred=True`) and nothing else.
3. No shadows, no gradients, no glassmorphism, no icons standing in for
   meaning. ASCII framing (`[ SECTION ]`) is used sparingly for a small
   number of fixed structural labels, not sprinkled everywhere - overusing
   it would itself become the kind of decorative tell the brief asked to
   avoid.
4. Grounding, change, and fusion results always carry their classical-CV
   disclosure text verbatim from the specialist's own `SpecialistOutput` /
   `ConfidenceResult.basis_description` - the UI never rephrases "v0
   classical baseline" into something that reads more like a trained model.

---

# Revision 2 - visual/UX refinement pass (approved-functionality, redesign only)

The functional v1 build above was approved. This revision is a **presentation-
only** pass: no routing, validation, specialist, confidence, or evidence
*computation* changed - only how the same real objects are laid out, labeled,
and typeset. Every "no backend changes" constraint from v1 still holds.

## What v1 got wrong (the brief's own diagnosis)

Sidebar too wide and dominated by raw internal tool identifiers; monospace
used almost everywhere, making the whole app read as a developer diagnostic
console rather than a finished product; long technical explanations (full
`basis_description` paragraphs, file-path references) sat directly under the
answer, competing with it; the router breadcrumb put an internal `tool_name`
string above every result before the user had even read the answer; tabs
were visually flat; imagery was not consistently the dominant visual element
(especially for VQA, where the analyzed image was buried under an "Evidence"
heading far below the answer).

## Skills used for this pass

`taste-skill:redesign-skill` (its whole premise - audit an existing UI,
diagnose generic/weak patterns, fix in place without rewriting the stack -
matches this task exactly: it is a redesign of an *existing, approved* app,
not a from-scratch build) and `frontend-design:frontend-design` (already
in use from v1, re-applied for hierarchy/typography judgment). `brutalist-
skill` was **not** re-applied this round - the brief explicitly named its
visible side effect (monospace overuse, "developer diagnostic console"
feeling) as a problem to fix, so leaning on it further would be reapplying
the cause of the complaint. Most of `redesign-skill`'s own catalogue targets
marketing sites (glassmorphism, parallax, testimonial carousels, pricing
tables) and does not apply to a single-operator analyst tool; the parts that
transfer directly are: one desaturated accent + neutral grays (already true
in v1, kept), a real spacing scale, visible hover/focus/pressed states,
tabular figures for data, and "remove border+shadow+white-card" genericism
(not applicable here since v1 never had white cards, but the underlying
principle - only add a panel boundary when it communicates real hierarchy -
carried forward into fewer, more purposeful panel divisions in v2).

## Wireframe (textual, produced before editing code)

```
HEADER
  SatQuery AI                                    <- display token, semibold
  Remote sensing analysis                        <- section-title token, dim
  SIH26167 · ISRO · Space Technology Programme    <- metadata token, dim, small

SIDEBAR (compact control rail, ~230px, was ~320px)
  SYSTEM
    * VQA      READY
    * GROUND   READY
    * CHANGE   READY
    * FUSION   READY
  [Technical system status v]  <- collapsed; raw registry rows + tool IDs live here

NAVIGATION (segmented, uppercase via CSS not literal shouting strings)
  01 Single image | 02 Grounding | 03 Change | 04 Optical + SAR

WORKSPACE (left column, per tab)
  Input
    imagery (upload or "use a bundled sample")
    modality / dates (only where the capability needs them)
    query
    [Run analysis]

RESULT (right column, per tab) - this is the hierarchy that changed most:
  Analysis type: <human label>            <- one quiet line, no tool_name here
  ── VQA only ──────────────────────────────────────────────
  [ source image (large) ] [ answer (large) ]
                            Confidence  0.71  Generation certainty
                            Model  SmolVLM-256M-Instruct · 29.9s
  ── grounding / change / fusion ─────────────────────────────
  answer (prose, real text, unmodified)
  EVIDENCE (hero image(s); change: no-change gets an explicit clean state;
            fusion: optical+SAR small pair above a larger fused image)
  CONFIDENCE (compact: number, short label, thin bar, "Details v")
  MODEL / TOOL (compact: role/method/latency, "Method details v")
  Export results  [Export JSON] [Export PDF]
  Execution trace >  (collapsed)
    Query -> Router -> Specialist -> Evidence -> Export (human-readable)
    Raw data (JSON) v   <- trace.to_dict(), unchanged, still the real object
```

## Typography tokens (v2)

No external font dependency (v1's Google Fonts `@import` is removed - the
brief requires the app to look correct with no internet access). System
stacks only:

- `--font-sans`: `Inter, ui-sans-serif, system-ui, -apple-system,
  BlinkMacSystemFont, 'Segoe UI', sans-serif` - used for nearly everything:
  the display title, section titles, body/answer text, labels, metadata.
- `--font-mono`: `ui-monospace, SFMono-Regular, Consolas, 'Liberation
  Mono', monospace` - used ONLY for: model/tool identifiers (`tool_name`,
  `model_name` when shown in a technical-details context), the execution
  trace's raw JSON, and the confidence engine's raw `method_version`
  string. Never for the answer, headings, help text, or normal labels -
  this is the single biggest change from v1, which used monospace almost
  everywhere.
- Explicit size/weight tokens: `--fs-display` (1.5rem/650, the app title),
  `--fs-title` (1.05rem/650, section titles like "Input"/"Result"),
  `--fs-section` (0.92rem/600, subsection labels), `--fs-body`
  (0.95rem/400, answer/prose), `--fs-label` (0.7rem/600, uppercase,
  compact chrome labels), `--fs-metadata` (0.78rem/500, small real-data
  fields), `--fs-mono` (0.78rem, the restricted monospace uses above).
- All data figures (confidence value, latency, percentages) render in
  `--font-sans` with `font-variant-numeric: tabular-nums` rather than
  switching typeface - keeps digits aligned without reintroducing
  monospace for ordinary numbers.

## Color / spacing tokens (v2)

Color system is unchanged from v1 above (it already satisfied "one
desaturated accent, near-black not pure black, red reserved for failure
only" - re-verified against `redesign-skill`'s color audit and left alone).
Added an explicit spacing scale, used everywhere instead of ad hoc pixel
values: `--space-1: 4px`, `--space-2: 8px`, `--space-3: 12px`, `--space-4:
16px`, `--space-6: 24px`, `--space-8: 32px`, `--space-12: 48px`.

## Hierarchy / disclosure rules (v2)

1. **Router visualization moved from "always visible above the answer" to
   "inside the collapsed execution trace."** v1's breadcrumb put an
   internal `tool_name` string above every answer - exactly the "internal
   tool IDs dominate the screen" problem. v2 shows one quiet, human-labeled
   line ("Analysis type: Single-image VQA") above the answer instead, using
   a small fixed lookup from `task_type` to a plain-English label (`single_
   image_vqa` -> "Single-image VQA", etc. - the lookup is copy, not data;
   `task_type` itself is untouched). The full breadcrumb (task/tool/router
   confidence/reasoning) still exists, verbatim, as the first thing inside
   the "Execution trace" expander.
2. **Confidence and model/tool disclosure are compact by default, with a
   "Details"/"Method details" expander holding the full real text.** The
   number and a short label are always visible; `ConfidenceResult.
   basis_description` (verbatim, unedited) and the raw `tool_name`/
   `method_version` strings move into the expander. The short label is a
   fixed mapping keyed on the real `method_version` value (`v0_classical`
   -> "Signal quality heuristic", `v1_vlm_mean_token_probability" ->
   "Generation certainty") - both method versions the system can ever
   produce are covered, so this is a real, exhaustive short-form of
   documented meaning (see `docs/confidence.md`), not an invented category;
   the unabridged sentence is always one click away, never deleted.
3. **VQA's evidence image and answer are shown side by side** (source
   image left, answer + compact confidence/model row right) instead of the
   image living far below under a separate "Evidence" heading - this is
   the one capability where the evidence artifact IS the exact image being
   discussed, so v1's structure was showing the same picture the answer is
   about, disconnected from the answer, for no reason.
4. **Bi-temporal change gets an explicit clean state.** When the real
   `changed_fraction` from `tool_change_v0`'s output rounds to 0.0%, the UI
   adds a plain "No significant change detected" line ABOVE the specialist's
   own real answer sentence - it does not remove or edit that real sentence
   (which still states the exact percentage/date range), it just gives the
   reader an immediate, honest headline instead of making them parse a
   percentage out of a paragraph to learn nothing changed.
5. **Sidebar became a compact capability-readiness rail**, computed for
   real from `registry.all_entries()` (a capability is READY if any of its
   registered tool names reports `AVAILABLE` - for VQA that's true whenever
   either the SmolVLM or the classical entry is available, which today is
   always at least the classical one), with the raw per-tool registry rows
   (internal IDs, exact `AVAILABLE`/`NOT_AVAILABLE_SANDBOX` strings) moved
   into a collapsed "Technical system status" expander underneath.
6. **Microcopy pass**: removed internal/meta language from user-facing
   text (references to "the real object", "not authored copy", file/module
   paths, "v0" as a headline badge, and the doubled "not a foundation
   model" phrasing that appeared both in the specialist's own real answer
   text AND in the UI's disclosure copy - the UI now states the baseline
   type once, since the specialist's own generated sentence already carries
   the fuller caveat and is never edited).

## What did NOT change

Every `pipeline.run_query(...)` call, every `session_state` key
(`vqa_trace`, `fu_extra`, etc.), every button's enable/disable condition,
and the shape of every object rendered are identical to v1 - re-verified by
rerunning `tests/integration/test_streamlit_app_logic.py` (the fake-
streamlit stub suite) against the rewritten file with no changes to its
own assertions beyond copy-text lookups (see that file's history).

---

# Revision 3 - FINAL PRODUCT UI/UX PASS

Triggered by real screenshots from the user's machine (v2 confirmed
working: real SmolVLM inference, the 0%-change clean state, grounding
overlays) that also surfaced two real bugs (duplicated "upload" text in
every file-uploader dropzone; severe character-by-character line-wrapping
of long tool-ID/status strings in the sidebar), followed immediately by a
large, 20-section "FINAL PRODUCT UI/UX PASS" specification covering visual
direction, input experience, conversational follow-up, agent visibility,
reporting, and error handling. This revision is presentation-plus-two-
small-additive-backend-helpers only - see "What did NOT change" below.

## Skill selection (re-evaluated, not assumed)

The brief again explicitly asked to inspect available skills rather than
reuse the prior choice by default. `frontend-design` (typography/layout/
restraint principles, already applied in v1/v2) continues to apply.
`taste-skill:redesign-skill`'s premise - audit an existing, already-running
UI against real screenshots and a written brief, then produce a wireframe
before touching code - matches this pass exactly, as it did for v2: this is
again a refinement of a live, working product, not a from-scratch layout.
`taste-skill:brutalist-skill` remains explicitly not reused (v1's own
brutalist output is what the user's problem list in the v2 brief was
describing). No accessibility- or responsive-design-specific plugin was
present in the available skill set beyond what `frontend-design` and the
brief's own explicit accessibility/responsive checklist already cover, so
those requirements are satisfied directly against that checklist rather
than a separate plugin.

## Visual direction

Kept v2's dark "professional earth-observation workstation" token system
(near-black surfaces, one warm amber primary accent, red reserved for
failure) and added one restrained secondary accent, `--accent2: #4FA8C9`
(a desaturated cyan), used ONLY for real, execution-driven "system/
telemetry" signals - the live registry-ready dot, the run timestamp badge,
the "done" glyph in the agent timeline, and the REAL MODEL registry label -
never as decoration. No glow, scanlines, starfields, or gradients were
added; the brief explicitly asked for "futuristic through precision, not
through visual effects."

## Wireframe (textual)

```
HEADER  (unchanged from v2)
SIDEBAR
  SYSTEM   - capability rows, now labeled REAL MODEL / CLASSICAL /
             UNAVAILABLE (not just READY/UNAVAILABLE)
  ACTIVE   - one row per capability: the model/method that actually,
             really executed most recently in this session ("-" if never)
  SESSION  - analyses run (real counter) + last query (real, truncated)
  [Technical system status] - collapsed, raw registry IDs (fixed wrapping)
  ABOUT
NAVIGATION  - 01 SINGLE IMAGE / 02 GROUNDING / 03 BI-TEMPORAL CHANGE /
              04 OPTICAL + SAR  (added "BI-TEMPORAL" per the brief's own
              suggested rewording; kept "CHANGE" too so the tab's intent
              reads unambiguously at a glance)
INPUT (per tab)
  one-line "what to provide" caption
  [About this analysis mode] - expander: why / format / then
  fixture checkbox -> upload(s) -> modality/dates -> query -> "Try:" chips
  inline validation category note (INSUFFICIENT INPUTS / MISSING SECOND
    IMAGE) shown above the Run button, not just a disabled button with no
    explanation
  Run analysis
RESULT (per tab, once a trace exists)
  run metadata badge (RUN <time> · <modality>)
  ANSWER  (or "No significant change detected" clean state, change tab only)
  EVIDENCE - hero imagery (VQA: source image beside answer; change: real
             before/after panels ABOVE the change-map composite; fusion:
             real optical/SAR panels ABOVE the fused map; grounding: the
             boxed detection image)
  CONFIDENCE (compact, Details expander)
  ACTIVE SPECIALIST (compact, Method details expander)
  [Agent process timeline] - expandable checklist, NEW (see below)
  Export JSON / PDF
  Ask a follow-up -> [Conversation history] once any exist
  [Technical details] - human-readable flow + raw JSON (renamed from
    "Execution trace"; same content)
EMPTY STATE (no trace yet) - one real, mode-specific "AWAITING INPUT" block
  instead of a bare caption
```

## New features and how each stays real

- **Two visual bug fixes (best-effort - this project still cannot run a
  real browser; please re-screenshot to confirm).** The sidebar's
  "Technical system status" rows changed from a narrow side-by-side
  flex row (label | value) to a stacked block (label above, value below,
  full panel width) - the character-by-character wrapping was a width
  problem, and a ~250px-wide stacked row gives a 20-character tool ID
  enough room to lay out normally. The file-uploader CSS's blanket
  `[data-testid="stFileUploaderDropzone"] *` universal selector (which can
  affect elements Streamlit itself keeps visually deduplicated) was
  replaced with specific, narrower selectors for the known instruction/
  button sub-elements.
- **Richer per-mode input guidance + example query chips.** `MODE_HELP` and
  `EXAMPLE_QUERIES` are fixed, hand-written copy (not generated), rendered
  as one always-visible line plus an opt-in expander (why/format/then) so
  the input column doesn't regain v1's wall-of-text problem. Chips set
  `session_state[query_key]` via a real Streamlit `on_click` callback -
  they never call `pipeline.run_query()` themselves, so a chip click can
  never be mistaken for an analysis run.
- **Inline, named validation states.** `INSUFFICIENT INPUTS` and `MISSING
  SECOND IMAGE` render as a small labeled note above the Run button,
  computed from the same real upload/query state that already drives the
  button's `disabled` flag - not a new validation system, just naming the
  existing one. (`UNSUPPORTED FORMAT` is enforced by the file picker's own
  `type=` allow-list before a file ever reaches the app; `AMBIGUOUS QUERY`
  / `MISSING MODALITY` for the two-image tabs are real backend outcomes
  the router/validator can still produce - see the numeric error-code
  panel below, which is where those actually surface today, since this
  UI's per-tab forms don't have a reachable path to fabricate them
  pre-run.)
- **Conversational follow-up**, implemented with zero backend changes: a
  follow-up is just another call to the same `pipeline.run_query()`,
  reusing the exact `image1_path`/`image2_path`/`modality`/`date` values
  cached from the tab's last real run (`session_state[f"{prefix}_inputs"]`)
  with only the new query text swapped in - never a transcript, never a
  second specialist call format. `_record_run()` pushes whatever was
  "current" into that tab's real history list before installing the new
  result, so "Conversation history" only ever contains real prior
  (query, ExecutionTrace) pairs.
- **Agent process timeline** (`routing/timeline.py`, new, shared with the
  PDF report) - a pure, read-only walk of the exact branch order
  `pipeline.run_query()` executes, reporting each of INPUT VALIDATED /
  QUERY INTERPRETED / SPECIALIST SELECTED / ANALYSIS EXECUTED / EVIDENCE
  GENERATED / CONFIDENCE COMPUTED / RESULT READY as done, failed, or not
  reached - strictly from fields already on `ExecutionTrace`. No step is
  ever marked done that did not really happen; a load failure, for
  example, marks step 1 failed and every later step "not reached," never
  guessed.
- **Analysis registry / active specialist indicator.** The sidebar's
  SYSTEM rail now reports REAL MODEL / CLASSICAL / UNAVAILABLE per
  capability (`_capability_backend_label()` - real model only when the
  live-available tool name is literally the one real pretrained-model tool
  in the registry, `tool_single_image_vqa_smolvlm_v1`; every other
  registered tool is a classical baseline by construction). The new
  ACTIVE section shows, per capability, what actually executed in the
  most recent real trace this session (`_active_label()`) - "-" until a
  real run happens, never lit just because a tab was opened.
- **Numeric error codes** (`routing/failure_classification.py`, new,
  shared with the PDF report) - maps the small, fixed set of real
  failure-string prefixes `pipeline.run_query()` can produce to an
  HTTP-style code (400 for bad/ambiguous/incompatible input, 500 for an
  internal specialist exception) and a named category (`INVALID_IMAGE`,
  `AMBIGUOUS_QUERY`, `MISSING_MODALITY`, `VALIDATION_ERROR`,
  `INTERNAL_EXECUTION_ERROR`, `UNEXPECTED_ERROR`). The full original
  failure text and traceback are still shown verbatim; this only adds a
  stable label on top.
- **Change tab BEFORE | AFTER panels.** Mirrors the fusion tab's existing
  pattern: the UI loads the two raw input arrays a second time (same
  `raster_io.load` + `composer.to_displayable_rgb` calls fusion already
  used) and displays them above `tool_change_v0`'s own before|after|
  highlighted composite - `tool_change_v0`'s evidence output shape is
  untouched.
- **Mission-style empty state** - one real, mode-specific "AWAITING INPUT"
  sentence per tab instead of a bare "No analysis run yet." caption.
- **PDF report rewrite** (`src/export/report.py`) - ten named sections
  (Analysis Summary, Input Data, Analysis Result, Visual Evidence, Active
  Specialist, Confidence, Agent Execution Timeline, Technical Trace,
  Limitations & Warnings, Export Metadata), a running header/footer with
  page numbers via a reportlab `onFirstPage`/`onLaterPages` canvas
  callback, and no hardcoded values - every cell is read off the same
  `ExecutionTrace` (plus the two shared derivation helpers above, so the
  PDF's timeline/error-code sections can never disagree with the UI's).

## What did NOT change

Routing, validation, specialist dispatch, confidence computation, and
evidence-image computation are byte-for-byte the same code as v2. The only
backend additions are `routing/timeline.py` and `routing/failure_
classification.py` - both pure, read-only functions that take an already-
built `ExecutionTrace` and return a summary of fields it already carries;
neither calls a model, computes a routing decision, or changes what
`pipeline.run_query()` returns. Every existing `session_state` key from v2
is preserved; new keys (`{prefix}_inputs`, `{prefix}_history`,
`{prefix}_query_used`, `_run_count`, `_last_query`) are additive. Re-
verified by `tests/integration/test_streamlit_app_logic.py` and two new
unit-test files (`tests/unit/test_timeline.py`,
`tests/unit/test_failure_classification.py`) - see docs/RUN_ON_WINDOWS.md
for the full pass/fail count.

## Revision 4 - FINAL UI REFINEMENT (visual-only pass)

This is the final visual refinement pass on top of Revision 3. The goal was
a noticeable composition/information-architecture change, not a spacing or
color tweak - and, unlike every prior pass, an explicit requirement to use
a real design/UI plugin rather than informal taste. **Backend code is
untouched**: no file under `src/` changed in this revision; every change
below is in `app/streamlit_app.py` (markup/CSS/render functions) plus test
and doc updates.

### Plugin selection

Four candidate skills were available: `taste-skill:imagegen-frontend-web`,
`taste-skill:brutalist-skill`, `taste-skill:redesign-skill`, and
`taste-skill:stitch-skill`. **`taste-skill:redesign-skill` was selected as
the single primary plugin.**

Why the other three were rejected:
- `taste-skill:brutalist-skill` was already tried and explicitly rejected
  in an earlier pass (Revision 2) - its industrial/telemetry aesthetic is
  what produced the "developer diagnostic console" complaint this whole
  UI/UX effort has been correcting. Re-applying it would regress, not
  refine.
- `taste-skill:stitch-skill` is built around Google Stitch's own
  DESIGN.md-driven generation workflow. This project hand-writes Streamlit
  markup and CSS directly; there is no Stitch project to drive from.
- `taste-skill:imagegen-frontend-web` is an image-generation skill for
  producing marketing-landing-page visual comps. It has no mechanism for
  editing a live, functional Streamlit application's CSS/Python, so it
  cannot apply here at all.
- `taste-skill:redesign-skill` matches directly: it is an audit-first
  Scan -> Diagnose -> Fix methodology for upgrading an *existing, already-
  working* product's visual layer without breaking functionality or
  migrating frameworks - exactly this task.

What the plugin actually contributed (not just "was invoked"): its
Diagnose checklist directly produced three concrete fixes applied here -
(1) "always one filled button + one ghost button" / inconsistent
color-role usage flagged the pre-Revision-4 bug where every button
(including TRY chips) rendered in the same amber, which is what Section 7
of the brief separately identified; the fix follows the skill's own
"Color and Surfaces" guidance ("pick one accent, remove the rest" - here,
one accent per *role*: amber for execution, cool cyan for suggestion/
secondary) via Streamlit's native `type="primary"`/`type="secondary"`
button-kind attribute. (2) Its "no loading/empty/error states" and
"Oops!-style" checks are why the empty state and error panel were rebuilt
as named, structured states (READY FOR ANALYSIS / ERROR 400) instead of a
single caption string. (3) Its "buttons not bottom-aligned" / "inconsistent
vertical rhythm" checks are why the Input/Result columns were given a real
two-level header (eyebrow + descriptor) instead of one ad hoc div, so both
columns start their content at a consistent position.

Two places where this project deliberately did **not** follow the skill's
generic advice, because a more specific constraint from this project (or
from the brief itself) overrides it:
- The skill's #1 fix-priority is a font swap (Geist/Outfit/Satoshi/Cabinet
  Grotesk). This project has a standing, explicit requirement to work with
  **zero external font dependency** (it must render correctly with no
  internet access on the analyst's machine) - established before this
  pass. The system-safe stack (`Inter, ui-sans-serif, system-ui,
  -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`) was kept.
- The skill recommends replacing all-caps subheaders with sentence case.
  The brief for this exact pass explicitly specifies all-caps instrument-
  console labeling throughout its own mockups (`SYSTEM READY`, `SUGGESTED
  QUERIES`, `ASK YOUR QUESTION`, `ERROR 400`, `WHAT TO DO`). The more
  specific, current instruction wins; labels stay upper-case (applied via
  CSS `text-transform: uppercase` on fixed-case source strings, not by
  typing labels in caps in Python, so the underlying text stays normal
  case for accessibility/copy-paste).

### Design changes (Header / Navigation / Sidebar / Input / Query / Result / Typography / Color / Evidence / Timeline / Conversation / Registry / Errors / Export)

- **Header** - now a two-part row: the existing display title/section/meta
  block, plus a real, live-computed status badge (`_system_status()`,
  which reuses the same registry-backed `_capability_ready()` check the
  sidebar's SYSTEM rail already used) reading "SYSTEM READY" when all four
  capabilities are ready, or "PARTIAL SYSTEM · N/4 READY" otherwise. Never
  a hardcoded string.
- **Navigation** - tab strip keeps Streamlit's native tabs (a full custom
  nav would have meant reimplementing tab state) but each tab's own Input
  column now opens with real per-capability navigation copy from the
  existing `TAB_INFO` table ("01 · INPUT" / "Single image — Ask &
  interpret") instead of a bare "Input" string - the fixed numbering/
  descriptor pairs the brief asked for, applied where Streamlit's tab
  widget itself cannot carry sub-text.
- **Sidebar** - unchanged in structure from Revision 3 (SYSTEM / ACTIVE /
  SESSION rails plus a collapsed "Technical system status" expander for
  raw tool IDs); already matched this brief's "compact system rail, IDs
  behind a details toggle" requirement.
- **Input area** - unchanged short provide/why/format/then guidance
  (Revision 3), now visually anchored under the new section header instead
  of a bare label.
- **Query experience** - reordered and relabeled: "SUGGESTED QUERIES"
  (secondary-styled chips, was "Try") now appears *before* an "ASK YOUR
  QUESTION" label over the query text box, matching the brief's exact
  ordering; the chips only ever set the text field's value via a real
  `on_click` callback and never execute analysis themselves.
- **Result hierarchy** - unchanged from Revision 3's order (answer above
  evidence above confidence/specialist above collapsible execution/
  technical detail); this was already correct per Section 10 of the brief
  and did not need restructuring.
- **Typography** - unchanged font stack (no external dependency); display
  size and body size both increased slightly (`--fs-display` 1.5rem ->
  1.65rem, `--fs-body` 0.95rem -> 1.02rem) for more presence, per the
  skill's typography-scale guidance.
- **Color system** - the four-role system (amber = primary action, cool
  muted cyan `--accent2` = secondary/suggestion, red = error/fallback,
  neutral = everything else) is now actually enforced by Streamlit's own
  `type="primary"`/`type="secondary"` button-kind attribute plus CSS
  targeting `button[kind="..."]`, not just a CSS rule that happened to
  color every button the same. This was the single REQUIRED fix (Section
  7) and the concrete reason the prior pass's TRY buttons all rendered
  amber (nothing distinguished them from Run analysis) - fixed here.
- **Evidence presentation** - unchanged composition (hero-sized evidence
  images, framed with a thin border for a more "instrument" feel); no
  fabricated telemetry added anywhere.
- **Timeline** - `render_agent_timeline()` rewritten from a flat glyph-per-
  row list to a single-HTML-string, single-`st.markdown()`-call vertical
  dot/connector structure (dot + line between steps), still deriving every
  status strictly from `routing/timeline.py`. There is intentionally **no**
  fourth "currently running / active-amber" visual state: an
  `ExecutionTrace` only exists once a run has already finished, so a live-
  progress indicator would be fabricated. Only the three real states
  (done / failed / not-reached) are rendered.
- **Conversation** - `render_conversation_history()` rewritten to a "You: /
  SatQuery:" transcript instead of a Q/A key-value table - still reading
  the exact same real `(query, ExecutionTrace)` history pairs, no
  synthetic chat content.
- **Registry** - unchanged from Revision 3 (already real, already
  distinguishing REAL MODEL/CLASSICAL/UNAVAILABLE, already driven only by
  actual execution for the ACTIVE row).
- **Error states** - `render_failure()` rewritten to the brief's exact
  ERROR/eyebrow/title/why/"WHAT TO DO"/next-step layout, and now includes
  a real, functional **"Return to analysis"** button (`type="secondary"`)
  that calls a new `_clear_tab_state()` helper (drops that tab's trace/run
  dir/inputs/history from `session_state`) and reruns - not a decorative
  label.
- **Export/PDF** - `render_export_row()` reordered so the PDF report is
  the first, primary-styled action and the JSON trace is the secondary
  action beside it (was JSON-first, both same default style). `report.py`
  itself is unchanged.

## What did NOT change (Revision 4)

No file under `src/` changed in this revision - `pipeline.run_query()`,
the router, the registry, every specialist, evidence computation,
confidence computation, `ExecutionTrace`/`SpecialistOutput`, and
`src/export/report.py`'s PDF/JSON generation are byte-for-byte the same as
Revision 3. `routing/timeline.py` and `routing/failure_classification.py`
are read as-is, not modified. Every `session_state` key from Revision 3 is
preserved; the only new key introduced is `_clear_tab_state()`'s own
targeted `.pop()` of five existing per-tab keys, and no new key is added.
Re-verified by the full suite plus four new tests targeting exactly the
new functional wiring - see docs/RUN_ON_WINDOWS.md.
