# Confidence methodology

Written before the confidence engine's UI so the number always has a defined
meaning, per the frozen Phase 0 decision ("implement the real calculation before
designing the UI around it"). If `src/confidence/engine.py` ever diverges from
this document, the document is wrong and must be fixed in the same change - they
must never drift apart.

## Formula

```
final_confidence = clamp(base_signal * (1 - total_downgrade), 0.0, 1.0)
```

`base_signal` is specialist-specific (see below) and always derived from a real,
computed number - never a constant, never invented.

`total_downgrade` is the sum of rule-based penalties (capped at 0.9 so confidence
never hits exactly zero from rules alone, since that would look like a bug rather
than a low-confidence answer):

| Condition | Downgrade |
|---|---|
| Any input's `modality_confidence == "unknown"` | 0.15 |
| Any validation warning present | 0.10 per warning, max 0.30 |
| Router itself routed with confidence < 0.6 (ambiguous query) | 0.20 |
| Image pair dimensions mismatched (validator warning) | already covered above |

## `method_version` and `basis_description`

`compute()` takes two labeling parameters, defaulted for the classical
specialists so their four existing call sites needed no changes:

- `method_version`: which base_signal methodology produced this result.
  `"v0_classical"` for the four classical-CV specialists below;
  `"v1_vlm_mean_token_probability"` for `tool_single_image_vqa_smolvlm_v1`
  (see its own entry below). **A specialist must never report a
  `method_version` that doesn't match what actually produced its
  `base_signal`** - this was a real bug, caught and fixed: SmolVLM's
  confidence was originally computed through the same `compute()` call as
  every classical specialist, with no way to override the label, so its
  result was stamped `method_version: "v0_classical"` even though its
  `base_signal` came from real generation-token probabilities, not a
  clustering/threshold score - exactly the confusion this document's
  original "Known limitation" section (below) was written to prevent.
- `basis_description`: a plain-English, per-result explanation of what
  `base_signal` actually measures, so a UI can display it verbatim without
  a separate lookup table keyed by `method_version`. Every `ConfidenceResult`
  carries one.

## Per-specialist `base_signal`

- **single_image_vqa (`tool_single_image_vqa_v0`)**: silhouette score of the
  k-means color/texture clustering used to derive the scene description,
  rescaled from its native `[-1, 1]` range to `[0, 1]`. A well-separated scene
  (clearly distinct water/vegetation/urban/soil clusters) scores high; a
  homogeneous or noisy image scores low. This is a real `sklearn.metrics.
  silhouette_score` call, not a placeholder.
- **grounding (`tool_grounding_v0`)**: fraction of the thresholded candidate
  region that survives morphological cleanup (contour area after
  open/close operations, divided by the raw thresholded area before cleanup),
  capped at 1.0. A clean, coherent blob scores high; a scattered, noisy
  threshold result (weak evidence for "there's really a discrete water body
  here") scores low.
- **bitemporal_change (`tool_change_v0`)**: normalized separation between the
  "changed" and "unchanged" pixel populations in the difference image, measured
  as `(otsu_threshold_between_class_variance / total_variance)` - a standard
  Otsu-quality metric. A clean bimodal difference image (obvious change vs. no
  change) scores high; a noisy, unimodal difference image (the algorithm isn't
  really sure where the boundary is) scores low.
- **optical_sar_fusion (`tool_fusion_v0`)**: silhouette score of the k-means
  clustering run on the fused (stacked) feature vector, same metric family as
  VQA above but computed on the joint optical+SAR feature space.
- **single_image_vqa (`tool_single_image_vqa_smolvlm_v1`, real model)**: mean
  of the top predicted token's softmax probability across every newly
  generated token in that specific answer's generation (from
  `model.generate(..., output_scores=True, return_dict_in_generate=True)`).
  This is a real, model-native number - not fabricated, not a fixed
  constant - but it measures **generation certainty, not factual
  correctness**: a fluently, confidently-worded wrong answer can still
  score high, and a correct answer using rarer wording can score lower. It
  is explicitly labeled `method_version: "v1_vlm_mean_token_probability"`
  (never `"v0_classical"`) so it is never mistaken for either a classical
  heuristic or a calibrated "this answer is correct" probability. See
  `src/specialists/vqa_smolvlm.py`.

## Model and fallback disclosure (`SpecialistOutput` fields)

Beyond confidence, every `SpecialistOutput` also carries, as first-class
fields a UI can read directly - never inferred from `tool_name` string
matching or from which keys happen to be present in `raw`:

- `model_name`: the real pretrained model's identity (e.g.
  `"HuggingFaceTB/SmolVLM-256M-Instruct"`) when one produced this output;
  `None` for classical-CV specialists, which are techniques, not models.
- `fallback_occurred` / `fallback_reason`: `True` and the real exception
  message only when a preferred real-model specialist was attempted and
  failed, and a different specialist's output is being returned instead
  (see `src/specialists/vqa_dispatch.py`). `False`/`None` otherwise -
  always present, never absent, so "no fallback happened" and "fallback
  status wasn't reported" can't be confused.

## What this deliberately does NOT do

- No specialist is allowed to hardcode a confidence number.
- No specialist is allowed to report confidence above what its own `base_signal`
  computation produced, before downgrades.
- The router's OWN confidence (how sure it was about which branch to run) is
  tracked separately (`RouterDecision.confidence`) and shown separately in the
  trace - it is never averaged into the specialist's answer confidence, because
  they answer different questions ("did we pick the right tool" vs "how much do
  we trust this tool's output").

## Known limitation

The four classical specialists' signals (cluster separation, threshold
quality) are not a calibrated probability from a trained deep model - see
`docs/network_constraints.md` for why. They are honestly labeled
`method_version: "v0_classical"` in every `ConfidenceResult` so nobody
mistakes them for a neural network's output. `tool_single_image_vqa_smolvlm_v1`
IS a real neural network's output (mean generated-token probability), and
carries its own distinct `method_version` for exactly that reason - see
above. Neither method version is a calibrated "this specific answer is
factually correct" probability; that would require ground-truth-labeled
evaluation data this project does not have (see
`src/evaluation/benchmarks.py`'s `NOT_YET_EVALUATED` status).
