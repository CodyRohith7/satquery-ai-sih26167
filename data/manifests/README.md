# Data manifests

Empty by design. Per the Phase 2B DATA requirements, every real dataset
subset pulled for RS adaptation must get a manifest file here recording:
source, version/date, subset, preprocessing, and license/source URL.

No manifest exists yet because no real dataset has been downloaded from any
environment this project has actually run in (see `docs/network_constraints.md`
and `docs/rs_adaptation.md` section 7 for why). The first real download
(done by a human on their own networked machine per `docs/RUN_ON_WINDOWS.md`)
should add `vrsbench.json` here, following this shape:

```json
{
  "source": "https://huggingface.co/datasets/xiang709/VRSBench",
  "downloaded_at_utc": null,
  "subset": "train split, first N examples (state exact N once chosen)",
  "measured_image_count": null,
  "measured_vqa_pair_count": null,
  "measured_referring_pair_count": null,
  "license": "CC BY-NC 4.0 (non-commercial)",
  "preprocessing": "describe exactly what was resized/filtered/cleaned"
}
```

Every `null` above is a placeholder for a value the real download must
measure - never invent one to fill this in early.
