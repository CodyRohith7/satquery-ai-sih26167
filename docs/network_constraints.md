# Network constraints (development sandbox)

Recorded 2026-09-05. This document exists so nobody re-discovers this the hard way
mid-demo-prep, and so every "v0" / "NOT_AVAILABLE_SANDBOX" label elsewhere in this
repo has a documented reason.

## What was tested

From the cloud development sandbox used to build the first cut of this repo:

| Host | Result |
|---|---|
| `pypi.org` / `files.pythonhosted.org` | `403 host_not_allowed` |
| `huggingface.co` | `403` (explicit policy denial, `connect_rejected`) |
| `download.pytorch.org` | `403 host_not_allowed` |
| `github.com`, `raw.githubusercontent.com` | `403 host_not_allowed` |
| `storage.googleapis.com`, `zenodo.org` | `403 host_not_allowed` |
| `dataspace.copernicus.eu` (Sentinel data) | `403 host_not_allowed` |
| `pip install <anything not already installed>` | fails instantly, 0 candidate links found — no index reachable at all |

## What this means

- No pretrained model weights (Hugging Face Hub, PyTorch Hub, or any other host) can
  be downloaded into this sandbox.
- No real satellite datasets (Copernicus/Sentinel, BigEarthNet, USGS, Bhuvan) can be
  downloaded into this sandbox.
- `torch`, `torchvision`, `transformers`, `peft`, `rasterio`, and `streamlit` are **not
  installed** and cannot be installed here (confirmed: not present in the base image,
  and `pip install` cannot reach any package index).

## What IS available and used instead

`opencv-python`, `numpy`, `Pillow`, `scikit-learn`, `scikit-image`, `scipy`, `pandas`,
`PyYAML`, and `reportlab` are preinstalled and confirmed working. Every specialist
branch in this repo's v0 implementation is built exclusively from that list — see
`src/specialists/`, `src/grounding/`, `src/change/`, `src/fusion/`.

## Consequence for the "remote-sensing adaptation" PS requirement

The genuine LoRA/PEFT fine-tuning of an open-source vision-language model on a
BigEarthNet subset — the approach frozen in the Phase 0 architecture doc — **has not
been executed or verified**, and is not claimed as working anywhere in this repo.
`docs/rs_adaptation.md` names the exact model and dataset this is designed for and
ships a runnable script, explicitly labeled not-yet-executed, for the first
environment that has internet access (a teammate's laptop, a Colab/Kaggle notebook,
or a machine where this project is re-opened with network access restored).

## How to unblock

1. Run this repo on a machine with normal internet access (a laptop is enough for the
   v0 classical-CV branches; a GPU is preferable but not required for LoRA on a small
   model/subset).
2. From there: `pip install -r requirements.txt -r requirements-optional.txt`
   (the LoRA/dataset packages — `peft`, `datasets` — live in
   `requirements-optional.txt`), then follow `docs/rs_adaptation.md` to run
   the adaptation script and verify it end-to-end.
3. Swap the resulting model into `src/models/registry.py` by changing its `status`
   from `NOT_AVAILABLE_SANDBOX` to `AVAILABLE` — the router and contracts do not
   change.

## Update: resolved for classical CV + SmolVLM on the target Windows machine

The blockers above applied to the isolated development environment described
in this document. On the user's real target machine (Windows, `conda`
environment `satquery`), `torch`, `transformers`, `accelerate`, and
`streamlit` install and run normally with real internet access — see
`docs/RUN_ON_WINDOWS.md` for the verified, real-hardware results. This let
`HuggingFaceTB/SmolVLM-256M-Instruct` be wired in as a real, working
single-image VQA specialist (`tool_single_image_vqa_smolvlm_v1`, with an
honest fallback to the classical baseline on failure).
The larger remote-sensing-domain LoRA adaptation described in
`docs/rs_adaptation.md` (originally targeting PaliGemma 2) remains
**not executed**: it was evaluated on the target machine and found
impractical on available CPU-only hardware, not blocked by network access.
`tool_rs_vlm_adapted_v1` therefore still reports `NOT_AVAILABLE_SANDBOX`,
and grounding/change/fusion remain classical CV baselines
(`tool_grounding_v0`, `tool_change_v0`, `tool_fusion_v0`) — see
`docs/rs_adaptation.md` for the full history and `README.md` for the
current-state summary.
