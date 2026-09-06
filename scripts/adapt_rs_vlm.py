#!/usr/bin/env python3
"""Reproducible LoRA/PEFT adaptation script for tool_rs_vlm_adapted_v1.

STATUS: NEVER EXECUTED. This script has not completed a training run in any
environment as part of this project. It exists so the adaptation procedure is
fully specified and reviewable (per the build brief's requirement to "create a
reproducible training/adaptation script" and to "not claim adaptation until it
is actually executed and verified") - not as evidence that adaptation happened.

What this script does, when it CAN run (see the guard below, which is why it
CANNOT run in the current development sandbox):
  1. Load configs/rs_adaptation.yaml (single source of truth for every
     hyperparameter - see docs/rs_adaptation.md).
  2. Load the base model `google/paligemma2-3b-pt-448` and processor from
     Hugging Face. This is a GATED model: whoever runs this must already have
     clicked "agree and access repository" on the model page and run
     `huggingface-cli login` (or set `HF_TOKEN`) beforehand - see
     docs/RUN_ON_WINDOWS.md. The precondition check below verifies a token is
     present, not that access has actually been granted (that can only be
     confirmed by the download itself).
  3. Load the VRSBench `vqa_train` and `referring_train` splits (never the
     `*_test` splits, which are reserved for src/evaluation/benchmarks.py).
  4. Wrap the base model with a peft LoraConfig and train for the configured
     number of epochs.
  5. Save the adapter weights plus a run_metadata.json matching the contract
     in docs/rs_adaptation.md section 5 - with every field filled from what
     ACTUALLY happened (measured dataset counts, real timestamps, real final
     loss), never copied from the config's estimates.

Why it cannot run here: this development sandbox has zero internet egress
(see docs/network_constraints.md) and none of torch/transformers/peft/datasets
are installed (see requirements-future.txt). Rather than let the script fail
midway with a confusing traceback - or worse, silently skip steps and produce
a fake-looking checkpoint - it checks its preconditions up front and exits
loudly and specifically.
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "rs_adaptation.yaml")


class AdaptationNotExecutableHere(RuntimeError):
    """Raised when this environment cannot honestly run the adaptation."""


def _check_preconditions() -> None:
    missing = []
    for pkg in ("torch", "transformers", "peft", "datasets"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise AdaptationNotExecutableHere(
            "RS_ADAPTATION_NOT_EXECUTABLE_HERE: missing required packages "
            f"{missing}. These are listed in requirements-future.txt "
            "specifically because they could not be installed in this "
            "sandbox (see docs/network_constraints.md). Install them on a "
            "networked machine before running this script for real."
        )

    # Even if the packages above were installed, a real run also needs
    # network access to pull the base model weights and the VRSBench
    # dataset. We check this explicitly rather than letting a download
    # call hang or fail deep inside a library.
    try:
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
             "--max-time", "5", "https://huggingface.co"],
            capture_output=True, text=True, timeout=10,
        )
        reachable = result.stdout.strip() not in ("", "000")
    except Exception:
        reachable = False
    if not reachable:
        raise AdaptationNotExecutableHere(
            "RS_ADAPTATION_NOT_EXECUTABLE_HERE: huggingface.co is not "
            "reachable from this environment (see docs/network_constraints.md "
            "and docs/rs_adaptation.md section 7 for the full egress test "
            "results, including the device-bridge VM checked in Phase 2B). "
            "Cannot download google/paligemma2-3b-pt-448 or VRSBench."
        )

    # google/paligemma2-3b-pt-448 is a GATED model: a bare download attempt
    # without an accepted-license token fails with a 403 from the Hub, not a
    # network error. Check for a token up front so that failure mode gets a
    # clear, specific message instead of a generic HTTP error deep in
    # transformers/huggingface_hub.
    has_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    if not has_token:
        try:
            from huggingface_hub import HfFolder  # type: ignore
            has_token = bool(HfFolder.get_token())
        except Exception:
            has_token = False
    if not has_token:
        raise AdaptationNotExecutableHere(
            "RS_ADAPTATION_NOT_EXECUTABLE_HERE: no Hugging Face auth token "
            "found (checked HF_TOKEN, HUGGING_FACE_HUB_TOKEN, and the local "
            "huggingface-cli login cache). google/paligemma2-3b-pt-448 is a "
            "gated model - visit its model page, click 'agree and access "
            "repository', then run `huggingface-cli login` before retrying. "
            "See docs/RUN_ON_WINDOWS.md."
        )


def _load_config() -> dict:
    import yaml  # PyYAML - available in this sandbox even though torch is not

    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main() -> int:
    print("adapt_rs_vlm.py: checking preconditions before touching any model "
          "or dataset code (never claim a run happened without one)...")
    try:
        _check_preconditions()
    except AdaptationNotExecutableHere as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        print(
            "Exiting without writing a checkpoint or a run_metadata.json. "
            "tool_rs_vlm_adapted_v1 remains status=NOT_AVAILABLE_SANDBOX in "
            "src/models/registry.py until this script actually completes on "
            "a suitable machine.",
            file=sys.stderr,
        )
        return 1

    cfg = _load_config()
    out_dir = os.path.join(REPO_ROOT, cfg["training"]["output_dir"], cfg["run_name"])
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # From here on this script would, on a real machine:
    #   from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
    #   from peft import LoraConfig, get_peft_model
    #   from datasets import load_dataset
    #   ... load base_model["repo_id"], load VRSBench splits_used, wrap with
    #   LoraConfig(**cfg["lora"]), train per cfg["training"], and save.
    # It intentionally stops at _check_preconditions() above in every
    # environment where those imports would fail, so this file can be read
    # end-to-end as the intended procedure without ever pretending to have
    # run it.
    # ------------------------------------------------------------------

    metadata = {
        "base_model": cfg["base_model"]["repo_id"],
        "adapter_type": "LoRA",
        "lora_config": cfg["lora"],
        "dataset": cfg["dataset"]["name"],
        "dataset_splits_used": cfg["dataset"]["splits_used"],
        "dataset_image_count_measured": None,
        "dataset_local_hash": None,
        "seed": cfg["seed"],
        "epochs_completed": 0,
        "train_loss_curve_path": None,
        "started_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "completed_at_utc": None,
        "git_commit": None,
        "executed_on": None,
        "status": "NOT_EXECUTED",
    }
    meta_path = os.path.join(out_dir, "run_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote a NOT_EXECUTED placeholder metadata file to {meta_path}. "
          "This is a record of intent, not a completed run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
