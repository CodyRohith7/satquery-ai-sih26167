"""Model / tool registry.

Every "specialist model or tool" the PS asks us to name has one entry here:
name, purpose, input modality, output, provenance, framework, and - the part
that keeps this honest - a `status` that is either `AVAILABLE` (it runs, today,
in this environment, and is covered by a test) or `NOT_AVAILABLE_SANDBOX` (it is
designed and documented but has not been executed here - see
docs/network_constraints.md for exactly why).

`load()` on an unavailable entry raises `ModelNotAvailableError` with the exact
reason rather than silently substituting a fake result - this is the "never
silently fallback from a real model to fake output" rule from the frozen spec.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Dict, Optional

AVAILABLE = "AVAILABLE"
NOT_AVAILABLE_SANDBOX = "NOT_AVAILABLE_SANDBOX"


class ModelNotAvailableError(RuntimeError):
    pass


@dataclasses.dataclass
class ModelEntry:
    name: str
    purpose: str
    input_modality: str
    output: str
    provenance: str
    framework: str
    status: str
    unavailable_reason: Optional[str] = None
    _loader: Optional[Callable[[], object]] = None

    def load(self):
        if self.status != AVAILABLE:
            raise ModelNotAvailableError(
                f"'{self.name}' is not available in this environment: "
                f"{self.unavailable_reason}"
            )
        if self._loader is None:
            raise ModelNotAvailableError(
                f"'{self.name}' is marked AVAILABLE but has no loader wired up "
                "- this is a registry bug, not a missing-dependency situation."
            )
        return self._loader()


_REGISTRY: Dict[str, ModelEntry] = {}


def register(entry: ModelEntry) -> None:
    _REGISTRY[entry.name] = entry


def get(name: str) -> ModelEntry:
    if name not in _REGISTRY:
        raise KeyError(f"No such model/tool registered: '{name}'")
    return _REGISTRY[name]


def all_entries() -> Dict[str, ModelEntry]:
    return dict(_REGISTRY)


def _smolvlm_deps_available() -> bool:
    """Real dependency check, not an assumption - True only if torch and
    transformers are actually importable in this environment."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _bootstrap() -> None:
    """Registers the v0 classical-CV tools (real, available today), the
    real deep-learning VQA specialist (SmolVLM-256M-Instruct - AVAILABLE
    only where torch/transformers are actually importable, checked live,
    not assumed), and the v1 RS-adapted specialist (designed, not executable
    in this sandbox)."""
    from specialists import vqa as vqa_mod
    from specialists import vqa_smolvlm as vqa_smolvlm_mod
    from grounding import grounding as grounding_mod
    from change import change as change_mod
    from fusion import fusion as fusion_mod

    register(
        ModelEntry(
            name="tool_single_image_vqa_v0",
            purpose="Answer natural-language questions about a single image via "
            "unsupervised color/texture clustering and a rule-based label map. "
            "Serves as the explicit, disclosed fallback for "
            "tool_single_image_vqa_smolvlm_v1 when that model cannot load or "
            "run (see specialists/vqa_dispatch.py) - not the primary VQA path "
            "wherever the real model is available.",
            input_modality="single image (any band count)",
            output="natural-language answer + cluster silhouette score",
            provenance="classical CV, no external weights; scikit-learn KMeans",
            framework="scikit-learn + OpenCV",
            status=AVAILABLE,
            _loader=lambda: vqa_mod.run,
        )
    )
    _smolvlm_available = _smolvlm_deps_available()
    register(
        ModelEntry(
            name="tool_single_image_vqa_smolvlm_v1",
            purpose="Answer open-ended natural-language questions about a single "
            "image via a real vision-language model - not a rule-based or "
            "clustering stand-in. This is the PRIMARY single-image VQA "
            "specialist wherever it is available; tool_single_image_vqa_v0 is "
            "its disclosed fallback (see specialists/vqa_dispatch.py).",
            input_modality="single image (RGB)",
            output="natural-language answer + mean generated-token confidence",
            provenance="HuggingFaceTB/SmolVLM-256M-Instruct, off-the-shelf "
            "(NOT remote-sensing-adapted); verified on real CPU hardware - "
            "see docs/rs_adaptation.md and scripts/first_success_test_vqa.py "
            "(load 46.891s, inference 38.457s, peak working set 2882.7MB)",
            framework="PyTorch + transformers",
            status=AVAILABLE if _smolvlm_available else NOT_AVAILABLE_SANDBOX,
            unavailable_reason=None if _smolvlm_available else (
                "requires torch + transformers; not importable in this environment"
            ),
            _loader=(lambda: vqa_smolvlm_mod.run) if _smolvlm_available else None,
        )
    )
    register(
        ModelEntry(
            name="tool_grounding_v0",
            purpose="Locate and box a queried feature (water/vegetation/urban/"
            "bare-soil) via colour-space thresholding and contour extraction.",
            input_modality="single image (>=3 bands assumed for colour heuristics)",
            output="bounding box(es) + cleanup-survival ratio",
            provenance="classical CV, no external weights; OpenCV thresholding",
            framework="OpenCV",
            status=AVAILABLE,
            _loader=lambda: grounding_mod.run,
        )
    )
    register(
        ModelEntry(
            name="tool_change_v0",
            purpose="Detect and map change between two co-registered images via "
            "image differencing, Otsu thresholding, and morphological cleanup.",
            input_modality="two images, same or resampled dimensions",
            output="change mask + change-area statistics + Otsu separation score",
            provenance="classical CV, no external weights; OpenCV/scikit-image",
            framework="OpenCV + scikit-image",
            status=AVAILABLE,
            _loader=lambda: change_mod.run,
        )
    )
    register(
        ModelEntry(
            name="tool_fusion_v0",
            purpose="Jointly cluster stacked optical+SAR-like features to "
            "identify built-up vs. water-covered regions.",
            input_modality="one optical (>=3 band) + one single-band image",
            output="fused segmentation map + cluster silhouette score",
            provenance="classical CV, no external weights; scikit-learn KMeans "
            "over a hand-stacked feature vector",
            framework="scikit-learn + OpenCV",
            status=AVAILABLE,
            _loader=lambda: fusion_mod.run,
        )
    )
    register(
        ModelEntry(
            name="tool_rs_vlm_adapted_v1",
            purpose="Remote-sensing-adapted vision-language model for VQA and "
            "referring-expression grounding, LoRA fine-tuned on the VRSBench "
            "VQA + grounding subset (see docs/rs_adaptation.md for the exact "
            "base model, exact dataset subset, and adaptation status).",
            input_modality="single image, GeoTIFF preferred",
            output="natural-language answer / bounding box with model-native "
            "confidence",
            provenance="intended base: google/paligemma2-3b-pt-448 + a LoRA "
            "adapter trained on VRSBench; NOT YET TRAINED - see "
            "docs/rs_adaptation.md for why and for the reproducible script",
            framework="PyTorch + transformers + peft (none installed here)",
            status=NOT_AVAILABLE_SANDBOX,
            unavailable_reason=(
                "requires torch/transformers/peft and a VRSBench download; "
                "this sandbox has zero internet egress (docs/network_constraints.md)"
            ),
        )
    )


_bootstrap()
