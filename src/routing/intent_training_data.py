"""Hand-written labeled training examples for the query-intent classifier.

This is real training data (not fabricated results) - short example queries an
analyst might type, labeled with which single-image task they express. It only
disambiguates between 'single_image_vqa' and 'grounding'; the change/fusion
branches are decided by hard input signals (image count, declared modality,
declared dates) in router.py, not by this classifier - see docs/architecture.md
for why those two are handled differently.

Grown/edited by hand is fine and expected; keep it balanced between the two
classes when adding examples.
"""
from __future__ import annotations

TRAINING_EXAMPLES = [
    # --- grounding: user wants a LOCATION / region highlighted ---
    ("locate the water body in this image and highlight it", "grounding"),
    ("where is the river in this scene", "grounding"),
    ("highlight the built-up area", "grounding"),
    ("point to the largest field in the image", "grounding"),
    ("show me where the forest is", "grounding"),
    ("draw a box around the lake", "grounding"),
    ("mark the urban region on the image", "grounding"),
    ("find and highlight all the roads", "grounding"),
    ("where are the buildings located", "grounding"),
    ("can you show the exact location of the water body", "grounding"),
    ("outline the vegetated area", "grounding"),
    ("identify the position of the reservoir", "grounding"),
    ("segment the water region", "grounding"),
    ("highlight where the bare soil is", "grounding"),
    ("show the bounding box for the agricultural field", "grounding"),
    # --- single_image_vqa: user wants a DESCRIPTION / answer, not a location ---
    ("what are the major features in this image", "single_image_vqa"),
    ("describe the land cover in this scene", "single_image_vqa"),
    ("what is the dominant terrain type here", "single_image_vqa"),
    ("is there any water visible in this image", "single_image_vqa"),
    ("how much of this image is vegetated", "single_image_vqa"),
    ("what kind of land use does this represent", "single_image_vqa"),
    ("summarize what you see in this satellite image", "single_image_vqa"),
    ("does this image show an urban or rural area", "single_image_vqa"),
    ("what objects are visible in the scene", "single_image_vqa"),
    ("give a scene description of this image", "single_image_vqa"),
    ("is this region mostly desert or forest", "single_image_vqa"),
    ("what percentage of the image is built-up", "single_image_vqa"),
    ("classify the land cover type", "single_image_vqa"),
    ("tell me about this satellite image", "single_image_vqa"),
    ("what stands out in this scene", "single_image_vqa"),
]
