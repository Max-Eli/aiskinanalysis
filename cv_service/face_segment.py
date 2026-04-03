"""
Face parsing via SegFormer (jonathandinu/face-parsing on HuggingFace).
Produces a binary skin mask — excludes hair, eyes, eyebrows, lips, nostrils,
ears, background.  Focuses on forehead, cheeks, nose, chin.

Label map (CelebAMask-HQ convention used by jonathandinu/face-parsing):
  0  background
  1  skin
  2  nose
  3  right eye glass
  4  left eye glass
  5  right eye
  6  left eye
  7  right eyebrow
  8  left eyebrow
  9  right ear
 10  left ear
 11  earring
 12  mouth (interior)
 13  upper lip
 14  lower lip
 15  neck
 16  necklace
 17  cloth
 18  hair
 19  hat
"""

from __future__ import annotations
import numpy as np
from PIL import Image
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# Labels we KEEP for skin analysis
SKIN_LABELS = {1, 2}   # skin + nose bridge

_processor = None
_model = None


def _load_model() -> None:
    global _processor, _model
    if _model is not None:
        return
    model_id = "jonathandinu/face-parsing"
    _processor = SegformerImageProcessor.from_pretrained(model_id)
    _model = SegformerForSemanticSegmentation.from_pretrained(model_id)
    _model.eval()


def get_skin_mask(img: Image.Image) -> np.ndarray:
    """
    Returns a boolean mask (H, W) where True = skin pixel.
    Image is returned at its original size.
    """
    _load_model()

    inputs = _processor(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits        # (1, num_labels, H', W')

    # Upsample to original size
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=(img.height, img.width),
        mode="bilinear",
        align_corners=False,
    )
    pred = upsampled.argmax(dim=1).squeeze(0).numpy()   # (H, W)

    mask = np.zeros_like(pred, dtype=bool)
    for label in SKIN_LABELS:
        mask |= (pred == label)

    return mask
