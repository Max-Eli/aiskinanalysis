"""
Face segmentation using MediaPipe Face Mesh.
Builds a precise skin mask from 478 facial landmarks — no heavy ML model needed.
RAM usage: ~80MB vs ~1.5GB for SegFormer.

Skin regions included: forehead, cheeks, nose, chin, jaw
Excluded: eyes, eyebrows, lips, nostrils (inner), ears, hair, background
"""

from __future__ import annotations
import numpy as np
from PIL import Image
import cv2
import mediapipe as mp

_face_mesh = None


def _load_model() -> None:
    global _face_mesh
    if _face_mesh is not None:
        return
    _face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )


# MediaPipe Face Mesh landmark indices for face outline (silhouette)
# These trace the outer boundary of the face
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Regions to EXCLUDE (punch holes in the skin mask)
LEFT_EYE = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466,
    388, 387, 386, 385, 384, 398,
]
RIGHT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173,
    157, 158, 159, 160, 161, 246,
]
LEFT_EYEBROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
RIGHT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
LIPS_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
]
LIPS_INNER = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
]
NOSTRILS = [2, 326, 327, 328, 294, 278, 344, 440, 275, 45, 220, 115, 48, 64, 98, 97]


def _landmarks_to_poly(landmarks, indices: list[int], w: int, h: int) -> np.ndarray:
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    return np.array(pts, dtype=np.int32)


def get_skin_mask(img: Image.Image) -> np.ndarray:
    """
    Returns a boolean mask (H, W) — True = skin pixel.
    Uses MediaPipe Face Mesh polygon fill, not a neural segmentation model.
    """
    _load_model()

    rgb = np.array(img.convert("RGB"))
    h, w = rgb.shape[:2]

    result = _face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        # Fallback: HSV-based skin color detection if no face detected
        return _hsv_skin_mask(rgb)

    lm = result.multi_face_landmarks[0].landmark
    mask = np.zeros((h, w), dtype=np.uint8)

    # Fill face oval
    face_poly = _landmarks_to_poly(lm, FACE_OVAL, w, h)
    cv2.fillPoly(mask, [face_poly], 255)

    # Punch out non-skin regions
    for region in [LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, LIPS_OUTER, NOSTRILS]:
        poly = _landmarks_to_poly(lm, region, w, h)
        if len(poly) >= 3:
            cv2.fillPoly(mask, [poly], 0)

    # Extra margin erosion to avoid edge artifacts (hair, jaw shadow)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.erode(mask, kernel, iterations=1)

    return mask.astype(bool)


def _hsv_skin_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Pure color-based fallback when no face landmarks detected.
    Works reasonably on well-lit frontal photos.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Skin tone range in HSV
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    # Also catch slightly darker skin tones
    lower2 = np.array([170, 20, 70], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask |= cv2.inRange(hsv, lower2, upper2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.astype(bool)
