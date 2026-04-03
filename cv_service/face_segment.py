"""
Face segmentation using MediaPipe Face Landmarker (Tasks API — mediapipe 0.10+).
Builds a precise skin mask from 478 facial landmarks.
RAM usage: ~80MB. No PyTorch required.
"""

from __future__ import annotations
import os
import numpy as np
from PIL import Image
import cv2
import mediapipe as mp

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

_landmarker = None


def _load_model() -> None:
    global _landmarker
    if _landmarker is not None:
        return
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    _landmarker = FaceLandmarker.create_from_options(options)


# Face oval silhouette landmark indices
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Regions to exclude from skin mask
LEFT_EYE       = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE      = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYEBROW   = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
RIGHT_EYEBROW  = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
LIPS_OUTER     = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
NOSTRILS       = [2, 326, 327, 328, 294, 278, 344, 440, 275, 45, 220, 115, 48, 64, 98, 97]


def _to_poly(landmarks, indices: list[int], w: int, h: int) -> np.ndarray:
    return np.array(
        [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices],
        dtype=np.int32,
    )


def get_skin_mask(img: Image.Image) -> np.ndarray:
    """Returns boolean mask (H, W) — True = skin pixel."""
    _load_model()

    rgb = np.array(img.convert("RGB"))
    h, w = rgb.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _landmarker.detect(mp_image)

    if not result.face_landmarks:
        return _hsv_skin_mask(rgb)

    lm = result.face_landmarks[0]
    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(mask, [_to_poly(lm, FACE_OVAL, w, h)], 255)

    for region in [LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, LIPS_OUTER, NOSTRILS]:
        poly = _to_poly(lm, region, w, h)
        if len(poly) >= 3:
            cv2.fillPoly(mask, [poly], 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.erode(mask, kernel, iterations=1)

    return mask.astype(bool)


def _hsv_skin_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([25, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([170, 20, 70]), np.array([180, 255, 255]))
    mask = m1 | m2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel).astype(bool)
