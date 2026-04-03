"""
Image quality checks: sharpness (Laplacian variance) and brightness.
Returns a quality gate result used before running the CV pipeline.
"""

import numpy as np
from PIL import Image, ImageFilter

SHARPNESS_THRESHOLD = 80.0   # Laplacian variance below this → blurry
BRIGHTNESS_MIN = 40.0        # Mean luminance below this → too dark
BRIGHTNESS_MAX = 220.0       # Mean luminance above this → overexposed


def _laplacian_variance(gray: np.ndarray) -> float:
    """Measure sharpness via Laplacian variance."""
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    from scipy.signal import convolve2d
    lap = convolve2d(gray.astype(np.float32), kernel, mode="valid")
    return float(np.var(lap))


def _mean_brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray))


def check_quality(img: Image.Image) -> dict:
    """
    Returns:
        {
            "quality": "good" | "poor",
            "issues": [...],
            "sharpness": float,
            "brightness": float
        }
    """
    # Resize to a fixed size for consistent measurement
    small = img.resize((256, 256)).convert("L")
    gray = np.array(small)

    sharpness = _laplacian_variance(gray)
    brightness = _mean_brightness(gray)

    issues = []
    if sharpness < SHARPNESS_THRESHOLD:
        issues.append("Image is blurry — please hold still and retake.")
    if brightness < BRIGHTNESS_MIN:
        issues.append("Image is too dark — move to a brighter area.")
    if brightness > BRIGHTNESS_MAX:
        issues.append("Image is overexposed — reduce lighting and retake.")

    return {
        "quality": "poor" if issues else "good",
        "issues": issues,
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
    }
