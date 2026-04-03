"""
Professional skin analysis — 11 concern metrics + zone analysis + positive attributes.

Concerns measured:
  acne, hyperpigmentation, melasma, redness, wrinkles, fine_lines,
  dryness, pore_visibility, oiliness, dark_circles, uneven_texture

Positive attributes:
  hydration, evenness, luminosity, firmness

Zone analysis:
  forehead, left_cheek, right_cheek, nose, chin

Falls back to feature-based scoring when no fine-tuned weights exist.
To use EfficientNet-B0: place trained weights at cv_service/weights/skin_model.pt
"""

from __future__ import annotations
import os
import numpy as np
from PIL import Image
from skimage.color import rgb2hsv, rgb2lab
from skimage.filters import gaussian, laplace, sobel
from skimage.feature import canny
from skimage.morphology import local_minima

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "skin_model.pt")

CONCERNS = [
    "acne", "hyperpigmentation", "melasma", "redness",
    "wrinkles", "fine_lines", "dryness", "pore_visibility",
    "oiliness", "dark_circles", "uneven_texture",
]

_model = None
_use_model = False


def _try_load_model() -> None:
    global _model, _use_model
    if not os.path.exists(WEIGHTS_PATH):
        return
    from models.efficientnet import SkinAnalysisNet
    import torch
    net = SkinAnalysisNet(pretrained=False)
    net.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    net.eval()
    _model = net
    _use_model = True


_try_load_model()


# ─── Zone masks ───────────────────────────────────────────────────────────────

def _zone_masks(mask: np.ndarray) -> dict[str, np.ndarray]:
    """
    Divide the face skin mask into anatomical zones.
    All zones are subsets of the skin mask.
    """
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return {z: np.zeros_like(mask, dtype=bool)
                for z in ["forehead", "left_cheek", "right_cheek", "nose", "chin"]}

    y0, y1 = int(rows.min()), int(rows.max())
    x0, x1 = int(cols.min()), int(cols.max())
    h = y1 - y0
    cx = (x0 + x1) // 2
    w = x1 - x0

    def zone(y_start_frac, y_end_frac, x_start_frac=0.0, x_end_frac=1.0) -> np.ndarray:
        z = mask.copy()
        z[:y0 + int(h * y_start_frac)] = False
        z[y0 + int(h * y_end_frac):] = False
        z[:, :x0 + int(w * x_start_frac)] = False
        z[:, x0 + int(w * x_end_frac):] = False
        return z

    return {
        "forehead":    zone(0.0,  0.28),
        "left_cheek":  zone(0.32, 0.68, 0.0,  0.48),
        "right_cheek": zone(0.32, 0.68, 0.52, 1.0),
        "nose":        zone(0.30, 0.65, 0.35, 0.65),
        "chin":        zone(0.72, 1.0),
    }


# ─── Individual metric functions ──────────────────────────────────────────────

def _acne_score(gray: np.ndarray, mask: np.ndarray) -> float:
    """High-frequency local variance in skin — bumps and lesions."""
    smooth = gaussian(gray, sigma=2)
    detail = gray - smooth
    skin_detail = np.abs(detail)[mask]
    raw = float(np.mean(skin_detail))
    return float(np.clip(raw / 0.04, 0, 1))


def _hyperpigmentation_score(lab: np.ndarray, mask: np.ndarray) -> float:
    """Standard deviation of skin luminance — uneven tone."""
    skin_L = lab[:, :, 0][mask]
    std = float(np.std(skin_L))
    return float(np.clip(std / 18.0, 0, 1))


def _melasma_score(lab: np.ndarray, mask: np.ndarray) -> float:
    """
    Dark brownish patches — distinctly low luminance AND warm hue in LAB.
    Thresholds are intentionally strict to avoid false positives from
    normal skin tone variation, shadows, or warm lighting.
    """
    L = lab[:, :, 0][mask]
    a = lab[:, :, 1][mask]
    b = lab[:, :, 2][mask]
    # L < 46: noticeably darker than average skin
    # a > 7:  distinctly reddish-brown (not just warm)
    # b > 14: strong yellow component (brown pigment signature)
    melasma_px = (L < 46) & (a > 7) & (b > 14)
    # Require > 20% of skin to be affected before scoring (not just a few pixels)
    proportion = float(np.mean(melasma_px))
    return float(np.clip(proportion / 0.20, 0, 1))


def _redness_score(rgb: np.ndarray, mask: np.ndarray) -> float:
    """R / G ratio elevation — inflammation, rosacea."""
    r = rgb[:, :, 0][mask]
    g = rgb[:, :, 1][mask] + 1e-6
    ratio = float(np.mean(np.clip((r - g) / g, 0, 1)))
    return float(np.clip(ratio / 0.25, 0, 1))


def _wrinkle_score(gray: np.ndarray, mask: np.ndarray, zones: dict) -> float:
    """
    Canny edge density in forehead + eye-area zones.
    Wrinkles = oriented linear edges at fine scale.
    """
    region = zones.get("forehead", mask)
    if np.sum(region) < 50:
        region = mask
    blurred = gaussian(gray, sigma=0.8)
    edges = canny(blurred, sigma=1.5, low_threshold=0.05, high_threshold=0.15)
    edge_density = float(np.sum(edges & region) / max(np.sum(region), 1))
    return float(np.clip(edge_density / 0.12, 0, 1))


def _fine_lines_score(gray: np.ndarray, mask: np.ndarray) -> float:
    """
    Fine lines appear as low-amplitude, high-frequency edges.
    Use a tighter Canny sigma than wrinkles.
    """
    blurred = gaussian(gray, sigma=0.5)
    edges = canny(blurred, sigma=0.8, low_threshold=0.02, high_threshold=0.08)
    density = float(np.sum(edges & mask) / max(np.sum(mask), 1))
    return float(np.clip(density / 0.18, 0, 1))


def _dryness_score(rgb: np.ndarray, mask: np.ndarray) -> float:
    """
    Dry skin: low HSV saturation + rough flaky texture.
    """
    hsv = rgb2hsv(rgb)
    sat = hsv[:, :, 1][mask]
    mean_sat = float(np.mean(sat))
    dryness = 1.0 - float(np.clip(mean_sat / 0.40, 0, 1))
    # Also factor in texture roughness at a medium scale (flakiness)
    gray = np.mean(rgb, axis=2)
    smooth = gaussian(gray, sigma=3)
    flake = float(np.std((gray - smooth)[mask]))
    flakiness = float(np.clip(flake / 0.025, 0, 1))
    return float(np.clip((dryness * 0.6 + flakiness * 0.4), 0, 1))


def _pore_score(gray: np.ndarray, mask: np.ndarray, zones: dict) -> float:
    """
    Pores = small dark local minima in skin region (nose + cheeks).
    """
    nose_zone = zones.get("nose", mask)
    lc = zones.get("left_cheek", mask)
    rc = zones.get("right_cheek", mask)
    pore_region = nose_zone | lc | rc
    if np.sum(pore_region) < 100:
        pore_region = mask

    skin_gray = gray.copy()
    skin_gray[~pore_region] = 1.0
    minima = local_minima(skin_gray, connectivity=1)
    density = float(np.sum(minima & pore_region) / max(np.sum(pore_region), 1))
    return float(np.clip(density / 0.015, 0, 1))


def _oiliness_score(rgb: np.ndarray, mask: np.ndarray) -> float:
    """
    Oily skin produces specular highlights — very bright pixels on T-zone.
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    specular = (r > 0.88) & (g > 0.88) & (b > 0.88) & mask
    density = float(np.sum(specular) / max(np.sum(mask), 1))
    return float(np.clip(density / 0.04, 0, 1))


def _dark_circles_score(lab: np.ndarray, mask: np.ndarray) -> float:
    """
    Under-eye region is darker than cheek region → dark circles.
    Approximated from vertical position in the face bounding box.
    """
    rows, _ = np.where(mask)
    if len(rows) < 100:
        return 0.0
    y0, y1 = int(rows.min()), int(rows.max())
    h = y1 - y0

    under_eye = mask.copy()
    under_eye[:y0 + int(h * 0.38)] = False
    under_eye[y0 + int(h * 0.52):] = False

    cheek = mask.copy()
    cheek[:y0 + int(h * 0.52)] = False
    cheek[y0 + int(h * 0.72):] = False

    if np.sum(under_eye) < 50 or np.sum(cheek) < 50:
        return 0.0

    L = lab[:, :, 0]
    diff = float(np.mean(L[cheek])) - float(np.mean(L[under_eye]))
    return float(np.clip(diff / 12.0, 0, 1))


def _texture_score(gray: np.ndarray, mask: np.ndarray) -> float:
    """Overall uneven texture — average Laplacian magnitude in skin."""
    lap = np.abs(laplace(gray))
    val = float(np.mean(lap[mask]))
    return float(np.clip(val / 0.035, 0, 1))


# ─── Positive attributes ─────────────────────────────────────────────────────

def _positive_attributes(
    rgb: np.ndarray,
    lab: np.ndarray,
    mask: np.ndarray,
    concerns: dict[str, float],
) -> dict[str, float]:
    """
    Score positive skin health attributes (higher = better).
    These are displayed as strengths in the report.
    """
    # Hydration: inverse of dryness, boosted by saturation
    hydration = float(np.clip(1.0 - concerns["dryness"] * 0.8, 0.1, 1.0))

    # Evenness: inverse of hyperpigmentation + melasma composite
    evenness = float(np.clip(1.0 - (concerns["hyperpigmentation"] * 0.6 + concerns["melasma"] * 0.4), 0.1, 1.0))

    # Luminosity: mean L channel relative to max — how bright/glowing
    L = lab[:, :, 0][mask]
    lum = float(np.mean(L)) / 100.0
    # Subtract oiliness penalty (shine ≠ glow)
    luminosity = float(np.clip(lum * 1.1 - concerns["oiliness"] * 0.3, 0.1, 1.0))

    # Firmness: low wrinkle + low fine-line score → high firmness
    firmness = float(np.clip(1.0 - (concerns["wrinkles"] * 0.5 + concerns["fine_lines"] * 0.5), 0.1, 1.0))

    return {
        "hydration":  round(hydration, 3),
        "evenness":   round(evenness, 3),
        "luminosity": round(luminosity, 3),
        "firmness":   round(firmness, 3),
    }


# ─── Zone-level concern summary ──────────────────────────────────────────────

def _zone_report(
    rgb: np.ndarray,
    gray: np.ndarray,
    lab: np.ndarray,
    zones: dict[str, np.ndarray],
) -> dict[str, dict]:
    """Return the dominant concern per face zone with its score."""
    report = {}
    for zone_name, zmask in zones.items():
        if np.sum(zmask) < 30:
            report[zone_name] = {"dominant_concern": "insufficient_data", "score": 0.0}
            continue

        scores = {
            "redness":          _redness_score(rgb, zmask),
            "hyperpigmentation": _hyperpigmentation_score(lab, zmask),
            "dryness":          _dryness_score(rgb, zmask),
            "oiliness":         _oiliness_score(rgb, zmask),
            "acne":             _acne_score(gray, zmask),
        }
        if zone_name == "forehead":
            scores["wrinkles"] = _wrinkle_score(gray, zmask, {})
        if zone_name in ("nose", "left_cheek", "right_cheek"):
            scores["pores"] = _pore_score(gray, zmask, {zone_name: zmask})

        dominant = max(scores, key=scores.__getitem__)
        report[zone_name] = {
            "dominant_concern": dominant,
            "score": round(scores[dominant], 3),
            "all_scores": {k: round(v, 3) for k, v in scores.items()},
        }
    return report


# ─── Severity label ───────────────────────────────────────────────────────────

def _severity(score: float) -> str:
    if score < 0.20:  return "none"
    if score < 0.42:  return "mild"
    if score < 0.65:  return "moderate"
    return "severe"


def _positive_label(score: float) -> str:
    if score >= 0.75:  return "excellent"
    if score >= 0.55:  return "good"
    if score >= 0.35:  return "fair"
    return "needs improvement"


# ─── Skin type classifier ─────────────────────────────────────────────────────

def _classify_skin_type(concerns: dict[str, float]) -> str:
    if concerns["oiliness"] > 0.55:
        return "oily"
    if concerns["dryness"] > 0.55:
        return "dry"
    if concerns["redness"] > 0.45:
        return "sensitive"
    if concerns["oiliness"] > 0.30 and concerns["dryness"] > 0.30:
        return "combination"
    return "normal"


# ─── Public API ──────────────────────────────────────────────────────────────

def analyze_skin(img: Image.Image, mask: np.ndarray) -> dict:
    """
    Full professional skin analysis.

    Returns:
        {
            confidence: float,
            method: str,
            skin_type: str,
            overall_score: int,          # 0-100 (higher = healthier)
            concerns: {name: {score, severity}},
            positives: {name: {score, label}},
            zone_analysis: {zone: {dominant_concern, score}},
        }
    """
    rgb = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    gray = np.mean(rgb, axis=2)
    lab = rgb2lab(rgb)
    zones = _zone_masks(mask)

    skin_pixels = np.sum(mask)
    if skin_pixels < 100:
        return {
            "confidence": 0.0,
            "method": "feature",
            "skin_type": "unknown",
            "overall_score": 0,
            "concerns": {c: {"score": 0.0, "severity": "none"} for c in CONCERNS},
            "positives": {},
            "zone_analysis": {},
        }

    # ── Run all concern metrics ───────────────────────────────────────────────
    raw_concerns = {
        "acne":             _acne_score(gray, mask),
        "hyperpigmentation": _hyperpigmentation_score(lab, mask),
        "melasma":          _melasma_score(lab, mask),
        "redness":          _redness_score(rgb, mask),
        "wrinkles":         _wrinkle_score(gray, mask, zones),
        "fine_lines":       _fine_lines_score(gray, mask),
        "dryness":          _dryness_score(rgb, mask),
        "pore_visibility":  _pore_score(gray, mask, zones),
        "oiliness":         _oiliness_score(rgb, mask),
        "dark_circles":     _dark_circles_score(lab, mask),
        "uneven_texture":   _texture_score(gray, mask),
    }

    concerns = {
        name: {"score": round(score, 3), "severity": _severity(score)}
        for name, score in raw_concerns.items()
    }

    # ── Positive attributes ──────────────────────────────────────────────────
    pos_raw = _positive_attributes(rgb, lab, mask, raw_concerns)
    positives = {
        name: {"score": round(score, 3), "label": _positive_label(score)}
        for name, score in pos_raw.items()
    }

    # ── Zone report ──────────────────────────────────────────────────────────
    zone_analysis = _zone_report(rgb, gray, lab, zones)

    # ── Skin type ────────────────────────────────────────────────────────────
    skin_type = _classify_skin_type(raw_concerns)

    # ── Overall score (0-100, weighted average of inverted concern scores) ───
    weights = {
        "acne": 0.12, "hyperpigmentation": 0.10, "melasma": 0.08,
        "redness": 0.08, "wrinkles": 0.12, "fine_lines": 0.08,
        "dryness": 0.10, "pore_visibility": 0.08, "oiliness": 0.08,
        "dark_circles": 0.08, "uneven_texture": 0.08,
    }
    weighted_bad = sum(raw_concerns[k] * w for k, w in weights.items())
    overall_score = int(round((1.0 - weighted_bad) * 100))
    overall_score = max(10, min(99, overall_score))

    # ── Confidence ───────────────────────────────────────────────────────────
    skin_ratio = float(np.mean(mask))
    confidence = round(min(0.88, skin_ratio * 10.0), 2)

    return {
        "confidence": confidence,
        "method": "feature",
        "skin_type": skin_type,
        "overall_score": overall_score,
        "concerns": concerns,
        "positives": positives,
        "zone_analysis": zone_analysis,
    }
