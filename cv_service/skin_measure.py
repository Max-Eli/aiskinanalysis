"""
Objective pixel-level skin measurements.
These are passed to Gemini Vision as hard data to anchor its scoring —
not used directly as final scores, but as calibration inputs.
"""

from __future__ import annotations
import numpy as np
from PIL import Image
import cv2
from skimage.color import rgb2lab
from skimage.filters import gaussian, laplace
from skimage.feature import canny


def _zone_masks(mask: np.ndarray) -> dict[str, np.ndarray]:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return {}
    h, w = mask.shape
    y0, y1 = int(rows.min()), int(rows.max())
    x0, x1 = int(cols.min()), int(cols.max())
    fh, fw = y1 - y0, x1 - x0

    def z(ys, ye, xs=0.0, xe=1.0):
        m = mask.copy()
        m[:y0 + int(fh * ys)] = False
        m[y0 + int(fh * ye):] = False
        m[:, :x0 + int(fw * xs)] = False
        m[:, x0 + int(fw * xe):] = False
        return m

    return {
        "forehead":    z(0.00, 0.28),
        "left_cheek":  z(0.32, 0.68, 0.0,  0.48),
        "right_cheek": z(0.32, 0.68, 0.52, 1.0),
        "nose":        z(0.30, 0.65, 0.35, 0.65),
        "chin":        z(0.72, 1.00),
        "under_eye":   z(0.35, 0.52),
    }


def measure_skin(img: Image.Image, mask: np.ndarray) -> dict:
    """
    Returns a dict of objective pixel measurements.
    All values are calibrated to meaningful ranges documented inline.
    """
    if np.sum(mask) < 100:
        return {}

    rgb  = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    gray = np.mean(rgb, axis=2)
    lab  = rgb2lab(rgb)
    L    = lab[:, :, 0]
    zones = _zone_masks(mask)

    out: dict = {}

    # ── 1. Redness index (R/G ratio − 1) ─────────────────────────────────────
    # Interpretation: >0.05 = slight, >0.12 = moderate, >0.20 = significant
    r = rgb[:, :, 0]
    g = rgb[:, :, 1] + 1e-6
    rg = (r / g)[mask]
    out["redness_index"] = round(float(np.mean(rg)) - 1.0, 3)

    zone_redness = {}
    for zn in ["forehead", "left_cheek", "right_cheek", "nose", "chin"]:
        zm = zones.get(zn)
        if zm is not None and np.sum(zm) > 50:
            zone_redness[zn] = round(float(np.mean((r / g)[zm])) - 1.0, 3)
    out["zone_redness"] = zone_redness

    # ── 2. Pigmentation variance (L* std dev) ─────────────────────────────────
    # Interpretation: <8 = even, 8–14 = minor unevenness, >14 = notable, >20 = significant
    out["pigmentation_std"] = round(float(np.std(L[mask])), 2)

    # ── 3. Dark spot count (blob detection on L channel) ─────────────────────
    # Detects discrete dark circular regions (spots, post-acne marks, freckles)
    mean_L  = float(np.mean(L[mask]))
    dark_px = ((L < mean_L * 0.80) & mask).astype(np.uint8) * 255
    n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(dark_px, connectivity=8)
    spots = [i for i in range(1, n_lbl) if 8 <= stats[i, cv2.CC_STAT_AREA] <= 600]
    out["dark_spot_count"]        = len(spots)
    out["dark_spot_area_pct"]     = round(
        sum(stats[i, cv2.CC_STAT_AREA] for i in spots) / max(int(np.sum(mask)), 1) * 100, 2
    )

    # ── 4. Red spot count (papules, pimples) ─────────────────────────────────
    # High redness AND medium darkness = inflammatory lesion
    red_lesion = ((rg > 1.25) if len(rg) else np.zeros(1, dtype=bool))
    red_px_map = np.zeros(mask.shape, dtype=bool)
    red_px_map[mask] = (r / g)[mask] > 1.25
    red_px_map = red_px_map.astype(np.uint8) * 255
    n_rl, _, rs, _ = cv2.connectedComponentsWithStats(red_px_map, connectivity=8)
    red_spots = [i for i in range(1, n_rl) if 6 <= rs[i, cv2.CC_STAT_AREA] <= 400]
    out["red_spot_count"]    = len(red_spots)
    out["red_spot_area_pct"] = round(
        sum(rs[i, cv2.CC_STAT_AREA] for i in red_spots) / max(int(np.sum(mask)), 1) * 100, 2
    )

    # ── 5. Dark circles (under-eye luminance vs cheek luminance) ─────────────
    # Interpretation: >4 L* = mild, >8 = moderate, >12 = significant
    ue = zones.get("under_eye")
    lc = zones.get("left_cheek")
    rc = zones.get("right_cheek")
    if ue is not None and np.sum(ue) > 50 and lc is not None and rc is not None:
        cheek_L    = float(np.mean(L[lc | rc])) if np.sum(lc | rc) > 50 else mean_L
        under_L    = float(np.mean(L[ue]))
        out["dark_circle_delta"] = round(cheek_L - under_L, 2)
    else:
        out["dark_circle_delta"] = 0.0

    # ── 6. Oiliness (specular highlight %) ───────────────────────────────────
    # Interpretation: >1.5% = slight, >4% = moderate, >8% = oily
    specular = (rgb[:, :, 0] > 0.88) & (rgb[:, :, 1] > 0.88) & (rgb[:, :, 2] > 0.88) & mask
    out["oiliness_pct"] = round(float(np.sum(specular) / max(int(np.sum(mask)), 1) * 100), 2)

    # ── 7. Dryness proxy (low saturation mean) ────────────────────────────────
    # Using HSV saturation: <0.15 avg = dry/dull, >0.30 = well-hydrated
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] /= 255.0
    out["mean_saturation"] = round(float(np.mean(hsv[:, :, 1][mask])), 3)

    # ── 8. Texture roughness (mean |Laplacian| in skin) ───────────────────────
    # Interpretation: <0.010 = smooth, 0.010–0.020 = moderate, >0.020 = rough
    lap = np.abs(laplace(gray))
    out["texture_roughness"] = round(float(np.mean(lap[mask])), 4)

    # ── 9. Edge density per zone (fine lines / wrinkles) ─────────────────────
    # Interpretation: <0.04 = smooth, 0.04–0.08 = some lines, >0.08 = notable
    blurred = gaussian(gray, sigma=0.8)
    edges   = canny(blurred, sigma=1.5, low_threshold=0.05, high_threshold=0.15)
    edge_d  = {}
    for zn in ["forehead", "left_cheek", "right_cheek", "chin"]:
        zm = zones.get(zn)
        if zm is not None and np.sum(zm) > 50:
            edge_d[zn] = round(float(np.sum(edges & zm) / max(int(np.sum(zm)), 1)), 4)
    out["edge_density"] = edge_d

    # ── 10. Pore density proxy (local minima density on nose / T-zone) ────────
    # Interpretation: <0.008 = fine pores, 0.008–0.018 = visible, >0.018 = enlarged
    nose_z = zones.get("nose")
    if nose_z is not None and np.sum(nose_z) > 100:
        nose_gray = gray.copy()
        nose_gray[~nose_z] = 1.0
        from skimage.morphology import local_minima
        minima = local_minima(nose_gray, connectivity=1)
        out["pore_density"] = round(float(np.sum(minima & nose_z) / max(int(np.sum(nose_z)), 1)), 4)
    else:
        out["pore_density"] = 0.0

    return out
