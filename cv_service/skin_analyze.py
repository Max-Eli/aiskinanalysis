"""
Skin analysis via Gemini 2.0 Flash Vision.
MediaPipe provides the face crop; Gemini scores 11 concern metrics.
Falls back to neutral zeros if the API call fails.
"""

from __future__ import annotations
import os
import json
import re
import logging
import numpy as np
from PIL import Image
import google.generativeai as genai

log = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

CONCERNS = [
    "acne", "hyperpigmentation", "melasma", "redness",
    "wrinkles", "fine_lines", "dryness", "pore_visibility",
    "oiliness", "dark_circles", "uneven_texture",
]

POSITIVES = ["hydration", "evenness", "luminosity", "firmness"]

_PROMPT = """You are a professional dermatology AI trained to assess skin from photographs.
Analyze this face image and return a JSON skin assessment.

Return ONLY valid JSON — no markdown, no explanation — with this exact structure:
{
  "skin_type": "<oily|dry|combination|normal|sensitive>",
  "overall_score": <integer 0-100, higher = healthier>,
  "confidence": <float 0.0-1.0, how clearly visible the skin is>,
  "concerns": {
    "acne": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"},
    "hyperpigmentation": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"},
    "melasma": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"},
    "redness": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"},
    "wrinkles": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"},
    "fine_lines": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"},
    "dryness": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"},
    "pore_visibility": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"},
    "oiliness": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"},
    "dark_circles": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"},
    "uneven_texture": {"score": <0.0-1.0>, "severity": "<none|mild|moderate|severe>"}
  },
  "positives": {
    "hydration": {"score": <0.0-1.0>, "label": "<needs improvement|fair|good|excellent>"},
    "evenness": {"score": <0.0-1.0>, "label": "<needs improvement|fair|good|excellent>"},
    "luminosity": {"score": <0.0-1.0>, "label": "<needs improvement|fair|good|excellent>"},
    "firmness": {"score": <0.0-1.0>, "label": "<needs improvement|fair|good|excellent>"}
  },
  "zone_analysis": {
    "forehead": {"dominant_concern": "<concern_name>", "score": <0.0-1.0>},
    "left_cheek": {"dominant_concern": "<concern_name>", "score": <0.0-1.0>},
    "right_cheek": {"dominant_concern": "<concern_name>", "score": <0.0-1.0>},
    "nose": {"dominant_concern": "<concern_name>", "score": <0.0-1.0>},
    "chin": {"dominant_concern": "<concern_name>", "score": <0.0-1.0>}
  }
}

Scoring rules:
- score 0.00–0.19 = none severity
- score 0.20–0.41 = mild severity
- score 0.42–0.64 = moderate severity
- score 0.65–1.00 = severe severity
- Severity label MUST match the score range above.
- Be objective. If the skin looks healthy, reflect that — do not invent concerns.
- overall_score: 100 = perfect skin, 0 = severe issues across all metrics.
- dominant_concern in zone_analysis must be one of the 11 concern keys above."""

_model = None


def _get_model() -> genai.GenerativeModel:
    global _model
    if _model is None:
        genai.configure(api_key=GEMINI_API_KEY)
        _model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=genai.types.GenerationConfig(temperature=0.1),
        )
    return _model


def _crop_face(img: Image.Image, mask: np.ndarray) -> Image.Image:
    """Crop to face bounding box with 10% padding."""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return img
    h, w = mask.shape
    y0, y1 = int(rows.min()), int(rows.max())
    x0, x1 = int(cols.min()), int(cols.max())
    pad_y = int((y1 - y0) * 0.10)
    pad_x = int((x1 - x0) * 0.10)
    y0 = max(0, y0 - pad_y)
    y1 = min(h, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(w, x1 + pad_x)
    return img.crop((x0, y0, x1, y1))


def _resize(img: Image.Image, max_side: int = 1024) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _parse(text: str) -> dict:
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def _fallback() -> dict:
    return {
        "confidence": 0.0,
        "method": "fallback",
        "skin_type": "unknown",
        "overall_score": 0,
        "concerns": {c: {"score": 0.0, "severity": "none"} for c in CONCERNS},
        "positives": {p: {"score": 0.5, "label": "fair"} for p in POSITIVES},
        "zone_analysis": {},
    }


def analyze_skin(img: Image.Image, mask: np.ndarray) -> dict:
    """Analyze skin using Gemini 2.0 Flash Vision."""
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not set")
        return _fallback()

    try:
        face_img = _resize(_crop_face(img, mask))
        model = _get_model()
        response = model.generate_content([_PROMPT, face_img])
        raw = _parse(response.text)

        return {
            "confidence": float(raw.get("confidence", 0.75)),
            "method": "gemini-vision",
            "skin_type": raw.get("skin_type", "normal"),
            "overall_score": int(raw.get("overall_score", 70)),
            "concerns": raw.get("concerns", {c: {"score": 0.0, "severity": "none"} for c in CONCERNS}),
            "positives": raw.get("positives", {p: {"score": 0.5, "label": "fair"} for p in POSITIVES}),
            "zone_analysis": raw.get("zone_analysis", {}),
        }
    except Exception as e:
        log.error(f"Gemini vision analysis failed: {e}")
        return _fallback()
