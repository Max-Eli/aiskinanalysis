"""
CV Service — FastAPI

POST /analyze
  Body: { "image": "<base64 data URL>" }
  Returns: structured skin analysis JSON
"""

from __future__ import annotations
import base64
import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from image_quality import check_quality
from face_segment import get_skin_mask
from skin_analyze import analyze_skin

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm-up: pre-load models on startup so first request is fast
    log.info("Loading SegFormer face-parsing model…")
    try:
        from face_segment import _load_model
        _load_model()
        log.info("SegFormer loaded.")
    except Exception as e:
        log.warning(f"Model warm-up failed (will retry on first request): {e}")
    yield


app = FastAPI(title="Skin CV Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    image: str   # data URL: "data:image/jpeg;base64,..."


def _decode_image(data_url: str) -> Image.Image:
    try:
        header, b64 = data_url.split(",", 1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image data URL format.")
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    # 1. Decode
    try:
        img = _decode_image(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Image quality gate
    quality = check_quality(img)
    if quality["quality"] == "poor":
        return {
            "image_quality": "poor",
            "quality_issues": quality["issues"],
            "confidence": 0.0,
            "concerns": None,
            "notes": quality["issues"],
        }

    # 3. Skin-region segmentation
    try:
        skin_mask = get_skin_mask(img)
    except Exception as e:
        log.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail="Segmentation failed.")

    skin_coverage = float(skin_mask.mean())
    if skin_coverage < 0.05:
        return {
            "image_quality": "poor",
            "quality_issues": ["No face detected in the image."],
            "confidence": 0.0,
            "concerns": None,
            "notes": ["Please ensure your face is clearly visible and try again."],
        }

    # 4. Skin analysis (EfficientNet-B0 or feature-based fallback)
    try:
        result = analyze_skin(img, skin_mask)
    except Exception as e:
        log.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed.")

    # 5. Build clinical notes from results
    notes = _build_notes(result["concerns"])

    return {
        "image_quality": quality["quality"],
        "confidence": result["confidence"],
        "analysis_method": result["method"],
        "skin_type": result["skin_type"],
        "overall_score": result["overall_score"],
        "concerns": result["concerns"],
        "positives": result["positives"],
        "zone_analysis": result["zone_analysis"],
        "notes": notes,
    }


def _build_notes(concerns: dict) -> list[str]:
    notes = []
    severity_map = {
        "acne": {
            "mild": "Mild acne patterns detected — likely comedones or small papules.",
            "moderate": "Visible inflammatory acne detected across facial zones.",
            "severe": "Significant acne lesions detected — professional treatment recommended.",
        },
        "hyperpigmentation": {
            "mild": "Slight uneven skin tone detected.",
            "moderate": "Visible uneven pigmentation on cheek or forehead areas.",
            "severe": "Significant dark patches or post-inflammatory pigmentation detected.",
        },
        "melasma": {
            "mild": "Possible early melasma patterns detected in cheek/forehead zones.",
            "moderate": "Moderate melasma-like pigmentation detected.",
            "severe": "Prominent melasma patches detected — sun protection is critical.",
        },
        "redness": {
            "mild": "Mild skin redness or flushing detected.",
            "moderate": "Moderate redness or inflammation visible — possible rosacea or irritation.",
            "severe": "Significant redness detected — may indicate active inflammation.",
        },
    }
    for name, data in concerns.items():
        sev = data["severity"]
        if sev != "none" and name in severity_map and sev in severity_map[name]:
            notes.append(severity_map[name][sev])
    return notes


@app.get("/health")
async def health():
    return {"status": "ok"}
